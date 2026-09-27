# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Memory-bounded, distributed SIGReg for MoT visual-token representations.

This implements the Epps--Pulley version of Sketched Isotropic Gaussian Regularization used by LeJEPA and
LeWorldModel. Random Cramer--Wold directions are synchronized by seed, and the empirical characteristic function
is reduced over the data-parallel group before it is compared with the characteristic function of N(0, 1).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.utils.checkpoint


def _distributed_sum(x: torch.Tensor, process_group: dist.ProcessGroup | None, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return x
    # Unlike dist.all_reduce, this functional collective has a backward definition. That is required because each
    # rank's local visual tokens contribute to the global empirical characteristic function.
    from torch.distributed.nn.functional import all_reduce

    return all_reduce(x, op=dist.ReduceOp.SUM, group=process_group)


def _epps_pulley_slice_chunk(
    x: torch.Tensor,
    directions: torch.Tensor,
    integration_points: torch.Tensor,
    target_cf: torch.Tensor,
    global_count: torch.Tensor,
    process_group: dist.ProcessGroup | None,
    world_size: int,
) -> torch.Tensor:
    """Return one Epps--Pulley statistic per column of ``directions``."""
    projected = x.float() @ directions  # [N_local, M_chunk]
    local_ecf_parts = []
    for t in integration_points:
        phase = projected * t
        local_ecf_parts.append(torch.stack((phase.cos().sum(dim=0), phase.sin().sum(dim=0)), dim=-1))
    # [M_chunk, T, 2], with real/imaginary components in the last dimension.
    global_ecf = _distributed_sum(torch.stack(local_ecf_parts, dim=1), process_group, world_size) / global_count
    error = (global_ecf[..., 0] - target_cf).square() + global_ecf[..., 1].square()
    weighted_error = error * target_cf
    return torch.trapezoid(weighted_error, integration_points, dim=1) * global_count


def sigreg_loss(
    visual_tokens: torch.Tensor,
    *,
    num_slices: int = 256,
    num_points: int = 17,
    integration_max: float = 5.0,
    slice_batch_size: int = 64,
    seed: int = 0,
    process_group: dist.ProcessGroup | None = None,
    normalize_by_count: bool = False,
) -> torch.Tensor:
    """Epps--Pulley SIGReg loss for a ``[num_visual_tokens, hidden_dim]`` tensor.

    ``normalize_by_count=False`` returns the test statistic ``N * integral(|ecf - cf|^2 w)`` (the papers' form; O(1)
    under the null, O(N) far from it). ``True`` returns the per-token integral, i.e. the statistic divided by the
    global token count ``N``: bounded (about [0, 1.8] for ``integration_max=5``) and independent of the batch size.

    The implementation is mathematically equivalent to materializing all ``[N, M, T]`` projected phases, but
    chunks the ``M`` random directions and activation-checkpoints each chunk. This is important for patch-token
    training, where ``N`` is tens of thousands even when the sample batch is only 128.
    """
    if visual_tokens.ndim != 2:
        raise ValueError(f"SIGReg expects [N,D] visual tokens, got {tuple(visual_tokens.shape)}")
    if num_slices < 1 or num_points < 2 or slice_batch_size < 1 or integration_max <= 0:
        raise ValueError(
            "SIGReg requires num_slices >= 1, num_points >= 2, slice_batch_size >= 1 and integration_max > 0"
        )

    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size(process_group) if distributed else 1
    global_count = torch.tensor(float(visual_tokens.shape[0]), device=visual_tokens.device, dtype=torch.float32)
    if world_size > 1:
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM, group=process_group)
    if global_count.item() == 0:
        return 0.0 * visual_tokens.sum()

    generator = torch.Generator(device=visual_tokens.device)
    generator.manual_seed(int(seed))
    directions = torch.randn(
        visual_tokens.shape[1], num_slices, generator=generator, device=visual_tokens.device, dtype=torch.float32
    )
    directions = directions / directions.norm(p=2, dim=0, keepdim=True).clamp_min(1.0e-12)
    integration_points = torch.linspace(
        -integration_max, integration_max, num_points, device=visual_tokens.device, dtype=torch.float32
    )
    target_cf = torch.exp(-0.5 * integration_points.square())

    statistics: list[torch.Tensor] = []
    for direction_chunk in directions.split(slice_batch_size, dim=1):

        def _chunk(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
            return _epps_pulley_slice_chunk(
                x,
                a,
                integration_points,
                target_cf,
                global_count,
                process_group,
                world_size,
            )

        if torch.is_grad_enabled() and visual_tokens.requires_grad:
            statistic = torch.utils.checkpoint.checkpoint(_chunk, visual_tokens, direction_chunk, use_reentrant=False)
        else:
            statistic = _chunk(visual_tokens, direction_chunk)
        statistics.append(statistic)
    statistic = torch.cat(statistics).mean()
    if normalize_by_count:
        statistic = statistic / global_count
    return statistic
