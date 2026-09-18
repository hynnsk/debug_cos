# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Learnable pieces of the V-JEPA 2.1 representation-alignment (REPA) loss.

REPA (Yu et al., "Representation Alignment for Generation: Training Diffusion Transformers Is Easier
Than You Think", 2024) regresses an intermediate diffusion-transformer hidden state, through a small MLP,
onto the patch features of a frozen self-supervised encoder and maximizes their cosine similarity.
Here the student is the Cosmos3 MoT generation pathway (video tokens at one decoder layer) and the
teacher is the V-JEPA 2.1 ``ema_encoder``. This module holds

* :class:`RepaProjector` -- the REPA 3-layer SiLU MLP (student side).
* three *target adapters* that bring the teacher token grid (per camera view ``T_t x H_p x W_p``, e.g.
  ``8x16x16`` for 16 frames at 256px) onto the MoT video-token grid of that view (e.g. ``4x5x5``):

  - ``"avgpool"``      (variant 1): parameter-free temporal pair averaging + spatial adaptive average pooling
    (one ``adaptive_avg_pool3d`` call; for 8->4 frames the temporal bins are exactly consecutive tubelet
    pairs, for 16->5 patches the spatial bins are the usual overlapping adaptive windows).
  - ``"avgpool_conv"`` (variant 2): variant 1 followed by a depthwise ``Conv3d`` and a ``1x1x1`` ``Conv3d``;
    both are initialized to the identity, so at step 0 the target equals variant 1 and the tiny
    (``D*k^3 + D^2``) parameter set can learn a spatio-temporal re-alignment.
  - ``"strided_conv"`` (variant 3): a single strided (depthwise by default) ``Conv3d`` whose kernel/stride
    are derived from ``(in_grid, out_grid)`` so its windows coincide with the adaptive-pooling bins, then a
    ``1x1x1`` ``Conv3d``. Initialized to box averaging + identity, i.e. also equal to variant 1 at step 0.

The teacher itself is frozen and kept outside the FSDP-wrapped network (see ``vjepa_teacher.py``); the
adapters/projector live inside ``Cosmos3VFMNetwork`` so FSDP, EMA, checkpointing and the optimizer treat
them like any other head (their parameter names carry the ``repa_`` prefix used by ``keys_to_select`` /
``keys_to_skip_loading``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_ADAPTER_VARIANTS: tuple[str, ...] = ("avgpool", "avgpool_conv", "strided_conv")

Grid3 = tuple[int, int, int]


# --------------------------------------------------------------------------------------------------
# DTensor-aware structured initialization
# --------------------------------------------------------------------------------------------------
def copy_full_tensor_into_param(param: torch.Tensor, full: torch.Tensor) -> None:
    """Write ``full`` (the complete, unsharded value) into ``param`` in place.

    ``Cosmos3VFMNetwork.init_weights`` runs after ``fully_shard`` + ``to_empty``, so the head parameters
    are FSDP2 ``DTensor`` shards. Random ``torch.nn.init`` calls work on those directly, but a structured
    (identity / box-filter) init needs the global offset of the local shard, which is what this helper
    resolves. Plain tensors (CPU tests, no FSDP) take the direct path.
    """
    if tuple(param.shape) != tuple(full.shape):
        raise ValueError(f"Shape mismatch: param {tuple(param.shape)} vs full {tuple(full.shape)}")
    with torch.no_grad():
        try:
            from torch.distributed.tensor import DTensor
        except ImportError:  # pragma: no cover - very old torch
            DTensor = ()  # type: ignore[assignment]
        if isinstance(param, DTensor):
            from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

            local = param.to_local()
            local_shape, global_offset = compute_local_shape_and_global_offset(
                tuple(param.shape), param.device_mesh, param.placements
            )
            slices = tuple(slice(o, o + s) for o, s in zip(global_offset, local_shape))
            piece = full[slices]
            if tuple(piece.shape) != tuple(local.shape):
                raise RuntimeError(
                    f"Local shard shape {tuple(local.shape)} does not match the computed slice {tuple(piece.shape)}"
                )
            local.copy_(piece.to(device=local.device, dtype=local.dtype))
        else:
            param.copy_(full.to(device=param.device, dtype=param.dtype))


def _identity_pointwise_conv3d_weight(dim: int) -> torch.Tensor:
    return torch.eye(dim, dtype=torch.float32).view(dim, dim, 1, 1, 1)


def _identity_depthwise_conv3d_weight(dim: int, kernel: Grid3) -> torch.Tensor:
    w = torch.zeros(dim, 1, *kernel, dtype=torch.float32)
    w[:, 0, kernel[0] // 2, kernel[1] // 2, kernel[2] // 2] = 1.0
    return w


def _box_depthwise_conv3d_weight(dim: int, kernel: Grid3) -> torch.Tensor:
    return torch.full((dim, 1, *kernel), 1.0 / math.prod(kernel), dtype=torch.float32)


def _box_full_conv3d_weight(dim: int, kernel: Grid3) -> torch.Tensor:
    """Full (non-depthwise) conv that averages each input channel onto the same output channel."""
    w = torch.zeros(dim, dim, *kernel, dtype=torch.float32)
    idx = torch.arange(dim)
    w[idx, idx] = 1.0 / math.prod(kernel)
    return w


# --------------------------------------------------------------------------------------------------
# Student-side projector (REPA MLP)
# --------------------------------------------------------------------------------------------------
class RepaProjector(nn.Module):
    """REPA projection head: ``Linear -> SiLU -> Linear -> SiLU -> Linear`` (``build_mlp`` in the REPA code)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.SiLU()

    def reset_parameters(self) -> None:
        # torch's default Linear init (kaiming-uniform weights, uniform biases); works on DTensor shards.
        for layer in (self.fc1, self.fc2, self.fc3):
            layer.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [N,in_dim] -> [N,out_dim]
        return self.fc3(self.act(self.fc2(self.act(self.fc1(x)))))


# --------------------------------------------------------------------------------------------------
# Teacher-side target adapters
# --------------------------------------------------------------------------------------------------
def adaptive_pool_teacher_grid(x: torch.Tensor, out_grid: Grid3) -> torch.Tensor:
    """Variant-1 pooling. ``x``: ``[B,D,T_t,H_p,W_p]`` -> ``[B,D,T,H,W]`` (computed in float32)."""
    if x.ndim != 5:
        raise ValueError(f"Expected [B,D,T,H,W] teacher tokens, got shape {tuple(x.shape)}")
    grid = tuple(int(v) for v in out_grid)
    if x.is_cuda:
        # The CUDA kernel accumulates in float32 for bf16/fp16 inputs, so pool in the native dtype: a float32 copy of
        # the full teacher tensor (e.g. 256 clips x 2048 tokens x 768 = 1.6 GB) is what pushed the 2-GPU recipe over
        # the 44 GiB limit.
        return F.adaptive_avg_pool3d(x, grid)
    orig_dtype = x.dtype
    return F.adaptive_avg_pool3d(x.float(), grid).to(orig_dtype)


class AvgPoolTargetAdapter(nn.Module):
    """Variant 1: temporal average of tubelet pairs + spatial adaptive average pooling. No parameters."""

    def reset_parameters(self) -> None:  # noqa: D401 - nothing to initialize
        return None

    def forward(self, x: torch.Tensor, out_grid: Grid3) -> torch.Tensor:
        return adaptive_pool_teacher_grid(x, out_grid)


class AvgPoolConvTargetAdapter(nn.Module):
    """Variant 2: variant-1 pooling -> depthwise Conv3d (k^3) -> 1x1x1 Conv3d, both identity-initialized."""

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd so the identity kernel has a center, got {kernel_size}")
        self.dim = dim
        self.kernel_size = kernel_size
        self.dwconv = nn.Conv3d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=True)
        self.pwconv = nn.Conv3d(dim, dim, 1, bias=True)

    def reset_parameters(self) -> None:
        k = (self.kernel_size,) * 3
        copy_full_tensor_into_param(self.dwconv.weight, _identity_depthwise_conv3d_weight(self.dim, k))
        copy_full_tensor_into_param(self.pwconv.weight, _identity_pointwise_conv3d_weight(self.dim))
        with torch.no_grad():
            self.dwconv.bias.zero_()
            self.pwconv.bias.zero_()

    def forward(self, x: torch.Tensor, out_grid: Grid3) -> torch.Tensor:
        y = adaptive_pool_teacher_grid(x, out_grid)
        y = y.to(self.dwconv.weight.dtype)
        return self.pwconv(self.dwconv(y))


def strided_conv_geometry(in_size: int, out_size: int) -> tuple[int, int]:
    """``(kernel, stride)`` of a padding-free strided conv mapping ``in_size -> out_size``.

    ``stride = in // out`` and ``kernel = in - stride * (out - 1)``; the resulting windows are exactly the
    bins of ``adaptive_avg_pool`` (e.g. 8->4: k=2,s=2; 16->5: k=4,s=3 -> [0,4),[3,7),[6,10),[9,13),[12,16)).
    """
    if in_size < out_size or out_size < 1:
        raise ValueError(f"Cannot downsample {in_size} -> {out_size} with a strided conv")
    stride = in_size // out_size
    kernel = in_size - stride * (out_size - 1)
    assert (in_size - kernel) // stride + 1 == out_size, (in_size, out_size, kernel, stride)
    return kernel, stride


class StridedConvTargetAdapter(nn.Module):
    """Variant 3: one strided Conv3d (``in_grid -> out_grid``) followed by a 1x1x1 Conv3d.

    ``depthwise=True`` (default) keeps it very small (``D * prod(kernel) + D^2`` weights); ``False`` uses a
    full ``D x D x kernel`` conv. Both start as box averaging + identity, i.e. equal to variant 1.
    """

    def __init__(self, dim: int, in_grid: Grid3, out_grid: Grid3, depthwise: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.in_grid = tuple(int(v) for v in in_grid)
        self.out_grid = tuple(int(v) for v in out_grid)
        self.depthwise = depthwise
        geom = [strided_conv_geometry(i, o) for i, o in zip(self.in_grid, self.out_grid)]
        self.kernel: Grid3 = tuple(k for k, _ in geom)  # type: ignore[assignment]
        self.stride: Grid3 = tuple(s for _, s in geom)  # type: ignore[assignment]
        self.conv = nn.Conv3d(
            dim, dim, kernel_size=self.kernel, stride=self.stride, groups=dim if depthwise else 1, bias=True
        )
        self.pwconv = nn.Conv3d(dim, dim, 1, bias=True)

    def reset_parameters(self) -> None:
        if self.depthwise:
            copy_full_tensor_into_param(self.conv.weight, _box_depthwise_conv3d_weight(self.dim, self.kernel))
        else:
            copy_full_tensor_into_param(self.conv.weight, _box_full_conv3d_weight(self.dim, self.kernel))
        copy_full_tensor_into_param(self.pwconv.weight, _identity_pointwise_conv3d_weight(self.dim))
        with torch.no_grad():
            self.conv.bias.zero_()
            self.pwconv.bias.zero_()

    def forward(self, x: torch.Tensor, out_grid: Grid3) -> torch.Tensor:
        if tuple(x.shape[2:]) != self.in_grid:
            raise ValueError(
                f"strided_conv adapter was built for teacher grid {self.in_grid} but got {tuple(x.shape[2:])}; "
                "set model.config.repa.teacher_num_frames / teacher_input_size to match the data."
            )
        if tuple(int(v) for v in out_grid) != self.out_grid:
            raise ValueError(
                f"strided_conv adapter was built for target grid {self.out_grid} but the MoT token grid of this "
                f"batch needs {tuple(out_grid)}; set model.config.repa.target_grid_thw accordingly."
            )
        y = self.conv(x.to(self.conv.weight.dtype))
        return self.pwconv(y)


def build_target_adapter(
    variant: str,
    dim: int,
    *,
    kernel_size: int = 3,
    in_grid: Grid3 = (8, 16, 16),
    out_grid: Grid3 = (4, 5, 5),
    depthwise: bool = True,
) -> nn.Module:
    if variant == "avgpool":
        return AvgPoolTargetAdapter()
    if variant == "avgpool_conv":
        return AvgPoolConvTargetAdapter(dim, kernel_size=kernel_size)
    if variant == "strided_conv":
        return StridedConvTargetAdapter(dim, in_grid=in_grid, out_grid=out_grid, depthwise=depthwise)
    raise ValueError(f"Unknown REPA target adapter {variant!r}; expected one of {TARGET_ADAPTER_VARIANTS}")


# --------------------------------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------------------------------
def repa_cosine_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """``(1 - mean cosine similarity, mean cosine similarity)`` over rows of ``[N,D]`` tensors (float32).

    REPA minimizes ``-cos``; ``1 - cos`` has the same gradient and a non-negative range that reads better on
    the loss curves. The teacher features arrive detached (frozen encoder); only the target *adapter*
    (variants 2/3) receives gradient through ``target``.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} must have the same shape")
    cos = F.cosine_similarity(pred.float(), target.float(), dim=-1, eps=eps)  # [N]
    cos_mean = cos.mean()
    return 1.0 - cos_mean, cos_mean.detach()


def concat_views_along_width(y: torch.Tensor, num_views: int) -> torch.Tensor:
    """``[B*V,D,T,H,Wv]`` (view-major batch) -> ``[B,T,H,V*Wv,D]``: the canvas layout of ``concat_view``."""
    bv, d, t, h, wv = y.shape
    if bv % num_views != 0:
        raise ValueError(f"Batch of {bv} view clips is not divisible by num_views={num_views}")
    b = bv // num_views
    y = y.view(b, num_views, d, t, h, wv).permute(0, 3, 4, 1, 5, 2)  # [B,T,H,V,Wv,D]
    return y.reshape(b, t, h, num_views * wv, d)


def split_flat_tokens(flat: torch.Tensor, token_shapes: Sequence[Sequence[int]]) -> list[torch.Tensor]:
    """Split ``[sum(T*H*W), D]`` (sample-major, then t, h, w -- the packing order) into ``[T,H,W,D]`` grids."""
    out: list[torch.Tensor] = []
    offset = 0
    for shape in token_shapes:
        t, h, w = (int(v) for v in shape)
        n = t * h * w
        out.append(flat[offset : offset + n].view(t, h, w, flat.shape[-1]))
        offset += n
    if offset != flat.shape[0]:
        raise ValueError(f"token_shapes cover {offset} tokens but {flat.shape[0]} were given")
    return out
