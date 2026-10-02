# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Learnable pieces of the V-JEPA 2.1 representation-alignment (REPA) loss.

REPA (Yu et al., "Representation Alignment for Generation: Training Diffusion Transformers Is Easier
Than You Think", 2024) regresses an intermediate diffusion-transformer hidden state, through a small MLP,
onto the patch features of a frozen self-supervised encoder and maximizes their cosine similarity.
Here the student is the Cosmos3 MoT generation pathway (video tokens at one decoder layer) and the
teacher is the V-JEPA 2.1 ``ema_encoder``. This module holds

* :class:`RepaProjector` -- the REPA 3-layer SiLU MLP (student side, ``projector_type="mlp"``) and
  :class:`RepaLinearProjector` -- a single ``Linear`` in its place (``projector_type="linear"``, the v4 recipe);
  :func:`build_projector` picks one by name.
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
import torch.utils.checkpoint
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


class RepaLinearProjector(nn.Module):
    """Linear REPA projection head: a single ``Linear(in_dim, out_dim)`` (no hidden layer, no nonlinearity).

    The aligned MoT hidden state is then constrained to be an *affine* image of the teacher features instead of an
    arbitrary MLP-reachable one, i.e. the student layer itself has to become V-JEPA-like.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def reset_parameters(self) -> None:
        self.fc.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [N,in_dim] -> [N,out_dim]
        return self.fc(x)


PROJECTOR_TYPES = ("mlp", "linear")


def build_projector(projector_type: str, in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    """``"mlp"`` -> :class:`RepaProjector` (``hidden_dim`` wide), ``"linear"`` -> :class:`RepaLinearProjector`
    (``hidden_dim`` is ignored)."""
    if projector_type == "mlp":
        return RepaProjector(in_dim, hidden_dim, out_dim)
    if projector_type == "linear":
        return RepaLinearProjector(in_dim, out_dim)
    raise ValueError(f"Unknown REPA projector_type {projector_type!r}; expected one of {PROJECTOR_TYPES}")


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


def _split_aligned_token_frames(
    x: torch.Tensor,
    num_tokens_per_sample: Sequence[int],
    frame_indexes_per_sample: Sequence[torch.Tensor],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Restore packed ``[N,D]`` aligned tokens to one ``[F,P,D]`` tensor per non-empty sample.

    REPA rows are sample-major, then frame-major, then spatial-patch-major. ``F`` is the number of selected noisy
    frames and ``P`` is inferred from the packed count. The paired frame-index tensor is returned on ``x.device``.
    """
    counts = [int(c) for c in num_tokens_per_sample]
    if len(counts) != len(frame_indexes_per_sample):
        raise ValueError(f"Got {len(counts)} token counts but {len(frame_indexes_per_sample)} frame-index tensors")
    if sum(counts) != x.shape[0]:
        raise ValueError(f"num_tokens_per_sample sums to {sum(counts)} but tensor has {x.shape[0]} rows")
    restored: list[tuple[torch.Tensor, torch.Tensor]] = []
    for rows, indexes in zip(torch.split(x, counts), frame_indexes_per_sample):
        indexes = indexes.to(device=x.device, dtype=torch.long)
        num_frames = indexes.numel()
        if rows.shape[0] == 0:
            if num_frames != 0:
                raise ValueError("A zero token count must have zero selected frame indexes")
            continue
        if num_frames == 0 or rows.shape[0] % num_frames:
            raise ValueError(f"Cannot restore {rows.shape[0]} token rows over {num_frames} selected frames")
        restored.append((rows.view(num_frames, rows.shape[0] // num_frames, x.shape[-1]), indexes))
    return restored


def repa_temporal_difference_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_tokens_per_sample: Sequence[int],
    frame_indexes_per_sample: Sequence[torch.Tensor],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align same-patch temporal transitions instead of absolute token features.

    For every genuinely consecutive selected frame pair, compares
    ``P(h[t+1,p]) - P(h[t,p])`` with ``y[t+1,p] - y[t,p]`` by cosine similarity. Non-consecutive selected frames
    are deliberately not bridged. Returns a graph-connected zero loss and zero cosine when no transition exists.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} must have the same shape")
    pred_frames = _split_aligned_token_frames(pred, num_tokens_per_sample, frame_indexes_per_sample)
    target_frames = _split_aligned_token_frames(target, num_tokens_per_sample, frame_indexes_per_sample)
    pred_deltas: list[torch.Tensor] = []
    target_deltas: list[torch.Tensor] = []
    for (p, indexes), (t, _) in zip(pred_frames, target_frames):
        consecutive = torch.nonzero(indexes[1:] == indexes[:-1] + 1, as_tuple=False).flatten()
        if consecutive.numel() == 0:
            continue
        pred_deltas.append(p.index_select(0, consecutive + 1) - p.index_select(0, consecutive))
        target_deltas.append(t.index_select(0, consecutive + 1) - t.index_select(0, consecutive))
    if not pred_deltas:
        zero = 0.0 * pred.sum()
        return zero, zero.detach()
    pred_delta = torch.cat(pred_deltas, dim=0).flatten(0, 1)
    target_delta = torch.cat(target_deltas, dim=0).flatten(0, 1)
    return repa_cosine_loss(pred_delta, target_delta, eps=eps)


def _spatial_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize each ``[P,D]`` frame independently over its spatial-patch axis."""
    x32 = x.float()
    mean = x32.mean(dim=1, keepdim=True)
    std = x32.std(dim=1, correction=0, keepdim=True)
    return (x32 - mean) / (std + eps)


def repa_spatial_normalized_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_tokens_per_sample: Sequence[int],
    frame_indexes_per_sample: Sequence[torch.Tensor],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame spatially normalize student and teacher tokens, then apply token-wise cosine loss.

    For every feature channel independently, ``SpatialNorm(x[t,p]) = (x[t,p] - mean_p(x[t,p])) /
    (std_p(x[t,p]) + eps)``. Statistics never cross frames or samples.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} must have the same shape")
    pred_frames = _split_aligned_token_frames(pred, num_tokens_per_sample, frame_indexes_per_sample)
    target_frames = _split_aligned_token_frames(target, num_tokens_per_sample, frame_indexes_per_sample)
    if not pred_frames:
        zero = 0.0 * pred.sum()
        return zero, zero.detach()
    pred_norm = torch.cat([_spatial_norm(frames, eps).flatten(0, 1) for frames, _ in pred_frames], dim=0)
    target_norm = torch.cat([_spatial_norm(frames, eps).flatten(0, 1) for frames, _ in target_frames], dim=0)
    return repa_cosine_loss(pred_norm, target_norm, eps=eps)


RELATION_DISTANCES = ("l2", "l1")
# Largest batched relation map (b x n x n entries, float32) before repa_relation_loss falls back to a per-sample loop.
RELATION_BATCHED_MAX_ENTRIES = 2**27  # 512 MB per matrix


def token_relation_matrix(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pairwise cosine similarities of the rows of ``x`` (``[..., n, D]`` -> ``[..., n, n]``, float32)."""
    xn = F.normalize(x.float(), dim=-1, eps=eps)
    return xn @ xn.transpose(-1, -2)


def repa_relation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_tokens_per_sample: Sequence[int],
    distance: str = "l2",
    eps: float = 1e-6,
    margin: float = 0.0,
    checkpoint: bool = False,
) -> torch.Tensor:
    """Token-relation distillation (VideoREPA TRD): match the *pairwise cosine-similarity map* of the projected
    student tokens to that of the teacher tokens, sample by sample.

    ``pred`` / ``target``: ``[N,D]`` rows in packing order (sample-major); ``num_tokens_per_sample`` says how many
    consecutive rows belong to each sample (``RepaAlignmentHead`` returns them as ``num_tokens_per_sample``). For
    every sample the ``n_i x n_i`` relation matrices ``R = normalize(x) normalize(x)^T`` over ALL tokens of the clip
    (spatial pairs within a frame and temporal pairs across frames alike -- the full map VideoREPA's
    ``token_relation_distillation`` uses) are compared entry-wise and averaged over all entries of all samples:

        l1: mean(relu(|R_s - R_t| - margin))        l2: mean(relu(|R_s - R_t| - margin)^2)

    ``margin`` (VideoREPA default 0.1) is the soft constraint: relation differences inside the margin are not
    penalized. ``margin = 0`` is the plain l1 / l2 distance. Unlike ``repa_cosine_loss`` this only constrains the
    *geometry* among the tokens of a video, never the absolute direction of a token, so it is invariant to any
    rotation of either feature space. ``checkpoint=True`` recomputes each sample's maps in backward (the per-sample
    path), which keeps the memory at one ``n_i x n_i`` map at a time (7200^2 / 8192^2 for the native-size recipes).
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} must have the same shape")
    if distance not in RELATION_DISTANCES:
        raise ValueError(f"Unknown relation distance {distance!r}; expected one of {RELATION_DISTANCES}")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError(f"margin must be a finite non-negative float, got {margin}")
    counts = [int(c) for c in num_tokens_per_sample]
    if sum(counts) != pred.shape[0]:
        raise ValueError(f"num_tokens_per_sample sums to {sum(counts)} but pred has {pred.shape[0]} rows")
    counts = [c for c in counts if c > 0]
    if not counts:
        raise ValueError("repa_relation_loss needs at least one sample with tokens")

    def _penalty(diff: torch.Tensor) -> torch.Tensor:
        d = diff.abs()
        if margin > 0:
            d = F.relu(d - margin)
        return d.square() if distance == "l2" else d

    def _sample_sum(p_i: torch.Tensor, t_i: torch.Tensor) -> torch.Tensor:
        return _penalty(token_relation_matrix(p_i, eps) - token_relation_matrix(t_i, eps)).sum()

    if len(set(counts)) == 1 and len(counts) * counts[0] ** 2 <= RELATION_BATCHED_MAX_ENTRIES:
        # Uniform grids (the LIBERO case): one batched matmul per side. With a target sub-grid the per-sample maps grow
        # to (200 S)^2, so beyond RELATION_BATCHED_MAX_ENTRIES the per-sample loop below is used instead (same value).
        b, n = len(counts), counts[0]
        diff = token_relation_matrix(pred.view(b, n, -1), eps) - token_relation_matrix(target.view(b, n, -1), eps)
        return _penalty(diff).sum() / (b * n * n)
    total = pred.new_zeros((), dtype=torch.float32)
    entries = 0
    use_ckpt = bool(checkpoint) and torch.is_grad_enabled() and (pred.requires_grad or target.requires_grad)
    for p_i, t_i in zip(torch.split(pred, counts), torch.split(target, counts)):
        if use_ckpt:
            total = total + torch.utils.checkpoint.checkpoint(_sample_sum, p_i, t_i, use_reentrant=False)
        else:
            total = total + _sample_sum(p_i, t_i)
        entries += p_i.shape[0] ** 2
    return total / entries


def repa_centered_cosine_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """``1 - cos`` between mean-centered student and teacher tokens.

    Each side has its own (detached) batch mean over all aligned tokens subtracted before the cosine, so a constant
    or per-position-mean prediction scores exactly ~0 and the projector is trained on the token-specific structure
    of the teacher features instead of their dominant shared direction (on LIBERO ~85% of the V-JEPA 2.1 token
    energy; a constant prediction already reads raw cos ~0.92). Centering each side by its own mean makes the loss
    invariant to a constant offset of the projector output. Same contract as :func:`repa_cosine_loss`.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} must have the same shape")
    p32, t32 = pred.float(), target.float()
    pc = p32 - p32.mean(dim=0, keepdim=True).detach()
    tc = t32 - t32.mean(dim=0, keepdim=True).detach()
    cos = F.cosine_similarity(pc, tc, dim=-1, eps=eps)  # [N]
    cos_mean = cos.mean()
    return 1.0 - cos_mean, cos_mean.detach()


@torch.no_grad()
def centered_cosine_similarity(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Diagnostic twin of :func:`repa_centered_cosine_loss`: mean cosine after each side's batch mean is removed.

    The raw cosine saturates near 1 on V-JEPA 2.1 targets because of their shared direction (a constant prediction
    already scores ~0.92, a per-position mean ~0.95) without saying how much sample-specific structure is matched;
    this value is ~0 for those shortcuts and 1 only for a true match of the token-specific part.
    """
    _, cos_mean = repa_centered_cosine_loss(pred, target, eps=eps)
    return cos_mean


def concat_views_along_width(y: torch.Tensor, num_views: int) -> torch.Tensor:
    """``[B*V,D,T,H,Wv]`` (view-major batch) -> ``[B,T,H,V*Wv,D]``: the canvas layout of ``concat_view``."""
    bv, d, t, h, wv = y.shape
    if bv % num_views != 0:
        raise ValueError(f"Batch of {bv} view clips is not divisible by num_views={num_views}")
    b = bv // num_views
    y = y.view(b, num_views, d, t, h, wv).permute(0, 3, 4, 1, 5, 2)  # [B,T,H,V,Wv,D]
    return y.reshape(b, t, h, num_views * wv, d)


class StudentTrilinearUpsampler(nn.Module):
    """Student-side "interpolation + conv" upsampler (cosmos_hs11 v12): trilinear interpolation of the MoT token grid of
    one camera view (``[N, D, T, H, W]`` -> the teacher grid ``out_grid``) followed by a depthwise 3x3x3 Conv3d that is
    identity-initialized, so at init the prediction at a teacher position is exactly the interpolated MoT feature and the
    conv can learn a local sharpening. Runs BEFORE the (per-position) REPA MLP, i.e. the MLP sees upsampled 2048-d
    features, which lets it mix the interpolated neighbours non-linearly.
    """

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.dim, self.kernel_size = int(dim), int(kernel_size)
        self.conv = nn.Conv3d(self.dim, self.dim, kernel_size=self.kernel_size, padding=self.kernel_size // 2, groups=self.dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        k = (self.kernel_size,) * 3
        copy_full_tensor_into_param(self.conv.weight, _identity_depthwise_conv3d_weight(self.dim, k))  # type: ignore[arg-type]
        copy_full_tensor_into_param(self.conv.bias, torch.zeros(self.dim))

    def forward(self, x: torch.Tensor, out_grid: Grid3) -> torch.Tensor:  # [N,D,T,H,W] -> [N,D,T_t,H_p,W_p]
        grid = tuple(int(v) for v in out_grid)
        if tuple(x.shape[2:]) != grid:
            x = F.interpolate(x.float(), size=grid, mode="trilinear", align_corners=False).to(x.dtype)
        return self.conv(x)


def repa_sample_sigmas(timesteps: torch.Tensor, num_samples: int, max_timestep: float) -> torch.Tensor:
    """Per-REPA-sample noise level ``sigma`` in [0, 1] from the vision timesteps of the training step.

    ``timesteps`` is ``[B_items, T]`` (or ``[B_items, 1]``), ``= sigma * max_timestep`` with one value per clip (all
    latent frames share it outside diffusion forcing); the per-item mean over frames is used. The REPA rows are
    sample-major in the same order as the vision items, which requires one clip per sample (LIBERO concat_view).
    """
    t = timesteps.detach().float().reshape(timesteps.shape[0], -1)
    if t.shape[0] != int(num_samples):
        raise ValueError(
            f"REPA sigma gating needs one vision item per REPA sample: got {t.shape[0]} timestep rows for "
            f"{num_samples} REPA samples (multi-clip batches are not supported)."
        )
    return (t.mean(dim=1) / float(max_timestep)).clamp_(0.0, 1.0)


def select_repa_samples(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_tokens_per_sample: Sequence[int],
    frame_indexes_per_sample: Sequence[torch.Tensor],
    keep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[torch.Tensor]]:
    """Keep only the rows of the samples flagged in ``keep`` (bool ``[B]``); returns the sub-packed rows, counts and
    frame indexes in the original sample order (so every REPA objective can be applied to the subset unchanged)."""
    counts = [int(c) for c in num_tokens_per_sample]
    if len(counts) != len(frame_indexes_per_sample) or len(counts) != int(keep.numel()):
        raise ValueError(
            f"select_repa_samples: {len(counts)} counts, {len(frame_indexes_per_sample)} frame-index sets, {int(keep.numel())} flags"
        )
    if sum(counts) != pred.shape[0] or pred.shape != target.shape:
        raise ValueError("select_repa_samples: counts do not match the packed rows")
    flags = [bool(k) for k in keep.detach().cpu().tolist()]
    p_chunks, t_chunks = torch.split(pred, counts), torch.split(target, counts)
    kept = [i for i, f in enumerate(flags) if f]
    if not kept:
        return pred[:0], target[:0], [], []
    return (
        torch.cat([p_chunks[i] for i in kept], dim=0),
        torch.cat([t_chunks[i] for i in kept], dim=0),
        [counts[i] for i in kept],
        [frame_indexes_per_sample[i] for i in kept],
    )


def concat_views_along_width_subgrid(y: torch.Tensor, num_views: int, subgrid: Grid3) -> torch.Tensor:
    """``[B*V,D,T*st,H*sh,Wv*sw]`` (view-major batch) -> ``[B,T,H,V*Wv,S,D]`` with ``S = st*sh*sw``.

    Sub-grid version of :func:`concat_views_along_width`: the adapter produced ``subgrid`` teacher cells per MoT token
    along (t, h, w); they are gathered as a trailing ``S`` axis in ``(dt, dh, dw)`` order so that flattening
    ``[..., S, D] -> [N*S, D]`` keeps rows token-major / sub-cell-minor. ``subgrid=(1,1,1)`` reproduces the plain
    layout with ``S = 1``. (Ported from cosmos_hs10, 2026-09-29.)
    """
    bv, d, tt, hh, ww = y.shape
    st, sh, sw = (int(v) for v in subgrid)
    if min(st, sh, sw) < 1 or tt % st or hh % sh or ww % sw:
        raise ValueError(f"Adapted grid {(tt, hh, ww)} is not a multiple of target_subgrid_thw={(st, sh, sw)}")
    if bv % num_views != 0:
        raise ValueError(f"Batch of {bv} view clips is not divisible by num_views={num_views}")
    b, t, h, wv = bv // num_views, tt // st, hh // sh, ww // sw
    y = y.view(b, num_views, d, t, st, h, sh, wv, sw).permute(0, 3, 5, 1, 7, 4, 6, 8, 2)  # [B,T,H,V,Wv,st,sh,sw,D]
    return y.reshape(b, t, h, num_views * wv, st * sh * sw, d)


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
