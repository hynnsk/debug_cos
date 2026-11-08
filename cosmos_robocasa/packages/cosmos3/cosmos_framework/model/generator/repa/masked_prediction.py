# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Pixel-masked, action-free teacher prediction for the Cosmos generation pathway.

The auxiliary input is rebuilt from native RGB, BEFORE resize/padding/VAE. Never
mask already encoded clean latents: their receptive fields include hidden pixels.
All operations on the teacher side are fixed and detached. See
``docs/action_policy_libero_masked_jepa.md`` for the differences from V-JEPA 2.1.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def validate_masked_prediction_config(cfg) -> None:
    if cfg.objective != "masked_prediction":
        return
    if cfg.target_adapter != "avgpool" or cfg.center_targets or cfg.relation_loss_weight != 0:
        raise ValueError(
            "masked_prediction requires fixed avgpool targets, center_targets=false, relation_loss_weight=0"
        )
    if not cfg.teacher.startswith("vjepa2_1_"):
        raise ValueError("masked_prediction requires a V-JEPA 2.1 teacher")
    if tuple(int(v) for v in getattr(cfg, "target_subgrid_thw", (1, 1, 1))) != (1, 1, 1):
        raise ValueError("masked_prediction requires target_subgrid_thw=[1,1,1] (its pixel mask is defined per MoT token)")
    if not 0 < cfg.masked_ratio_min <= cfg.masked_ratio_max < 1:
        raise ValueError("Require 0 < masked_ratio_min <= masked_ratio_max < 1")
    if cfg.masked_max_samples < 1 or cfg.masked_warmup_steps < 0:
        raise ValueError("masked_max_samples must be positive and masked_warmup_steps nonnegative")
    if not math.isfinite(cfg.loss_weight) or cfg.loss_weight < 0:
        raise ValueError("loss_weight must be finite and nonnegative")
    if not math.isfinite(cfg.masked_visible_weight) or cfg.masked_visible_weight < 0:
        raise ValueError("masked_visible_weight must be finite and nonnegative")
    if len(cfg.target_grid_thw) != 3 or min(cfg.target_grid_thw) < 1 or math.prod(cfg.target_grid_thw[1:]) < 2:
        raise ValueError("masked_prediction needs a positive T,H,W grid with at least two spatial cells")


def sample_tube_mask(
    num_views: int,
    height: int,
    width: int,
    ratio_min: float,
    ratio_max: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Boolean [V,H,W] masks: unions of rectangles, constant across the entire clip.

    Ratios are approximate on small grids. Keep at least one target and one context
    cell per camera. Sampling uses a private CPU generator, independent of FM noise.
    """
    if num_views < 1 or height * width < 2 or not 0 < ratio_min <= ratio_max < 1:
        raise ValueError("Invalid tube-mask geometry or ratios")
    mask = torch.zeros(num_views, height, width, dtype=torch.bool)
    for view in mask:
        ratio = ratio_min + (ratio_max - ratio_min) * torch.rand((), generator=generator).item()
        count = min(height * width - 1, max(1, round(height * width * ratio)))
        # Rectangle proposals accumulate into a block mask; the last proposal is
        # truncated in random order to attain the requested cell count exactly.
        for _ in range(64):
            missing = count - int(view.sum())
            if missing == 0:
                break
            bh = int(torch.randint(1, height + 1, (), generator=generator))
            bw = int(torch.randint(1, width + 1, (), generator=generator))
            top = int(torch.randint(height - bh + 1, (), generator=generator))
            left = int(torch.randint(width - bw + 1, (), generator=generator))
            proposal = torch.zeros_like(view)
            proposal[top : top + bh, left : left + bw] = True
            candidates = (proposal & ~view).flatten().nonzero().flatten()
            if len(candidates) > missing:
                candidates = candidates[torch.randperm(len(candidates), generator=generator)[:missing]]
            view.flatten()[candidates] = True
        missing = count - int(view.sum())
        if missing:
            candidates = (~view).flatten().nonzero().flatten()
            view.flatten()[candidates[torch.randperm(len(candidates), generator=generator)[:missing]]] = True
    return mask


def expand_pool_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Hide the full support of each selected adaptive-average-pool target cell.

    PyTorch adaptive pooling overlaps bins when sizes do not divide (16 -> 5).
    Nearest interpolation of a 5x5 mask would leave some target pixels visible.
    """
    views, gh, gw = mask.shape
    result = torch.zeros(views, height, width, dtype=torch.bool, device=mask.device)
    for v, y, x in mask.nonzero().tolist():
        result[
            v, y * height // gh : math.ceil((y + 1) * height / gh), x * width // gw : math.ceil((x + 1) * width / gw)
        ] = True
    return result


def masked_native_clip(native: torch.Tensor, mask: torch.Tensor, teacher_side: int) -> torch.Tensor:
    """[C,T,H,V*W] RGB uint8 -> masked float RGB, including conditioning frame 0.

    Mask in teacher-patch coordinates, then conservatively map to native pixels.
    Teacher preprocessing uses a full-frame resize, so this preserves its geometry.
    """
    while isinstance(native, (list, tuple)):
        if len(native) != 1:
            raise ValueError("Expected one native clip per sample")
        native = native[0]
    if native.ndim == 5 and native.shape[0] == 1:
        native = native[0]
    if native.ndim != 4 or native.shape[0] != 3 or native.dtype != torch.uint8:
        raise ValueError("Expected native uint8 [3,T,H,V*W] RGB")
    views = mask.shape[0]
    if native.shape[-1] % views or teacher_side % 16:
        raise ValueError("Canvas width must divide into views; teacher size must divide into 16px patches")
    teacher_mask = expand_pool_mask(mask, teacher_side // 16, teacher_side // 16)
    teacher_pixels = teacher_mask.repeat_interleave(16, -2).repeat_interleave(16, -1)
    h, w = native.shape[-2], native.shape[-1] // views
    # adaptive max pooling conservatively covers all resized teacher mask pixels.
    pixel_mask = F.adaptive_max_pool2d(teacher_pixels.float()[:, None], (h, w))[:, 0].bool()
    pixel_mask = pixel_mask.permute(1, 0, 2).reshape(h, views * w)
    return native.float().masked_fill(pixel_mask.to(native.device)[None, None], 127.5)


def make_masked_canvas(
    native: torch.Tensor, mask: torch.Tensor, image_size: torch.Tensor, teacher_side: int
) -> torch.Tensor:
    """Mask -> aspect-preserving resize -> reflection padding -> [-1,1].

    Return [1,C,T,H_pad,W_pad]. Matches the LIBERO canvas metadata used by the
    main branch; masking precedes every spatial mixing operation, including padding.
    """
    x = masked_native_clip(native, mask, teacher_side)
    th, tw, h, w = (int(v) for v in image_size.reshape(-1).tolist())
    if min(th, tw, h, w) < 1 or h > th or w > tw:
        raise ValueError("Invalid image_size: expected [canvas_h,canvas_w,content_h,content_w]")
    x = x.permute(1, 0, 2, 3)  # T,C,H,W
    if x.shape[-2:] != (h, w):
        x = F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False, antialias=True)
    x = x.clamp(0, 255)  # same RGB range as the uint8 main-branch transform
    if (th, tw) != (h, w):
        mode = "replicate" if tw - w >= w or th - h >= h else "reflect"
        x = F.pad(x, (0, tw - w, 0, th - h), mode=mode)
    return (x.permute(1, 0, 2, 3).unsqueeze(0) / 127.5 - 1).contiguous()


def flatten_target_mask(mask: torch.Tensor, temporal: int) -> torch.Tensor:
    """[V,H,W] -> [T*H*(V*W)], exactly the concat-view REPA row order."""
    spatial = mask.permute(1, 0, 2).flatten(1)
    return spatial.unsqueeze(0).expand(temporal, -1, -1).reshape(-1)


def masked_prediction_loss(pred, target, mask, counts, visible_weight: float) -> tuple[torch.Tensor, dict]:
    """Channel-normalized teacher L1, with per-sample masked/context means.

    Only the target is layer-normalized, as in JEPA regression. Separate means
    keep the loss scale independent of the sampled mask ratio. Teacher gradients
    are forbidden even if a caller accidentally supplies a requires_grad tensor.
    """
    if pred.shape != target.shape or pred.ndim != 2 or mask.shape != pred.shape[:1]:
        raise ValueError("Expected matching [N,D] predictions/targets and [N] mask")
    if mask.dtype != torch.bool or sum(counts) != len(pred) or not counts or min(counts) <= 0:
        raise ValueError("Invalid mask dtype or per-sample counts")
    if not math.isfinite(visible_weight) or visible_weight < 0:
        raise ValueError("visible_weight must be finite and nonnegative")
    y = F.layer_norm(target.detach().float(), (target.shape[-1],))
    x = pred.float()
    token_loss = (x - y).abs().mean(-1)
    masked, visible, centered, spread = [], [], [], []
    for err, m, px, ty in zip(token_loss.split(counts), mask.split(counts), x.split(counts), y.split(counts)):
        if not bool(m.any()) or not bool((~m).any()):
            raise ValueError("Every auxiliary sample needs both masked and context targets")
        masked.append(err[m].mean())
        visible.append(err[~m].mean())
        with torch.no_grad():
            centered.append(F.cosine_similarity(px - px.mean(0), ty - ty.mean(0), dim=-1)[m].mean())
            spread.append(px.std(dim=0, unbiased=False).mean())
    ml, vl = torch.stack(masked).mean(), torch.stack(visible).mean()
    loss = ml + visible_weight * vl
    return loss, {
        "jepa_loss": loss,
        "jepa_masked_loss": ml.detach(),
        "jepa_visible_loss": vl.detach(),
        "jepa_mask_fraction": mask.float().mean(),
        "jepa_centered_cos": torch.stack(centered).mean(),
        "jepa_pred_std": torch.stack(spread).mean(),
    }


def auxiliary_weight(weight: float, warmup_steps: int, iteration: int) -> float:
    return weight * (min(1.0, max(0, iteration) / warmup_steps) if warmup_steps else 1.0)


def prepare_masked_batch(model, data_batch, sequence_plans, clean, iteration: int) -> dict:
    """Build a small, independently packed vision-only auxiliary batch.

    No original latent, action, caption, or unmasked conditioning image is reused.
    The auxiliary prefix and the ordinary full pass execute within one root FSDP
    forward; the teacher holder is consumed before either activation graph grows.
    """
    from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan
    from cosmos_framework.model.generator.repa.alignment import RepaTeacherTokens
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

    cfg = model.config.repa
    if (
        clean.is_image_batch
        or clean.num_views_per_vision_item is not None
        or clean.num_vision_items_per_sample is not None
    ):
        raise ValueError("masked_prediction currently supports single-clip concat-view video batches only")
    if any(not p.has_vision or list(p.condition_frame_indexes_vision) != [0] for p in sequence_plans):
        raise ValueError("masked_prediction requires one video with conditioning frame [0] per sample")
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    # Validation uses fixed masks independent of checkpoint iteration. Training
    # remains reproducible on resume and does not advance any global RNG state.
    step = iteration if torch.is_grad_enabled() else 0
    generator = torch.Generator().manual_seed(cfg.masked_seed + 1000003 * step + 9176 * rank)
    size = min(clean.batch_size, cfg.masked_max_samples)
    indexes = torch.randperm(clean.batch_size, generator=generator)[:size].tolist()
    if not indexes:
        raise ValueError("Empty auxiliary batch")
    t, h, w = cfg.target_grid_thw
    latents, masks, plans = [], [], []
    native = data_batch[cfg.native_video_key]
    with torch.no_grad():
        for idx in indexes:
            mask = sample_tube_mask(cfg.num_views, h, w, cfg.masked_ratio_min, cfg.masked_ratio_max, generator)
            # Reject masks which erase every teacher pixel in a view after the
            # conservative 16->5 pooling-support expansion.
            for _ in range(32):
                if (
                    not expand_pool_mask(mask, cfg.teacher_input_size // 16, cfg.teacher_input_size // 16)
                    .flatten(1)
                    .all(1)
                    .any()
                ):
                    break
                mask = sample_tube_mask(cfg.num_views, h, w, cfg.masked_ratio_min, cfg.masked_ratio_max, generator)
            else:
                raise ValueError("Mask leaves no pixel context; lower masked_ratio_max or use a larger target grid")
            canvas = make_masked_canvas(native[idx], mask, data_batch["image_size"][idx], cfg.teacher_input_size)
            canvas = canvas.to(**model.tensor_kwargs_fp32)
            latent = model.encode(canvas).contiguous().float()
            latent = model._remove_padding_from_latent([latent], [data_batch["image_size"][idx]])[0]
            if latent.shape != clean.x0_tokens_vision[idx].shape:
                raise ValueError("Auxiliary VAE geometry differs from the main branch")
            latents.append(latent)
            masks.append(flatten_target_mask(mask, t))
            # Empty text content retains the packer's structural special tokens
            # and causal/full split invariant, without the task caption.
            plans.append(SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[0]))

    aux_clean = GenerationDataClean(
        batch_size=size,
        is_image_batch=False,
        x0_tokens_vision=latents,
        fps_vision=clean.fps_vision[indexes] if clean.fps_vision is not None else None,
    )
    # sigma=0: clean but occluded context, no diffusion noise; no FM loss on this branch.
    packed = model._pack_input_sequence(plans, [[] for _ in plans], aux_clean, torch.zeros(size, 1))
    expected_shape = (t + 1, h, w * cfg.num_views)
    if packed.vision is None or any(tuple(s) != expected_shape for s in packed.vision.token_shapes):
        raise ValueError(f"Auxiliary token geometry must be {expected_shape}; check target_grid_thw")
    packed.to_cuda()
    teacher_batch = {cfg.native_video_key: [native[i] for i in indexes]}
    teacher = RepaTeacherTokens(model._compute_repa_teacher_tokens(teacher_batch, plans))
    return {"packed_seq": packed, "teacher_tokens": teacher, "mask": torch.cat(masks).to(latents[0].device)}
