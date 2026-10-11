# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Action-language hinge ranking on paired, identically noised FM predictions."""

from __future__ import annotations

import math

import torch


def action_language_ranking_loss(
    positive: list[torch.Tensor],
    negative: list[torch.Tensor],
    target: list[torch.Tensor],
    condition_mask: list[torch.Tensor],
    valid_samples: list[bool],
    margin: float,
    raw_action_dim: list[torch.Tensor] | None = None,
    action_valid_mask: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Average ``relu(margin + E_pos - E_neg)`` over the original batch.

    E is fp32 MSE over generated timesteps and real, valid action channels,
    without diffusion-time weights or the main action-loss multiplier. Padded
    channels and conditioned actions do not affect the margin. CFG-dropped or
    fully conditioned samples contribute zero, retaining both autograd paths.
    Averaging over the original batch gives equal per-sample weighting under
    DDP even when ranks have different numbers of dropped prompts.
    """
    count = len(positive)
    if not count or any(len(x) != count for x in (negative, target, condition_mask, valid_samples)):
        raise ValueError("Ranking predictions, targets, masks and validity must have the same nonzero batch size")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("Ranking margin must be finite and nonnegative")
    errors_pos, errors_neg, valid = [], [], []
    for i, (pos, neg, tgt, cond) in enumerate(zip(positive, negative, target, condition_mask)):
        if pos.shape != neg.shape or pos.shape != tgt.shape or pos.ndim != 2:
            raise ValueError("Ranking action tensors must have identical [T,D] shapes")
        width = int(raw_action_dim[i]) if raw_action_dim is not None and raw_action_dim[i] is not None else pos.shape[1]
        if not 0 < width <= pos.shape[1]:
            raise ValueError("Invalid raw action dimension for ranking")
        mask = (1.0 - cond.to(device=pos.device, dtype=torch.float32)).expand_as(pos)[:, :width]
        if action_valid_mask is not None and action_valid_mask[i] is not None:
            channels = action_valid_mask[i].to(device=pos.device, dtype=torch.float32).reshape(1, -1)
            if channels.shape[1] < width:
                raise ValueError("Action validity mask is narrower than the ranking action dimension")
            mask = mask * channels[:, :width]
        denominator = mask.sum()
        errors_pos.append(
            ((pos[:, :width].float() - tgt[:, :width].float()).square() * mask).sum() / denominator.clamp_min(1)
        )
        errors_neg.append(
            ((neg[:, :width].float() - tgt[:, :width].float()).square() * mask).sum() / denominator.clamp_min(1)
        )
        valid.append((denominator > 0).float() * float(valid_samples[i]))
    e_pos, e_neg, keep = torch.stack(errors_pos), torch.stack(errors_neg), torch.stack(valid)
    gap = e_neg - e_pos
    loss = (torch.relu(margin - gap) * keep).mean()
    n_valid = keep.sum().clamp_min(1)
    metrics = {
        "language_ranking_loss": loss.detach(),
        "language_ranking_positive_error": ((e_pos * keep).sum() / n_valid).detach(),
        "language_ranking_negative_error": ((e_neg * keep).sum() / n_valid).detach(),
        "language_ranking_gap": ((gap * keep).sum() / n_valid).detach(),
        "language_ranking_margin_satisfied": (((gap >= margin).float() * keep).sum() / n_valid).detach(),
        "language_ranking_valid_fraction": keep.mean().detach(),
    }
    return loss, metrics
