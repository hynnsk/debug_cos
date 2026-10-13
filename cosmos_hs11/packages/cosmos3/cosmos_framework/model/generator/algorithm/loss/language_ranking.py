# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Action-language hinge ranking on paired, identically noised FM predictions.

cosmos_hs11 v24 family (object v24-v27, LIBERO-10 v75-v78) adds three optional modifications, all off by default:

* ``negative_grad_scale`` (asymmetric / positive-only gradient): the hinge value is unchanged, but the gradient that
  flows through ``E_neg`` is multiplied by this factor (1 = original, 0 = positive-only: the model can satisfy the
  hinge only by lowering ``E_pos``, never by making the wrong-instruction prediction worse).
* ``margin_mode="relative"``: the required gap is ``margin * sg(E_pos)`` per sample instead of the absolute
  ``margin`` (``E_neg >= (1 + margin) * E_pos``), so samples whose own error is already small are not asked for an
  impossible absolute gap.
* ``self_paced_fraction``: only the easiest fraction of the valid samples (largest ``gap - margin_i``, detached)
  receives the hinge; pairs that never separate (instruction-independent motion) drop out of the loss. The loss is
  normalized by ``batch * fraction`` so a selected sample keeps the same weight as in the full-batch mean.
"""

from __future__ import annotations

import math

import torch

MARGIN_MODES = ("absolute", "relative")


def action_language_ranking_loss(
    positive: list[torch.Tensor],
    negative: list[torch.Tensor],
    target: list[torch.Tensor],
    condition_mask: list[torch.Tensor],
    valid_samples: list[bool],
    margin: float,
    raw_action_dim: list[torch.Tensor] | None = None,
    action_valid_mask: list[torch.Tensor] | None = None,
    negative_grad_scale: float = 1.0,
    margin_mode: str = "absolute",
    self_paced_fraction: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Average ``relu(margin_i + E_pos - E_neg)`` over the original batch.

    E is fp32 MSE over generated timesteps and real, valid action channels,
    without diffusion-time weights or the main action-loss multiplier. Padded
    channels and conditioned actions do not affect the margin. CFG-dropped or
    fully conditioned samples contribute zero, retaining both autograd paths.
    Averaging over the original batch gives equal per-sample weighting under
    DDP even when ranks have different numbers of dropped prompts.

    ``margin_i`` is ``margin`` (absolute) or ``margin * sg(E_pos_i)`` (relative). ``negative_grad_scale`` scales the
    gradient through ``E_neg`` only; ``self_paced_fraction`` keeps the easiest fraction of the valid samples (see the
    module docstring). The defaults reproduce the original loss exactly.
    """
    count = len(positive)
    if not count or any(len(x) != count for x in (negative, target, condition_mask, valid_samples)):
        raise ValueError("Ranking predictions, targets, masks and validity must have the same nonzero batch size")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("Ranking margin must be finite and nonnegative")
    if not math.isfinite(negative_grad_scale) or not 0.0 <= negative_grad_scale <= 1.0:
        raise ValueError(f"negative_grad_scale must be in [0, 1], got {negative_grad_scale}")
    if margin_mode not in MARGIN_MODES:
        raise ValueError(f"margin_mode must be one of {MARGIN_MODES}, got {margin_mode!r}")
    if not math.isfinite(self_paced_fraction) or not 0.0 <= self_paced_fraction <= 1.0:
        raise ValueError(f"self_paced_fraction must be in [0, 1], got {self_paced_fraction}")
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
    # Asymmetric gradient: same value as e_neg, gradient scaled by negative_grad_scale (0 = stop-gradient).
    e_neg_grad = negative_grad_scale * e_neg + (1.0 - negative_grad_scale) * e_neg.detach()
    gap = e_neg - e_pos  # diagnostics (value identical to e_neg_grad - e_pos)
    margins = margin * e_pos.detach() if margin_mode == "relative" else torch.full_like(e_pos, margin)
    hinge = torch.relu(margins - (e_neg_grad - e_pos))
    # Self-paced selection: keep the easiest fraction of the VALID samples (largest gap - margin_i, detached).
    n_valid_int = int(keep.sum().item())
    if self_paced_fraction > 0.0 and n_valid_int > 0:
        n_select = min(n_valid_int, max(1, math.ceil(self_paced_fraction * n_valid_int)))
        score = torch.where(keep > 0, (gap - margins).detach(), torch.full_like(gap, -float("inf")))
        selected = torch.zeros_like(keep)
        selected[torch.argsort(score, descending=True)[:n_select]] = 1.0
        selected = selected * keep
        normalizer = count * self_paced_fraction
    else:
        selected = keep
        normalizer = float(count)
    loss = (hinge * selected).sum() / normalizer
    n_valid = keep.sum().clamp_min(1)
    metrics = {
        "language_ranking_loss": loss.detach(),
        "language_ranking_positive_error": ((e_pos * keep).sum() / n_valid).detach(),
        "language_ranking_negative_error": ((e_neg * keep).sum() / n_valid).detach(),
        "language_ranking_gap": ((gap * keep).sum() / n_valid).detach(),
        "language_ranking_margin_satisfied": (((gap >= margins).float() * keep).sum() / n_valid).detach(),
        "language_ranking_valid_fraction": keep.mean().detach(),
        "language_ranking_effective_margin": ((margins * keep).sum() / n_valid).detach(),
        "language_ranking_selected_fraction": (selected.sum() / n_valid).detach(),
    }
    return loss, metrics
