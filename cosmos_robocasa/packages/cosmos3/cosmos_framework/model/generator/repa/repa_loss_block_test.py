# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests of ``OmniMoTModel._repa_loss_terms`` (the REPA / relation block of ``_compute_losses``) with a stand-in
``self``: every recipe family (token cosine, relation-only TRD, sigma gate, ramp, probe) must run and return the
expected terms. Added after an UnboundLocalError in this block reached a GPU run (2026-10-02)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.configs.base.defaults.model_config import RepaConfig
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.repa.adapters import repa_cosine_loss, repa_relation_loss

B, T, HW, D = 4, 4, 6, 8  # 4 windows, 4 predicted latent frames, 6 "tokens" per frame, 8-dim teacher space


def _self(**repa):
    return SimpleNamespace(config=SimpleNamespace(repa=RepaConfig(enabled=True, **repa), rectified_flow_inference_config=SimpleNamespace(num_train_timesteps=1000)))


def _out_net(seed=0, empty=False):
    g = torch.Generator().manual_seed(seed)
    if empty:
        probe = torch.zeros((), requires_grad=True)
        return {"repa_pred": probe.view(1, 1), "repa_target": probe.detach().view(1, 1) + 1.0, "repa_empty": True}
    pred = torch.randn(B * T * HW, D, generator=g, requires_grad=True)
    target = torch.randn(B * T * HW, D, generator=g)
    return {
        "repa_pred": pred, "repa_target": target, "repa_empty": False,
        "repa_num_tokens_per_sample": [T * HW] * B,
        "repa_frame_indexes_per_sample": [torch.arange(1, T + 1)] * B,
    }


def _timesteps(sigmas):  # [B_items, T_vis], sigma * 1000
    return (torch.tensor(sigmas) * 1000.0)[:, None].expand(B, T + 1).clone()


def _run(self_, out_net, sigmas=(0.1, 0.4, 0.6, 0.9), iteration=10):
    losses = {}
    term = OmniMoTModel._repa_loss_terms(self_, out_net, _timesteps(sigmas), iteration, losses)
    return term, losses


def test_v2_token_cosine_matches_direct_loss_and_backprops():
    out = _out_net()
    term, L = _run(_self(loss_weight=5.0), out)
    direct, cos = repa_cosine_loss(out["repa_pred"], out["repa_target"])
    torch.testing.assert_close(L["repa_loss"], direct)
    torch.testing.assert_close(L["repa_cos_sim"], cos)
    torch.testing.assert_close(term, 5.0 * direct + 0.0 * L["repa_rel_loss"])
    assert float(L["repa_weight"]) == 5.0 and "repa_sigma_frac" not in L
    assert L["repa_rel_loss"].requires_grad is False  # weight 0 -> monitor only, no graph
    term.backward()
    assert out["repa_pred"].grad is not None and torch.isfinite(out["repa_pred"].grad).all()


def test_relation_only_trd_with_margin():
    out = _out_net(seed=1)
    term, L = _run(_self(loss_weight=0.0, relation_loss_weight=5.0, relation_distance="l1", relation_margin=0.1), out)
    direct = repa_relation_loss(out["repa_pred"], out["repa_target"], [T * HW] * B, distance="l1", margin=0.1)
    torch.testing.assert_close(L["repa_rel_loss"], direct)
    torch.testing.assert_close(term, 5.0 * direct)
    assert L["repa_loss"].requires_grad is False and L["repa_cos_sim"].ndim == 0  # cosine logged only
    term.backward()
    assert out["repa_pred"].grad is not None and out["repa_pred"].grad.abs().sum() > 0


def test_both_terms_on_sum_and_ramp():
    out = _out_net(seed=2)
    self_ = _self(loss_weight=5.0, relation_loss_weight=5.0, relation_distance="l1", relation_margin=0.1, loss_weight_warmup_steps=200)
    term, L = _run(self_, out, iteration=100)  # half-way through the ramp -> cosine weight 2.5
    assert float(L["repa_weight"]) == pytest.approx(2.5)
    torch.testing.assert_close(term, 2.5 * L["repa_loss"] + 5.0 * L["repa_rel_loss"])
    term.backward()


def test_sigma_gate_selects_samples_and_logs_fraction():
    out = _out_net(seed=3)
    term, L = _run(_self(loss_weight=5.0, sigma_max=0.5), out, sigmas=(0.1, 0.4, 0.6, 0.9))
    assert float(L["repa_sigma_frac"]) == pytest.approx(0.5)
    n = T * HW
    kept_pred, kept_tgt = out["repa_pred"][: 2 * n], out["repa_target"][: 2 * n]  # samples 0 and 1 (sigma 0.1, 0.4)
    direct, _ = repa_cosine_loss(kept_pred, kept_tgt)
    torch.testing.assert_close(L["repa_loss"], direct)
    _, cos_all = repa_cosine_loss(out["repa_pred"].detach(), out["repa_target"])
    torch.testing.assert_close(L["repa_cos_sim"], cos_all)  # logged over ALL rows
    term.backward()
    g = out["repa_pred"].grad.view(B, n, D)
    assert g[:2].abs().sum() > 0 and g[2:].abs().sum() == 0  # no gradient for the gated-out samples
    # empty window -> terms x 0, parameters still in the graph
    out2 = _out_net(seed=4)
    term2, L2 = _run(_self(loss_weight=5.0, sigma_min=0.95), out2, sigmas=(0.1, 0.4, 0.6, 0.9))
    assert float(L2["repa_sigma_frac"]) == 0.0 and float(term2) == 0.0 and term2.requires_grad
    term2.backward()
    assert out2["repa_pred"].grad.abs().sum() == 0


def test_other_objectives_and_probe_paths_run():
    for kw in (dict(objective="temporal_difference"), dict(objective="spatial_normalized"), dict(center_targets=True)):
        out = _out_net(seed=5)
        term, L = _run(_self(loss_weight=5.0, **kw), out)
        assert torch.isfinite(term) and "repa_cos_sim" in L and "repa_cos_sim_centered" in L
        term.backward()
    # both weights zero: the probe term keeps the projector output in the graph
    out = _out_net(seed=6)
    term, L = _run(_self(loss_weight=0.0, relation_loss_weight=0.0), out)
    assert float(term) == 0.0 and term.requires_grad
    # repa_empty probe (no alignable tokens on this rank)
    term, L = _run(_self(loss_weight=5.0), _out_net(empty=True))
    assert float(term) == 0.0 and term.requires_grad and float(L["repa_loss"]) == 0.0
