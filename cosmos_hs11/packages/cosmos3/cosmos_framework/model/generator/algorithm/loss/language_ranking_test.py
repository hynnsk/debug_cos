# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.model.generator.algorithm.loss.language_ranking import action_language_ranking_loss


def test_hinge_is_per_sample_and_gradients_improve_positive_and_penalize_negative():
    pos = [torch.tensor([[2.0]], requires_grad=True), torch.tensor([[0.0]], requires_grad=True)]
    neg = [torch.tensor([[1.0]], requires_grad=True), torch.tensor([[3.0]], requires_grad=True)]
    loss, metrics = action_language_ranking_loss(
        pos, neg, [torch.zeros(1, 1)] * 2, [torch.zeros(1, 1)] * 2, [True, True], 0.1
    )
    assert loss.item() == pytest.approx(1.55)  # mean(relu([3.1, -8.9])), not relu(mean(...))
    assert float(metrics["language_ranking_margin_satisfied"]) == 0.5
    loss.backward()
    assert pos[0].grad.item() > 0 and neg[0].grad.item() < 0
    assert pos[1].grad.item() == 0 and neg[1].grad.item() == 0


@pytest.mark.parametrize("valid", [True, False])
def test_padding_invalid_channels_and_conditioned_actions_are_excluded(valid):
    pos = torch.tensor([[100.0, 100.0, 100.0], [2.0, 200.0, 200.0]], requires_grad=True)
    neg = torch.tensor([[300.0, 300.0, 300.0], [1.0, 300.0, 300.0]], requires_grad=True)
    loss, metrics = action_language_ranking_loss(
        [pos],
        [neg],
        [torch.zeros_like(pos)],
        [torch.tensor([[1.0], [0.0]])],
        [valid],
        0.1,
        raw_action_dim=[torch.tensor(2)],
        action_valid_mask=[torch.tensor([1.0, 0.0, 0.0])],
    )
    assert loss.item() == pytest.approx(3.1 if valid else 0.0)
    loss.backward()
    for pred in (pos, neg):
        assert pred.grad[0].abs().sum() == 0
        assert pred.grad[:, 1:].abs().sum() == 0
        if not valid:
            assert pred.grad.abs().sum() == 0
    assert float(metrics["language_ranking_valid_fraction"]) == float(valid)


def test_fully_conditioned_samples_have_zero_loss_and_finite_metrics():
    pred = torch.randn(3, 4, requires_grad=True)
    loss, metrics = action_language_ranking_loss(
        [pred], [pred + 1], [torch.zeros_like(pred)], [torch.ones(3, 1)], [True], 0.1
    )
    assert loss.item() == 0 and all(torch.isfinite(v) for v in metrics.values())
    loss.backward()
    assert pred.grad.abs().sum() == 0


def test_production_loss_assembly_adds_ranking_without_changing_fm():
    from cosmos_framework.configs.base.defaults.model_config import LanguageRankingConfig
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    cfg = SimpleNamespace(
        language_ranking=LanguageRankingConfig(enabled=True, loss_weight=2.0, margin=0.1),
        vision_gen=False,
        action_gen=True,
        sound_gen=False,
        rectified_flow_training_config=SimpleNamespace(
            normalize_loss_by_active=False, sample_level_loss_averaging=False, action_loss_weight=10.0
        ),
    )
    model = SimpleNamespace(
        config=cfg,
        repa_enabled=False,
        masked_prediction_enabled=False,
        sigreg_enabled=False,
        tensor_kwargs_fp32={"device": "cpu", "dtype": torch.float32},
        rectified_flow_action=SimpleNamespace(train_time_weight=lambda t, kw: torch.ones_like(t)),
        _get_load_balancing_loss_meshes=lambda: (None, None),
    )
    pos = torch.full((2, 2), 2.0, requires_grad=True)
    neg = torch.full((2, 2), 1.0, requires_grad=True)
    action = SimpleNamespace(
        condition_mask=[torch.zeros(2, 1)],
        noisy_frame_indexes=[torch.arange(2)],
        tokens=[torch.zeros(2, 2)],
        mse_loss_indexes=torch.arange(2),
        raw_action_dim=[torch.tensor(2)],
        action_valid_mask=[torch.ones(2)],
    )
    outputs = dict(preds_action=[pos], ranking_preds_action=[neg], ranking_valid=[True])
    args = dict(
        out_net=outputs,
        data_batch_packed=SimpleNamespace(action=action),
        gen_data_noised=SimpleNamespace(vt_target_action=[torch.zeros(2, 2)]),
        timesteps=torch.ones(1, 1),
        is_image_batch=False,
    )
    total, metrics = OmniMoTModel._compute_losses(model, **args)
    assert total.item() == pytest.approx(40.0 + 2 * 3.1)
    assert metrics["flow_matching_loss_action"].item() == 4.0
    total.backward()
    assert pos.grad.abs().sum() > 0 and neg.grad.abs().sum() > 0
    cfg.language_ranking.enabled = False
    baseline, baseline_metrics = OmniMoTModel._compute_losses(model, **args)
    assert baseline.item() == 40.0
    assert not any(k.startswith("language_ranking") for k in baseline_metrics)
