# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU contracts: leakage, frozen targets, packing, normalization and gradients."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from cosmos_framework.model.generator.repa.alignment import RepaAlignmentHead
from cosmos_framework.model.generator.repa.masked_prediction import (
    auxiliary_weight,
    expand_pool_mask,
    flatten_target_mask,
    make_masked_canvas,
    masked_native_clip,
    masked_prediction_loss,
    prepare_masked_batch,
    sample_tube_mask,
    validate_masked_prediction_config,
)


def test_tubes_reproducible_and_leave_context_without_advancing_global_rng():
    state = torch.get_rng_state().clone()
    for seed in range(20):
        mask = sample_tube_mask(2, 5, 5, 0.4, 0.7, torch.Generator().manual_seed(seed))
        assert torch.equal(mask, sample_tube_mask(2, 5, 5, 0.4, 0.7, torch.Generator().manual_seed(seed)))
        assert ((mask.sum((1, 2)) >= 10) & (mask.sum((1, 2)) <= 18)).all()
        flat = flatten_target_mask(mask, 4).reshape(4, 5, 10)
        assert torch.equal(flat[0], flat[3])
        assert torch.equal(flat[0, :, :5], mask[0])
        assert torch.equal(flat[0, :, 5:], mask[1])
    assert torch.equal(state, torch.get_rng_state())


def test_mask_covers_overlapping_teacher_pool_bins():
    mask = torch.zeros(2, 5, 5, dtype=torch.bool)
    mask[0, 1, 2] = True
    mask[1, 4, 4] = True
    patches = expand_pool_mask(mask, 16, 16)
    # A selected 5x5 target cell pools only masked teacher patches.
    pooled = F.adaptive_avg_pool2d(patches.float()[:, None], (5, 5))[:, 0]
    assert (pooled[mask] == 1).all()
    assert patches[0, 3:7, 6:10].all()
    assert not patches[0, :3].any()


def test_hidden_pixels_cannot_leak_via_vae_condition_frame_or_reflection_padding():
    raw = torch.randint(0, 256, (3, 17, 64, 128), dtype=torch.uint8)
    mask = torch.zeros(2, 5, 5, dtype=torch.bool)
    mask[0, 1:4, 2:4] = True
    mask[1, 3:, 3:] = True
    hidden = masked_native_clip(raw, mask, 64) == 127.5
    assert hidden[:, 0].any()  # frame 0 must be masked too
    changed = torch.where(hidden, 255 - raw, raw)
    geometry = torch.tensor([64, 96, 48, 96])  # resized content + reflected bottom
    canvas = make_masked_canvas(raw, mask, geometry, 64)
    changed_canvas = make_masked_canvas(changed, mask, geometry, 64)
    torch.testing.assert_close(canvas, changed_canvas, rtol=0, atol=0)
    # A spatial+temporal mixing encoder cannot recover the changed hidden pixels.
    kernel = torch.randn(4, 3, 3, 3, 3)
    torch.testing.assert_close(F.conv3d(canvas, kernel), F.conv3d(changed_canvas, kernel), rtol=0, atol=0)
    assert raw.dtype == torch.uint8 and not torch.equal(raw, changed)


def test_loss_matches_sample_balanced_l1_and_teacher_is_detached():
    target = torch.randn(10, 8, requires_grad=True)
    pred = torch.randn(10, 8, requires_grad=True)
    mask = torch.tensor([True, False, True, False, False, False, False, True, True, True])
    loss, metrics = masked_prediction_loss(pred, target, mask, [2, 8], 0.25)
    error = (pred - F.layer_norm(target.detach(), (8,))).abs().mean(-1)
    expected = 0.5 * (error[:2][mask[:2]].mean() + error[2:][mask[2:]].mean())
    expected += 0.125 * (error[:2][~mask[:2]].mean() + error[2:][~mask[2:]].mean())
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert target.grad is None
    assert pred.grad is not None and pred.grad[mask].abs().sum() > 0 and pred.grad[~mask].abs().sum() > 0
    assert all(torch.isfinite(value).all() for value in metrics.values())


def test_masked_only_has_no_visible_gradient_and_detects_constant_prediction():
    pred = torch.ones(12, 8, requires_grad=True)
    target = torch.randn_like(pred)
    mask = torch.arange(12) % 2 == 0
    loss, metrics = masked_prediction_loss(pred, target, mask, [6, 6], 0.0)
    loss.backward()
    assert pred.grad[~mask].count_nonzero() == 0
    assert pred.grad[mask].abs().sum() > 0
    assert metrics["jepa_centered_cos"] == 0
    assert metrics["jepa_pred_std"] == 0


def test_view_mask_order_matches_teacher_alignment_and_backbone_gets_gradient():
    head = RepaAlignmentHead(
        hidden_size=12,
        teacher_embed_dim=8,
        projector_hidden_dim=16,
        target_grid_thw=(4, 2, 2),
        teacher_grid_thw=(8, 4, 4),
    )
    hidden = torch.randn(5 * 2 * 4, 12, requires_grad=True)
    teacher = torch.randn(1, 2, 8, 4, 4, 8, requires_grad=True)
    mask = torch.tensor([[[True, False], [False, True]], [[False, True], [True, False]]])
    out = head(hidden, [(5, 2, 4)], [torch.arange(1, 5)], teacher)
    loss, _ = masked_prediction_loss(out["pred"], out["target"], flatten_target_mask(mask, 4), [32], 0.25)
    loss.backward()
    assert hidden.grad[:8].count_nonzero() == 0  # clean frame has no direct target
    assert hidden.grad[8:].abs().sum() > 0
    assert teacher.grad is None
    assert all(p.grad is not None for p in head.projector.parameters())


def test_warmup_and_invalid_loss_masks():
    assert auxiliary_weight(0.5, 200, 0) == 0
    assert auxiliary_weight(0.5, 200, 100) == 0.25
    assert auxiliary_weight(0.5, 200, 500) == 0.5
    assert auxiliary_weight(0.5, 0, 0) == 0.5
    with pytest.raises(ValueError, match="both masked and context"):
        masked_prediction_loss(torch.randn(4, 8), torch.randn(4, 8), torch.ones(4, dtype=torch.bool), [4], 0.25)


@pytest.mark.parametrize(
    "bad",
    [
        {"target_adapter": "avgpool_conv"},
        {"center_targets": True},
        {"relation_loss_weight": 1.0},
        {"masked_ratio_min": 0.8, "masked_ratio_max": 0.4},
        {"masked_ratio_max": 1.0},
        {"masked_max_samples": 0},
        {"masked_visible_weight": -1},
        {"masked_warmup_steps": -1},
        {"loss_weight": float("nan")},
        {"teacher": "dinov2_vitb14"},
    ],
)
def test_invalid_configuration_is_rejected(bad):
    from cosmos_framework.configs.base.defaults.model_config import RepaConfig

    with pytest.raises(ValueError):
        validate_masked_prediction_config(RepaConfig(objective="masked_prediction", **bad))


def test_prepare_uses_real_packer_and_original_teacher_clips(monkeypatch):
    from cosmos_framework.configs.base.defaults.model_config import RepaConfig
    from cosmos_framework.data.generator.sequence_packing.packers import pack_input_sequence
    from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequence, SequencePlan
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

    monkeypatch.setattr(PackedSequence, "to_cuda", lambda self: None)
    cfg = RepaConfig(
        objective="masked_prediction",
        target_grid_thw=(4, 2, 2),
        teacher_input_size=64,
        masked_max_samples=2,
        masked_ratio_max=0.5,
    )
    raw = [torch.randint(0, 256, (3, 17, 64, 128), dtype=torch.uint8) for _ in range(3)]
    originals = [x.clone() for x in raw]
    geometry = [torch.tensor([64, 128, 64, 128])] * 3
    clean = GenerationDataClean(
        batch_size=3,
        is_image_batch=False,
        x0_tokens_vision=[torch.randn(1, 3, 5, 4, 8) for _ in raw],
        fps_vision=torch.tensor([20.0, 20.0, 20.0]),
        x0_tokens_action=[torch.randn(1, 16, 64) for _ in raw],
    )
    encoded = []
    teacher_clips = []

    def encode(canvas):
        assert not torch.is_grad_enabled()
        encoded.append(canvas)
        return F.adaptive_avg_pool3d(canvas, (5, 4, 8))

    def pack(plans, text, aux_clean, timesteps):
        assert all(not p.has_action and not p.has_sound for p in plans)
        assert text == [[], []] and aux_clean.x0_tokens_action is None
        assert not timesteps.any()
        return pack_input_sequence(
            plans,
            text,
            aux_clean,
            timesteps,
            {"bos_token_id": 1, "eos_token_id": 2, "start_of_generation": 3},
            latent_patch_size=2,
        )

    def teacher(batch, plans):
        for x in batch["video_native"]:
            assert any(x is original for original in raw)
            teacher_clips.append(x)
        return torch.randn(2, 2, 8, 4, 4, 8)

    model = SimpleNamespace(
        config=SimpleNamespace(repa=cfg),
        tensor_kwargs_fp32={"device": "cpu", "dtype": torch.float32},
        encode=encode,
        _pack_input_sequence=pack,
        _remove_padding_from_latent=lambda latents, sizes: latents,
        _compute_repa_teacher_tokens=teacher,
    )
    plans = [
        SequencePlan(has_text=True, has_vision=True, has_action=True, condition_frame_indexes_vision=[0]) for _ in raw
    ]
    batch = {"video_native": raw, "image_size": geometry}
    state = torch.get_rng_state().clone()
    result = prepare_masked_batch(model, batch, plans, clean, 10)
    assert result["packed_seq"].action is None
    assert result["packed_seq"].vision.token_shapes == [(5, 2, 4)] * 2
    assert result["mask"].shape == (64,)
    assert result["teacher_tokens"].take().shape == (2, 2, 8, 4, 4, 8)
    assert len(encoded) == len(teacher_clips) == 2
    for before, after in zip(originals, raw):
        torch.testing.assert_close(before, after)
    # Retry at the same step reproduces student input and mask independently of global RNG.
    retry = prepare_masked_batch(model, batch, plans, clean, 10)
    assert torch.equal(result["mask"], retry["mask"])
    torch.testing.assert_close(encoded[0], encoded[2])


def test_new_recipes_route_and_fewshot_splits_are_disjoint():
    import json

    import tomllib

    from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig
    from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides

    root = Path(__file__).resolve().parents[4]
    recipes = sorted((root / "examples/toml/sft_config").glob("action_policy_libero_10_edge_masked_jepa*.toml"))
    assert len(recipes) == 3
    for path in recipes:
        data = tomllib.loads(path.read_text())
        cfg = SFTExperimentConfig.model_validate(data)
        assert cfg.model.repa.objective == "masked_prediction"
        overrides = build_hydra_overrides(data)
        assert "model.config.repa.objective=masked_prediction" in overrides
        assert "model.config.repa.masked_max_samples=16" in overrides
        assert "trainer.seed=42" in overrides
    subsets = root / "cosmos_framework/data/generator/action/episode_subsets"
    train = json.loads((subsets / "libero_10_3ep_per_task_seed42.json").read_text())
    val = json.loads((subsets / "libero_10_val_5ep_per_task_seed42_excl3ep.json").read_text())
    assert len(train["tasks"]) == 10
    assert all(len(x["episodes"]) == 3 for x in train["tasks"].values())
    train_ids = {i for x in train["tasks"].values() for i in x["episodes"]}
    val_ids = {i for x in val["tasks"].values() for i in x["episodes"]}
    assert len(train_ids) == 30 and len(val_ids) == 50 and not train_ids & val_ids
