# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU tests for the cosmos_hs11 v115-v123 INTERNAL REPA teachers: the identity projector, precomputed token targets
(``RepaTokenTargets``) through the network forward, the capture-only forward of the frozen ``self_init`` teacher, the
``wan_vae`` targets (clean latent patches), the relation loss with different widths, and the model-side plumbing with a
stand-in ``self``. Every external-teacher path must stay bit-identical (defaults unchanged)."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.configs.base.defaults.model_config import RepaConfig
from cosmos_framework.data.generator.sequence_packing.packers import pack_input_sequence
from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.repa.adapters import (
    PROJECTOR_TYPES,
    RepaIdentityProjector,
    build_projector,
    repa_relation_loss,
    split_flat_tokens,
)
from cosmos_framework.model.generator.repa.alignment import RepaAlignmentHead, RepaTokenTargets, select_noisy_vision_rows
from cosmos_framework.model.generator.repa.masked_network_test import TinyLanguage
from cosmos_framework.model.generator.repa.teachers import (
    INTERNAL_REPA_TEACHERS,
    InternalRepaTeacher,
    build_repa_teacher,
    is_internal_repa_teacher,
    resolve_repa_teacher_spec,
)
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

HIDDEN = 16  # TinyLanguage hidden size
LATENT_C, PATCH = 3, 2


def _net(projector_type="identity", teacher_embed_dim=0, layer_index=8):
    language = TinyLanguage(False)
    config = Cosmos3VFMNetworkConfig(
        vlm_config=language.config,
        action_gen=True,
        action_dim=4,
        num_embodiment_domains=1,
        latent_channel_size=LATENT_C,
        latent_patch_size=PATCH,
        repa_enabled=True,
        repa_layer_index=layer_index,
        repa_teacher_embed_dim=teacher_embed_dim,  # 0 = hidden size (self_init)
        repa_projector_type=projector_type,
        repa_projector_hidden_dim=16,
        repa_teacher_grid_thw=(4, 2, 2),  # internal teachers: the MoT grid itself
        repa_target_grid_thw=(4, 2, 2),
    )
    return Cosmos3VFMNetwork(language, config)


def _clean(batch=1):
    return GenerationDataClean(
        batch_size=batch,
        is_image_batch=False,
        x0_tokens_vision=[torch.randn(1, LATENT_C, 5, 4, 8) for _ in range(batch)],
        x0_tokens_action=[torch.randn(16, 4) for _ in range(batch)],
        action_domain_id=[torch.tensor(0) for _ in range(batch)],
    )


def _pack(clean):
    plans = [SequencePlan(has_text=True, has_vision=True, has_action=True, condition_frame_indexes_vision=[0])] * clean.batch_size
    return pack_input_sequence(
        plans, [[4, 5]] * clean.batch_size, clean, torch.tensor([[0.5]] * clean.batch_size),
        {"bos_token_id": 1, "eos_token_id": 2, "start_of_generation": 3}, latent_patch_size=PATCH,
    )


# ---------------------------------------------------------------------------------------------
# registry / projector / loss pieces
# ---------------------------------------------------------------------------------------------
def test_internal_teacher_registry_and_identity_projector():
    assert INTERNAL_REPA_TEACHERS == ("self_init", "wan_vae")
    assert is_internal_repa_teacher("self_init") and is_internal_repa_teacher(" WAN_VAE ") and not is_internal_repa_teacher("dinov2_vitb14")
    spec = resolve_repa_teacher_spec("self_init")
    assert spec.family == "internal" and spec.embed_dim == 0
    assert resolve_repa_teacher_spec("dinov2_vitb14").family == "dinov2"  # external teachers untouched
    with pytest.raises(ValueError, match="internal"):
        build_repa_teacher("wan_vae", checkpoint_path=None, input_size=224, num_frames=16, dtype=torch.float32, device="cpu", chunk_size=1)
    handle = InternalRepaTeacher("self_init", 16, (4, 2, 2), 28)
    assert handle.spec.name == "self_init" and handle.embed_dim == 16 and handle.layer_index == 28
    assert PROJECTOR_TYPES == ("mlp", "linear", "identity")
    proj = build_projector("identity", 16, 32, 192)
    assert isinstance(proj, RepaIdentityProjector) and sum(p.numel() for p in proj.parameters()) == 0
    x = torch.randn(5, 16)
    assert torch.equal(proj(x), x) and proj.reset_parameters() is None
    assert isinstance(build_projector("mlp", 16, 32, 8), nn.Module)  # unchanged


def test_relation_loss_accepts_different_widths_and_rejects_row_mismatch():
    torch.manual_seed(0)
    pred, target = torch.randn(8, 16), torch.randn(8, 4)
    loss = repa_relation_loss(pred, target, [4, 4], distance="l1")
    assert torch.isfinite(loss) and loss.ndim == 0
    assert torch.allclose(repa_relation_loss(pred, pred, [4, 4], distance="l1"), torch.zeros(()), atol=1e-6)
    with pytest.raises(ValueError, match="same N"):
        repa_relation_loss(pred, target[:6], [3, 3], distance="l1")


def test_token_targets_and_noisy_row_selection_match_the_head():
    with pytest.raises(ValueError, match="counts"):
        RepaTokenTargets(torch.zeros(4, 3), [3])
    net = _net()
    pack = _pack(_clean(2))
    vision = pack.vision
    hidden = torch.randn(int(sum(t * h * w for t, h, w in vision.token_shapes)), HIDDEN)
    rows, counts = select_noisy_vision_rows(hidden, vision.token_shapes, vision.noisy_frame_indexes)
    ref = net.repa_head.project(hidden, vision.token_shapes, vision.noisy_frame_indexes)  # identity projector
    torch.testing.assert_close(rows, ref)
    assert counts == [32, 32]  # 4 predicted latent frames x 2 x 4 tokens per sample
    # frame 0 (clean conditioning) is rejected like the head does
    with pytest.raises(ValueError, match="frame 0"):
        select_noisy_vision_rows(hidden, vision.token_shapes, [torch.tensor([0, 1])] * 2)
    assert select_noisy_vision_rows(hidden, vision.token_shapes, [torch.tensor([], dtype=torch.long)] * 2) is None


# ---------------------------------------------------------------------------------------------
# network: capture-only forward + precomputed targets
# ---------------------------------------------------------------------------------------------
def test_capture_only_forward_returns_the_block_output_and_resets_flags():
    torch.manual_seed(1)
    net = _net(layer_index=8)
    pack = _pack(_clean(1))
    with torch.no_grad():
        captured = net(pack, capture_only_layer=8)
    assert set(captured) == {"captured_hidden"}
    text = net.language_model.model
    assert text._repa_capture_layer is None and text._repa_captured is None and not text._repa_stop_after_capture
    # the same residual the REPA head reads at block 8 (identity projector): compare the predicted-frame rows
    vision = pack.vision
    rows, _ = select_noisy_vision_rows(captured["captured_hidden"][vision.sequence_indexes], vision.token_shapes, vision.noisy_frame_indexes)
    with torch.no_grad():
        out = net(pack, repa_teacher_tokens=RepaTokenTargets(torch.zeros_like(rows), [rows.shape[0]]))
    torch.testing.assert_close(out["repa_pred"], rows)
    # block 10 (the last) differs from block 8; out-of-range / combined calls are rejected
    with torch.no_grad():
        last = net(pack, capture_only_layer=10)["captured_hidden"]
    assert not torch.allclose(last, captured["captured_hidden"])
    with pytest.raises(ValueError):
        net(pack, capture_only_layer=11)


def test_precomputed_targets_flow_through_forward_and_losses():
    torch.manual_seed(2)
    net = _net()
    pack = _pack(_clean(2))
    vision = pack.vision
    n_rows = 64
    target = torch.randn(n_rows, HIDDEN)
    out = net(pack, repa_teacher_tokens=RepaTokenTargets(target, [32, 32]))
    assert out["repa_pred"].shape == (n_rows, HIDDEN) and torch.equal(out["repa_target"], target)
    assert out["repa_num_tokens_per_sample"] == [32, 32] and not out.get("repa_empty", False)
    assert len(out["repa_frame_indexes_per_sample"]) == 2
    # a width mismatch is fine for the relation term only (the loss block guards the feature term)
    out2 = net(pack, repa_teacher_tokens=RepaTokenTargets(torch.randn(n_rows, 4), [32, 32]))
    assert out2["repa_pred"].shape == (n_rows, HIDDEN) and out2["repa_target"].shape == (n_rows, 4)
    # row mismatch is rejected
    with pytest.raises(RuntimeError, match="row mismatch"):
        net(pack, repa_teacher_tokens=RepaTokenTargets(torch.randn(10, HIDDEN), [10]))


def _loss_model(repa_cfg):
    m = SimpleNamespace(config=SimpleNamespace(repa=repa_cfg, rectified_flow_inference_config=SimpleNamespace(num_train_timesteps=1000)))
    m._repa_loss_terms = MethodType(OmniMoTModel._repa_loss_terms, m)
    return m


def test_loss_block_handles_the_width_mismatch_and_relation_only_arms():
    torch.manual_seed(3)
    pred = torch.randn(8, 16, requires_grad=True)
    out = {"repa_pred": pred, "repa_target": torch.randn(8, 4), "repa_num_tokens_per_sample": [4, 4], "repa_frame_indexes_per_sample": [torch.tensor([1, 2])] * 2}
    losses = {}
    m = _loss_model(RepaConfig(enabled=True, loss_weight=0.0, relation_loss_weight=5.0, relation_distance="l1", teacher="wan_vae", projector_type="identity"))
    total = m._repa_loss_terms(out, torch.tensor([[500.0]] * 2), 0, losses)
    assert torch.isfinite(total) and total.requires_grad and torch.isnan(losses["repa_cos_sim"])
    total.backward()
    assert pred.grad is not None and float(pred.grad.abs().sum()) > 0  # the relation term trains h_k
    with pytest.raises(ValueError, match="one width"):
        _loss_model(RepaConfig(enabled=True, loss_weight=5.0, relation_loss_weight=0.0, teacher="wan_vae"))._repa_loss_terms(out, torch.tensor([[500.0]] * 2), 0, {})
    # same width: unchanged behaviour (cosine + relation both computed)
    out_same = dict(out, repa_target=torch.randn(8, 16))
    losses = {}
    m = _loss_model(RepaConfig(enabled=True, loss_weight=5.0, relation_loss_weight=5.0, relation_distance="l1", teacher="self_init", teacher_layer_index=10, projector_type="identity"))
    total = m._repa_loss_terms(out_same, torch.tensor([[500.0]] * 2), 0, losses)
    assert torch.isfinite(total) and torch.isfinite(losses["repa_cos_sim"]) and float(losses["repa_weight"]) == 5.0


# ---------------------------------------------------------------------------------------------
# model-side plumbing (stand-in self)
# ---------------------------------------------------------------------------------------------
def _model(repa_cfg, net):
    model = SimpleNamespace(
        config=SimpleNamespace(
            repa=repa_cfg,
            state_ch=LATENT_C,
            diffusion_expert_config=SimpleNamespace(patch_spatial=PATCH),
            parallelism=SimpleNamespace(cp_enabled=False),
        ),
        net=net,
        parallel_dims=None,
        precision=torch.float32,
        masked_prediction_enabled=False,
    )
    for name in ("repa_internal_teacher", "_repa_teacher_embed_dim", "_repa_teacher_grid_thw", "_set_up_internal_repa_teacher",
                 "_internal_repa_targets", "repa_enabled"):
        attr = getattr(OmniMoTModel, name)
        if isinstance(attr, property):
            setattr(type(model), name, attr) if False else None
    # properties cannot be attached to a SimpleNamespace instance: emulate them as plain attributes / bound methods
    model.repa_enabled = bool(repa_cfg.enabled)
    model.repa_internal_teacher = OmniMoTModel.repa_internal_teacher.fget(model)
    for name in ("_repa_teacher_embed_dim", "_repa_teacher_grid_thw", "_set_up_internal_repa_teacher", "_internal_repa_targets"):
        setattr(model, name, MethodType(getattr(OmniMoTModel, name), model))
    return model


def test_internal_teacher_setup_validation_and_embed_dims():
    net = _net()
    cfg = RepaConfig(enabled=True, teacher="wan_vae", projector_type="identity", loss_weight=0.0, relation_loss_weight=5.0)
    m = _model(cfg, net)
    assert m.repa_internal_teacher == "wan_vae" and m._repa_teacher_embed_dim() == LATENT_C * PATCH * PATCH
    assert m._repa_teacher_grid_thw() == tuple(cfg.target_grid_thw)
    m._set_up_internal_repa_teacher("wan_vae")
    assert isinstance(m.repa_teacher, InternalRepaTeacher) and m.repa_teacher.embed_dim == 12
    m = _model(RepaConfig(enabled=True, teacher="self_init", teacher_layer_index=10, projector_type="identity"), net)
    assert m._repa_teacher_embed_dim() == 0  # = hidden size, resolved by the network
    m._set_up_internal_repa_teacher("self_init")
    assert m.repa_teacher.embed_dim == HIDDEN and m.repa_teacher.layer_index == 10
    with pytest.raises(ValueError, match="teacher_layer_index"):
        _model(RepaConfig(enabled=True, teacher="self_init", projector_type="identity"), net)._set_up_internal_repa_teacher("self_init")
    with pytest.raises(ValueError, match="only meaningful"):
        _model(RepaConfig(enabled=True, teacher="wan_vae", teacher_layer_index=3), net)._set_up_internal_repa_teacher("wan_vae")
    with pytest.raises(NotImplementedError, match="token"):
        _model(RepaConfig(enabled=True, teacher="wan_vae", objective="temporal_difference"), net)._set_up_internal_repa_teacher("wan_vae")
    # external teachers: not internal, embed dim from the registry (unchanged)
    m = _model(RepaConfig(enabled=True, teacher="dinov2_vitb14", teacher_input_size=224), net)
    assert m.repa_internal_teacher is None and m._repa_teacher_embed_dim() == 768
    assert _model(RepaConfig(enabled=False, teacher="self_init"), net).repa_internal_teacher is None


def test_wan_vae_targets_are_the_clean_latent_patches_of_the_predicted_frames():
    torch.manual_seed(4)
    net = _net(projector_type="identity")
    clean = _clean(2)
    pack = _pack(clean)
    clean_tokens = list(pack.vision.tokens)  # what training_step_from_inputs keeps before the noise replacement
    pack.vision.tokens = [t + torch.randn_like(t) for t in clean_tokens]  # "noised" latents in the pack
    m = _model(RepaConfig(enabled=True, teacher="wan_vae", projector_type="identity", loss_weight=0.0, relation_loss_weight=5.0), net)
    targets = m._internal_repa_targets(pack, clean_tokens, 0)
    assert isinstance(targets, RepaTokenTargets) and targets.counts == [32, 32]
    assert targets.target.shape == (64, LATENT_C * PATCH * PATCH) and not targets.target.requires_grad
    # reference: patchify the CLEAN latents exactly like _encode_vision, keep the predicted frames
    flat, _ = net.patchify_and_pack_latents(clean_tokens, pack.vision.token_shapes, latent_channel=LATENT_C)
    ref, _ = select_noisy_vision_rows(flat, pack.vision.token_shapes, pack.vision.noisy_frame_indexes)
    torch.testing.assert_close(targets.target, ref)
    with pytest.raises(RuntimeError, match="clean latents"):
        m._internal_repa_targets(pack, None, 0)


def test_self_init_targets_come_from_the_frozen_teacher_capture():
    torch.manual_seed(5)
    net = _net(projector_type="identity")
    teacher = _net(projector_type="identity")
    with torch.no_grad():
        for p in teacher.parameters():
            p.add_(0.1 * torch.randn_like(p))
    pack = _pack(_clean(1))
    m = _model(RepaConfig(enabled=True, teacher="self_init", teacher_layer_index=10, projector_type="identity", loss_weight=5.0), net)
    built = []
    m._get_self_init_teacher_net = lambda iteration: (built.append(iteration), teacher)[1]
    targets = m._internal_repa_targets(pack, None, 7)
    assert built == [7] and targets.counts == [32] and targets.target.shape == (32, HIDDEN)
    with torch.no_grad():
        ref_hidden = teacher(pack, capture_only_layer=10)["captured_hidden"][pack.vision.sequence_indexes]
    ref, _ = select_noisy_vision_rows(ref_hidden, pack.vision.token_shapes, pack.vision.noisy_frame_indexes)
    torch.testing.assert_close(targets.target, ref)
    # the student forward with these targets: pred = its own block-8 rows, target = the teacher's block-10 rows
    out = net(pack, repa_teacher_tokens=targets)
    assert out["repa_pred"].shape == out["repa_target"].shape == (32, HIDDEN)
    assert not torch.allclose(out["repa_pred"], out["repa_target"])


def test_reptile_inner_loop_path_rejects_internal_teachers():
    net = _net()
    m = _model(RepaConfig(enabled=True, teacher="self_init", teacher_layer_index=10, projector_type="identity"), net)
    m._compute_repa_teacher_tokens = MethodType(OmniMoTModel._compute_repa_teacher_tokens, m)
    with pytest.raises(NotImplementedError, match="internal"):
        m._compute_repa_teacher_tokens({"video_native": torch.zeros(1)}, [])


# ---------------------------------------------------------------------------------------------
# end to end on the stand-in model: training_step_from_inputs with the internal teachers
# ---------------------------------------------------------------------------------------------
def _step_model(net, repa_cfg, kind, teacher_net=None):
    from cosmos_framework.configs.base.defaults.model_config import LanguageRankingConfig
    from cosmos_framework.data.generator.sequence_packing import PackedSequence
    from cosmos_framework.model.generator.language_ranking_integration_test import UnitWeightFlow

    rf = SimpleNamespace(
        independent_action_schedule=False, independent_sound_schedule=False, normalize_loss_by_active=False,
        sample_level_loss_averaging=False, loss_scale=10.0, image_loss_scale=None, action_loss_weight=10.0,
    )
    packs = []

    def pack(plans, text, clean, times, **kwargs):
        result = pack_input_sequence(
            plans, text, clean, times, {"bos_token_id": 1, "eos_token_id": 2, "start_of_generation": 3},
            latent_patch_size=PATCH, **kwargs,
        )
        packs.append(result)
        return result

    model = SimpleNamespace(
        config=SimpleNamespace(
            language_ranking=LanguageRankingConfig(enabled=False),
            repa=repa_cfg,
            rectified_flow_training_config=rf,
            rectified_flow_inference_config=SimpleNamespace(num_train_timesteps=1000),
            vision_gen=True, action_gen=True, sound_gen=False, lidar_state_ch=None,
            state_ch=LATENT_C, diffusion_expert_config=SimpleNamespace(patch_spatial=PATCH),
        ),
        net=net,
        parallel_dims=None,
        precision=torch.float32,
        tensor_kwargs={"device": "cpu", "dtype": torch.float32},
        tensor_kwargs_fp32={"device": "cpu", "dtype": torch.float32},
        rectified_flow_video=UnitWeightFlow(),
        rectified_flow_image=UnitWeightFlow(),
        rectified_flow_action=UnitWeightFlow(),
        repa_enabled=True,
        repa_internal_teacher=kind,
        masked_prediction_enabled=False,
        sigreg_enabled=False,
        sigreg_on_residual_tokens=False,
        _get_train_noise_level_vision=lambda **kw: (torch.tensor([[500.0]]), torch.tensor([[0.5]])),
        _get_train_noise_level_action=lambda **kw: (torch.tensor([[250.0]]), torch.tensor([[0.25]])),
        _pack_input_sequence=pack,
        pre_noise_memory_hook=lambda pack, clean, memory: memory,
        _resolve_repa_teacher_tokens=lambda *a, **k: (_ for _ in ()).throw(AssertionError("external path must not run")),
        _get_self_init_teacher_net=lambda iteration: teacher_net,
        build_memory_state=lambda *args: None,
        _get_load_balancing_loss_meshes=lambda: (None, None),
        packs=packs,
    )
    for name in (
        "_replace_clean_with_noised", "_compute_flow_matching_loss", "_compute_losses", "_repa_loss_terms", "denoise",
        "_add_noise_to_input", "_internal_repa_targets",
    ):
        setattr(model, name, MethodType(getattr(OmniMoTModel, name), model))
    return model


@pytest.mark.parametrize("kind,projector,loss_weight,relation_weight", [
    ("self_init", "identity", 5.0, 0.0),
    ("self_init", "identity", 0.0, 5.0),
    ("self_init", "identity", 5.0, 5.0),
    ("wan_vae", "mlp", 5.0, 0.0),
    ("wan_vae", "identity", 0.0, 5.0),
    ("wan_vae", "mlp", 5.0, 5.0),
])
def test_training_step_with_internal_teachers(monkeypatch, kind, projector, loss_weight, relation_weight):
    from cosmos_framework.data.generator.sequence_packing import PackedSequence

    torch.manual_seed(11)
    monkeypatch.setattr(PackedSequence, "to_cuda", lambda self: None)
    teacher_dim = 0 if kind == "self_init" else LATENT_C * PATCH * PATCH
    net = _net(projector_type=projector, teacher_embed_dim=teacher_dim, layer_index=8)
    teacher_net = _net(projector_type="identity", layer_index=8) if kind == "self_init" else None
    if teacher_net is not None:
        with torch.no_grad():
            for p in teacher_net.parameters():
                p.add_(0.1 * torch.randn_like(p))
    with torch.no_grad():
        for p in net.llm2action.parameters():
            p.normal_(std=0.1)
    cfg = RepaConfig(
        enabled=True, teacher=kind, teacher_layer_index=10 if kind == "self_init" else None, layer_index=8,
        projector_type=projector, loss_weight=loss_weight, relation_loss_weight=relation_weight, relation_distance="l1",
    )
    model = _step_model(net, cfg, kind, teacher_net)
    clean = _clean(1)
    plans = [SequencePlan(has_text=True, has_vision=True, has_action=True, condition_frame_indexes_vision=[0])]
    batch = {"video_native": torch.zeros(1), "other": 1}
    forwards = []
    net.register_forward_hook(lambda module, args, output: forwards.append(output))
    output, loss = OmniMoTModel.training_step_from_inputs(
        model, ([[4, 5]], plans, clean, dict(skip_text=False, initial_temporal_offset=0), None, []), 3, data_batch=batch
    )
    assert "video_native" not in batch and batch["other"] == 1  # native frames are dropped (not needed)
    assert torch.isfinite(loss) and "repa_loss" in output and "repa_rel_loss" in output
    assert float(output["repa_weight"]) == loss_weight
    assert len(forwards) == 1  # the student forward (the frozen teacher is a different module)
    net_out = forwards[0]
    assert not net_out.get("repa_empty", False)
    pred, target = net_out.get("repa_pred"), net_out.get("repa_target")
    assert pred is not None and target is not None and pred.shape[0] == target.shape[0] == 32
    if kind == "wan_vae":
        # the target is the CLEAN latent patches (not the noised ones): recompute from the clean data
        flat, _ = net.patchify_and_pack_latents(clean.x0_tokens_vision, model.packs[0].vision.token_shapes, latent_channel=LATENT_C)
        ref, _ = select_noisy_vision_rows(flat, model.packs[0].vision.token_shapes, model.packs[0].vision.noisy_frame_indexes)
        torch.testing.assert_close(target, ref)
        assert target.shape[-1] == LATENT_C * PATCH * PATCH
        assert pred.shape[-1] == (HIDDEN if projector == "identity" else LATENT_C * PATCH * PATCH)
    else:
        assert target.shape[-1] == HIDDEN and pred.shape[-1] == HIDDEN
    loss.backward()
    grads = [p.grad for p in net.language_model.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)
    if teacher_net is not None:
        assert all(p.grad is None for p in teacher_net.parameters())  # frozen teacher never receives gradient
