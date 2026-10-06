# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for REPA inside the Reptile inner loop (cosmos_hs11 v11): the precomputed-teacher-token path of
``OmniMoTModel.training_step_from_inputs`` (stand-in self), the trainer helpers, the loader settings, the theta group
of the projector, and the v11 TOML / launcher recipe. Every other recipe must stay bit-identical (REPA off)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomllib
import torch

from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig
from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides
from cosmos_framework.data.generator.action.meta.reptile_meta import GROUP_REPA_HEAD, reptile_param_group
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.repa.alignment import RepaTeacherTokens
from cosmos_framework.scripts.train_action_meta import SAMPLER_OVERRIDE_KEYS
from cosmos_framework.scripts.train_action_reptile import (
    ReptileTrainConfig,
    _episode_repa_tokens,
    _fm_part,
    _repa_scalars,
    apply_repa_loader_settings,
)

REPO = Path(__file__).resolve().parents[2]
TOML_DIR = REPO / "examples/toml/sft_config"
V4 = TOML_DIR / "action_reptile_meta_edge_v4.toml"
V11 = TOML_DIR / "action_reptile_meta_edge_v11.toml"
LAUNCHER_V4 = REPO / "examples/launch_reptile_meta_edge_v4.sh"
LAUNCHER_V11 = REPO / "examples/launch_reptile_meta_edge_v11.sh"


# ---------------------------------------------------------------------------------------------
# OmniMoTModel._resolve_repa_teacher_tokens (stand-in self)
# ---------------------------------------------------------------------------------------------
def _model(enabled=True, masked=False, teacher=object(), computed=None):
    calls = []

    def _compute(data_batch, sequence_plans):
        calls.append((data_batch, sequence_plans))
        return computed if computed is not None else torch.zeros(2, 1, 16, 16, 16, 8)

    m = SimpleNamespace(
        repa_enabled=enabled,
        masked_prediction_enabled=masked,
        repa_teacher=teacher,
        config=SimpleNamespace(repa=SimpleNamespace(native_video_key="video_native")),
        _compute_repa_teacher_tokens=_compute,
        _calls=calls,
    )
    return m


def test_precomputed_tokens_are_wrapped_in_a_fresh_holder_and_not_consumed():
    m = _model()
    raw = torch.randn(2, 1, 16, 16, 16, 8)
    h1 = OmniMoTModel._resolve_repa_teacher_tokens(m, None, [], raw)
    h2 = OmniMoTModel._resolve_repa_teacher_tokens(m, None, [], raw)
    assert isinstance(h1, RepaTeacherTokens) and isinstance(h2, RepaTeacherTokens) and h1 is not h2
    assert h1.take() is raw and h2.take() is raw  # k inner steps can share one tensor
    assert m._calls == []  # the teacher was not run again


def test_precomputed_holder_is_unwrapped_and_rewrapped():
    m = _model()
    raw = torch.zeros(1, 1, 16, 16, 16, 8)
    given = RepaTeacherTokens(raw)
    out = OmniMoTModel._resolve_repa_teacher_tokens(m, None, [], given)
    assert given.consumed and out is not given and out.take() is raw


def test_raw_batch_path_runs_the_teacher_and_precomputed_wins_over_it():
    m = _model()
    batch, plans = {"video_native": [torch.zeros(3, 17, 8, 8, dtype=torch.uint8)]}, ["plan"]
    out = OmniMoTModel._resolve_repa_teacher_tokens(m, batch, plans, None)
    assert isinstance(out, RepaTeacherTokens) and m._calls == [(batch, plans)]
    raw = torch.ones(1)
    out2 = OmniMoTModel._resolve_repa_teacher_tokens(m, batch, plans, raw)
    assert out2.take() is raw and len(m._calls) == 1  # not called a second time


def test_neither_source_raises_and_disabled_or_masked_returns_none():
    with pytest.raises(ValueError, match="repa_teacher_tokens"):
        OmniMoTModel._resolve_repa_teacher_tokens(_model(), None, [], None)
    assert OmniMoTModel._resolve_repa_teacher_tokens(_model(enabled=False), None, [], None) is None
    assert OmniMoTModel._resolve_repa_teacher_tokens(_model(masked=True), None, [], None) is None
    assert OmniMoTModel._resolve_repa_teacher_tokens(_model(teacher=None), None, [], None) is None
    # the old call signature (no tokens kwarg) is what every non-v11 caller uses: still the same objects
    assert OmniMoTModel._resolve_repa_teacher_tokens(_model(enabled=False), {"x": 1}, []) is None


def _teacher_model(num_views=1, layouts=None, patches=4, frames=4):
    """Stand-in self for OmniMoTModel._compute_repa_teacher_tokens with a fake per-frame teacher (D=2: the first channel
    is the clip's mean intensity, the second counts the clips encoded so far)."""
    state = {"n": 0}

    def teacher(batch):  # [b,C,T,H,W] uint8 -> [b,T,P,P,2]
        b, _, t, h, w = batch.shape
        assert (h, w) == (8, 8), (h, w)  # every clip arrives at the teacher input size
        out = torch.zeros(b, t, patches, patches, 2)
        for j in range(b):
            out[j, ..., 0] = batch[j].float().mean()
            out[j, ..., 1] = state["n"]
            state["n"] += 1
        return out

    repa = SimpleNamespace(
        native_video_key="video_native", num_views=num_views, teacher_num_frames=frames, teacher_batch_size=2,
        teacher_input_size=8, view_layouts=layouts if layouts is not None else {},
    )
    return SimpleNamespace(config=SimpleNamespace(repa=repa), repa_teacher=teacher)


def _native(h, w, frames=4, fill=None):
    x = torch.zeros(3, frames + 1, h, w, dtype=torch.uint8)
    if fill is not None:
        for box, val in fill:
            y0, y1, x0, x1 = (int(round(f * n)) for f, n in zip(box, (h, h, w, w)))
            x[:, :, y0:y1, x0:x1] = val
    return x


def test_teacher_tokens_stock_path_is_unchanged_without_layouts(monkeypatch):
    import cosmos_framework.model.generator.omni_mot_model as omm

    monkeypatch.setattr(omm, "DEVICE", "cpu")
    m = _teacher_model(num_views=2)
    batch = {"video_native": [_native(8, 16), _native(8, 16)], "dataset_name": ["libero", "libero"]}
    plans = [SimpleNamespace(condition_frame_indexes_vision=[0])] * 2
    out = OmniMoTModel._compute_repa_teacher_tokens(m, batch, plans)
    assert isinstance(out, torch.Tensor) and tuple(out.shape) == (2, 2, 4, 4, 4, 2)
    assert "video_native" not in batch  # popped after the teacher


def test_teacher_tokens_compose_the_yam_layout_and_keep_other_samples_stock(monkeypatch):
    import cosmos_framework.model.generator.omni_mot_model as omm

    monkeypatch.setattr(omm, "DEVICE", "cpu")
    P2 = [(0.0, 2 / 3, 0.0, 1.0), (2 / 3, 1.0, 0.0, 0.5), (2 / 3, 1.0, 0.5, 1.0)]
    m = _teacher_model(num_views=1, layouts={"molmoact2_yam": "primary_over_two", "fractal": "canvas"})
    yam = _native(24, 16, fill=[(P2[0], 10), (P2[1], 20), (P2[2], 30)])  # 2x2 composite: top 16x16 px, wrists 8x8 px
    rt1 = _native(8, 8, fill=[((0, 1, 0, 1), 40)])
    batch = {"video_native": [yam, rt1], "dataset_name": ["molmoact2_yam", "fractal"]}
    plans = [SimpleNamespace(condition_frame_indexes_vision=[0])] * 2
    out = OmniMoTModel._compute_repa_teacher_tokens(m, batch, plans)
    assert isinstance(out, list) and len(out) == 2
    canvas, single = out
    assert tuple(canvas.shape) == (1, 4, 6, 4, 2)  # layout_canvas_grid(P2, (4, 4)) = (6, 4): top 4x4, wrists 2x2
    assert tuple(single.shape) == (1, 4, 4, 4, 2)
    top, lw, rw = canvas[0, ..., 0][:, :4, :], canvas[0, ..., 0][:, 4:, :2], canvas[0, ..., 0][:, 4:, 2:]
    assert torch.allclose(top, torch.full_like(top, 10.0)) and torch.allclose(lw, torch.full_like(lw, 20.0))
    assert torch.allclose(rw, torch.full_like(rw, 30.0)) and torch.allclose(single[..., 0], torch.full_like(single[..., 0], 40.0))
    # 3 view clips for YAM then 1 for RT-1, in sample order (teacher chunks of 2)
    assert sorted(set(canvas[0, ..., 1].flatten().tolist())) == [0.0, 1.0, 2.0] and float(single[..., 1].max()) == 3.0
    assert "video_native" not in batch


# ---------------------------------------------------------------------------------------------
# trainer helpers
# ---------------------------------------------------------------------------------------------
def test_episode_repa_tokens_runs_the_teacher_once_per_sub_batch_with_its_plans():
    m = _model()
    batches = [{"b": 0}, {"b": 1}]
    inputs = [("ids0", ["plan0"], None), ("ids1", ["plan1"], None)]
    toks = _episode_repa_tokens(m, batches, inputs)
    assert len(toks) == 2 and m._calls == [({"b": 0}, ["plan0"]), ({"b": 1}, ["plan1"])]
    assert _episode_repa_tokens(m, [], []) == []
    with pytest.raises(ValueError):
        _episode_repa_tokens(m, batches, inputs[:1])


def test_repa_scalars_and_fm_part():
    out = {
        "repa_loss": torch.tensor(0.8),
        "repa_cos_sim_centered": torch.tensor(0.3),
        "repa_weight": 2.0,
        "repa_weighted_loss": torch.tensor(1.6),
        "repa_rel_loss": torch.tensor(0.5),
        "flow_matching_loss_action": torch.tensor(0.1),
    }
    sc = _repa_scalars(out)
    assert sc == pytest.approx(
        {"repa_loss": 0.8, "repa_cos_sim_centered": 0.3, "repa_weight": 2.0, "repa_weighted_loss": 1.6, "repa_rel_loss": 0.5}
    )
    assert all(isinstance(v, float) for v in sc.values())
    assert _fm_part(3.0, sc, relation_weight=0.0) == pytest.approx(1.4)
    assert _fm_part(3.0, sc, relation_weight=2.0) == pytest.approx(0.4)
    assert _repa_scalars({"flow_matching_loss_action": torch.tensor(0.1)}) == {} and _fm_part(3.0, {}, 5.0) == 3.0


def _config(enabled, num_views=1, size=224, keep=False, native_size=None, with_fields=True, layouts=None, full_res=None):
    dl = {"keep_native_video": keep, "native_video_size": native_size, "native_video_full_res": full_res} if with_fields else {}
    repa = {"enabled": enabled, "num_views": num_views, "teacher_input_size": size}
    if layouts is not None:
        repa["view_layouts"] = layouts
    return SimpleNamespace(model=SimpleNamespace(config={"repa": repa}), dataloader_train=dl)


def test_apply_repa_loader_settings():
    plain = {"keep_native_video": True, "native_video_size": 224, "native_video_full_res": None}
    c = _config(True)
    assert apply_repa_loader_settings(c) == {"keep_native_video": True, "native_video_size": 224}
    assert c.dataloader_train == plain
    c = _config(True, num_views=2)  # multi-view canvas: the teacher splits the native width itself -> no shrink
    assert apply_repa_loader_settings(c) == {"keep_native_video": True}
    assert c.dataloader_train["native_video_size"] is None
    c = _config(True, keep=True, native_size=256)  # explicit [custom.meta] values win
    assert apply_repa_loader_settings(c) == {} and c.dataloader_train == {**plain, "native_video_size": 256}
    c = _config(False)  # REPA off: untouched (every existing recipe)
    assert apply_repa_loader_settings(c) == {} and c.dataloader_train == {"keep_native_video": False, "native_video_size": None, "native_video_full_res": None}
    # composite layouts: those embodiments keep camera resolution (the teacher crops the views), "canvas" ones do not
    c = _config(True, layouts={"molmoact2_yam": "primary_over_two", "fractal": "canvas"})
    assert apply_repa_loader_settings(c) == {"keep_native_video": True, "native_video_size": 224, "native_video_full_res": ["molmoact2_yam"]}
    c = _config(True, layouts={"molmoact2_yam": "primary_over_two"}, full_res=["x"])  # explicit list wins
    assert "native_video_full_res" not in apply_repa_loader_settings(c) and c.dataloader_train["native_video_full_res"] == ["x"]
    assert apply_repa_loader_settings(SimpleNamespace(model=SimpleNamespace(config={}), dataloader_train={})) == {}
    with pytest.raises(KeyError):
        apply_repa_loader_settings(_config(True, with_fields=False))


def test_repa_head_has_its_own_theta_group():
    assert reptile_param_group("repa_head.projector.0.weight") == GROUP_REPA_HEAD
    assert reptile_param_group("net.repa_head.target_adapter.conv.weight") == GROUP_REPA_HEAD
    assert reptile_param_group("blocks.0.mlp_moe_gen.up_proj.weight") == "moe_gen"
    assert reptile_param_group("something_else.weight") == "other"
    assert {"keep_native_video", "native_video_size", "native_video_full_res"} <= SAMPLER_OVERRIDE_KEYS


# ---------------------------------------------------------------------------------------------
# recipe files
# ---------------------------------------------------------------------------------------------
def _meta(raw):
    meta = dict(raw["custom"]["meta"])
    sampler = {k: meta.pop(k) for k in list(meta) if k in SAMPLER_OVERRIDE_KEYS}
    return ReptileTrainConfig.from_dict(meta), sampler


def test_v11_toml_is_v4_plus_inner_loop_repa():
    v4, v11 = tomllib.loads(V4.read_text()), tomllib.loads(V11.read_text())
    SFTExperimentConfig.model_validate(v11)
    overrides = build_hydra_overrides(v11)
    assert "experiment=action_reptile_meta_edge" in overrides
    assert "model.config.repa.enabled=true" in overrides or "model.config.repa.enabled=True" in overrides
    repa = v11["model"]["repa"]
    assert repa["enabled"] is True and repa["teacher"].startswith("dinov2") and repa["teacher_input_size"] % 14 == 0
    assert 1 <= repa["layer_index"] <= 28 and repa["loss_weight"] > 0 and repa["num_views"] == 1
    assert repa["objective"] != "masked_prediction" and repa["target_adapter"] == "avgpool"
    assert repa["teacher_num_frames"] == 16
    assert "repa_" in v11["optimizer"]["keys_to_select"] and "repa_" in v11["checkpoint"]["keys_to_skip_loading"]
    assert [k for k in v11["optimizer"]["keys_to_select"] if k != "repa_"] == v4["optimizer"]["keys_to_select"]
    assert "repa" not in v4["model"], "v4 must stay the plain recipe"
    assert "model.config.repa.view_layouts.molmoact2_yam=primary_over_two" in overrides
    assert repa["view_layouts"] == {"molmoact2_yam": "primary_over_two"}
    cfg11, sampler11 = _meta(v11)
    cfg4, sampler4 = _meta(v4)
    assert cfg11.loss_mode == "total"
    assert sampler11.pop("keep_native_video") is True
    assert sampler11.pop("native_video_size") == repa["teacher_input_size"]
    assert sampler11.pop("native_video_full_res") == ["molmoact2_yam"]  # composite canvas cropped by the teacher
    assert sampler11 == sampler4, "same episode geometry as v4"
    assert {k: v for k, v in cfg11.__dict__.items()} == {k: v for k, v in cfg4.__dict__.items()}, "same Reptile knobs as v4"
    for section in ("optimizer", "trainer", "parallelism", "activation_checkpointing"):
        a = v11.get(section, v11["model"].get(section))
        b = v4.get(section, v4["model"].get(section))
        if section == "optimizer":
            a, b = {k: v for k, v in a.items() if k != "keys_to_select"}, {k: v for k, v in b.items() if k != "keys_to_select"}
        assert a == b, section
    assert v11["job"]["name"] != v4["job"]["name"]


def test_v11_launcher_is_the_v4_launcher_pointed_at_v11():
    assert os.access(LAUNCHER_V11, os.X_OK)
    s11, s4 = LAUNCHER_V11.read_text(), LAUNCHER_V4.read_text()
    assert "action_reptile_meta_edge_v11.toml" in s11 and "action_reptile_meta_edge_v11.toml" not in s4
    assert "action_reptile_meta_edge_v4.toml" in s4 and 'TOML_FILE:-examples/toml/sft_config/action_reptile_meta_edge_v4.toml' not in s11
    assert "MASTER_PORT:-50082" in s11 and "MASTER_PORT:-50016" in s4
    assert "HF_HUB_CACHE" in s11 and "dinov2-base" in s11
    assert "cosmos_framework.scripts.train_action_reptile" in s11
    # no other launcher claims the port
    others = [p for p in (REPO / "examples").glob("*.sh") if p != LAUNCHER_V11]
    assert not [p.name for p in others if "50082" in p.read_text()]
