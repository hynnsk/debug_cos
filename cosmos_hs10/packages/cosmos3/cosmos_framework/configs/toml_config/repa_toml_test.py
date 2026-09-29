# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path

import pytest
import tomllib

from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig
from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides

_TOML_DIR = Path(__file__).resolve().parents[3] / "examples" / "toml" / "sft_config"
_REPA_TOMLS = sorted(
    list(_TOML_DIR.glob("action_policy_libero_10_edge_repa*.toml"))
    + list(_TOML_DIR.glob("action_policy_libero_10_nano_repa*.toml"))
)


@pytest.mark.parametrize("toml_path", _REPA_TOMLS, ids=[p.stem for p in _REPA_TOMLS])
def test_repa_tomls_validate_and_route_to_model_config_repa(toml_path: Path):
    raw = tomllib.load(open(toml_path, "rb"))
    cfg = SFTExperimentConfig.model_validate(raw)
    assert cfg.model.repa.enabled is True
    overrides = build_hydra_overrides(raw)
    tier = "nano" if "_nano_" in toml_path.stem else "edge"
    assert f"experiment=action_policy_libero_{tier}_repa" in overrides
    if tier == "nano":
        # Nano: ViT-L teachers, a block of the 36-block Qwen3-VL-8B MoT (recipe default 10; user variants scan it),
        # shard 8 x 32, compile off (Ampere smem), see docs section 8
        assert raw["model"]["repa"]["teacher"].startswith(("vjepa2_1_", "dinov2_"))  # user variants also scan ViT-B DINOv2
        assert 1 <= raw["model"]["repa"]["layer_index"] <= 36
        assert raw["model"]["parallelism"]["data_parallel_shard_degree"] in (4, 8)
        assert raw["model"]["compile"]["enabled"] is False
    assert "model.config.repa.enabled=true" in overrides
    assert any(o.startswith("model.config.repa.layer_index=") for o in overrides)
    assert any(o.startswith("model.config.repa.target_adapter=") for o in overrides)
    assert "model.config.repa.target_grid_thw=[4,5,5]" in overrides
    # Optional knobs route 1:1 (present in the TOML <-> present as an override; absent -> experiment default).
    repa_raw = raw["model"]["repa"]
    for key in (
        "projector_type",
        "relation_loss_weight",
        "relation_distance",
        "objective",
        "spatial_norm_eps",
        "spatial_norm_scale",
        "normalize_student",
        "center_targets",
        "teacher_layer_index",
    ):
        got = [o for o in overrides if o.startswith(f"model.config.repa.{key}=")]
        want = [f"model.config.repa.{key}={str(repa_raw[key]).lower()}"] if key in repa_raw else []
        assert got == want, key
    # pinned recipes: v4 = linear projector, v5 = VideoREPA-style token-relation loss only (direct cos weight 0)
    if toml_path.stem.endswith("_v4"):
        assert repa_raw["projector_type"] == "linear"
    if toml_path.stem.endswith("_v5"):
        assert (repa_raw["loss_weight"], repa_raw["relation_loss_weight"], repa_raw["relation_distance"]) == (
            0.0,
            5.0,
            "l2",
        )
        assert repa_raw.get("projector_type", "mlp") == "mlp"
    if toml_path.stem.endswith("_v8"):
        assert repa_raw["objective"] == "temporal_difference"
    if toml_path.stem.endswith("_v9"):
        assert raw["model"]["sigreg"]["enabled"] is True
        # v9 = SIGReg on the REPA projector output (shared space), per-token normalized statistic
        assert "model.config.sigreg.input=repa_projection" in overrides
        assert "model.config.sigreg.normalize_by_count=true" in overrides
        assert any(o.startswith("model.config.sigreg.loss_weight=") for o in overrides)
    if toml_path.stem.endswith("_v10"):
        assert repa_raw["objective"] == "spatial_normalized"
    # v6 / v10 normalize both sides (defaults); v6.1 / v10.1 normalize the teacher target only
    if toml_path.stem.endswith(("_v6", "_v10")):
        assert repa_raw.get("normalize_student", True) is True
    if toml_path.stem.endswith("_v6_1"):
        assert repa_raw["center_targets"] is True and repa_raw["normalize_student"] is False
        assert "model.config.repa.normalize_student=false" in overrides
    if toml_path.stem.endswith("_v10_1"):
        assert repa_raw["objective"] == "spatial_normalized"
        assert repa_raw["normalize_student"] is False and repa_raw["spatial_norm_scale"] is False
        assert "model.config.repa.spatial_norm_scale=false" in overrides
    # v12 = ViT-L block 23, plain token cosine; v13 = v12 + 2x2 sub-cells; v14 = v7 (DINOv2-B) + 2x2 sub-cells
    if toml_path.stem.endswith(("_v12", "_v13")):
        assert repa_raw["teacher"] == "vjepa2_1_vit_large_384" and repa_raw["teacher_layer_index"] == 23
        assert repa_raw["objective"] == "token" and "spatial_norm_eps" not in repa_raw
        assert "model.config.repa.teacher_layer_index=23" in overrides
    if toml_path.stem.endswith(("_v13", "_v14")):
        assert repa_raw["target_subgrid_thw"] == [1, 2, 2]
        assert "model.config.repa.target_subgrid_thw=[1,2,2]" in overrides
    elif toml_path.stem.endswith(("nano_repa_v10.6", "nano_repa_v7.11")):
        # Nano v10.6 / v7.11 = the Edge v13 / v14 "less pooling" (2x2 spatial sub-cells per MoT token)
        assert repa_raw["target_subgrid_thw"] == [1, 2, 2]
        assert "model.config.repa.target_subgrid_thw=[1,2,2]" in overrides
    else:
        assert "target_subgrid_thw" not in repa_raw
    if toml_path.stem.endswith(("nano_repa_v10.5", "nano_repa_v10.6")):
        assert repa_raw["objective"] == "token" and "spatial_norm_eps" not in repa_raw
        assert repa_raw["teacher"] == "vjepa2_1_vit_large_384" and repa_raw["teacher_layer_index"] == 23
        assert repa_raw["teacher_input_size"] == 256
    if toml_path.stem.endswith("nano_repa_v7.11"):
        assert repa_raw["teacher"] == "dinov2_vitb14" and repa_raw["teacher_input_size"] == 224
        assert repa_raw["objective"] == "token" and "teacher_layer_index" not in repa_raw
    if toml_path.stem.endswith("_v14"):
        assert repa_raw["teacher"] == "dinov2_vitb14" and "teacher_layer_index" not in repa_raw


def test_repa_projector_type_is_validated_by_the_model_config():
    from cosmos_framework.configs.base.defaults.model_config import RepaConfig, SigRegConfig

    assert RepaConfig().projector_type == "mlp"
    assert RepaConfig(projector_type="linear").projector_type == "linear"
    with pytest.raises(ValueError):
        RepaConfig(projector_type="conv")
    assert RepaConfig().relation_loss_weight == 0.0 and RepaConfig().relation_distance == "l2"
    assert RepaConfig(relation_distance="l1").relation_distance == "l1"
    with pytest.raises(ValueError):
        RepaConfig(relation_distance="cosine")
    assert RepaConfig(objective="temporal_difference").objective == "temporal_difference"
    assert RepaConfig(objective="spatial_normalized").objective == "spatial_normalized"
    with pytest.raises(ValueError):
        RepaConfig(objective="difference")
    # v6 / v10 behaviour is the default; v6.1 / v10.1 flip these
    assert RepaConfig().normalize_student is True and RepaConfig().spatial_norm_scale is True
    assert RepaConfig().teacher_layer_index is None and tuple(RepaConfig().target_subgrid_thw) == (1, 1, 1)
    assert RepaConfig(normalize_student=False, spatial_norm_scale=False).normalize_student is False
    sig = SigRegConfig()
    assert (sig.layer_index, sig.loss_weight, sig.input, sig.normalize_by_count) == (8, 1.0, "repa_projection", True)
    assert SigRegConfig(input="residual_tokens").input == "residual_tokens"
    with pytest.raises(ValueError):
        SigRegConfig(input="projector")


def test_vlm_task_skips_repa_block():
    raw = {
        "job": {"task": "vlm", "experiment": "x"},
        "model": {"repa": {"enabled": True}, "sigreg": {"enabled": True}},
    }
    overrides = build_hydra_overrides(raw)
    assert not any("repa" in o or "sigreg" in o for o in overrides)


def test_unknown_repa_key_is_rejected():
    with pytest.raises(Exception):
        SFTExperimentConfig.model_validate({"model": {"repa": {"weight": 0.5}}})
