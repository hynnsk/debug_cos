# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path

import pytest
import tomllib

from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig
from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides

_TOML_DIR = Path(__file__).resolve().parents[3] / "examples" / "toml" / "sft_config"
_REPA_TOMLS = sorted(_TOML_DIR.glob("action_policy_libero_10_edge_repa*.toml"))


@pytest.mark.parametrize("toml_path", _REPA_TOMLS, ids=[p.stem for p in _REPA_TOMLS])
def test_repa_tomls_validate_and_route_to_model_config_repa(toml_path: Path):
    raw = tomllib.load(open(toml_path, "rb"))
    cfg = SFTExperimentConfig.model_validate(raw)
    assert cfg.model.repa.enabled is True
    overrides = build_hydra_overrides(raw)
    assert "experiment=action_policy_libero_edge_repa" in overrides
    assert "model.config.repa.enabled=true" in overrides
    assert any(o.startswith("model.config.repa.layer_index=") for o in overrides)
    assert any(o.startswith("model.config.repa.target_adapter=") for o in overrides)
    assert "model.config.repa.target_grid_thw=[4,5,5]" in overrides
    # Optional knobs route 1:1 (present in the TOML <-> present as an override; absent -> experiment default).
    repa_raw = raw["model"]["repa"]
    for key in ("projector_type", "relation_loss_weight", "relation_distance", "objective", "spatial_norm_eps"):
        got = [o for o in overrides if o.startswith(f"model.config.repa.{key}=")]
        assert got == ([f"model.config.repa.{key}={repa_raw[key]}"] if key in repa_raw else []), key
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
