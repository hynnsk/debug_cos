# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import tomllib
from pathlib import Path

import pytest

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


def test_vlm_task_skips_repa_block():
    raw = {"job": {"task": "vlm", "experiment": "x"}, "model": {"repa": {"enabled": True}}}
    assert not any("repa" in o for o in build_hydra_overrides(raw))


def test_unknown_repa_key_is_rejected():
    with pytest.raises(Exception):
        SFTExperimentConfig.model_validate({"model": {"repa": {"weight": 0.5}}})
