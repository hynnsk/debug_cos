# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for the ordinary-source-SFT ablation control of the edge2 Reptile run (cosmos_hs11)."""

from __future__ import annotations

from pathlib import Path

import tomllib

from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig
from cosmos_framework.scripts.train_action_meta import SAMPLER_OVERRIDE_KEYS
from cosmos_framework.scripts.train_action_reptile import ReptileTrainConfig

REPO = Path(__file__).resolve().parents[2]
TOML_DIR = REPO / "examples/toml/sft_config"
EDGE2 = TOML_DIR / "action_reptile_meta_edge2.toml"
CONTROL = TOML_DIR / "action_reptile_meta_edge2_joint_sft.toml"
REPTILEINIT = TOML_DIR / "action_policy_libero_10_edge_reptileinit.toml"
SFTINIT = TOML_DIR / "action_policy_libero_10_edge_sftinit.toml"

# the only [custom.meta] keys the control may change: they turn Reptile into plain Adam training (k = 1, eps = 1)
CONTROL_KEYS = {"inner_steps": 1, "inner_warmup_steps": 0, "inner_reset_optimizer": False, "meta_lr": 1.0, "meta_lr_min_ratio": 1.0}


def _strip(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if k not in SAMPLER_OVERRIDE_KEYS}


def test_control_is_edge2_with_only_the_joint_keys_changed() -> None:
    edge2 = tomllib.loads(EDGE2.read_text())
    ctrl = tomllib.loads(CONTROL.read_text())
    SFTExperimentConfig.model_validate(ctrl)
    em, cm = edge2["custom"]["meta"], ctrl["custom"]["meta"]
    assert set(em) == set(cm)
    for k in em:
        if k in CONTROL_KEYS:
            assert cm[k] == CONTROL_KEYS[k], k
        else:
            assert cm[k] == em[k], k  # same sampler (k_shot 16 x 16 windows, 128/rank), theta, loss, seed, diagnostics
    # same data budget and topology: 256 windows per iteration on 2 ranks, 1000 iterations, same optimizer
    assert (cm["k_shot"] * cm["windows_per_demo"], cm["max_samples_per_batch"]) == (256, 128)
    assert ctrl["model"]["parallelism"]["data_parallel_shard_degree"] == 2
    for section in ("model", "optimizer", "trainer", "checkpoint"):
        assert ctrl[section] == edge2[section], section
    assert ctrl["job"]["experiment"] == edge2["job"]["experiment"] == "action_reptile_meta_edge"
    assert ctrl["job"]["name"] != edge2["job"]["name"] and ctrl["job"]["group"] != edge2["job"]["group"]
    cfg = ReptileTrainConfig.from_dict(_strip(cm))
    assert cfg.inner_steps == 1 and cfg.meta_lr == 1.0 and cfg.meta_lr_min_ratio == 1.0 and cfg.inner_reset_optimizer is False
    assert cfg.inner_warmup_steps == 0 and cfg.meta_optimizer == "sgd" and cfg.theta_mode == "full"


def test_sftinit_downstream_recipe_equals_reptileinit_except_the_name() -> None:
    a = tomllib.loads(REPTILEINIT.read_text())
    b = tomllib.loads(SFTINIT.read_text())
    SFTExperimentConfig.model_validate(b)
    assert b["job"]["name"] == "edge_libero10_3ep_fullft_sourcesftinit" != a["job"]["name"]
    assert {k: v for k, v in b["job"].items() if k != "name"} == {k: v for k, v in a["job"].items() if k != "name"}
    for section in a:
        if section != "job":
            assert b[section] == a[section], section
