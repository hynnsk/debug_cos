# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""[dataloader_train].episode_balanced_sampling (cosmos_hs11 v43) routes to the nested LIBERO dataset node and is absent
from every other recipe."""

from pathlib import Path

import tomllib

from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig
from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides

_TOML_DIR = Path(__file__).resolve().parents[3] / "examples" / "toml" / "sft_config"
_KEY = "dataloader_train.dataloader.datasets.libero.dataset.episode_balanced_sampling"


def test_v43_routes_the_knob_and_v3_does_not():
    v43 = tomllib.load(open(_TOML_DIR / "action_policy_libero_10_edge_reptileinit_v43.toml", "rb"))
    v3 = tomllib.load(open(_TOML_DIR / "action_policy_libero_10_edge_reptileinit_v3.toml", "rb"))
    SFTExperimentConfig.model_validate(v43)
    o43, o3 = build_hydra_overrides(v43), build_hydra_overrides(v3)
    assert f"{_KEY}=true" in o43 and not any(o.startswith(_KEY) for o in o3)
    assert v43["dataloader_train"]["episode_balanced_sampling"] is True
    # v43 = v3 + that one key (+ name); validation loader untouched
    assert {k: v for k, v in v43["dataloader_train"].items() if k != "episode_balanced_sampling"} == v3["dataloader_train"]
    assert v43["dataloader_val"] == v3["dataloader_val"]
    for section in ("model", "optimizer", "scheduler", "trainer", "checkpoint"):
        assert v43[section] == v3[section], section
    assert v43["job"]["name"] != v3["job"]["name"]


def test_no_other_recipe_sets_the_knob():
    for p in sorted(_TOML_DIR.glob("*.toml")):
        if p.stem.endswith("_v43"):
            continue
        raw = tomllib.load(open(p, "rb"))
        assert "episode_balanced_sampling" not in raw.get("dataloader_train", {}), p.name


def test_experiment_default_is_off():
    import cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_edge as m

    node = m.action_policy_libero_edge["dataloader_train"]["dataloader"]["datasets"]["libero"]["dataset"]
    assert node["episode_balanced_sampling"] is False and node["iterable_shuffle"] is True
    assert m.action_policy_libero_edge["dataloader_val"]["dataloader"]["datasets"]["libero"]["dataset"]["episode_balanced_sampling"] is False
