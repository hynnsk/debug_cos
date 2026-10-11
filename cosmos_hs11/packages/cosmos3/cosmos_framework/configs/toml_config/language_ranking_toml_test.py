# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy
import re
import subprocess
from pathlib import Path

import pytest
import tomllib
from pydantic import ValidationError

from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig
from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides

ROOT = Path(__file__).resolve().parents[3]
RECIPES = [
    ("10", "v58", "10", "v3"),
    ("goal", "v23", "goal", "v13"),
    ("object", "v23", "object", "v3"),
    ("spatial", "v23", "object", "v7.1"),
]


@pytest.mark.parametrize("suite,version,source_suite,source_version", RECIPES)
def test_new_recipes_preserve_source_settings_and_route_ranking(suite, version, source_suite, source_version):
    stem = f"action_policy_libero_{suite}_edge_reptileinit_{version}"
    source = f"action_policy_libero_{source_suite}_edge_reptileinit_{source_version}"
    directory = ROOT / "examples/toml/sft_config"
    raw = tomllib.loads((directory / f"{stem}.toml").read_text())
    original = tomllib.loads((directory / f"{source}.toml").read_text())
    validated = SFTExperimentConfig.model_validate(raw)
    assert validated.model.language_ranking.enabled is True
    for section in ("optimizer", "scheduler", "trainer", "checkpoint"):
        assert raw[section] == original[section]
    model = copy.deepcopy(raw["model"])
    del model["language_ranking"]
    assert model == original["model"]
    for section in ("dataloader_train", "dataloader_val"):
        expected = copy.deepcopy(original[section])
        expected["episode_subset_path"] = expected["episode_subset_path"].replace(
            f"libero_{source_suite}_", f"libero_{suite}_"
        )
        assert raw[section] == expected
        assert (
            ROOT / "cosmos_framework/data/generator/action/episode_subsets" / expected["episode_subset_path"]
        ).is_file()
    overrides = build_hydra_overrides(raw)
    assert "model.config.language_ranking.enabled=true" in overrides
    assert "model.config.language_ranking.margin=0.1" in overrides
    assert "model.config.language_ranking.loss_weight=1.0" in overrides
    assert "repa" not in raw["model"]
    launcher = ROOT / "examples" / f"launch_sft_{stem}.sh"
    assert f"{stem}.toml" in launcher.read_text()
    subprocess.run(["bash", "-n", str(launcher)], check=True)


def test_ranking_is_off_by_default_and_bad_hyperparameters_fail():
    from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig

    assert OmniMoTModelConfig().language_ranking.enabled is False
    base = dict(job=dict(task="vfm", experiment="action_policy_libero_edge"))
    for key in ("margin", "loss_weight"):
        for value in (-0.1, float("nan"), float("inf")):
            with pytest.raises(ValidationError):
                SFTExperimentConfig.model_validate({**base, "model": {"language_ranking": {key: value}}})


def test_loader_enables_negative_sampling_for_train_and_validation():
    from omegaconf import OmegaConf

    from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_edge import (
        action_policy_libero_edge,
    )

    cfg = OmegaConf.create(action_policy_libero_edge, flags={"allow_objects": True})
    cfg.model.config.language_ranking.enabled = True
    for split in ("dataloader_train", "dataloader_val"):
        assert cfg[split].dataloader.datasets.libero.dataset.language_ranking is True


def test_launcher_ports_are_distinct():
    ports = []
    for suite, version, _, _ in RECIPES:
        shell = (ROOT / f"examples/launch_sft_action_policy_libero_{suite}_edge_reptileinit_{version}.sh").read_text()
        ports.append(re.search(r"MASTER_PORT:=(\d+)", shell).group(1))
    assert len(set(ports)) == 4
