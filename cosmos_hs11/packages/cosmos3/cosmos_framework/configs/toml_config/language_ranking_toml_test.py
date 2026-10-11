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
    # action-similarity hard negatives (cosmos_hs11 v27 family): goal = v23 + knobs, LIBERO-10 = v3 + section
    ("goal", "v27", "goal", "v13"),
    ("goal", "v28", "goal", "v13"),
    ("goal", "v29", "goal", "v13"),
    ("10", "v63", "10", "v3"),
    ("10", "v64", "10", "v3"),
    ("10", "v65", "10", "v3"),
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
    ranking = raw["model"]["language_ranking"]
    assert f"model.config.language_ranking.margin={ranking['margin']}" in overrides
    assert f"model.config.language_ranking.loss_weight={ranking['loss_weight']}" in overrides
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
    assert len(set(ports)) == len(RECIPES)


# ---------------------------------------------------------------------------------------------------------------
# cosmos_hs11 v27: action-similarity hard negatives
# ---------------------------------------------------------------------------------------------------------------
_HARD_KEYS = ("negative_sampling", "hard_negative_fraction", "hard_negative_temperature", "hard_negative_prefix_frames")


# (suite, version) -> (uniform sibling version, fraction, temperature, loss_weight, launcher port)
_HARD_FAMILY = {
    ("goal", "v27"): ("v23", 0.7, 0.5, 1.0, 50127),
    ("goal", "v28"): ("v23", 0.7, 0.5, 5.0, 50128),
    ("goal", "v29"): ("v23", 1.0, 0.25, 1.0, 50129),
    ("10", "v63"): ("v58", 0.7, 0.5, 1.0, 50130),
    ("10", "v64"): ("v58", 0.7, 0.5, 5.0, 50131),
    ("10", "v65"): ("v58", 1.0, 0.25, 1.0, 50132),
}


@pytest.mark.parametrize("suite,version", sorted(_HARD_FAMILY))
def test_hard_negative_recipes_are_the_uniform_sibling_plus_the_knobs_and_route_them(suite, version):
    sibling_version, fraction, temperature, loss_weight, port = _HARD_FAMILY[(suite, version)]
    directory = ROOT / "examples/toml/sft_config"
    raw = tomllib.loads((directory / f"action_policy_libero_{suite}_edge_reptileinit_{version}.toml").read_text())
    sibling = tomllib.loads((directory / f"action_policy_libero_{suite}_edge_reptileinit_{sibling_version}.toml").read_text())
    ranking = raw["model"]["language_ranking"]
    assert ranking["enabled"] is True and ranking["margin"] == 0.1
    assert ranking["negative_sampling"] == "action_similarity"
    assert ranking["hard_negative_fraction"] == fraction
    assert ranking["hard_negative_temperature"] == temperature
    assert ranking["hard_negative_prefix_frames"] == 64
    assert ranking["loss_weight"] == loss_weight
    stripped = copy.deepcopy(raw)
    stripped["model"]["language_ranking"] = {
        **{k: v for k, v in ranking.items() if k not in _HARD_KEYS}, "loss_weight": sibling["model"]["language_ranking"]["loss_weight"]
    }
    stripped["job"]["name"] = sibling["job"]["name"]
    assert stripped == sibling  # nothing but the ranking knobs and the run name differs from the uniform sibling
    SFTExperimentConfig.model_validate(raw)
    overrides = build_hydra_overrides(raw)
    assert "model.config.language_ranking.negative_sampling=action_similarity" in overrides
    assert f"model.config.language_ranking.hard_negative_fraction={fraction}" in overrides
    assert f"model.config.language_ranking.hard_negative_temperature={temperature}" in overrides
    assert "model.config.language_ranking.hard_negative_prefix_frames=64" in overrides
    assert f"model.config.language_ranking.loss_weight={loss_weight}" in overrides
    assert not any("hard_negative" in o or "negative_sampling" in o for o in build_hydra_overrides(sibling))
    launcher = (ROOT / f"examples/launch_sft_action_policy_libero_{suite}_edge_reptileinit_{version}.sh").read_text()
    assert f"action_policy_libero_{suite}_edge_reptileinit_{version}.toml" in launcher
    assert f"MASTER_PORT:={port}" in launcher


def test_run_names_in_the_family_are_distinct():
    names = set()
    for suite, version in _HARD_FAMILY:
        raw = tomllib.loads((ROOT / f"examples/toml/sft_config/action_policy_libero_{suite}_edge_reptileinit_{version}.toml").read_text())
        names.add((raw["job"]["group"], raw["job"]["name"]))
    assert len(names) == len(_HARD_FAMILY)


def test_no_other_recipe_sets_the_hard_negative_knobs():
    family = {f"action_policy_libero_{s}_edge_reptileinit_{v}" for s, v in _HARD_FAMILY}
    for p in sorted((ROOT / "examples/toml/sft_config").glob("*.toml")):
        if p.stem in family:
            continue
        ranking = tomllib.loads(p.read_text()).get("model", {}).get("language_ranking", {})
        assert not (set(ranking) & set(_HARD_KEYS)), p.name


def test_hard_negative_validation_and_defaults():
    from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig

    defaults = OmniMoTModelConfig().language_ranking
    assert defaults.negative_sampling == "uniform" and defaults.hard_negative_fraction == 0.0
    assert defaults.hard_negative_temperature == 0.5 and defaults.hard_negative_prefix_frames == 64
    base = dict(job=dict(task="vfm", experiment="action_policy_libero_edge"))
    bad = [
        ("negative_sampling", "text_overlap"),
        ("hard_negative_fraction", -0.1),
        ("hard_negative_fraction", 1.5),
        ("hard_negative_fraction", float("nan")),
        ("hard_negative_temperature", 0.0),
        ("hard_negative_temperature", float("inf")),
        ("hard_negative_prefix_frames", 0),
    ]
    for key, value in bad:
        with pytest.raises(ValidationError):
            SFTExperimentConfig.model_validate({**base, "model": {"language_ranking": {key: value}}})
    SFTExperimentConfig.model_validate(
        {**base, "model": {"language_ranking": {"negative_sampling": "uniform", "hard_negative_fraction": 1.0}}}
    )


def test_experiment_routes_the_hard_negative_knobs_to_both_loaders():
    from omegaconf import OmegaConf

    from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_edge import (
        action_policy_libero_edge,
    )

    cfg = OmegaConf.create(action_policy_libero_edge, flags={"allow_objects": True})
    node = cfg.model.config.language_ranking
    assert node.negative_sampling == "uniform" and node.hard_negative_fraction == 0.0
    node.enabled = True
    node.negative_sampling = "action_similarity"
    node.hard_negative_fraction = 0.7
    node.hard_negative_temperature = 0.25
    node.hard_negative_prefix_frames = 32
    for split in ("dataloader_train", "dataloader_val"):
        dataset = cfg[split].dataloader.datasets.libero.dataset
        assert dataset.language_ranking is True
        assert dataset.language_ranking_negative_sampling == "action_similarity"
        assert dataset.language_ranking_hard_fraction == 0.7
        assert dataset.language_ranking_hard_temperature == 0.25
        assert dataset.language_ranking_hard_prefix_frames == 32
