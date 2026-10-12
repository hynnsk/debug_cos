# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""1-demo-per-task LIBERO recipes (cosmos_hs11 1ep v1): the subset jsons are strict subsets of the 3-demo subsets, the
TOMLs are the goal v7 baseinit / reptileinit recipes with only the suite, run name and subset changed, launchers point at
them with distinct ports."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import tomllib

from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig
from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides

ROOT = Path(__file__).resolve().parents[3]
TOML_DIR = ROOT / "examples/toml/sft_config"
SUBSETS = ROOT / "cosmos_framework/data/generator/action/episode_subsets"
PARENTS = {
    "10": "libero_10_3ep_per_task_seed42.json",
    "goal": "libero_goal_3ep_per_task_v2.json",
    "object": "libero_object_3ep_per_task_seed42.json",
    "spatial": "libero_spatial_3ep_per_task_v3.json",
}
SOURCES = {"baseline": "action_policy_libero_goal_edge_baseinit_v7.toml", "reptileinit": "action_policy_libero_goal_edge_reptileinit_v7.toml"}
GROUPS = {"10": "edge_libero10_fewshot", "goal": "edge_libero_goal_fewshot", "object": "edge_libero_object_fewshot", "spatial": "edge_libero_spatial_fewshot"}


def _episodes(entry):
    return [int(e) for e in (entry["episodes"] if isinstance(entry, dict) else entry)]


@pytest.mark.parametrize("suite", sorted(PARENTS))
def test_1ep_json_is_one_episode_per_task_drawn_from_the_3ep_parent(suite):
    child = json.loads((SUBSETS / f"libero_{suite}_1ep_per_task_v1.json").read_text())
    parent = json.loads((SUBSETS / PARENTS[suite]).read_text())
    assert child["based_on"] == PARENTS[suite]
    assert child["episodes_per_task"] == 1 and child["n_tasks"] == 10 and child["n_episodes"] == 10
    assert child["source_root_basename"] == parent["source_root_basename"]
    assert child["n_episodes_total_in_dataset"] == parent["n_episodes_total_in_dataset"]
    assert sorted(child["tasks"], key=int) == sorted(parent["tasks"], key=int)
    chosen = []
    for task, entry in child["tasks"].items():
        eps = _episodes(entry)
        assert len(eps) == 1
        assert eps[0] in _episodes(parent["tasks"][task]), (suite, task)
        assert child["v1_selection"][task]["chosen"] == eps[0]
        assert sorted(child["v1_selection"][task]["candidates"]) == sorted(_episodes(parent["tasks"][task]))
        if isinstance(parent["tasks"][task], dict) and "task" in parent["tasks"][task]:
            assert entry["task"] == parent["tasks"][task]["task"]
        chosen.append(eps[0])
    assert len(set(chosen)) == 10
    val = json.loads((SUBSETS / f"libero_{suite}_val_5ep_per_task_seed42_excl3ep.json").read_text())
    assert not set(chosen) & {e for entry in val["tasks"].values() for e in _episodes(entry)}


@pytest.mark.parametrize("init", sorted(SOURCES))
@pytest.mark.parametrize("suite", sorted(PARENTS))
def test_1ep_recipes_are_the_goal_v7_recipe_with_only_data_and_name_changed(suite, init):
    stem = f"action_policy_1ep_libero_{suite}_edge_{init}_v1"
    raw = tomllib.loads((TOML_DIR / f"{stem}.toml").read_text())
    src = tomllib.loads((TOML_DIR / SOURCES[init]).read_text())
    SFTExperimentConfig.model_validate(raw)
    assert raw["job"]["group"] == GROUPS[suite]
    assert raw["job"]["name"].endswith(f"_1ep_fullft_{'baseinit' if init == 'baseline' else 'reptile'}_v1")
    assert raw["job"]["name"] != src["job"]["name"]
    for section in ("model", "optimizer", "scheduler", "trainer", "checkpoint"):
        assert raw[section] == src[section], (stem, section)
    assert raw["dataloader_train"]["episode_subset_path"] == f"libero_{suite}_1ep_per_task_v1.json"
    assert raw["dataloader_val"]["episode_subset_path"] == f"libero_{suite}_val_5ep_per_task_seed42_excl3ep.json"
    for section in ("dataloader_train", "dataloader_val"):
        assert {k: v for k, v in raw[section].items() if k != "episode_subset_path"} == {
            k: v for k, v in src[section].items() if k != "episode_subset_path"
        }
        assert (SUBSETS / raw[section]["episode_subset_path"]).is_file()
    overrides = build_hydra_overrides(raw)
    assert f"dataloader_train.dataloader.datasets.libero.dataset.episode_subset_path=libero_{suite}_1ep_per_task_v1.json" in overrides
    launcher = ROOT / "examples" / f"launch_sft_1ep_libero_{suite}_edge_{init}_v1.sh"
    text = launcher.read_text()
    assert f"{stem}.toml" in text
    basename = "libero_10" if suite == "10" else f"libero_{suite}"
    assert f'== "{basename}" ]]' in text  # suite guard on LIBERO_ROOT
    if init == "reptileinit":
        assert "REPTILE_CKPT_PATH/model" in text and "META_ACTION_INIT_PATH" in text
    subprocess.run(["bash", "-n", str(launcher)], check=True)


def test_1ep_launcher_ports_are_distinct_and_unused_elsewhere():
    mine: dict[str, int] = {}
    others: set[int] = set()
    for p in sorted((ROOT / "examples").glob("*.sh")):
        found = {int(m.group(1)) for m in re.finditer(r"MASTER_PORT[:=]{1,2}-?(\d+)", p.read_text())}
        if p.name.startswith("launch_sft_1ep_"):
            assert len(found) == 1, p.name
            mine[p.name] = next(iter(found))
        else:
            others |= found
    assert len(mine) == 8 and len(set(mine.values())) == 8
    assert not set(mine.values()) & others
