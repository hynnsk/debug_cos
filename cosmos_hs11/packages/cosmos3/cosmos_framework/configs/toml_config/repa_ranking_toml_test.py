# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""cosmos_hs11 v105-v107: LIBERO-10 Reptile post-training with the action-language ranking loss (v104) PLUS the DINOv2
REPA distillation. Both losses only coexist on experiment ``action_policy_libero_edge_repa``, which therefore carries the
ranking wiring of ``action_policy_libero_edge`` (off by default)."""

import copy
import subprocess
import tomllib
from pathlib import Path

import pytest

from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig
from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides

ROOT = Path(__file__).resolve().parents[3]
TOML_DIR = ROOT / "examples/toml/sft_config"
COMBINED = ["v105", "v106", "v107"]
SOURCE = "v104"  # ranking-only sibling, plain experiment


def _load(version: str) -> dict:
    return tomllib.loads((TOML_DIR / f"action_policy_libero_10_edge_reptileinit_{version}.toml").read_text())


@pytest.mark.parametrize("version", COMBINED)
def test_combined_recipe_routes_both_losses_to_the_repa_experiment(version):
    raw, src = _load(version), _load(SOURCE)
    cfg = SFTExperimentConfig.model_validate(raw)
    assert cfg.model.repa.enabled is True and cfg.model.language_ranking.enabled is True
    overrides = build_hydra_overrides(raw)
    assert "experiment=action_policy_libero_edge_repa" in overrides  # the plain experiment has no model.config.repa
    assert "model.config.repa.enabled=true" in overrides
    assert "model.config.repa.teacher=dinov2_vitb14" in overrides
    assert "model.config.language_ranking.enabled=true" in overrides
    assert "experiment=action_policy_libero_edge" in build_hydra_overrides(src)  # v104 itself is untouched
    # Everything except the REPA section, the experiment, the run name and the ranking margin equals v104.
    for section in ("optimizer", "scheduler", "trainer", "dataloader_train", "dataloader_val"):
        assert raw[section] == src[section], section
    model, src_model = copy.deepcopy(raw["model"]), copy.deepcopy(src["model"])
    del model["repa"]
    model["language_ranking"].pop("margin"), src_model["language_ranking"].pop("margin")
    assert model == src_model
    ckpt = copy.deepcopy(raw["checkpoint"])
    assert "repa_" in ckpt["keys_to_skip_loading"]  # repa_head is absent from the Reptile DCP
    ckpt["keys_to_skip_loading"] = [k for k in ckpt["keys_to_skip_loading"] if k != "repa_"]
    assert ckpt == src["checkpoint"]


def test_combined_recipes_have_distinct_names_and_the_intended_repa_arms():
    names = {_load(v)["job"]["name"] for v in COMBINED}
    assert len(names) == len(COMBINED)
    v105, v106, v107 = (_load(v)["model"]["repa"] for v in COMBINED)
    assert v105["layer_index"] == 24 and v106["layer_index"] == 28  # block sweep
    assert v107["relation_loss_weight"] == 5.0 and v107["relation_distance"] == "l1"  # VideoREPA TRD arm


@pytest.mark.parametrize("version", COMBINED)
def test_combined_launcher_sets_up_the_dinov2_teacher(version):
    launcher = ROOT / f"examples/launch_sft_action_policy_libero_10_edge_reptileinit_{version}.sh"
    text = launcher.read_text()
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    assert f"action_policy_libero_10_edge_reptileinit_{version}.toml" in text
    assert 'export HF_HOME="${HF_HOME:-$COSMOS_STORAGE/hf_cache}"' in text  # frozen facebook/dinov2-base from the hub cache
    assert "models--facebook--dinov2-base" in text
    for var in ("LIBERO_ROOT", "REPTILE_CKPT_PATH", "META_ACTION_INIT_PATH"):
        assert var in text


def test_repa_experiment_routes_ranking_to_both_loaders_and_keeps_native_frames():
    from omegaconf import OmegaConf

    from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_edge_repa import (
        action_policy_libero_edge_repa,
    )

    cfg = OmegaConf.create(action_policy_libero_edge_repa, flags={"allow_objects": True})
    # Off by default: the existing REPA recipes (repa_dinov2_v*, masked_jepa) are unchanged.
    assert cfg.model.config.language_ranking.enabled is False
    assert cfg.model.config.repa.enabled is True
    for split in ("dataloader_train", "dataloader_val"):
        ds = cfg[split].dataloader.datasets.libero.dataset
        assert ds.language_ranking is False and ds.keep_native_video is True
        assert ds.episode_balanced_sampling is False
    cfg.model.config.language_ranking.enabled = True
    cfg.model.config.language_ranking.negative_sampling = "action_similarity"
    cfg.model.config.language_ranking.hard_negative_fraction = 0.5
    for split in ("dataloader_train", "dataloader_val"):
        ds = cfg[split].dataloader.datasets.libero.dataset
        assert ds.language_ranking is True
        assert ds.language_ranking_negative_sampling == "action_similarity"
        assert ds.language_ranking_hard_fraction == 0.5
    keys = cfg.trainer.callbacks.val_loss_breakdown["keys"]  # ["keys"]: .keys is the DictConfig method
    assert "language_ranking_loss" in keys and "repa_loss" in keys and "repa_rel_loss" in keys


def test_plain_experiment_is_untouched_by_the_repa_wiring():
    from omegaconf import OmegaConf

    from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_edge import (
        action_policy_libero_edge,
    )

    cfg = OmegaConf.create(action_policy_libero_edge, flags={"allow_objects": True})
    assert "repa" not in cfg.model.config
    # The plain loaders ship no native frames (the REPA teacher would have nothing to encode).
    assert not cfg.dataloader_train.dataloader.datasets.libero.dataset.get("keep_native_video", False)
    assert cfg.model.config.language_ranking.enabled is False
