# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""cosmos_hs11: every Reptile meta-training TOML (Edge and Nano) validates, routes to its experiment and has a
consistent episode / inner-batch geometry; the Nano reptile-init post-training TOML is wired like the Edge one."""

from pathlib import Path

import pytest
import tomllib

from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig
from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides

_TOML_DIR = Path(__file__).resolve().parents[3] / "examples" / "toml" / "sft_config"
_REPTILE_TOMLS = sorted(_TOML_DIR.glob("action_reptile_meta_*.toml"))


def _meta_cfg(raw: dict):
    from cosmos_framework.scripts.train_action_meta import SAMPLER_OVERRIDE_KEYS
    from cosmos_framework.scripts.train_action_reptile import ReptileTrainConfig

    meta = dict(raw["custom"]["meta"])
    meta.pop("dataset_kwargs", None)
    sampler = {k: meta.pop(k) for k in list(meta) if k in SAMPLER_OVERRIDE_KEYS}
    return ReptileTrainConfig.from_dict(meta), sampler


@pytest.mark.parametrize("toml_path", _REPTILE_TOMLS, ids=[p.stem for p in _REPTILE_TOMLS])
def test_reptile_tomls_validate_and_are_geometrically_consistent(toml_path: Path) -> None:
    raw = tomllib.load(open(toml_path, "rb"))
    SFTExperimentConfig.model_validate(raw)
    overrides = build_hydra_overrides(raw)
    nano = "nano" in toml_path.stem
    assert f"experiment=action_reptile_meta_{'nano' if nano else 'edge'}" in overrides
    cfg, sampler = _meta_cfg(raw)
    assert cfg.theta_mode in ("full", "lora")
    shard = raw["model"]["parallelism"]["data_parallel_shard_degree"]
    support = sampler["k_shot"] * sampler["windows_per_demo"]
    assert support >= shard, "every rank needs at least one support window"
    keys = raw["optimizer"]["keys_to_select"]
    if cfg.theta_mode == "lora":
        assert raw["model"]["lora_enabled"] is True and "lora_" in keys and "moe_gen" not in keys
        assert "lora_" in raw["checkpoint"]["keys_to_skip_loading"]
    if nano:
        assert "k_norm_und_for_gen" not in keys, "Nano has no k_norm_und_for_gen"
        assert "model.config.compile.enabled=false" in overrides, "compile must be off for Nano on 48 GB Ampere"
        assert "action_pos_embed" in raw["checkpoint"]["keys_to_skip_loading"]
        if cfg.theta_mode == "lora":
            assert "mlp_moe_gen.gate_proj" in raw["model"]["lora_target_modules"]
    if "smoke" not in toml_path.stem:
        # production recipes: the support splits into a whole number of inner-step minibatches (1 = full-batch inner loop)
        step_batch = shard * sampler["max_samples_per_batch"]
        assert support % step_batch == 0 and support >= step_batch, (support, shard, sampler["max_samples_per_batch"])
        if toml_path.stem.endswith("_h200"):
            # matched to a downstream gbs of 8 x 128: one full-batch inner step of 1024 windows, downstream lr
            assert support == step_batch == 1024 and cfg.inner_steps == 20 and raw["optimizer"]["lr"] == 5e-5
        assert cfg.save_at_end is True
    else:
        assert raw["trainer"]["max_iter"] <= 3
        if nano:  # the Nano smoke must not write the ~63 GB final DCP (the older Edge smoke predates the knob)
            assert cfg.save_at_end is False


def test_nano_recipes_mirror_the_edge_recipe() -> None:
    edge = tomllib.load(open(_TOML_DIR / "action_reptile_meta_edge2.toml", "rb"))
    nano = tomllib.load(open(_TOML_DIR / "action_reptile_meta_nano.toml", "rb"))
    e, n = edge["custom"]["meta"], nano["custom"]["meta"]
    for k in (
        "k_shot",
        "q_query",
        "windows_per_demo",
        "inner_steps",
        "inner_warmup_steps",
        "meta_lr",
        "meta_lr_min_ratio",
        "loss_mode",
        "disjoint_tasks",
    ):
        assert e[k] == n[k], k
    # same 256-window inner batch, different topology
    assert edge["model"]["parallelism"]["data_parallel_shard_degree"] * e["max_samples_per_batch"] == 256
    assert nano["model"]["parallelism"]["data_parallel_shard_degree"] * n["max_samples_per_batch"] == 256
    assert edge["optimizer"]["lr"] == nano["optimizer"]["lr"] == 5e-5
    h200 = tomllib.load(open(_TOML_DIR / "action_reptile_meta_nano_h200.toml", "rb"))
    b = h200["custom"]["meta"]
    assert b["max_samples_per_batch"] == 128 and b["k_shot"] * b["windows_per_demo"] == 8 * 128
    assert b["windows_per_demo"] == 16  # bridge / fractal demos have 20-28 windows: K grows, windows_per_demo does not
    assert h200["optimizer"]["lr"] == pytest.approx(5e-5)  # the downstream LR; Cosmos3 keeps 5e-5 from gbs 256 to 2048


def test_nano_reptileinit_downstream_toml() -> None:
    raw = tomllib.load(open(_TOML_DIR / "action_policy_libero_10_nano_reptileinit.toml", "rb"))
    SFTExperimentConfig.model_validate(raw)
    overrides = build_hydra_overrides(raw)
    assert "experiment=action_policy_libero_nano" in overrides
    ckpt = raw["checkpoint"]
    assert ckpt["load_path"] == "${oc.env:REPTILE_CKPT_PATH}"
    assert (
        ckpt["meta_action_init_path"].startswith("${oc.env:META_ACTION_INIT_PATH}")
        and ckpt["meta_action_init_domain_id"] == 5
    )
    assert ckpt["meta_action_init_include_lora"] is False and ckpt["meta_action_init_include_time_embedder"] is False
    assert {"net_ema.", "action2llm", "llm2action", "action_modality_embed", "action_pos_embed"} <= set(
        ckpt["keys_to_skip_loading"]
    )
    assert all(
        raw["optimizer"]["lr_multipliers"][k] == 1.0 for k in ("action2llm", "llm2action", "action_modality_embed")
    )
    assert (
        raw["model"]["parallelism"]["data_parallel_shard_degree"] == 8 and raw["model"]["compile"]["enabled"] is False
    )
    base = tomllib.load(open(_TOML_DIR / "action_policy_libero_10_nano_fewshot.toml", "rb"))
    for section in ("model", "scheduler", "trainer", "dataloader_train", "dataloader_val"):
        assert raw[section] == base[section], section
