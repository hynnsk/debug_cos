# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for the Cosmos3-Nano FOMAML LoRA meta-training recipes (``action_fewshot_meta_lora_nano``)."""

from __future__ import annotations

from pathlib import Path

import pytest
import tomllib

from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig
from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides
from cosmos_framework.scripts.train_action_meta import SAMPLER_OVERRIDE_KEYS, MetaTrainConfig

REPO = Path(__file__).resolve().parents[2]
TOML_DIR = REPO / "examples/toml/sft_config"
NANO_TOML = TOML_DIR / "action_fewshot_meta_lora_nano.toml"
NANO_SMOKE_TOML = TOML_DIR / "action_fewshot_meta_lora_nano_smoke.toml"
EDGE_TOML = TOML_DIR / "action_fewshot_meta_lora_edge.toml"
NANO_TARGETS = (
    "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen,"
    "mlp_moe_gen.gate_proj,mlp_moe_gen.up_proj,mlp_moe_gen.down_proj"
)


def _strip(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if k not in SAMPLER_OVERRIDE_KEYS}


@pytest.mark.parametrize("toml_path", [NANO_TOML, NANO_SMOKE_TOML], ids=["nano", "nano_smoke"])
def test_nano_meta_tomls_validate_and_parse(toml_path: Path) -> None:
    raw = tomllib.loads(toml_path.read_text())
    SFTExperimentConfig.model_validate(raw)
    overrides = build_hydra_overrides(raw)
    assert "experiment=action_fewshot_meta_lora_nano" in overrides
    assert raw["model"]["lora_enabled"] is True and raw["model"]["lora_target_modules"] == NANO_TARGETS
    assert (raw["model"]["lora_rank"], raw["model"]["lora_alpha"]) == (32, 64)
    assert raw["model"]["compile"]["enabled"] is False
    assert raw["checkpoint"]["load_path"] == "${oc.env:BASE_CHECKPOINT_PATH}"
    meta = raw["custom"]["meta"]
    cfg = MetaTrainConfig.from_dict(_strip(meta))  # unknown keys raise
    assert cfg.include_lora and cfg.include_time_embedder and cfg.inner_optimizer == "adam" and cfg.loss_mode == "action"
    sampler = {k for k in meta if k in SAMPLER_OVERRIDE_KEYS}
    assert {"k_shot", "q_query", "windows_per_demo", "max_samples_per_batch"} <= sampler
    assert meta["max_samples_per_batch"] == meta["k_shot"] * meta["windows_per_demo"]  # support = one packed batch


def test_nano_meta_recipe_matches_the_h200_sizing() -> None:
    raw = tomllib.loads(NANO_TOML.read_text())
    meta = raw["custom"]["meta"]
    assert (meta["k_shot"], meta["q_query"], meta["windows_per_demo"], meta["max_samples_per_batch"]) == (8, 4, 8, 64)
    assert (meta["inner_steps"], meta["inner_lr"], meta["meta_lr"]) == (10, 1.0e-4, 1.0e-4)
    assert raw["trainer"]["max_iter"] in (1000, 2000)
    # the Nano recipe differs from the Edge one only where documented: targets (gate_proj), episode sizing, inner steps
    edge = tomllib.loads(EDGE_TOML.read_text())
    em = edge["custom"]["meta"]
    same = ("disjoint_tasks", "include_modality_embed", "include_lora", "include_time_embedder", "init_source",
            "scratch_domain_id", "inner_optimizer", "inner_grad_clip", "meta_lr", "meta_grad_clip",
            "meta_lr_warmup_iters", "meta_lr_min_ratio", "loss_mode", "eval_zero_shot_every")
    for key in same:
        assert meta[key] == em[key], key
    assert "mlp_moe_gen.gate_proj" in raw["model"]["lora_target_modules"]
    assert "gate_proj" not in edge["model"]["lora_target_modules"]  # Nemotron MLP has no gate


def test_nano_meta_experiment_config() -> None:
    import cosmos_framework.configs.base.experiment.action.meta.action_fewshot_meta_lora_nano as m

    cfg = m.action_fewshot_meta_lora_nano
    mc = cfg["model"]["config"]
    assert mc["parallelism"]["enable_inference_mode"] is True and mc["parallelism"]["data_parallel_shard_degree"] == 1
    assert mc["parallelism"]["fsdp_master_dtype"] == "bfloat16"
    assert mc["ema"]["enabled"] is False and mc["compile"]["enabled"] is False
    assert mc["activation_checkpointing"]["mode"] == "full"
    assert mc["lora_enabled"] is True and mc["lora_target_modules"] == m.LORA_TARGET_MODULES_NANO == NANO_TARGETS
    assert (mc["lora_rank"], mc["lora_alpha"], mc["lora_freeze_base"]) == (32, 64, True)
    # Nano tier deltas of the LIBERO Nano recipe
    assert mc["rectified_flow_training_config"]["loss_scale"] == 10.0
    assert mc["rectified_flow_training_config"]["image_loss_scale"] is None
    assert mc["diffusion_expert_config"]["load_weights_from_pretrained"] is False
    assert mc["tokenizer"]["encode_exact_durations"] == [17, 61, 73]
    assert mc["max_num_tokens_after_packing"] == 45056
    assert {"net_ema.", "lora_", "action_pos_embed"} <= set(cfg["checkpoint"]["keys_to_skip_loading"])
    assert cfg["optimizer"]["keys_to_select"] == ["action2llm", "llm2action", "action_modality_embed", "lora_", "time_embedder"]
    dl = cfg["dataloader_train"]
    assert (dl["k_shot"], dl["q_query"], dl["windows_per_demo"], dl["max_samples_per_batch"]) == (8, 4, 8, 64)
    # every sampler key a TOML may override exists on the loader call (apply_sampler_overrides raises otherwise)
    raw = tomllib.loads(NANO_TOML.read_text())
    for k in raw["custom"]["meta"]:
        if k in SAMPLER_OVERRIDE_KEYS and k != "loader_timeout_s":
            assert k in dl, k
