# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""cosmos_hs11: the REPA / masked-JEPA post-training recipes on the Reptile init validate and route to model.config.repa."""

from pathlib import Path

import pytest
import tomllib

from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig
from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides

_TOML_DIR = Path(__file__).resolve().parents[3] / "examples" / "toml" / "sft_config"
_REPTILE_TOMLS = sorted(_TOML_DIR.glob("action_policy_libero_10_edge_reptileinit*.toml"))
_REPA_TOMLS = [p for p in _REPTILE_TOMLS if "repa" in tomllib.load(open(p, "rb")).get("model", {})]
_PLAIN_TOMLS = [p for p in _REPTILE_TOMLS if p not in _REPA_TOMLS]

_MASKED_KEYS = (
    "masked_ratio_min",
    "masked_ratio_max",
    "masked_visible_weight",
    "masked_max_samples",
    "masked_warmup_steps",
    "masked_seed",
)


def _reptileinit_asserts(raw: dict, overrides: list[str]) -> None:
    ckpt = raw["checkpoint"]
    assert ckpt["load_path"] == "${oc.env:REPTILE_CKPT_PATH}"
    assert ckpt["meta_action_init_path"].startswith("${oc.env:META_ACTION_INIT_PATH}")
    assert ckpt["meta_action_init_domain_id"] == 5
    assert ckpt["meta_action_init_include_lora"] is False and ckpt["meta_action_init_include_time_embedder"] is False
    assert raw["model"]["lora_enabled"] is False
    assert {"net_ema.", "action2llm", "llm2action", "action_modality_embed"} <= set(ckpt["keys_to_skip_loading"])
    assert all(
        raw["optimizer"]["lr_multipliers"][k] == 1.0 for k in ("action2llm", "llm2action", "action_modality_embed")
    )
    assert "checkpoint.meta_action_init_domain_id=5" in overrides


def test_two_recipes_exist() -> None:
    stems = {p.stem for p in _REPA_TOMLS}
    assert {
        "action_policy_libero_10_edge_reptileinit_repa_dinov2",
        "action_policy_libero_10_edge_reptileinit_masked_jepa",
    } <= stems


@pytest.mark.parametrize("toml_path", _REPA_TOMLS, ids=[p.stem for p in _REPA_TOMLS])
def test_repa_reptileinit_tomls_validate_and_route(toml_path: Path) -> None:
    raw = tomllib.load(open(toml_path, "rb"))
    cfg = SFTExperimentConfig.model_validate(raw)
    assert cfg.model.repa.enabled is True
    overrides = build_hydra_overrides(raw)
    assert "experiment=action_policy_libero_edge_repa" in overrides
    assert "model.config.repa.enabled=true" in overrides
    assert "model.config.repa.target_grid_thw=[4,5,5]" in overrides
    assert any(o.startswith("model.config.repa.layer_index=") for o in overrides)
    _reptileinit_asserts(raw, overrides)
    assert "repa_" in raw["checkpoint"]["keys_to_skip_loading"]  # repa_head is absent from the Reptile DCP
    repa_raw = raw["model"]["repa"]
    # optional knobs route 1:1 (present in the TOML <-> present as an override)
    for key in (
        "projector_type",
        "projector_hidden_dim",
        "objective",
        "center_targets",
        "teacher_batch_size",
        *_MASKED_KEYS,
    ):
        got = [o for o in overrides if o.startswith(f"model.config.repa.{key}=")]
        if key not in repa_raw:
            assert got == [], key
            continue
        val = repa_raw[key]
        assert got == [f"model.config.repa.{key}={str(val).lower() if isinstance(val, bool) else val}"], key
    if toml_path.stem.endswith("_repa_dinov2"):
        assert (repa_raw["teacher"], repa_raw["teacher_input_size"], repa_raw["target_adapter"]) == (
            "dinov2_vitb14",
            224,
            "avgpool",
        )
        assert (repa_raw["loss_weight"], repa_raw["layer_index"]) == (5.0, 8)
        assert repa_raw.get("objective", "token") == "token"
    if toml_path.stem.endswith("_masked_jepa"):
        assert repa_raw["objective"] == "masked_prediction"
        assert (repa_raw["teacher"], repa_raw["teacher_input_size"], repa_raw["target_adapter"]) == (
            "vjepa2_1_vit_base_384",
            256,
            "avgpool",
        )
        assert (repa_raw["loss_weight"], repa_raw["layer_index"], repa_raw["projector_hidden_dim"]) == (0.5, 8, 512)
        assert 0 < repa_raw["masked_ratio_min"] < repa_raw["masked_ratio_max"] < 1
        assert raw["trainer"]["seed"] == 42 and "trainer.seed=42" in overrides


@pytest.mark.parametrize("toml_path", _PLAIN_TOMLS, ids=[p.stem for p in _PLAIN_TOMLS])
def test_plain_reptileinit_tomls_do_not_touch_repa(toml_path: Path) -> None:
    raw = tomllib.load(open(toml_path, "rb"))
    SFTExperimentConfig.model_validate(raw)
    overrides = build_hydra_overrides(raw)
    assert "experiment=action_policy_libero_edge" in overrides
    assert not any(o.startswith("model.config.repa") for o in overrides)
    _reptileinit_asserts(raw, overrides)


def test_repa_and_sigreg_model_configs_validate() -> None:
    from cosmos_framework.configs.base.defaults.model_config import RepaConfig, SigRegConfig

    assert (
        RepaConfig().objective == "token" and RepaConfig(objective="masked_prediction").objective == "masked_prediction"
    )
    with pytest.raises(ValueError):
        RepaConfig(objective="difference")
    assert (RepaConfig().masked_ratio_min, RepaConfig().masked_ratio_max, RepaConfig().masked_max_samples) == (
        0.4,
        0.7,
        16,
    )
    assert RepaConfig(projector_type="linear").projector_type == "linear"
    with pytest.raises(ValueError):
        RepaConfig(projector_type="conv")
    sig = SigRegConfig()  # the hs10 2026-09-22 SIGReg (input / normalize_by_count) is the version carried here
    assert (sig.layer_index, sig.loss_weight, sig.input, sig.normalize_by_count) == (8, 1.0, "repa_projection", True)


def test_vlm_task_skips_repa_block() -> None:
    raw = {"job": {"task": "vlm", "experiment": "x"}, "model": {"repa": {"enabled": True}, "sigreg": {"enabled": True}}}
    assert not any("repa" in o or "sigreg" in o for o in build_hydra_overrides(raw))


def test_unknown_repa_key_is_rejected() -> None:
    with pytest.raises(Exception):
        SFTExperimentConfig.model_validate({"model": {"repa": {"weight": 0.5}}})
