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
    # heads are meta-initialized: 1.0 in the main recipes; the user's *_v2 arm re-tests the fresh-head 5x boost
    assert all(
        raw["optimizer"]["lr_multipliers"][k] in (1.0, 5.0) for k in ("action2llm", "llm2action", "action_modality_embed")
    )
    assert "checkpoint.meta_action_init_domain_id=5" in overrides


def test_two_recipes_exist() -> None:
    stems = {p.stem for p in _REPA_TOMLS}
    assert {
        "action_policy_libero_10_edge_reptileinit_repa_dinov2",
        "action_policy_libero_10_edge_reptileinit_repa_dinov2_v3",
        "action_policy_libero_10_edge_reptileinit_repa_dinov2_v5",
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
        "loss_weight_warmup_steps",
        "sigma_min",
        "sigma_max",
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
    if toml_path.stem.endswith("_repa_dinov2_v3"):
        # v3 = the dinov2 recipe with the cosine-term weight ramped linearly 0 -> 5.0 over the first 200 iterations
        assert (repa_raw["teacher"], repa_raw["loss_weight"], repa_raw["layer_index"]) == ("dinov2_vitb14", 5.0, 8)
        assert repa_raw["loss_weight_warmup_steps"] == 200
        assert "model.config.repa.loss_weight_warmup_steps=200" in overrides
    elif not toml_path.stem.endswith("_repa_dinov2_v4"):  # v4 (user variant) also ramps; the others keep the constant weight
        assert repa_raw.get("loss_weight_warmup_steps", 0) == 0
    # v6 / v7 = v2 + noise-level gate (sigma <= 0.5 / sigma >= 0.5); nothing else may change
    if toml_path.stem.endswith(("_repa_dinov2_v6", "_repa_dinov2_v7")):
        v2 = tomllib.load(open(_TOML_DIR / "action_policy_libero_10_edge_reptileinit_repa_dinov2_v2.toml", "rb"))
        gate = {"sigma_max": 0.5} if toml_path.stem.endswith("_v6") else {"sigma_min": 0.5}
        assert {k: v for k, v in repa_raw.items() if k not in gate} == v2["model"]["repa"]
        assert all(repa_raw[k] == v for k, v in gate.items())
        for section in ("model", "optimizer", "trainer", "checkpoint", "dataloader_train", "dataloader_val"):
            if section == "model":
                assert {k: v for k, v in raw[section].items() if k != "repa"} == {k: v for k, v in v2[section].items() if k != "repa"}
            else:
                assert raw[section] == v2[section], section
        key, val = next(iter(gate.items()))
        assert f"model.config.repa.{key}={val}" in overrides
    else:
        assert "sigma_min" not in repa_raw and "sigma_max" not in repa_raw
    if toml_path.stem.endswith("_repa_dinov2_v5"):
        # v5 = v2 (DINOv2-B, block 24, w 5.0) + less pooling: 2x2 teacher sub-cells per MoT token
        assert (repa_raw["teacher"], repa_raw["layer_index"], repa_raw["loss_weight"]) == ("dinov2_vitb14", 24, 5.0)
        assert repa_raw["target_subgrid_thw"] == [1, 2, 2]
        assert "model.config.repa.target_subgrid_thw=[1,2,2]" in overrides
    else:
        assert "target_subgrid_thw" not in repa_raw
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
    assert RepaConfig().loss_weight_warmup_steps == 0  # constant weight unless a recipe opts in (v3)
    assert tuple(RepaConfig().target_subgrid_thw) == (1, 1, 1)
    assert (RepaConfig().sigma_min, RepaConfig().sigma_max) == (0.0, 1.0)
    assert RepaConfig(sigma_max=0.5).sigma_max == 0.5 and RepaConfig(sigma_min=0.5).sigma_min == 0.5
    with pytest.raises(ValueError):
        RepaConfig(sigma_max=1.5)
    from cosmos_framework.model.generator.repa.masked_prediction import validate_masked_prediction_config

    with pytest.raises(ValueError, match="target_subgrid_thw"):
        validate_masked_prediction_config(RepaConfig(objective="masked_prediction", target_subgrid_thw=(1, 2, 2)))
    validate_masked_prediction_config(RepaConfig(objective="masked_prediction"))  # default sub-grid is fine
    with pytest.raises(ValueError):
        RepaConfig(projector_type="conv")
    sig = SigRegConfig()  # the hs10 2026-09-22 SIGReg (input / normalize_by_count) is the version carried here
    assert (sig.layer_index, sig.loss_weight, sig.input, sig.normalize_by_count) == (8, 1.0, "repa_projection", True)


def test_repa_loss_weight_ramp() -> None:
    """v3: the cosine-term weight follows the masked-JEPA ramp (0 at iteration 0); warmup 0 keeps the constant weight."""
    from cosmos_framework.model.generator.repa.masked_prediction import auxiliary_weight

    assert [auxiliary_weight(5.0, 200, it) for it in (0, 50, 100, 200, 1999)] == [0.0, 1.25, 2.5, 5.0, 5.0]
    assert auxiliary_weight(5.0, 0, 0) == 5.0 and auxiliary_weight(5.0, 0, 10**6) == 5.0


def test_vlm_task_skips_repa_block() -> None:
    raw = {"job": {"task": "vlm", "experiment": "x"}, "model": {"repa": {"enabled": True}, "sigreg": {"enabled": True}}}
    assert not any("repa" in o or "sigreg" in o for o in build_hydra_overrides(raw))


def test_unknown_repa_key_is_rejected() -> None:
    with pytest.raises(Exception):
        SFTExperimentConfig.model_validate({"model": {"repa": {"weight": 0.5}}})


_NANO_REPTILE_TOMLS = sorted(_TOML_DIR.glob("action_policy_libero_10_nano_reptileinit*.toml"))


@pytest.mark.parametrize("toml_path", _NANO_REPTILE_TOMLS, ids=[p.stem for p in _NANO_REPTILE_TOMLS])
def test_nano_reptileinit_tomls_validate_and_route(toml_path: Path) -> None:
    """Nano Reptile-init recipes: plain -> action_policy_libero_nano; v2 -> action_policy_libero_nano_repa + DINOv2-L on block 36."""
    raw = tomllib.load(open(toml_path, "rb"))
    SFTExperimentConfig.model_validate(raw)
    overrides = build_hydra_overrides(raw)
    ckpt = raw["checkpoint"]
    assert ckpt["load_path"] == "${oc.env:REPTILE_CKPT_PATH}" and ckpt["meta_action_init_domain_id"] == 5
    assert {"net_ema.", "action2llm", "llm2action", "action_modality_embed", "action_pos_embed"} <= set(ckpt["keys_to_skip_loading"])
    assert raw["model"]["compile"]["enabled"] is False  # Ampere smem limit (see docs)
    if "repa" in raw["model"]:
        repa_raw = raw["model"]["repa"]
        assert "experiment=action_policy_libero_nano_repa" in overrides
        assert "repa_" in ckpt["keys_to_skip_loading"]
        assert (repa_raw["teacher"], repa_raw["teacher_input_size"], repa_raw["layer_index"]) == ("dinov2_vitl14", 224, 36)
        assert repa_raw["loss_weight"] > 0 and repa_raw.get("objective", "token") == "token"  # v2/v3/v4 = weight sweep
        assert "model.config.repa.layer_index=36" in overrides and "model.config.repa.teacher=dinov2_vitl14" in overrides
        if toml_path.stem.endswith("_v5"):  # v5 = v2 + less pooling (2x2 teacher sub-cells per MoT token)
            assert repa_raw["target_subgrid_thw"] == [1, 2, 2]
            assert "model.config.repa.target_subgrid_thw=[1,2,2]" in overrides
        else:
            assert "target_subgrid_thw" not in repa_raw
    else:
        assert "experiment=action_policy_libero_nano" in overrides
        assert not any(o.startswith("model.config.repa") for o in overrides)


def test_nano_repa_experiment_is_registered() -> None:
    import cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_nano_repa as m

    cfg = m.action_policy_libero_nano_repa
    assert "repa_" in cfg["optimizer"]["keys_to_select"] and "repa_" in cfg["checkpoint"]["keys_to_skip_loading"]
    repa = cfg["model"]["config"]["repa"]
    assert repa["enabled"] is True and repa["teacher"] == "vjepa2_1_vit_large_384" and repa["layer_index"] == 10
    # every default key must exist on the hs11 RepaConfig (which the TOML overrides land on)
    from cosmos_framework.configs.base.defaults.model_config import RepaConfig
    import attrs

    assert set(repa) <= {f.name for f in attrs.fields(RepaConfig)}, set(repa) - {f.name for f in attrs.fields(RepaConfig)}
    RepaConfig(**repa)
