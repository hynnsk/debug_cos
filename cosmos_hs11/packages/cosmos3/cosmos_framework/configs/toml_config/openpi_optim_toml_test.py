"""cosmos_hs11 v114: openpi-matched optimization knobs -- `[scheduler].lr_scheduler_type` and `[model.ema].beta`."""

from __future__ import annotations

import inspect
import tomllib
import types
from pathlib import Path

import pytest

from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig
from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides

_TOML_DIR = Path(__file__).parents[3] / "examples" / "toml" / "sft_config"
_TOML = _TOML_DIR / "action_policy_libero_10_edge_reptileinit_v114.toml"
_OPENPI_MATCHED = [
    _TOML,
    _TOML_DIR / "action_policy_libero_goal_edge_reptileinit_v105.toml",
    _TOML_DIR / "action_policy_libero_spatial_edge_reptileinit_v103.toml",
    _TOML_DIR / "action_policy_libero_object_edge_reptileinit_v99.toml",
]


@pytest.mark.parametrize("toml_path", _OPENPI_MATCHED, ids=lambda p: p.stem)
def test_openpi_matched_recipes_emit_the_expected_overrides(toml_path: Path) -> None:
    raw = tomllib.loads(toml_path.read_text())
    cfg = SFTExperimentConfig.model_validate(raw)
    assert cfg.scheduler.lr_scheduler_type == "LambdaCosine" and cfg.model.ema.beta == 0.999
    overrides = build_hydra_overrides(raw)
    for expected in (
        "optimizer.betas=[0.9,0.95]",
        "optimizer.eps=1e-08",
        "optimizer.weight_decay=1e-10",
        "scheduler.lr_scheduler_type=LambdaCosine",
        "scheduler.cycle_lengths=[5000]",
        "scheduler.warm_up_steps=[200]",
        "scheduler.f_min=[0.1]",
        "model.config.ema.beta=0.999",
        "trainer.max_iter=5000",
        "dataloader_train.max_samples_per_batch=32",
    ):  # save_iter / shard degree / wandb_mode are run-level knobs the user tunes in place, not part of the match
        assert expected in overrides, (expected, overrides)
    assert cfg.optimizer.lr == 5e-5 and cfg.optimizer.lr_multipliers["action2llm"] == 5.0  # untouched Cosmos knobs
    assert "lr_scheduler_type" in raw["scheduler"]  # launchers exist for every openpi-matched recipe
    launcher = toml_path.parents[2] / toml_path.name.replace("action_policy_", "launch_sft_action_policy_").replace(".toml", ".sh")
    assert launcher.is_file() and toml_path.name in launcher.read_text(), launcher


def test_schema_rejects_bad_values() -> None:
    from pydantic import ValidationError

    base = tomllib.loads(_TOML.read_text())
    bad_type = {**base, "scheduler": {**base["scheduler"], "lr_scheduler_type": "cosine"}}
    with pytest.raises(ValidationError):
        SFTExperimentConfig.model_validate(bad_type)
    bad_beta = {**base, "model": {**base["model"], "ema": {**base["model"]["ema"], "beta": 1.0}}}
    with pytest.raises(ValidationError):
        SFTExperimentConfig.model_validate(bad_beta)


def test_vlm_remap_skips_the_vfm_only_knobs() -> None:
    raw = tomllib.loads(_TOML.read_text())
    raw["job"]["task"] = "vlm"
    overrides = build_hydra_overrides(raw)
    assert not any("lr_scheduler_type" in o or "ema.beta" in o for o in overrides), overrides


def test_ema_beta_is_constant_when_set_and_power_otherwise() -> None:
    try:
        from cosmos_framework.model.generator import omni_mot_model
    except ImportError as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"model stack not importable here: {exc!r}")
    cls = next(c for _, c in inspect.getmembers(omni_mot_model, inspect.isclass) if "ema_beta" in c.__dict__)
    fake = types.SimpleNamespace(config=types.SimpleNamespace(ema=types.SimpleNamespace(iteration_shift=0, beta=0.999)), ema_exp_coefficient=6.94)
    assert cls.ema_beta(fake, 0) == 0.0  # first update still copies the live weights
    assert cls.ema_beta(fake, 1) == cls.ema_beta(fake, 5000) == pytest.approx(0.999)
    fake.config.ema.beta = None
    assert cls.ema_beta(fake, 1) == pytest.approx((1 - 1 / 2) ** 7.94)
