import os
import pathlib
import re

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from openpi.hs11 import toml_config

ROOT = pathlib.Path(__file__).resolve().parents[3]
TOML_DIR = ROOT / "scripts" / "hs11" / "toml"
LIBERO_TOMLS = sorted(p for p in TOML_DIR.glob("*.toml") if not p.name.startswith("reptile_"))
REPTILE_TOMLS = sorted(TOML_DIR.glob("reptile_*.toml"))


_SUITES = ("libero10", "goal", "object", "spatial")
_GENERATED = sorted(f"{s}_{d}_{i}_v1" for s in _SUITES for d in ("all", "3ep", "1ep") for i in ("base", "reptile"))


def test_every_generated_experiment_has_a_toml_and_a_launcher():
    # the user may add further TOMLs (e.g. *_v2 copies run through TOML_FILE=...), so only the generated set is pinned
    assert len(LIBERO_TOMLS) >= 24 and len(REPTILE_TOMLS) >= 2
    for stem in [*_GENERATED, "reptile_pi05_v1", "reptile_pi05_joint_v1"]:
        assert (TOML_DIR / f"{stem}.toml").is_file(), stem
        launcher = ROOT / "scripts" / "hs11" / f"launch_{stem}.sh"
        assert launcher.is_file() and f"{stem}.toml" in launcher.read_text(), stem


@pytest.mark.parametrize("path", LIBERO_TOMLS, ids=lambda p: p.stem)
def test_libero_tomls_build_train_configs(path, tmp_path, monkeypatch):
    overrides = []
    if "_reptile_" in path.stem:
        try:  # default init_params points at a Reptile run that may not have been trained yet
            toml_config.load_train_config(path)
        except FileNotFoundError:
            pass
        (tmp_path / "params").mkdir()
        overrides = [f"checkpoint.init_params={tmp_path / 'params'}"]
    import tomllib

    raw = tomllib.loads(path.read_text())
    cfg = toml_config.load_train_config(path, overrides)
    suite, data, init = path.stem.split("_")[:3]
    assert cfg.name == raw["job"]["config"] == f"pi05_libero_v3_{ {'libero10': 'libero_10', 'goal': 'libero_goal', 'object': 'libero_object', 'spatial': 'libero_spatial'}[suite] }_{data}_{init}"
    assert cfg.exp_name == raw["job"]["exp_name"]
    wandb = raw.get("wandb", {})
    assert cfg.wandb_enabled is bool(wandb.get("enabled", False))
    assert os.environ["WANDB_MODE"] == (wandb.get("mode", "offline") if cfg.wandb_enabled else "disabled")
    assert cfg.batch_size == raw["train"]["batch_size"] and cfg.num_train_steps == raw["train"]["num_train_steps"]
    assert cfg.optimizer.clip_gradient_norm == raw["optimizer"]["clip_gradient_norm"] and cfg.lr_schedule.peak_lr == raw["optimizer"]["peak_lr"]
    assert cfg.ema_decay == (raw["train"]["ema_decay"] if raw["train"].get("ema_enabled", True) else None)
    assert cfg.fsdp_devices >= 1  # 0 in the TOML -> device count
    assert (cfg.data.episode_subset is None) == (raw["data"]["episode_subset"] == "")
    if init == "base":
        assert cfg.weight_loader.params_path.startswith("gs://openpi-assets/checkpoints/pi05_base")
    else:
        assert cfg.weight_loader.params_path == str(tmp_path / "params")


@pytest.mark.parametrize("path", REPTILE_TOMLS, ids=lambda p: p.stem)
def test_reptile_tomls_build_reptile_configs(path):
    cfg = toml_config.load_reptile_config(path)
    assert cfg.exp_name == path.stem.replace("reptile_pi05_", "reptile_") and cfg.wandb_enabled is True
    assert cfg.inner_lr == 5e-5 and cfg.inner_clip == 1.0 and cfg.model.action_horizon == 10
    assert cfg.weight_loader.params_path.startswith("gs://") and cfg.embodiments == ("fractal", "bridge", "robomind_ur", "robomind_franka", "molmoact2_yam")
    # train_reptile.py splits the support / query sets into equal micro-batches sharded over the (4) GPUs
    assert cfg.sampler.support_size % cfg.inner_batch_size == 0 and cfg.sampler.query_size % cfg.inner_batch_size == 0
    assert cfg.inner_batch_size % 4 == 0
    if path.stem in ("reptile_pi05_v1", "reptile_pi05_joint_v1"):  # the cosmos_hs11 v4 defaults
        assert cfg.sampler.k_shot == 32 and cfg.sampler.windows_per_demo == 8 and cfg.sampler.support_size == 256
        assert cfg.inner_batch_size == 32
    elif re.fullmatch(r"reptile_pi05_v[2-9]", path.stem):  # 2026-10-22 variants: v1 compute budget, diverse support, query diagnostics every iteration
        assert cfg.inner_steps * cfg.sampler.support_size * cfg.max_iter == 10 * 256 * 1000
        assert cfg.sampler.windows_per_demo <= 2 and cfg.sampler.k_shot >= 128 and cfg.sampler.disjoint_query_tasks
        assert cfg.eval_query_every == 1 and cfg.max_iter % cfg.save_interval == 0
    if "joint" in path.stem:
        assert cfg.inner_steps == 1 and cfg.meta_lr == 1.0 and cfg.meta_lr_min_ratio == 1.0 and not cfg.inner_reset_optimizer and cfg.max_iter == 10000
    elif path.stem == "reptile_pi05_v1":
        assert cfg.inner_steps == 10 and cfg.meta_lr == 0.5 and cfg.meta_lr_min_ratio == 0.1 and cfg.inner_reset_optimizer and cfg.max_iter == 1000


def test_overrides_wandb_and_validation(tmp_path):
    path = TOML_DIR / "goal_3ep_base_v1.toml"
    cfg = toml_config.load_train_config(path, ["train.num_train_steps=20", "wandb.enabled=false", "train.ema_enabled=false", "optimizer.peak_lr=1e-5", "train.batch_size=8"])
    assert cfg.num_train_steps == 20 and cfg.wandb_enabled is False and os.environ["WANDB_MODE"] == "disabled"
    assert cfg.ema_decay is None and cfg.lr_schedule.peak_lr == 1e-5 and cfg.batch_size == 8
    cfg = toml_config.load_train_config(path, ["wandb.mode=online", 'data.episode_subset=""'])
    assert os.environ["WANDB_MODE"] == "online" and cfg.data.episode_subset is None
    assert toml_config.parse_override_value("checkpoints/x/params") == "checkpoints/x/params"
    assert toml_config.parse_override_value("[1, 2]") == [1, 2] and toml_config.parse_override_value("3e-5") == 3e-5
    with pytest.raises(ValueError, match="unknown keys"):
        toml_config.load_train_config(path, ["train.batch_sz=8"])
    with pytest.raises(ValueError, match="unknown TOML sections"):
        toml_config.load_train_config(path, ["trainer.x=1"])
    with pytest.raises(ValueError, match="suite"):
        toml_config.load_train_config(path, ["data.suite=libero_10"])
    with pytest.raises(FileNotFoundError, match="episode_subset"):
        toml_config.load_train_config(path, ["data.episode_subset=nope.json"])
    with pytest.raises(ValueError, match="wandb"):
        toml_config.load_train_config(path, ["wandb.mode=sometimes"])
    rep = toml_config.load_reptile_config(TOML_DIR / "reptile_pi05_v1.toml", ["outer.max_iter=2", "inner.steps=3", "sampler.k_shot=4", "wandb.enabled=false"])
    assert rep.max_iter == 2 and rep.inner_steps == 3 and rep.sampler.k_shot == 4 and rep.wandb_enabled is False
    with pytest.raises(ValueError, match="unknown keys"):
        toml_config.load_reptile_config(TOML_DIR / "reptile_pi05_v1.toml", ["inner.step=3"])


def test_resume_defaults_to_true_and_existing_dirs_resume(tmp_path, monkeypatch):
    """A checkpoint dir that already exists must continue (user rule), never raise openpi's FileExistsError."""
    import tomllib

    src = TOML_DIR / "goal_3ep_base_v1.toml"
    raw = tomllib.loads(src.read_text())
    assert raw["train"]["resume"] is True and raw["train"]["overwrite"] is False
    for t in [*LIBERO_TOMLS, *REPTILE_TOMLS]:
        sec = tomllib.loads(t.read_text()).get("infra" if t.name.startswith("reptile_") else "train", {})
        assert sec.get("resume", True) is True and not sec.get("overwrite", False), t.name
    # key omitted -> still true
    stripped = tmp_path / "no_resume_key.toml"
    stripped.write_text("\n".join(l for l in src.read_text().splitlines() if not l.startswith("resume")))
    cfg = toml_config.load_train_config(stripped)
    assert cfg.resume is True and cfg.overwrite is False
    assert toml_config.load_train_config(stripped, ["train.resume=false"]).resume is False

    from openpi.training import checkpoints as _checkpoints

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "wandb_id.txt").write_text("abc")  # the crashed-before-first-save case
    mngr, resuming = _checkpoints.initialize_checkpoint_dir(ckpt, keep_period=None, overwrite=False, resume=True)
    assert resuming is False  # nothing to restore -> fresh start in place
    assert "fresh start in place" in toml_config.checkpoint_status(ckpt, overwrite=False, resume=True)
    (ckpt / "1000").mkdir()
    assert "RESUMING from step 1000" in toml_config.checkpoint_status(ckpt, overwrite=False, resume=True)
    assert "refuse" in toml_config.checkpoint_status(ckpt, overwrite=False, resume=False)
    with pytest.raises(FileExistsError):
        _checkpoints.initialize_checkpoint_dir(ckpt, keep_period=None, overwrite=False, resume=False)
