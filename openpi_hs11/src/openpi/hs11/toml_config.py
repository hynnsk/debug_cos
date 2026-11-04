"""TOML-driven run configuration for openpi_hs11 (cosmos_hs11 style: one TOML per experiment, CLI ``key=value`` overrides).

A LIBERO post-training TOML names a registered base recipe (``[job].config``, which fixes the model and the data
pipeline) and re-states the run settings explicitly::

    [job]        config, exp_name, project_name
    [wandb]      enabled, mode (offline|online|disabled), dir, entity
    [data]       suite, episode_subset ("" = all episodes), prompt_dropout (0 = off; 0.1 trains an unconditional branch for CFG)
    [checkpoint] init (base|reptile), init_params (reptile: <ckpt>/<iter>/params), base_dir
    [train]      batch_size, num_train_steps, fsdp_devices (0 = all visible GPUs), num_workers, save_interval,
                 keep_period, log_interval, seed, ema_enabled, ema_decay, overwrite, resume (default true: an existing
                 checkpoint dir continues from its latest step and wandb run id), assets_base_dir
    [optimizer]  peak_lr, warmup_steps, decay_steps, decay_lr, weight_decay, clip_gradient_norm, b1, b2
    [model]      action_horizon; sub-table [model.language_ranking] enabled, loss_weight, margin, raw_action_dim,
                 negative_grad_scale, margin_mode, self_paced_fraction (cosmos_hs11 action-language hinge ranking; pi0_config.LanguageRankingConfig)

A Reptile TOML maps 1:1 onto ``scripts/train_reptile.py::ReptileConfig`` (sections job / wandb / model / checkpoint /
data / sampler / inner / outer / infra). Unknown sections or keys are errors, so typos cannot silently fall back to
defaults. Overrides use dotted paths: ``train.num_train_steps=20 wandb.enabled=false checkpoint.init_params=/x``.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import tomllib
from collections.abc import Iterable, Mapping
from typing import Any

import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as _weight_loaders
from openpi.hs11 import configs as hs11_configs
from openpi.hs11 import libero_v3

# ---------------------------------------------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------------------------------------------
def parse_override_value(text: str) -> Any:
    """``"20"`` -> 20, ``"false"`` -> False, ``'"x"'`` -> "x", ``"[1, 2]"`` -> [1, 2]; anything else stays a string."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def apply_overrides(raw: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` (or deeper) overrides to the parsed TOML dict (copy)."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in raw.items()}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not of the form section.key=value")
        path, value = item.split("=", 1)
        keys = path.strip().split(".")
        if len(keys) < 2:
            raise ValueError(f"override {item!r}: use a dotted path like train.batch_size=32")
        node = out
        for k in keys[:-1]:
            node = node.setdefault(k, {})
            if not isinstance(node, dict):
                raise ValueError(f"override {item!r}: {k!r} is not a table")
        node[keys[-1]] = parse_override_value(value.strip())
    return out


def _section(raw: Mapping[str, Any], name: str, allowed: Iterable[str], *, required: Iterable[str] = ()) -> dict[str, Any]:
    sec = dict(raw.get(name, {}) or {})
    unknown = sorted(set(sec) - set(allowed))
    if unknown:
        raise ValueError(f"[{name}]: unknown keys {unknown}; allowed: {sorted(allowed)}")
    missing = [k for k in required if k not in sec]
    if missing:
        raise ValueError(f"[{name}]: missing required keys {missing}")
    return sec


_LANGUAGE_RANKING_KEYS = ("enabled", "loss_weight", "margin", "raw_action_dim", "negative_grad_scale", "margin_mode", "self_paced_fraction")


def _language_ranking_config(base, model_section: Mapping[str, Any]):
    """``[model.language_ranking]`` -> ``pi0_config.LanguageRankingConfig`` (unknown keys error, like every section)."""
    sec = _section(model_section, "language_ranking", _LANGUAGE_RANKING_KEYS)
    kw: dict[str, Any] = {}
    for key, value in sec.items():
        if key == "enabled":
            kw[key] = bool(value)
        elif key in ("loss_weight", "margin", "negative_grad_scale", "self_paced_fraction"):
            kw[key] = float(value)
        elif key == "raw_action_dim":
            kw[key] = None if value is None or int(value) <= 0 else int(value)  # 0 = infer from the batch
        else:
            kw[key] = str(value)
    return dataclasses.replace(base, **kw)


def _check_sections(raw: Mapping[str, Any], allowed: Iterable[str]) -> None:
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        raise ValueError(f"unknown TOML sections {unknown}; allowed: {sorted(allowed)}")


def resolve_fsdp_devices(value: int) -> int:
    """``0`` means every device JAX can see (the SLURM allocation)."""
    value = int(value)
    if value > 0:
        return value
    import jax

    return int(jax.device_count())


def apply_wandb_section(wandb: Mapping[str, Any]) -> bool:
    """Export WANDB_MODE / WANDB_DIR / WANDB_ENTITY from ``[wandb]``; returns ``enabled``."""
    enabled = bool(wandb.get("enabled", False))
    mode = str(wandb.get("mode", "offline"))
    if mode not in ("offline", "online", "disabled"):
        raise ValueError(f"[wandb].mode must be offline|online|disabled, got {mode!r}")
    os.environ["WANDB_MODE"] = mode if enabled else "disabled"
    if wandb.get("dir"):
        os.environ["WANDB_DIR"] = str(pathlib.Path(str(wandb["dir"])).expanduser())
    if wandb.get("entity"):
        os.environ["WANDB_ENTITY"] = str(wandb["entity"])
    return enabled


def _resolve_init_params(path: str) -> str:
    """Local checkpoint dirs must exist (gs:// paths are left to the downloader)."""
    if path.startswith("gs://"):
        return path
    p = pathlib.Path(path).expanduser()
    if not (p / "_METADATA").exists() and not p.is_dir():
        raise FileNotFoundError(f"init params dir not found: {path}")
    return str(p.resolve())


# ---------------------------------------------------------------------------------------------------------------
# LIBERO post-training
# ---------------------------------------------------------------------------------------------------------------
_TRAIN_SECTIONS = ("job", "wandb", "data", "checkpoint", "train", "optimizer", "model")


def load_train_config(path: str | pathlib.Path, overrides: Iterable[str] = ()):
    """Build an openpi ``TrainConfig`` from a LIBERO TOML (+ overrides)."""
    from openpi.training import config as _config

    raw = apply_overrides(tomllib.loads(pathlib.Path(path).read_text()), overrides)
    _check_sections(raw, _TRAIN_SECTIONS)
    job = _section(raw, "job", ("config", "exp_name", "project_name"), required=("config", "exp_name"))
    wandb = _section(raw, "wandb", ("enabled", "mode", "dir", "entity"))
    data = _section(raw, "data", ("suite", "episode_subset", "prompt_dropout"))
    ckpt = _section(raw, "checkpoint", ("init", "init_params", "base_dir"))
    train = _section(
        raw,
        "train",
        ("batch_size", "num_train_steps", "fsdp_devices", "num_workers", "save_interval", "keep_period", "log_interval",
         "seed", "ema_enabled", "ema_decay", "overwrite", "resume", "assets_base_dir"),
    )
    opt = _section(raw, "optimizer", ("peak_lr", "warmup_steps", "decay_steps", "decay_lr", "weight_decay", "clip_gradient_norm", "b1", "b2"))
    model = _section(raw, "model", ("action_horizon", "language_ranking"))

    cfg = _config.get_config(str(job["config"]))
    updates: dict[str, Any] = {"exp_name": str(job["exp_name"])}
    if "project_name" in job:
        updates["project_name"] = str(job["project_name"])
    updates["wandb_enabled"] = apply_wandb_section(wandb)

    # [data] -> the registered LiberoV3DataConfig (suite must match the base recipe; subset may be changed)
    data_cfg = cfg.data
    if "suite" in data:
        suite = str(data["suite"])
        if getattr(data_cfg, "suite", suite) != suite:
            raise ValueError(f"[data].suite={suite!r} but [job].config={job['config']!r} is a {data_cfg.suite!r} recipe")
    if "episode_subset" in data:
        subset = str(data["episode_subset"]) or None
        if subset is not None:
            libero_v3_root = pathlib.Path(__file__).resolve().parents[3] / "assets" / "episode_subsets"
            if not pathlib.Path(subset).is_absolute() and not (libero_v3_root / subset).is_file():
                raise FileNotFoundError(f"[data].episode_subset {subset!r} not found under {libero_v3_root}")
        data_cfg = dataclasses.replace(data_cfg, episode_subset=subset)
    if "prompt_dropout" in data:
        pd = float(data["prompt_dropout"])
        if not 0.0 <= pd < 1.0:
            raise ValueError(f"[data].prompt_dropout must be in [0, 1), got {pd}")
        data_cfg = dataclasses.replace(data_cfg, prompt_dropout=pd)
    updates["data"] = data_cfg

    # [checkpoint]
    init = str(ckpt.get("init", "base"))
    if init == "base":
        updates["weight_loader"] = _weight_loaders.CheckpointWeightLoader(hs11_configs.PI05_BASE_PARAMS)
    elif init == "reptile":
        params = str(ckpt.get("init_params", "") or os.environ.get(hs11_configs.ENV_REPTILE_PARAMS, ""))
        if not params:
            raise ValueError("[checkpoint].init = 'reptile' needs init_params (or the REPTILE_PARAMS / HS11_REPTILE_PARAMS env)")
        updates["weight_loader"] = _weight_loaders.CheckpointWeightLoader(_resolve_init_params(params))
    else:
        raise ValueError(f"[checkpoint].init must be base|reptile, got {init!r}")
    if "base_dir" in ckpt:
        updates["checkpoint_base_dir"] = str(ckpt["base_dir"])

    # [train]
    for key in ("batch_size", "num_train_steps", "num_workers", "save_interval", "keep_period", "log_interval", "seed", "overwrite", "resume", "assets_base_dir"):
        if key in train:
            updates[key] = train[key]
    updates.setdefault("resume", True)  # an existing checkpoint dir continues from its latest step (and wandb run); overwrite wipes it
    if "fsdp_devices" in train:
        updates["fsdp_devices"] = resolve_fsdp_devices(int(train["fsdp_devices"]))
    if "ema_enabled" in train or "ema_decay" in train:
        enabled = bool(train.get("ema_enabled", True))
        updates["ema_decay"] = float(train.get("ema_decay", cfg.ema_decay or 0.999)) if enabled else None

    # [optimizer]
    if opt:
        sched = cfg.lr_schedule if isinstance(cfg.lr_schedule, _optimizer.CosineDecaySchedule) else _optimizer.CosineDecaySchedule()
        sched_kw = {k: opt[k] for k in ("peak_lr", "warmup_steps", "decay_steps", "decay_lr") if k in opt}
        updates["lr_schedule"] = dataclasses.replace(sched, **sched_kw)
        adam = cfg.optimizer if isinstance(cfg.optimizer, _optimizer.AdamW) else _optimizer.AdamW()
        adam_kw = {k: opt[k] for k in ("weight_decay", "clip_gradient_norm", "b1", "b2") if k in opt}
        updates["optimizer"] = dataclasses.replace(adam, **adam_kw)

    # [model]
    model_updates: dict[str, Any] = {}
    if "action_horizon" in model:
        model_updates["action_horizon"] = int(model["action_horizon"])
    if "language_ranking" in model:
        model_updates["language_ranking"] = _language_ranking_config(cfg.model.language_ranking, model)
    if model_updates:
        updates["model"] = dataclasses.replace(cfg.model, **model_updates)

    return dataclasses.replace(cfg, **updates)


# ---------------------------------------------------------------------------------------------------------------
# Reptile meta-training
# ---------------------------------------------------------------------------------------------------------------
_REPTILE_SECTIONS = ("job", "wandb", "model", "checkpoint", "data", "sampler", "inner", "outer", "infra")


def load_reptile_config(path: str | pathlib.Path, overrides: Iterable[str] = ()):
    """Build ``scripts.train_reptile.ReptileConfig`` from a Reptile TOML (+ overrides)."""
    from openpi.hs11 import reptile as _reptile

    try:
        from scripts.train_reptile import ReptileConfig
    except ImportError:  # pragma: no cover
        from train_reptile import ReptileConfig  # type: ignore[no-redef]

    raw = apply_overrides(tomllib.loads(pathlib.Path(path).read_text()), overrides)
    _check_sections(raw, _REPTILE_SECTIONS)
    job = _section(raw, "job", ("name", "exp_name", "project_name"), required=("exp_name",))
    wandb = _section(raw, "wandb", ("enabled", "mode", "dir", "entity"))
    model = _section(raw, "model", ("action_horizon", "language_ranking"))
    ckpt = _section(raw, "checkpoint", ("init_params", "base_dir", "save_interval", "keep_period"))
    data = _section(raw, "data", ("embodiments", "data_root", "assets_dir", "num_workers", "prefetch_episodes"))
    sampler = _section(raw, "sampler", ("k_shot", "windows_per_demo", "q_query", "query_windows_per_demo", "disjoint_query_tasks", "min_tasks_for_disjoint", "min_frames"))
    inner = _section(raw, "inner", ("steps", "batch_size", "lr", "warmup_steps", "weight_decay", "clip", "reset_optimizer"))
    outer = _section(raw, "outer", ("meta_lr", "meta_lr_min_ratio", "max_iter", "eval_query_every"))
    infra = _section(raw, "infra", ("seed", "fsdp_devices", "log_interval", "overwrite", "resume"))

    kw: dict[str, Any] = {"exp_name": str(job["exp_name"])}
    if "name" in job:
        kw["name"] = str(job["name"])
    if "project_name" in job:
        kw["project_name"] = str(job["project_name"])
    kw["wandb_enabled"] = apply_wandb_section(wandb)
    if "language_ranking" in model:
        raise ValueError("[model.language_ranking] is a LIBERO post-training knob; the Reptile stream has no negative instructions")
    if "action_horizon" in model:
        kw["model"] = dataclasses.replace(hs11_configs.pi05_model(), action_horizon=int(model["action_horizon"]))
    if "init_params" in ckpt:
        kw["weight_loader"] = _weight_loaders.CheckpointWeightLoader(_resolve_init_params(str(ckpt["init_params"])))
    if "base_dir" in ckpt:
        kw["checkpoint_base_dir"] = str(ckpt["base_dir"])
    for key in ("save_interval", "keep_period"):
        if key in ckpt:
            kw[key] = ckpt[key]
    if "embodiments" in data:
        kw["embodiments"] = tuple(str(e) for e in data["embodiments"])
    if "data_root" in data:
        kw["data_root"] = str(data["data_root"]) or None
    for key in ("assets_dir", "num_workers", "prefetch_episodes"):
        if key in data:
            kw[key] = data[key]
    if sampler:
        s = dict(sampler)
        if "query_windows_per_demo" in s and int(s["query_windows_per_demo"]) <= 0:
            s["query_windows_per_demo"] = None
        kw["sampler"] = _reptile.SamplerConfig(**s)
    inner_map = {"steps": "inner_steps", "batch_size": "inner_batch_size", "lr": "inner_lr", "warmup_steps": "inner_warmup_steps",
                 "weight_decay": "inner_weight_decay", "clip": "inner_clip", "reset_optimizer": "inner_reset_optimizer"}
    for k, v in inner.items():
        kw[inner_map[k]] = v
    for k, v in outer.items():
        kw[k] = v
    for k, v in infra.items():
        kw[k] = resolve_fsdp_devices(int(v)) if k == "fsdp_devices" else v
    kw.setdefault("resume", True)  # same auto-resume semantics as the LIBERO TOMLs
    return ReptileConfig(**kw)


def checkpoint_status(checkpoint_dir, *, overwrite: bool, resume: bool) -> str:
    """One line for the launch log: what an existing checkpoint directory means for this run."""
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return f"checkpoint dir {checkpoint_dir} is new -> fresh start"
    steps = sorted(int(d.name) for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit())
    if overwrite:
        return f"checkpoint dir {checkpoint_dir} exists (steps {steps}) and overwrite=true -> WIPED, fresh start"
    if not resume:
        return f"checkpoint dir {checkpoint_dir} exists and resume=false -> openpi will refuse to run (set resume=true or overwrite=true)"
    if steps:
        return f"checkpoint dir {checkpoint_dir} exists -> RESUMING from step {max(steps)} (saved steps {steps}), same wandb run id"
    return f"checkpoint dir {checkpoint_dir} exists but holds no checkpoint yet -> fresh start in place (new wandb run)"


def describe(config) -> str:
    """Flat ``field = value`` listing of a (nested) config dataclass for logs / --dryrun."""
    lines = []

    def walk(prefix: str, obj: Any) -> None:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            for f in dataclasses.fields(obj):
                walk(f"{prefix}{f.name}.", getattr(obj, f.name))
        else:
            lines.append(f"{prefix[:-1]} = {obj!r}")

    walk("", config)
    return "\n".join(lines)
