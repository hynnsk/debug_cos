"""Reptile meta-training of pi05 between ``pi05_base`` and target fine-tuning (cosmos_hs11 port, JAX / FSDP).

One meta-iteration ("meta-episode", cf. ``cosmos_framework/scripts/train_action_reptile.py``):

    e ~ Uniform(embodiments);  support = k_shot demos x windows_per_demo windows of e;  query = q_query other demos
    params <- theta;  Adam state reset;  for s in 1..inner_steps:  params <- AdamW step on the FULL support set
        (the 256 support windows are processed as support_size / inner_batch_size micro-batches whose gradients are
        averaged, so one inner step == one pass over the support set, exactly as in cosmos_hs11)
    theta <- theta + eps_j * (params - theta),   eps_j: meta_lr -> meta_lr * meta_lr_min_ratio linearly over max_iter

theta is the FULL parameter set (``nnx.Param``), i.e. the "full" theta_mode of cosmos_hs11; the inner optimizer mirrors the
downstream LIBERO recipe (AdamW, clip 1.0, lr 5e-5 with a short linear ramp). Checkpoints are standard openpi
``<checkpoint_dir>/<step>/params`` directories, so a downstream config warm-starts with
``CheckpointWeightLoader("<checkpoint_dir>/<step>/params")`` (the ``pi05_libero_v3_*_reptile`` recipes read
``$HS11_REPTILE_PARAMS``).

Usage (4 GPUs, FSDP over all of them)::

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_reptile.py --exp-name reptile_v1 --fsdp-devices 4
"""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import pathlib
import platform
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.shared.array_typing as at
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms
from openpi.hs11 import configs as hs11_configs
from openpi.hs11 import embodiments as _emb
from openpi.hs11 import reptile as _reptile

# Reuse the helpers of scripts/train.py (importable as ``scripts.train`` under pytest / the repo root, or as ``train``
# when this file is executed directly and sys.path[0] is scripts/).
try:
    from scripts import train as _train
except ImportError:  # pragma: no cover
    import train as _train  # type: ignore[no-redef]


@dataclasses.dataclass(frozen=True)
class InnerSchedule(_optimizer.LRScheduleConfig):
    """Linear ramp over ``warmup_steps`` optimizer steps, then constant (restarts with every Adam reset)."""

    peak_lr: float = 5e-5
    warmup_steps: int = 2

    def create(self) -> optax.Schedule:
        if self.warmup_steps <= 0:
            return optax.constant_schedule(self.peak_lr)
        ramp = optax.linear_schedule(self.peak_lr / (self.warmup_steps + 1), self.peak_lr, self.warmup_steps)
        return optax.join_schedules([ramp, optax.constant_schedule(self.peak_lr)], [self.warmup_steps])


@dataclasses.dataclass
class ReptileConfig:
    # Experiment name (checkpoints land in <checkpoint_base_dir>/<name>/<exp_name>).
    exp_name: str
    name: str = "pi05_reptile"
    project_name: str = "openpi_hs11"
    # Model + initial weights (the downstream recipes use the same model config).
    model: pi0_config.Pi0Config = dataclasses.field(default_factory=hs11_configs.pi05_model)
    weight_loader: _weight_loaders.CheckpointWeightLoader = dataclasses.field(
        default_factory=lambda: _weight_loaders.CheckpointWeightLoader(hs11_configs.PI05_BASE_PARAMS)
    )
    # Data.
    embodiments: tuple[str, ...] = _emb.DEFAULT_EMBODIMENTS
    data_root: str | None = None  # defaults to $ROBOT_FEWSHOT_ROOT
    assets_dir: str = "./assets/pi05_reptile"  # <assets_dir>/<embodiment>/norm_stats.json
    sampler: _reptile.SamplerConfig = dataclasses.field(default_factory=_reptile.SamplerConfig)
    num_workers: int = 8
    prefetch_episodes: int = 2
    # Inner loop (= downstream optimizer).
    inner_steps: int = 10
    inner_batch_size: int = 32  # micro-batch; gradients are averaged over support_size / inner_batch_size micro-batches
    inner_lr: float = 5e-5
    inner_warmup_steps: int = 2
    inner_weight_decay: float = 1e-10
    inner_clip: float = 1.0
    inner_reset_optimizer: bool = True  # False + inner_steps=1 + meta_lr=1 == plain sequential multi-task SFT (control)
    # Outer loop.
    meta_lr: float = 0.5
    meta_lr_min_ratio: float = 0.1
    max_iter: int = 1000
    eval_query_every: int = 5
    # Infra.
    seed: int = 42
    fsdp_devices: int = 1
    checkpoint_base_dir: str = "./checkpoints"
    save_interval: int = 250
    keep_period: int | None = 250
    log_interval: int = 1
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = False

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    def train_config(self) -> _config.TrainConfig:
        """A ``TrainConfig`` describing the INNER optimizer, used to build the train state with openpi's own code."""
        return _config.TrainConfig(
            name=self.name,
            project_name=self.project_name,
            exp_name=self.exp_name,
            model=self.model,
            weight_loader=self.weight_loader,
            lr_schedule=InnerSchedule(peak_lr=self.inner_lr, warmup_steps=self.inner_warmup_steps),
            optimizer=_optimizer.AdamW(weight_decay=self.inner_weight_decay, clip_gradient_norm=self.inner_clip),
            ema_decay=None,
            data=_config.FakeDataConfig(),
            batch_size=self.inner_batch_size,
            seed=self.seed,
            fsdp_devices=self.fsdp_devices,
            checkpoint_base_dir=self.checkpoint_base_dir,
            wandb_enabled=self.wandb_enabled,
        )


class _NoAssets:
    """Stand-in data loader for ``checkpoints.save_state`` (no norm stats are stored with the Reptile theta)."""

    def data_config(self) -> _config.DataConfig:
        return _config.DataConfig()


# ---------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------
def _make_factory(data_config: _config.DataConfig, action_horizon: int):
    """Picklable factory: (raw v3 dataset, transform) built inside each worker."""
    transforms = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    return functools.partial(_build_pair, data_config.dataset_builder, action_horizon, transforms)


def _build_pair(builder, action_horizon, transforms):
    return builder(action_horizon), _transforms.compose(transforms)


def _identity(x):
    return x


def build_data(config: ReptileConfig, start_spec: int):
    model_config = config.model
    data_configs = {
        name: _emb.make_embodiment_data_config(name, model_config, config.assets_dir, data_root=config.data_root)
        for name in config.embodiments
    }
    tables = {}
    for name, dc in data_configs.items():
        raw = dc.dataset_builder(model_config.action_horizon)
        tables[name] = _reptile.EmbodimentTable.from_dataset(raw, with_tasks=config.sampler.disjoint_query_tasks)
    sampler = _reptile.EpisodicSampler(tables, config.sampler, action_horizon=model_config.action_horizon, seed=config.seed)
    factories = {name: _make_factory(dc, model_config.action_horizon) for name, dc in data_configs.items()}
    stream = _reptile.ReptileStream(factories, sampler, start_spec=start_spec)
    loader = torch.utils.data.DataLoader(
        stream,
        batch_size=None,
        num_workers=config.num_workers,
        multiprocessing_context=torch.multiprocessing.get_context("spawn") if config.num_workers > 0 else None,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=4 if config.num_workers > 0 else None,
        worker_init_fn=_data_loader._worker_init_fn,  # noqa: SLF001
        collate_fn=_identity,  # items pass through untouched (a lambda would not pickle into spawned workers)
    )
    collector = _reptile.EpisodeCollector(iter(loader), sampler, start_spec=start_spec)
    return sampler, _reptile.EpisodePrefetcher(collector, maxsize=config.prefetch_episodes)


# ---------------------------------------------------------------------------------------------------------------
# JAX step functions
# ---------------------------------------------------------------------------------------------------------------
def _loss_fn(model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions):
    return jnp.mean(model.compute_loss(rng, observation, actions, train=True))


def make_step_fns(train_config: _config.TrainConfig, mesh, state_sharding, data_sharding):
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    trainable = train_config.trainable_filter

    def grad_step(rng, state: training_utils.TrainState, batch):
        model = nnx.merge(state.model_def, state.params)
        model.train()
        observation, actions = batch
        loss, grads = nnx.value_and_grad(_loss_fn, argnums=nnx.DiffState(0, trainable))(model, rng, observation, actions)
        return loss, grads

    def loss_step(rng, state: training_utils.TrainState, batch):
        model = nnx.merge(state.model_def, state.params)
        model.train()  # same objective as the support loss (incl. image augmentation)
        observation, actions = batch
        return _loss_fn(model, rng, observation, actions)

    def accumulate(acc, grads, scale):
        return jax.tree.map(lambda a, g: a + g * scale, acc, grads)

    def apply_step(state: training_utils.TrainState, grads):
        model = nnx.merge(state.model_def, state.params)
        params = state.params.filter(trainable)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(model, new_params)
        new_params = nnx.state(model)
        return dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state), optax.global_norm(grads)

    def reset_step(state: training_utils.TrainState, theta):
        """params <- theta; fresh inner optimizer state."""
        model = nnx.merge(state.model_def, state.params)
        nnx.update(model, theta)
        params = nnx.state(model)
        return dataclasses.replace(state, params=params, opt_state=state.tx.init(params.filter(trainable)))

    def set_params(state: training_utils.TrainState, theta):
        model = nnx.merge(state.model_def, state.params)
        nnx.update(model, theta)
        return dataclasses.replace(state, params=nnx.state(model))

    def meta_step(theta, params, eps):
        """theta + eps * (params - theta) on the trainable subset; returns (new_theta, |params - theta|, |step|)."""
        delta = jax.tree.map(lambda p, t: p - t, params, theta)
        delta_norm = optax.global_norm(delta)
        new_theta = jax.tree.map(lambda t, d: t + eps * d, theta, delta)
        return new_theta, delta_norm, eps * delta_norm

    theta_sharding = state_sharding.params.filter(trainable)
    fns = {
        "grad": jax.jit(grad_step, in_shardings=(replicated, state_sharding, data_sharding), out_shardings=(replicated, theta_sharding)),
        "loss": jax.jit(loss_step, in_shardings=(replicated, state_sharding, data_sharding), out_shardings=replicated),
        "accumulate": jax.jit(accumulate, in_shardings=(theta_sharding, theta_sharding, replicated), out_shardings=theta_sharding, donate_argnums=(0,)),
        "apply": jax.jit(apply_step, in_shardings=(state_sharding, theta_sharding), out_shardings=(state_sharding, replicated), donate_argnums=(0,)),
        "reset": jax.jit(reset_step, in_shardings=(state_sharding, theta_sharding), out_shardings=state_sharding, donate_argnums=(0,)),
        "set_params": jax.jit(set_params, in_shardings=(state_sharding, theta_sharding), out_shardings=state_sharding, donate_argnums=(0,)),
        "meta": jax.jit(meta_step, in_shardings=(theta_sharding, theta_sharding, replicated), out_shardings=(theta_sharding, replicated, replicated), donate_argnums=(0,)),
        "copy": jax.jit(lambda p: jax.tree.map(lambda x: x, p), in_shardings=(theta_sharding,), out_shardings=theta_sharding),
    }
    return fns


def _to_device(batch: dict[str, Any], data_sharding) -> tuple[_model.Observation, _model.Actions]:
    arrays = jax.tree.map(lambda x: jax.make_array_from_process_local_data(data_sharding, np.asarray(x)), batch)
    return _model.Observation.from_dict(arrays), arrays["actions"]


# ---------------------------------------------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------------------------------------------
def main(config: ReptileConfig) -> None:
    _train.init_logging()
    logging.info(f"Running on: {platform.node()}; devices: {jax.devices()}")
    if config.sampler.support_size % config.inner_batch_size != 0:
        raise ValueError(f"support size {config.sampler.support_size} must be a multiple of inner_batch_size {config.inner_batch_size}")
    if config.sampler.query_size % config.inner_batch_size != 0 and config.sampler.q_query > 0:
        raise ValueError(f"query size {config.sampler.query_size} must be a multiple of inner_batch_size {config.inner_batch_size}")
    if config.inner_batch_size % jax.device_count() != 0:
        raise ValueError(f"inner_batch_size {config.inner_batch_size} must be divisible by the device count {jax.device_count()}")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    train_config = config.train_config()
    rng = jax.random.key(config.seed)
    rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=config.overwrite, resume=config.resume
    )
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (config.checkpoint_dir / "reptile_config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2, default=str))
    _train.init_wandb(train_config, resuming=resuming, enabled=config.wandb_enabled)
    if config.wandb_enabled:
        import wandb

        wandb.config.update({"reptile": dataclasses.asdict(config)}, allow_val_change=True)

    # ---- model / train state --------------------------------------------------------------------------------
    train_state, state_sharding = _train.init_train_state(train_config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    start_iter = 0
    if resuming:
        # Checkpoints hold theta (as params) and the meta-iteration (as step); the inner optimizer is rebuilt.
        with at.disable_typechecking():  # the shape state holds ShapeDtypeStructs, which TrainState's typecheck rejects
            shape_state = dataclasses.replace(train_state, opt_state=())
            restored = _checkpoints.restore_state(checkpoint_manager, shape_state, _NoAssets())
            start_iter = int(restored.step)
            params = jax.jit(lambda p: p, out_shardings=state_sharding.params)(restored.params)
            init_opt = jax.jit(
                lambda p: train_state.tx.init(p.filter(train_config.trainable_filter)), out_shardings=state_sharding.opt_state
            )
            train_state = dataclasses.replace(
                train_state, step=jnp.asarray(start_iter, dtype=jnp.int32), params=params, opt_state=init_opt(params)
            )
        logging.info(f"Resumed theta at meta-iteration {start_iter}")
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    fns = make_step_fns(train_config, mesh, state_sharding, data_sharding)
    trainable = train_config.trainable_filter
    theta = fns["copy"](train_state.params.filter(trainable))

    # ---- data -----------------------------------------------------------------------------------------------
    sampler, prefetcher = build_data(config, start_iter)
    n_micro = config.sampler.support_size // config.inner_batch_size
    log_path = config.checkpoint_dir / "reptile_train_log.jsonl"

    def query_loss(state, batches, rng_key) -> float:
        losses = [fns["loss"](jax.random.fold_in(rng_key, i), state, b) for i, b in enumerate(batches)]
        return float(np.mean([float(x) for x in losses]))

    for it in range(start_iter, config.max_iter):
        t0 = time.monotonic()
        episode = prefetcher.get()
        if episode.spec.spec_id != it:
            raise RuntimeError(f"data stream out of sync: expected spec {it}, got {episode.spec.spec_id}")
        if episode.failed:
            logging.warning(f"iter {it}: meta-episode has undecodable windows; skipping (theta unchanged)")
            continue
        support = [_to_device(b, data_sharding) for b in _reptile.micro_batches(episode.support, config.inner_batch_size)]
        query = [_to_device(b, data_sharding) for b in _reptile.micro_batches(episode.query, config.inner_batch_size)] if episode.query else []
        t_data = time.monotonic() - t0
        iter_rng = jax.random.fold_in(rng, it)

        # params <- theta (+ Adam reset)
        with sharding.set_mesh(mesh):
            if config.inner_reset_optimizer or it == start_iter:
                train_state = fns["reset"](train_state, theta)
            else:
                train_state = fns["set_params"](train_state, theta)
            do_query = bool(query) and (it % config.eval_query_every == 0)
            q_zero = query_loss(train_state, query, jax.random.fold_in(iter_rng, 1_000_000)) if do_query else None

            support_losses, grad_norms = [], []
            for s in range(config.inner_steps):
                acc = None
                step_losses = []
                for m, batch in enumerate(support):
                    loss, grads = fns["grad"](jax.random.fold_in(iter_rng, s * n_micro + m), train_state, batch)
                    acc = jax.tree.map(lambda g: g * (1.0 / n_micro), grads) if acc is None else fns["accumulate"](acc, grads, 1.0 / n_micro)
                    step_losses.append(loss)
                train_state, gnorm = fns["apply"](train_state, acc)
                support_losses.append(float(np.mean([float(x) for x in step_losses])))
                grad_norms.append(float(gnorm))
            q_adapt = query_loss(train_state, query, jax.random.fold_in(iter_rng, 2_000_000)) if do_query else None

            eps = _reptile.meta_lr_at(it, config.max_iter, config.meta_lr, config.meta_lr_min_ratio)
            theta, adapt_delta, meta_step_norm = fns["meta"](theta, train_state.params.filter(trainable), jnp.float32(eps))
            jax.block_until_ready(theta)

        info = {
            "iter": it,
            "embodiment": episode.spec.embodiment,
            "eps": eps,
            "support_loss_first": support_losses[0],
            "support_loss_last": support_losses[-1],
            "grad_norm_first": grad_norms[0],
            "grad_norm_last": grad_norms[-1],
            "adapt_delta": float(adapt_delta),
            "meta_step_norm": float(meta_step_norm),
            "query_loss_zero_shot": q_zero,
            "query_loss_adapted": q_adapt,
            "t_data_s": t_data,
            "t_iter_s": time.monotonic() - t0,
        }
        if it % config.log_interval == 0:
            msg = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in info.items() if v is not None)
            logging.info(f"[meta {it}/{config.max_iter}] {msg}")
        with log_path.open("a") as f:
            f.write(json.dumps(info) + "\n")
        if config.wandb_enabled:
            import wandb

            wandb.log({k: v for k, v in info.items() if isinstance(v, (int, float)) and v is not None}, step=it)

        is_last = it == config.max_iter - 1
        if (config.save_interval > 0 and (it + 1) % config.save_interval == 0) or is_last:
            # theta is the checkpointed parameter set; the inner optimizer state is not needed to resume.
            with sharding.set_mesh(mesh):
                train_state = fns["set_params"](train_state, theta)
            save_state = dataclasses.replace(train_state, step=it + 1, opt_state=(), ema_params=None)
            with at.disable_typechecking():
                _checkpoints.save_state(checkpoint_manager, save_state, _NoAssets(), it + 1)
            checkpoint_manager.wait_until_finished()
            logging.info(f"saved theta at {config.checkpoint_dir / str(it + 1) / 'params'}")

    prefetcher.stop()
    logging.info("Reptile meta-training finished")


if __name__ == "__main__":
    main(tyro.cli(ReptileConfig))
