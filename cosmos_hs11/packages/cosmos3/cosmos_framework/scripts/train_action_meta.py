# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cross-embodiment few-shot meta-training of the Cosmos3 action heads (first-order MAML / ANIL).

Sits between Cosmos3 mid-training and the LIBERO few-shot post-training. The backbone is frozen;
only a shared initialization ``theta_meta`` of the action I/O projectors (``action2llm``,
``llm2action``, ``action_modality_embed``) is learned so that a *new* embodiment can be adapted
from a handful of demonstrations::

    for meta_iter in range(max_iter):
        embodiment ~ Uniform(RT-1, Bridge, RoboMIND-UR, RoboMIND-Franka, MolmoAct2-YAM)
        support (K demos x w windows), query (Q demos x w windows) <- EpisodicEmbodimentSampler
        fast <- theta_meta                                   # written into one scratch DomainAwareLinear row
        for _ in range(inner_steps):                          # inner loop (support)
            fast <- fast - inner_lr * grad_fast  L_support(fast)
        g <- grad_fast L_query(fast)                          # first-order meta-gradient (no 2nd-order terms)
        theta_meta <- AdamW(theta_meta, all_reduce_mean(g))    # outer loop

Every rank runs its own meta-episode on a full bf16 replica of the frozen backbone (no FSDP); only
the tiny theta_meta gradient is averaged across ranks. The VAE encode + text tokenization of a
meta-episode is done once (``OmniMoTModel._get_training_inputs``) and reused by every inner step
(``training_step_from_inputs``), which draws fresh noise levels each time.

Usage::

    torchrun --nproc_per_node=4 -m cosmos_framework.scripts.train_action_meta \\
        --sft-toml=examples/toml/sft_config/action_fewshot_meta_edge.toml [-- key.path=value ...]

Outputs (``$IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>/``)::

    meta_action_init.pt                 latest theta_meta  -> checkpoint.meta_action_init_path downstream
    meta_action_init_iter_XXXXXX.pt     snapshots every checkpoint.save_iter
    meta_state_latest.pt                theta_meta + meta optimizer + iteration (auto-resume)

    theta_meta groups: action heads (+ action_modality_embed) always; ``[custom.meta] include_lora`` /
    ``include_time_embedder`` add the moe_gen LoRA adapters and the time_embedder (cosmos_hs09, model.config.lora_enabled).
    meta_train_log.jsonl                per-iteration scalars (also sent to W&B when enabled)

Meta-specific knobs live in the ``[custom.meta]`` TOML table (see :class:`MetaTrainConfig`); sampler
knobs with the same name override the ``dataloader_train`` LazyCall (``build_meta_episode_loader``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from loguru import logger as logging

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.data.generator.action.meta.meta_action_adapter import (
    MetaActionAdapter,
    MetaActionInitSpec,
    save_meta_action_init,
)
from cosmos_framework.utils import distributed, misc
from cosmos_framework.utils.config import Config
from cosmos_framework.utils.context_managers import data_loader_init, distributed_init, model_init
from cosmos_framework.utils.lazy_config import LazyConfig, instantiate

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Length of IterableDataset .* was reported to be .* samples have been fetched.*",
    category=UserWarning,
)

# ``[custom.meta]`` keys that belong to the sampler / loader (``build_meta_episode_loader`` kwargs).
SAMPLER_OVERRIDE_KEYS = {
    "embodiments",
    "k_shot",
    "q_query",
    "windows_per_demo",
    "query_windows_per_demo",
    "disjoint_tasks",
    "min_tasks_for_disjoint",
    "min_windows_per_demo",
    "embodiment_weights",
    "max_samples_per_batch",
    "num_workers",
    "prefetch_factor",
    "loader_timeout_s",
    "seed",
    "embodiment_override",
    "resolution",
    "cfg_dropout_rate",
    "chunk_length",
    "mode",
    "data_root",
    "dataset_kwargs",
    "root_overrides",
    "format_prompt_as_json",
    "append_idle_frames",
    # cosmos_hs11 v11 (Reptile loader only): native frames for the REPA teacher in the inner loop
    "keep_native_video",
    "native_video_size",
    "native_video_full_res",
}


@dataclass
class MetaTrainConfig:
    """Adapter / inner-loop / outer-loop knobs from ``[custom.meta]``."""

    # inner loop (support)
    inner_steps: int = 5
    inner_lr: float = 0.05
    inner_optimizer: str = "sgd"  # "sgd" | "adam"
    inner_adam_betas: tuple[float, float] = (0.9, 0.999)
    inner_grad_clip: float | None = 1.0
    # outer loop (query)
    meta_lr: float = 1.0e-4
    meta_weight_decay: float = 0.0
    meta_betas: tuple[float, float] = (0.9, 0.99)
    meta_grad_clip: float | None = 1.0
    meta_lr_warmup_iters: int = 50
    meta_lr_min_ratio: float = 0.1  # linear decay from meta_lr down to meta_lr * ratio at max_iter
    # loss used for both loops: "action" = flow-matching action loss only (ANIL-style), "total" = model loss
    loss_mode: str = "action"
    # which parameters / row
    scratch_domain_id: int = 31
    include_modality_embed: bool = True
    # cosmos_hs09: extend theta_meta from the action heads to the modules the downstream LoRA
    # post-training optimizes. Needs model.config.lora_enabled=True for include_lora.
    include_lora: bool = False
    include_time_embedder: bool = False
    init_source: str = "fresh"  # fresh | checkpoint_row | checkpoint_mean | zeros
    init_domain_ids: list[int] = field(default_factory=list)
    init_seed: int = 0
    # diagnostics
    eval_zero_shot_every: int = 10  # also measure the query loss BEFORE adaptation every N iters (0 = never)
    log_every: int | None = None  # defaults to trainer.logging_iter
    save_every: int | None = None  # defaults to checkpoint.save_iter
    sync_meta_every: int = 1  # broadcast theta_meta from rank 0 every N iters (guards against drift)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MetaTrainConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - names
        if unknown:
            raise ValueError(f"[custom.meta] has unknown keys {sorted(unknown)}; valid: {sorted(names | SAMPLER_OVERRIDE_KEYS)}")
        kwargs = dict(raw)
        for key in ("inner_adam_betas", "meta_betas"):
            if key in kwargs:
                kwargs[key] = tuple(float(x) for x in kwargs[key])
        for key in ("inner_grad_clip", "meta_grad_clip"):
            if key in kwargs and kwargs[key] is not None and float(kwargs[key]) <= 0:
                kwargs[key] = None
        cfg = cls(**kwargs)
        if cfg.inner_optimizer not in ("sgd", "adam"):
            raise ValueError(f"inner_optimizer must be 'sgd' or 'adam', got {cfg.inner_optimizer!r}")
        if cfg.loss_mode not in ("action", "total"):
            raise ValueError(f"loss_mode must be 'action' or 'total', got {cfg.loss_mode!r}")
        if cfg.inner_steps < 0:
            raise ValueError("inner_steps must be >= 0")
        return cfg


def split_custom_meta(config: Config) -> tuple[MetaTrainConfig, dict[str, Any]]:
    """Split ``config.custom['meta']`` into trainer knobs and sampler/loader overrides."""
    custom = getattr(config, "custom", None) or {}
    raw = dict(custom.get("meta", {}) or {})
    sampler = {k: raw.pop(k) for k in list(raw) if k in SAMPLER_OVERRIDE_KEYS}
    return MetaTrainConfig.from_dict(raw), sampler


def apply_sampler_overrides(config: Config, overrides: dict[str, Any]) -> None:
    for k, v in overrides.items():
        if k not in config.dataloader_train:
            raise KeyError(f"dataloader_train has no field {k!r} to override from [custom.meta]")
        config.dataloader_train[k] = v


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _select_loss(output_batch: dict[str, Any], total_loss: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "total":
        return total_loss
    loss = output_batch["flow_matching_loss_action"]
    if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
        raise RuntimeError("flow_matching_loss_action carries no gradient -- is action_gen enabled and the batch action-bearing?")
    return loss


def _route_to_domain(batch: dict[str, Any], domain_id: int) -> None:
    """Point every sample of a packed batch at the scratch DomainAwareLinear row."""
    ids = batch.get("domain_id")
    if ids is None:
        raise KeyError("packed batch has no domain_id")
    batch["domain_id"] = [torch.full_like(torch.as_tensor(d), int(domain_id)) for d in ids]


def _meta_lr_at(cfg: MetaTrainConfig, iteration: int, max_iter: int) -> float:
    warm = max(0, int(cfg.meta_lr_warmup_iters))
    if warm > 0 and iteration < warm:
        return cfg.meta_lr * float(iteration + 1) / warm
    span = max(1, max_iter - warm)
    progress = min(1.0, max(0.0, (iteration - warm) / span))
    return cfg.meta_lr * (1.0 - (1.0 - cfg.meta_lr_min_ratio) * progress)


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


class _InnerAdam:
    """Per-episode Adam state for the fast weights (reset at every meta-episode)."""

    def __init__(self, names: list[str], betas: tuple[float, float], eps: float = 1e-8) -> None:
        self.names = names
        self.b1, self.b2 = betas
        self.eps = eps
        self.m: dict[str, torch.Tensor] = {}
        self.v: dict[str, torch.Tensor] = {}
        self.t = 0

    def step(self, fast: dict[str, torch.Tensor], grads: dict[str, torch.Tensor], lr: float) -> dict[str, torch.Tensor]:
        self.t += 1
        out = {}
        for n in self.names:
            g = grads[n]
            if n not in self.m:
                self.m[n] = torch.zeros_like(g)
                self.v[n] = torch.zeros_like(g)
            self.m[n].mul_(self.b1).add_(g, alpha=1 - self.b1)
            self.v[n].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            m_hat = self.m[n] / (1 - self.b1**self.t)
            v_hat = self.v[n] / (1 - self.b2**self.t)
            out[n] = fast[n] - lr * m_hat / (v_hat.sqrt() + self.eps)
        return out


def _all_gather_stats(stats: dict[str, Any], world_size: int) -> list[dict[str, Any]]:
    if world_size <= 1:
        return [stats]
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, stats)
    return gathered  # type: ignore[return-value]


# --------------------------------------------------------------------------------------------
# main loop
# --------------------------------------------------------------------------------------------
@logging.catch(reraise=True)
def _prepare_dataloader_process_limits() -> None:
    """Avoid ``OSError: [Errno 24] Too many open files`` in the DataLoader workers.

    One meta-episode is ~1k small tensors (per-window video/action/text lists for support + query).
    With the default ``file_descriptor`` sharing strategy every tensor in flight pins one fd in the
    worker AND in the trainer, so ``num_workers x prefetch_factor`` episodes blow past the 1024
    soft limit of the compute nodes; the worker's queue feeder then silently drops the item and the
    trainer blocks forever on it (rank 0 spins in all_reduce, the others in ``next(loader)``).
    ``file_system`` shares via named shm segments (same memory, no fd per tensor). Also lift the
    soft nofile limit to the hard limit as belt-and-braces (inherited by forked workers).
    """
    torch.multiprocessing.set_sharing_strategy("file_system")
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = hard if hard != resource.RLIM_INFINITY else max(soft, 65536)
        if target > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            logging.info(f"Raised RLIMIT_NOFILE soft limit {soft} -> {target}")
    except (ValueError, OSError) as e:  # noqa: PERF203
        logging.warning(f"Could not raise RLIMIT_NOFILE ({e}); relying on the file_system sharing strategy")


@logging.catch(reraise=True)
def launch(config: Config, meta_cfg: MetaTrainConfig, args: argparse.Namespace) -> None:
    _prepare_dataloader_process_limits()
    with distributed_init():
        distributed.init()
    config.validate()
    config.freeze()

    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    is_rank0 = rank == 0
    job_dir = Path(config.job.path_local)
    if is_rank0:
        job_dir.mkdir(parents=True, exist_ok=True)
        LazyConfig.save_yaml(config, str(job_dir / "config.yaml"))
        (job_dir / "meta_config.json").write_text(json.dumps(dataclasses.asdict(meta_cfg), indent=2) + "\n")
        logging.info(f"Job dir: {job_dir}")
        logging.info(f"Meta config: {json.dumps(dataclasses.asdict(meta_cfg))}")
    dist.barrier()
    misc.set_random_seed(seed=config.trainer.seed, by_rank=True)

    # ---- model (frozen backbone, replicated) --------------------------------------------
    with model_init():
        model = instantiate(config.model)
    model = model.to("cuda", memory_format=config.trainer.memory_format)
    model.on_train_start(config.trainer.memory_format)
    if model.config.ema.enabled:
        raise ValueError("action_fewshot_meta_edge must run with model.config.ema.enabled=False")
    checkpointer = instantiate(config.checkpoint.type, config.checkpoint, config.job, callbacks=None)
    checkpointer.load(model)  # warm start: base Cosmos3-Edge weights (model only)
    del checkpointer
    model.train()

    adapter = MetaActionAdapter(
        model.net,
        scratch_domain_id=meta_cfg.scratch_domain_id,
        include_modality_embed=meta_cfg.include_modality_embed,
        include_lora=meta_cfg.include_lora,
        include_time_embedder=meta_cfg.include_time_embedder,
        init=MetaActionInitSpec(
            source=meta_cfg.init_source, domain_ids=list(meta_cfg.init_domain_ids), seed=meta_cfg.init_seed
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(
        f"Trainable network params (fast-weight carriers): {trainable:,}; theta_meta numel: {adapter.numel:,} "
        f"by group: {adapter.numel_by_group()}"
    )
    meta_opt = torch.optim.AdamW(
        adapter.meta_parameters, lr=meta_cfg.meta_lr, betas=meta_cfg.meta_betas, weight_decay=meta_cfg.meta_weight_decay
    )

    # ---- resume theta_meta / optimizer -------------------------------------------------
    start_iter = 0
    state_path = job_dir / "meta_state_latest.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        adapter.load_meta(state["meta_params"])
        meta_opt.load_state_dict(state["optimizer"])
        start_iter = int(state["iteration"])
        logging.info(f"Resumed theta_meta + meta optimizer from {state_path} at iteration {start_iter}")
    # Keep theta_meta bit-identical across ranks.
    for t in adapter.meta_parameters:
        dist.broadcast(t.data, src=0)

    # ---- data --------------------------------------------------------------------------------
    with data_loader_init():
        loader = instantiate(config.dataloader_train)
    loader_iter = iter(loader)

    # ---- logging ---------------------------------------------------------------------------
    max_iter = int(config.trainer.max_iter)
    log_every = int(meta_cfg.log_every or config.trainer.logging_iter)
    save_every = int(meta_cfg.save_every or config.checkpoint.save_iter)
    wandb_run = None
    if is_rank0 and config.job.wandb_mode != "disabled":
        try:
            import wandb

            id_file = job_dir / "wandb_id.txt"
            run_id = id_file.read_text().strip() if id_file.exists() else wandb.util.generate_id()
            id_file.write_text(run_id)
            wandb_run = wandb.init(
                project=config.job.project,
                group=config.job.group,
                name=config.job.name,
                id=run_id,
                resume="allow",
                dir=str(job_dir),
                mode=config.job.wandb_mode,
                config={"meta": dataclasses.asdict(meta_cfg), "sft_toml": args.sft_toml},
            )
        except Exception as e:  # noqa: BLE001
            logging.warning(f"wandb init failed ({e}); continuing without wandb")
    jsonl = open(job_dir / "meta_train_log.jsonl", "a") if is_rank0 else None

    def _save(iteration: int) -> None:
        if not is_rank0:
            return
        metadata = {
            **adapter.metadata(),
            "iteration": iteration,
            "max_iter": max_iter,
            "meta_config": dataclasses.asdict(meta_cfg),
            "embodiments": list(config.dataloader_train.get("embodiments", [])),
            "base_checkpoint": str(config.checkpoint.load_path),
            "job": {"project": config.job.project, "group": config.job.group, "name": config.job.name},
        }
        meta_params = adapter.export_meta()
        save_meta_action_init(job_dir / f"meta_action_init_iter_{iteration:06d}.pt", meta_params, metadata)
        save_meta_action_init(job_dir / "meta_action_init.pt", meta_params, metadata)
        tmp = state_path.with_suffix(".tmp")
        torch.save({"meta_params": meta_params, "optimizer": meta_opt.state_dict(), "iteration": iteration}, tmp)
        tmp.replace(state_path)
        logging.info(f"Saved theta_meta at iteration {iteration} -> {job_dir / 'meta_action_init.pt'}")

    logging.info(f"Starting meta-training: iterations {start_iter} -> {max_iter}, world_size={world_size}")
    iteration = start_iter
    ema_query: float | None = None
    while iteration < max_iter:
        t_iter = time.time()
        t0 = time.time()
        episode = next(loader_iter)
        t_data = time.time() - t0
        embodiment = episode["embodiment"]
        support_batches = episode["support"]
        query_batches = episode["query"]
        for b in support_batches + query_batches:
            _route_to_domain(b, meta_cfg.scratch_domain_id)
        support_batches = [misc.to(b, device="cuda") for b in support_batches]
        query_batches = [misc.to(b, device="cuda") for b in query_batches]

        # One VAE encode / tokenization per meta-episode; reused by every inner step.
        t0 = time.time()
        with torch.no_grad():
            support_inputs = [model._get_training_inputs(b, iteration) for b in support_batches]
            query_inputs = [model._get_training_inputs(b, iteration) for b in query_batches]
        t_encode = time.time() - t0

        adapter.start_episode()
        adapter.zero_grads()

        # Optional diagnostic: query loss at theta_meta before any adaptation.
        query_zero_shot = float("nan")
        if meta_cfg.eval_zero_shot_every > 0 and (iteration % meta_cfg.eval_zero_shot_every == 0):
            with torch.no_grad():
                vals = []
                for inp in query_inputs:
                    out, total = model.training_step_from_inputs(inp, iteration)
                    vals.append(float(out["flow_matching_loss_action"] if meta_cfg.loss_mode == "action" else total))
                query_zero_shot = _mean(vals)

        # ---- inner loop on the support set ----
        t0 = time.time()
        inner_losses: list[float] = []
        inner_gnorms: list[float] = []
        inner_adam = _InnerAdam(list(adapter.names), meta_cfg.inner_adam_betas) if meta_cfg.inner_optimizer == "adam" else None
        for k in range(meta_cfg.inner_steps):
            inp = support_inputs[k % len(support_inputs)]
            out, total = model.training_step_from_inputs(inp, iteration)
            loss = _select_loss(out, total, meta_cfg.loss_mode)
            loss.backward()
            grads = adapter.collect_grads()
            inner_gnorms.append(adapter.clip_grads_(grads, meta_cfg.inner_grad_clip))
            if inner_adam is None:
                adapter.inner_update(grads, meta_cfg.inner_lr)
            else:
                adapter.set_fast(inner_adam.step(adapter.fast, grads, meta_cfg.inner_lr))
            inner_losses.append(float(loss.detach()))
        t_inner = time.time() - t0

        # ---- outer loop on the query set (first-order meta-gradient at the adapted weights) ----
        t0 = time.time()
        adapter.zero_grads()
        query_losses: list[float] = []
        for inp in query_inputs:
            out, total = model.training_step_from_inputs(inp, iteration)
            loss = _select_loss(out, total, meta_cfg.loss_mode)
            (loss / len(query_inputs)).backward()
            query_losses.append(float(loss.detach()))
        meta_grads = adapter.collect_grads()
        if world_size > 1:
            for g in meta_grads.values():
                dist.all_reduce(g, op=dist.ReduceOp.SUM)
                g.div_(world_size)
        meta_gnorm = adapter.clip_grads_(meta_grads, meta_cfg.meta_grad_clip)
        adapter.assign_meta_grads(meta_grads)
        lr = _meta_lr_at(meta_cfg, iteration, max_iter)
        for group in meta_opt.param_groups:
            group["lr"] = lr
        meta_opt.step()
        meta_opt.zero_grad(set_to_none=True)
        if meta_cfg.sync_meta_every > 0 and (iteration % meta_cfg.sync_meta_every == 0):
            for t in adapter.meta_parameters:
                dist.broadcast(t.data, src=0)
        t_outer = time.time() - t0
        iteration += 1

        # ---- logging ----
        query_loss = _mean(query_losses)
        stats = {
            "embodiment": embodiment,
            "query_loss": query_loss,
            "query_loss_zero_shot": query_zero_shot,
            "support_loss_first": inner_losses[0] if inner_losses else float("nan"),
            "support_loss_last": inner_losses[-1] if inner_losses else float("nan"),
            "inner_grad_norm": _mean(inner_gnorms),
            "meta_grad_norm": meta_gnorm,
            "adapt_delta": adapter.fast_delta_norm(),
            "adapt_delta_by_group": adapter.fast_delta_norms_by_group(),
            "num_support": int(episode["spec"]["num_support_windows"]),
            "num_query": int(episode["spec"]["num_query_windows"]),
            "disjoint_tasks": bool(episode["spec"]["disjoint_tasks"]),
            "t_data": t_data,
            "t_encode": t_encode,
            "t_inner": t_inner,
            "t_outer": t_outer,
        }
        gathered = _all_gather_stats(stats, world_size)
        if is_rank0:
            mean_q = _mean([g["query_loss"] for g in gathered])
            ema_query = mean_q if ema_query is None else 0.98 * ema_query + 0.02 * mean_q
            zs = [g["query_loss_zero_shot"] for g in gathered if not math.isnan(g["query_loss_zero_shot"])]
            record: dict[str, Any] = {
                "iteration": iteration,
                "lr": lr,
                "query_loss": mean_q,
                "query_loss_ema": ema_query,
                "query_loss_zero_shot": _mean(zs) if zs else None,
                "support_loss_first": _mean([g["support_loss_first"] for g in gathered]),
                "support_loss_last": _mean([g["support_loss_last"] for g in gathered]),
                "meta_grad_norm": _mean([g["meta_grad_norm"] for g in gathered]),
                "inner_grad_norm": _mean([g["inner_grad_norm"] for g in gathered]),
                "adapt_delta": _mean([g["adapt_delta"] for g in gathered]),
                "iter_time": time.time() - t_iter,
                "t_data": _mean([g["t_data"] for g in gathered]),
                "t_encode": _mean([g["t_encode"] for g in gathered]),
                "t_inner": _mean([g["t_inner"] for g in gathered]),
                "t_outer": _mean([g["t_outer"] for g in gathered]),
                "embodiments": [g["embodiment"] for g in gathered],
            }
            per_emb: dict[str, list[float]] = {}
            for g in gathered:
                per_emb.setdefault(g["embodiment"], []).append(g["query_loss"])
            for emb, vals in per_emb.items():
                record[f"query_loss/{emb}"] = _mean(vals)
            # adaptation magnitude per theta_meta group (action_heads / action_modality_embed / lora / time_embedder)
            for grp in sorted({k for g in gathered for k in g.get("adapt_delta_by_group", {})}):
                record[f"adapt_delta/{grp}"] = _mean([g["adapt_delta_by_group"][grp] for g in gathered if grp in g.get("adapt_delta_by_group", {})])
            if jsonl is not None:
                jsonl.write(json.dumps(record) + "\n")
                jsonl.flush()
            if iteration % log_every == 0 or iteration == start_iter + 1:
                logging.info(
                    f"iter {iteration}/{max_iter} | query {mean_q:.4f} (ema {ema_query:.4f}"
                    + (f", zero-shot {record['query_loss_zero_shot']:.4f}" if record["query_loss_zero_shot"] is not None else "")
                    + f") | support {record['support_loss_first']:.4f} -> {record['support_loss_last']:.4f} "
                    f"| meta-grad {record['meta_grad_norm']:.3e} lr {lr:.2e} | adapt {record['adapt_delta']:.3f} "
                    f"| {record['iter_time']:.1f}s (data {record['t_data']:.1f} enc {record['t_encode']:.1f} "
                    f"inner {record['t_inner']:.1f} outer {record['t_outer']:.1f}) | {record['embodiments']}"
                )
            if wandb_run is not None:
                wandb_run.log({k: v for k, v in record.items() if isinstance(v, (int, float)) and v is not None}, step=iteration)

        if iteration % save_every == 0 or iteration == max_iter:
            _save(iteration)
        dist.barrier()

    if jsonl is not None:
        jsonl.close()
    if wandb_run is not None:
        wandb_run.finish()
    logging.success(f"Meta-training finished at iteration {iteration}. theta_meta: {job_dir / 'meta_action_init.pt'}")
    dist.barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-embodiment few-shot meta-training of the Cosmos3 action heads")
    parser.add_argument("--sft-toml", required=True, help="Run TOML (experiment action_fewshot_meta_edge + [custom.meta]).")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[], help="Extra Hydra dotted-path overrides (after --).")
    parser.add_argument("--dryrun", action="store_true", help="Print the resolved config and exit.")
    args = parser.parse_args()

    config = load_experiment_from_toml(args.sft_toml, extra_overrides=args.opts)
    meta_cfg, sampler_overrides = split_custom_meta(config)
    apply_sampler_overrides(config, sampler_overrides)
    args.config = args.sft_toml

    if args.dryrun:
        logging.info("Config:\n" + config.pretty_print(use_color=True))
        logging.info("Meta config: " + json.dumps(dataclasses.asdict(meta_cfg), indent=2))
        logging.info(f"Sampler overrides applied: {sampler_overrides}")
        return
    launch(config, meta_cfg, args)
    if os.environ.get("COSMOS_EXIT_WITHOUT_FINALIZE", "").lower() in ("1", "true"):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
