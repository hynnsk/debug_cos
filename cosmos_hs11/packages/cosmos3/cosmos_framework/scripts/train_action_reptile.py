# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Reptile meta-training of Cosmos3-Edge for few-shot robot post-training (cosmos_hs11).

Replaces the FOMAML/ANIL loop of ``train_action_meta.py`` (cosmos_hs07/hs09) with Reptile
(Nichol, Achiam & Schulman 2018), whose objective -- "an initialization close to where k steps of
ordinary fine-tuning end up on every embodiment" -- matches the downstream use (1000-2000 steps of
LIBERO post-training) far better than "good after 5 gradient steps"::

    for meta-iteration:
        embodiment e ~ uniform;  K demonstrations -> support windows (sharded across ranks), Q -> query
        theta_tilde = k inner optimizer steps on the support set, starting from theta      (standard FSDP training)
        theta      <- theta + eps(it) * (theta_tilde - theta)                                (Reptile meta step)
        (diagnostics only) query loss at theta ("zero-shot") and at theta_tilde ("adapted")

Because the meta step is a plain interpolation, the model's own FSDP-sharded fp32 parameters ARE the fast
weights and the whole standard training stack is reused (``model.init_optimizer_scheduler``, FusedAdam with
``keys_to_select`` / ``lr_multipliers``, activation checkpointing, ``DistributedCheckpointer``). theta is one
extra local-shard copy of the trainable parameters (:class:`ReptileMetaState`). Two modes, chosen by the TOML:

* ``full`` (default): theta = moe_gen + time_embedder + vae2llm + llm2vae + k_norm_und_for_gen + action heads
  -- the trainable set of the Cosmos full-FT recipe ``action_policy_libero_edge``; downstream loads the saved
  DCP checkpoint as ``load_path`` and the heads via ``meta_action_init_path``.
* ``lora``: theta = LoRA adapters + heads + time_embedder -- the cosmos_hs09 set; downstream unchanged
  (``meta_action_init.pt`` carries all groups).

Outputs (``<IMAGINAIRE_OUTPUT_ROOT>/<project>/<group>/<name>/``)::

    checkpoints/iter_XXXXXXXXX/{model,optim,scheduler,trainer}   DCP checkpoint = theta (standard format; auto-resume)
    meta_action_init_iter_XXXXXX.pt, meta_action_init.pt         heads (+ LoRA / time_embedder in lora mode)
    reptile_train_log.jsonl                                       per-iteration scalars (also sent to W&B)

Run::

    torchrun --nproc_per_node=4 -m cosmos_framework.scripts.train_action_reptile \\
        --sft-toml=examples/toml/sft_config/action_reptile_meta_edge.toml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from loguru import logger as logging

from cosmos_framework.callbacks.grad_clip import (
    _clip_grads_with_global_norm,
    _group_params_by_mesh,
    _total_norm_by_mesh,
)
from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.data.generator.action.meta.meta_action_adapter import save_meta_action_init
from cosmos_framework.data.generator.action.meta.reptile_meta import (
    GROUP_LORA,
    GROUP_MOE_GEN,
    ReptileMetaState,
    capture_base_lrs,
    reset_optimizer_state,
    set_lr_scale,
)
from cosmos_framework.scripts.train_action_meta import (
    SAMPLER_OVERRIDE_KEYS,
    _mean,
    _prepare_dataloader_process_limits,
    _route_to_domain,
    _select_loss,
    apply_sampler_overrides,
)
from cosmos_framework.utils import distributed, misc
from cosmos_framework.utils.config import Config
from cosmos_framework.utils.context_managers import data_loader_init, distributed_init, model_init
from cosmos_framework.utils.lazy_config import LazyConfig, instantiate

warnings.filterwarnings(
    "ignore",
    message=r"Length of IterableDataset .* was reported to be .* samples have been fetched.*",
    category=UserWarning,
)


# --------------------------------------------------------------------------------------------
# [custom.meta]
# --------------------------------------------------------------------------------------------
@dataclass
class ReptileTrainConfig:
    """Inner / outer loop knobs from ``[custom.meta]`` (sampler keys are routed to the loader)."""

    # which parameters theta covers ("full" | "lora") -- only affects sanity checks and what meta_action_init.pt exports;
    # the trainable set itself is optimizer.keys_to_select of the experiment / TOML.
    theta_mode: str = "full"  # "full" | "lora" ("mode" itself is the dataset mode "wam" -- a sampler override key)
    # inner loop = k steps of the standard optimizer (FusedAdam, lr / multipliers from [optimizer])
    inner_steps: int = 20
    inner_warmup_steps: int = (
        3  # linear LR ramp over the first steps of every inner loop (Adam starts from empty state)
    )
    inner_grad_clip: float | None = 1.0
    inner_reset_optimizer: bool = True  # forget Adam moments between meta-episodes (each episode = a fresh fine-tune)
    # outer loop: theta <- theta + eps * (theta_tilde - theta)
    meta_optimizer: str = "sgd"  # "sgd" (Reptile) | "adam" (Adam on the displacement)
    meta_lr: float = 0.5  # eps at the start
    meta_lr_min_ratio: float = 0.1  # linear decay of eps to meta_lr * ratio at max_iter (Reptile anneals eps to ~0)
    meta_lr_warmup_iters: int = 0
    meta_betas: tuple[float, float] = (0.9, 0.999)
    meta_batch_embodiments: int = 1  # episodes per meta step (displacements averaged); 1 = serial Reptile
    # loss used in the inner loop / diagnostics: "total" = the recipe's objective (vision x scale + action)
    loss_mode: str = "total"
    scratch_domain_id: int = 31  # DomainAwareLinear row that holds the meta action heads
    # diagnostics
    eval_query_every: int = 5  # query loss at theta (zero-shot) and at theta_tilde (adapted) every N iters (0 = never)
    log_every: int | None = None  # defaults to trainer.logging_iter
    save_every: int | None = None  # defaults to checkpoint.save_iter

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ReptileTrainConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - names
        if unknown:
            raise ValueError(
                f"[custom.meta] has unknown keys {sorted(unknown)}; valid: {sorted(names | SAMPLER_OVERRIDE_KEYS)}"
            )
        kwargs = dict(raw)
        if "meta_betas" in kwargs:
            kwargs["meta_betas"] = tuple(float(x) for x in kwargs["meta_betas"])
        if (
            "inner_grad_clip" in kwargs
            and kwargs["inner_grad_clip"] is not None
            and float(kwargs["inner_grad_clip"]) <= 0
        ):
            kwargs["inner_grad_clip"] = None
        cfg = cls(**kwargs)
        if cfg.theta_mode not in ("full", "lora"):
            raise ValueError(f"theta_mode must be 'full' or 'lora', got {cfg.theta_mode!r}")
        if cfg.meta_optimizer not in ("sgd", "adam"):
            raise ValueError(f"meta_optimizer must be 'sgd' or 'adam', got {cfg.meta_optimizer!r}")
        if cfg.loss_mode not in ("action", "total"):
            raise ValueError(f"loss_mode must be 'action' or 'total', got {cfg.loss_mode!r}")
        if cfg.inner_steps < 1:
            raise ValueError("inner_steps must be >= 1")
        if cfg.meta_batch_embodiments < 1:
            raise ValueError("meta_batch_embodiments must be >= 1")
        if not (0.0 < cfg.meta_lr <= 1.0):
            raise ValueError("meta_lr (Reptile eps) must be in (0, 1]")
        return cfg


def split_custom_meta(config: Config) -> tuple[ReptileTrainConfig, dict[str, Any]]:
    custom = getattr(config, "custom", None) or {}
    raw = dict(custom.get("meta", {}) or {})
    sampler = {k: raw.pop(k) for k in list(raw) if k in SAMPLER_OVERRIDE_KEYS}
    return ReptileTrainConfig.from_dict(raw), sampler


def _eps_at(cfg: ReptileTrainConfig, iteration: int, max_iter: int) -> float:
    warm = max(0, int(cfg.meta_lr_warmup_iters))
    if warm > 0 and iteration < warm:
        return cfg.meta_lr * float(iteration + 1) / warm
    span = max(1, max_iter - warm)
    progress = min(1.0, max(0.0, (iteration - warm) / span))
    return cfg.meta_lr * (1.0 - (1.0 - cfg.meta_lr_min_ratio) * progress)


def _clip_grads(params: list[torch.nn.Parameter], max_norm: float | None) -> float:
    """Global-norm clip over FSDP/DTensor grads (mesh-aware, same helpers as the GradClip callback)."""
    with_grad = [p for p in params if p.grad is not None]
    if not with_grad:
        return float("nan")
    groups = _group_params_by_mesh(with_grad)
    total, _ = _total_norm_by_mesh(groups)
    if max_norm is not None and max_norm > 0:
        _clip_grads_with_global_norm(groups, float(max_norm), total)
    return float(total)


def _reduce_mean(values: list[float]) -> float:
    """Mean of per-rank lists of losses across all ranks (weighted by count); nan when empty everywhere."""
    t = torch.tensor([float(sum(values)), float(len(values))], dtype=torch.float64, device="cuda")
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t[0] / t[1]) if t[1] > 0 else float("nan")


def _assert_same_across_ranks(value: int, what: str) -> None:
    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return
    vals: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(vals, int(value))
    if len(set(vals)) != 1:
        raise RuntimeError(f"ranks disagree on {what}: {vals} -- FSDP collectives would deadlock")


def _warm_up_checkpoint_collectives(world_size: int) -> None:
    """Open the NCCL point-to-point connections that ``dcp.save`` needs while the GPU is still empty.

    The training loop only uses all_reduce / all_gather / FSDP collectives. ``dcp.save`` additionally runs
    ``gather_object`` + ``scatter_object_list`` (send/recv based), and NCCL sets up those P2P connections
    lazily on first use with fresh cudaMallocs. Doing that for the first time at the first save_iter, when
    the caching allocator already holds ~44 GiB of a 45 GiB A40, fails with
    "NCCL Error 1: unhandled cuda error". Issuing the same collectives once here (before the model is
    built) allocates the connection buffers up front; NCCL keeps them for the rest of the run.
    """
    if world_size <= 1:
        return
    rank = dist.get_rank()
    gathered: list[Any] | None = [None] * world_size if rank == 0 else None
    dist.gather_object({"rank": rank}, gathered, dst=0)
    scattered: list[Any] = [None]
    dist.scatter_object_list(scattered, [{"rank": r} for r in range(world_size)] if rank == 0 else None, src=0)
    dist.barrier()


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------
@logging.catch(reraise=True)
def launch(config: Config, cfg: ReptileTrainConfig, args: argparse.Namespace) -> None:
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
        (job_dir / "reptile_config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2) + "\n")
        logging.info(f"Job dir: {job_dir}")
        logging.info(f"Reptile config: {json.dumps(dataclasses.asdict(cfg))}")
    dist.barrier()
    _warm_up_checkpoint_collectives(world_size)
    misc.set_random_seed(seed=config.trainer.seed, by_rank=True)

    # ---- model (FSDP-sharded, fp32 master weights) + the standard optimizer ------------------
    with model_init():
        model = instantiate(config.model)
    model = model.to("cuda", memory_format=config.trainer.memory_format)
    model.on_train_start(config.trainer.memory_format)
    if model.config.ema.enabled:
        raise ValueError("action_reptile_meta_edge must run with model.config.ema.enabled=False")
    optimizer, scheduler = model.init_optimizer_scheduler(config.optimizer, config.scheduler)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=False)
    checkpointer = instantiate(config.checkpoint.type, config.checkpoint, config.job, callbacks=None)
    # warm start from the base DCP (load_path, keys_to_skip_loading) or auto-resume from this job's latest checkpoint
    start_iter = int(checkpointer.load(model, optimizer, scheduler, grad_scaler))
    model.train()

    state = ReptileMetaState(model.net, meta_optimizer=cfg.meta_optimizer, meta_betas=cfg.meta_betas)
    groups_present = set(state.groups.values())
    if cfg.theta_mode == "lora" and GROUP_LORA not in groups_present:
        raise ValueError(
            "mode='lora' but no trainable LoRA parameters: set model.lora_enabled=true and keys_to_select accordingly"
        )
    if cfg.theta_mode == "full" and GROUP_MOE_GEN not in groups_present:
        raise ValueError("mode='full' but moe_gen is not trainable: check optimizer.keys_to_select")
    if cfg.theta_mode == "full" and GROUP_LORA in groups_present:
        logging.warning(
            "mode='full' with trainable LoRA parameters -- they are meta-learned too (unusual; check the TOML)"
        )
    trainable_params = list(state.params.values())
    base_lrs = capture_base_lrs(optimizer)
    if cfg.inner_reset_optimizer:
        reset_optimizer_state(optimizer)
    logging.info(
        f"theta: {state.numel:,} params in {len(state.params)} tensors, by group {state.numel_by_group()}; "
        f"inner base lrs per group: {[[f'{lr:.2e}' for lr in b] for b in base_lrs]}; start_iter={start_iter}"
    )

    # ---- data: rank-synchronized meta-episodes ---------------------------------------------------
    dl_cfg = config.dataloader_train
    k_shot, w = int(dl_cfg.k_shot), int(dl_cfg.windows_per_demo)
    if k_shot * w < world_size:
        raise ValueError(
            f"k_shot * windows_per_demo = {k_shot * w} < world size {world_size}: some rank would get no support windows"
        )
    with data_loader_init():
        loader = instantiate(dl_cfg)
    loader.dataset.spec_offset = start_iter  # a resumed run continues with new episodes
    loader_iter = iter(loader)

    # ---- logging ---------------------------------------------------------------------------------
    max_iter = int(config.trainer.max_iter)
    log_every = int(cfg.log_every or config.trainer.logging_iter)
    save_every = int(cfg.save_every or config.checkpoint.save_iter)
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
                config={"reptile": dataclasses.asdict(cfg), "sft_toml": args.sft_toml},
            )
        except Exception as e:  # noqa: BLE001
            logging.warning(f"wandb init failed ({e}); continuing without wandb")
    jsonl = open(job_dir / "reptile_train_log.jsonl", "a") if is_rank0 else None

    def _save(iteration: int) -> None:
        # model == theta here (meta_step restores it). Drop the inner Adam moments so the checkpoint holds
        # only what a resume / downstream warm start needs.
        if cfg.inner_reset_optimizer:
            reset_optimizer_state(optimizer)
        # Return the allocator's cached-but-free blocks (~5 GiB at peak) to CUDA so the save's NCCL
        # collectives and DCP staging buffers have headroom on a near-full GPU.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
        meta = state.export_meta_action_init(
            cfg.scratch_domain_id,
            include_lora=(cfg.theta_mode == "lora"),
            include_time_embedder=(cfg.theta_mode == "lora"),
        )
        if is_rank0:
            md = {
                **state.metadata(cfg.theta_mode, cfg.scratch_domain_id),
                "iteration": iteration,
                "max_iter": max_iter,
                "reptile_config": dataclasses.asdict(cfg),
                "embodiments": list(dl_cfg.get("embodiments", [])),
                "base_checkpoint": str(config.checkpoint.load_path),
                "dcp_checkpoint": str(job_dir / "checkpoints" / f"iter_{iteration:09d}"),
                "job": {"project": config.job.project, "group": config.job.group, "name": config.job.name},
            }
            save_meta_action_init(job_dir / f"meta_action_init_iter_{iteration:06d}.pt", meta, md)
            save_meta_action_init(job_dir / "meta_action_init.pt", meta, md)
            logging.info(f"Saved theta at iteration {iteration}: DCP {md['dcp_checkpoint']} + meta_action_init.pt")

    def _fetch_synced() -> dict[str, Any] | None:
        """Next episode; every rank must hold the same spec. None when any rank failed to materialize it."""
        item = next(loader_iter)
        flags: list[Any] = [None] * world_size
        if world_size > 1:
            dist.all_gather_object(flags, (int(item["spec_id"]), bool(item.get("failed", False))))
        else:
            flags = [(int(item["spec_id"]), bool(item.get("failed", False)))]
        if len({f[0] for f in flags}) != 1:
            raise RuntimeError(f"meta-episode desync across ranks: spec ids {[f[0] for f in flags]}")
        if any(f[1] for f in flags):
            logging.warning(
                f"skipping meta-episode {item['spec_id']} ({item['embodiment']}): materialization failed on some rank"
            )
            return None
        return item

    def _eval_query(query_inputs: list, iteration: int) -> float:
        vals: list[float] = []
        with torch.no_grad():
            for inp in query_inputs:
                out, total = model.training_step_from_inputs(inp, iteration)
                vals.append(float(_select_loss(out, total, cfg.loss_mode).detach()))
        return _reduce_mean(vals)

    logging.info(
        f"Starting Reptile meta-training: iterations {start_iter} -> {max_iter}, world_size={world_size}, mode={cfg.theta_mode}"
    )
    iteration = start_iter
    ema_adapted: float | None = None
    while iteration < max_iter:
        t_iter = time.time()
        do_eval = cfg.eval_query_every > 0 and (iteration % cfg.eval_query_every == 0)
        episodes: list[dict[str, Any]] = []
        t0 = time.time()
        while len(episodes) < cfg.meta_batch_embodiments:
            item = _fetch_synced()
            if item is not None:
                episodes.append(item)
        t_data = time.time() - t0

        per_ep: list[dict[str, Any]] = []
        t_encode = t_inner = t_eval = 0.0
        for ep in episodes:
            support_batches, query_batches = ep["support"], ep["query"]
            for b in support_batches + query_batches:
                _route_to_domain(b, cfg.scratch_domain_id)
            _assert_same_across_ranks(len(query_batches), "number of query sub-batches")
            support_batches = [misc.to(b, device="cuda") for b in support_batches]
            query_batches = [misc.to(b, device="cuda") for b in query_batches]

            # VAE encode / tokenize once per episode (model == theta here)
            t0 = time.time()
            with torch.no_grad():
                support_inputs = [model._get_training_inputs(b, iteration) for b in support_batches]
                query_inputs = [model._get_training_inputs(b, iteration) for b in query_batches]
            t_encode += time.time() - t0

            t0 = time.time()
            q_zero = _eval_query(query_inputs, iteration) if do_eval and query_inputs else float("nan")
            t_eval += time.time() - t0

            # ---- inner loop: k standard optimizer steps from theta -------------------------------
            t0 = time.time()
            if cfg.inner_reset_optimizer:
                reset_optimizer_state(optimizer)
            inner_losses: list[float] = []
            inner_gnorms: list[float] = []
            for s in range(cfg.inner_steps):
                scale = min(1.0, (s + 1) / cfg.inner_warmup_steps) if cfg.inner_warmup_steps > 0 else 1.0
                set_lr_scale(optimizer, base_lrs, scale)
                inp = support_inputs[s % len(support_inputs)]
                out, total = model.training_step_from_inputs(inp, iteration)
                loss = _select_loss(out, total, cfg.loss_mode)
                backward_loss = out.get("_backward_loss", total) if cfg.loss_mode == "total" else loss
                backward_loss.backward()
                model.on_after_backward()
                inner_gnorms.append(_clip_grads(trainable_params, cfg.inner_grad_clip))
                model.on_before_optimizer_step(optimizer, scheduler, iteration=iteration)
                optimizer.step()
                model.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
                optimizer.zero_grad(set_to_none=True)
                inner_losses.append(float(loss.detach()))
            t_inner += time.time() - t0

            t0 = time.time()
            q_adapted = _eval_query(query_inputs, iteration) if do_eval and query_inputs else float("nan")
            t_eval += time.time() - t0
            disp = state.displacement_norms_by_group()
            per_ep.append(
                {
                    "embodiment": ep["embodiment"],
                    "support_loss_first": _reduce_mean([inner_losses[0]]),
                    "support_loss_last": _reduce_mean([inner_losses[-1]]),
                    "inner_grad_norm": _mean(inner_gnorms),
                    "query_loss_zero_shot": q_zero,
                    "query_loss": q_adapted,
                    "adapt_delta": math.sqrt(sum(v * v for v in disp.values())),
                    "adapt_delta_by_group": disp,
                }
            )
            if cfg.meta_batch_embodiments > 1:
                state.accumulate_displacement()
                state.restore_to_model()

        # ---- Reptile meta step -----------------------------------------------------------------
        t0 = time.time()
        eps = _eps_at(cfg, iteration, max_iter)
        meta_step_norm = state.meta_step(eps)  # theta <- theta + eps * d ; model <- theta
        t_meta = time.time() - t0
        iteration += 1

        # ---- logging (all values are already rank-averaged; every rank saw the same embodiment) -----
        if is_rank0:
            q_ad = _mean([e["query_loss"] for e in per_ep if not math.isnan(e["query_loss"])])
            if not math.isnan(q_ad):
                ema_adapted = q_ad if ema_adapted is None else 0.98 * ema_adapted + 0.02 * q_ad
            record: dict[str, Any] = {
                "iteration": iteration,
                "eps": eps,
                "query_loss": q_ad,
                "query_loss_ema": ema_adapted,
                "query_loss_zero_shot": _mean(
                    [e["query_loss_zero_shot"] for e in per_ep if not math.isnan(e["query_loss_zero_shot"])]
                ),
                "support_loss_first": _mean([e["support_loss_first"] for e in per_ep]),
                "support_loss_last": _mean([e["support_loss_last"] for e in per_ep]),
                "inner_grad_norm": _mean([e["inner_grad_norm"] for e in per_ep]),
                "adapt_delta": _mean([e["adapt_delta"] for e in per_ep]),
                "meta_step_norm": meta_step_norm,
                "iter_time": time.time() - t_iter,
                "t_data": t_data,
                "t_encode": t_encode,
                "t_inner": t_inner,
                "t_eval": t_eval,
                "t_meta": t_meta,
                # peak VRAM of this rank since process start (GiB); the first logged value is the steady-state footprint
                "mem_peak_alloc_gb": (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0,
                "mem_peak_reserved_gb": (torch.cuda.max_memory_reserved() / 2**30)
                if torch.cuda.is_available()
                else 0.0,
                "embodiments": [e["embodiment"] for e in per_ep],
            }
            if not (math.isnan(record["query_loss"]) or math.isnan(record["query_loss_zero_shot"])):
                record["query_loss_gain"] = record["query_loss_zero_shot"] - record["query_loss"]
            for grp in sorted({k for e in per_ep for k in e["adapt_delta_by_group"]}):
                record[f"adapt_delta/{grp}"] = _mean(
                    [e["adapt_delta_by_group"][grp] for e in per_ep if grp in e["adapt_delta_by_group"]]
                )
            for e in per_ep:
                if not math.isnan(e["query_loss"]):
                    record[f"query_loss/{e['embodiment']}"] = e["query_loss"]
                record[f"support_loss_last/{e['embodiment']}"] = e["support_loss_last"]
            if iteration % log_every == 0 or iteration == start_iter + 1:
                theta_norms = state.theta_norms_by_group()
                record.update({f"theta_norm/{g}": v for g, v in theta_norms.items()})
                q_str = (
                    f"query {record['query_loss']:.4f} (zero-shot {record['query_loss_zero_shot']:.4f}) | "
                    if do_eval
                    else ""
                )
                adapt_str = ", ".join(
                    f"{k.split('/')[1]} {v:.3f}" for k, v in record.items() if k.startswith("adapt_delta/")
                )
                logging.info(
                    f"iter {iteration}/{max_iter} | {record['embodiments']} | {q_str}"
                    f"support {record['support_loss_first']:.4f} -> {record['support_loss_last']:.4f} | "
                    f"inner-grad {record['inner_grad_norm']:.3f} | adapt {record['adapt_delta']:.3f} ({adapt_str}) | "
                    f"eps {eps:.3f} step {meta_step_norm:.3f} | {record['iter_time']:.1f}s (data {t_data:.1f} enc {t_encode:.1f} "
                    f"inner {t_inner:.1f} eval {t_eval:.1f} meta {t_meta:.1f}) | "
                    f"peak mem {record['mem_peak_alloc_gb']:.1f} GiB alloc / {record['mem_peak_reserved_gb']:.1f} GiB reserved"
                )
            if jsonl is not None:
                jsonl.write(json.dumps(record) + "\n")
                jsonl.flush()
            if wandb_run is not None:
                wandb_run.log(
                    {
                        k: v
                        for k, v in record.items()
                        if isinstance(v, (int, float))
                        and v is not None
                        and not (isinstance(v, float) and math.isnan(v))
                    },
                    step=iteration,
                )
        else:
            if iteration % log_every == 0 or iteration == start_iter + 1:
                state.theta_norms_by_group()  # collective: keep in step with rank 0
        if iteration % save_every == 0 or iteration == max_iter:
            _save(iteration)

    if jsonl is not None:
        jsonl.close()
    if wandb_run is not None:
        wandb_run.finish()
    logging.success(
        f"Reptile meta-training finished at iteration {iteration}. theta: {job_dir / 'checkpoints'} (DCP), heads: {job_dir / 'meta_action_init.pt'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reptile meta-training of Cosmos3-Edge for few-shot robot post-training"
    )
    parser.add_argument(
        "--sft-toml", required=True, help="Run TOML (experiment action_reptile_meta_edge + [custom.meta])."
    )
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, default=[], help="Extra Hydra dotted-path overrides (after --)."
    )
    parser.add_argument("--dryrun", action="store_true", help="Print the resolved config and exit.")
    args = parser.parse_args()

    config = load_experiment_from_toml(args.sft_toml, extra_overrides=args.opts)
    cfg, sampler_overrides = split_custom_meta(config)
    apply_sampler_overrides(config, sampler_overrides)
    args.config = args.sft_toml

    if args.dryrun:
        logging.info("Config:\n" + config.pretty_print(use_color=True))
        logging.info("Reptile config: " + json.dumps(dataclasses.asdict(cfg), indent=2))
        logging.info(f"Sampler overrides applied: {sampler_overrides}")
        return
    launch(config, cfg, args)
    if os.environ.get("COSMOS_EXIT_WITHOUT_FINALIZE", "").lower() in ("1", "true"):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
