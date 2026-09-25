# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Decoupled L2-SP: keep selected parameters close to their STARTING POINT during post-training (cosmos_hs09).

L2-SP (Li, Grandvalet & Davoine, ICML 2018) replaces the usual L2 penalty toward zero by a penalty toward the
pre-trained starting point, ``(alpha / 2) * ||p - p0||^2``. A penalty added to the *loss* is rescaled per coordinate by
Adam's second-moment estimate, so its effective strength is no longer ``alpha``; following the AdamW argument this
callback applies it *decoupled*, right after every optimizer step::

    p <- p - lr_t * alpha * (p - p0)      (== p <- lerp(p, p0, lr_t * alpha))

``lr_t`` is the current learning rate of ``p``'s optimizer group (schedule and ``optimizer.lr_multipliers`` included), so
a group with multiplier 0 (the frozen meta LoRA) is never touched. Under a persistent, sign-consistent Adam drift of
about ``lr`` per step the deviation ``|p - p0|`` saturates at about ``1 / alpha`` per coordinate, independent of ``lr``:
``alpha = 100`` caps the sustained drift of a ~0.02-scale meta-initialised head weight at roughly half its size, while a
non-persistent gradient moves it much less. The pull acts on the LOCAL FSDP shards in place: no autograd, no
communication, and it runs before the EMA update (``EMAModelCallback.on_training_step_end``), so ``net_ema`` follows.

``p0`` is the weights right after checkpoint load + ``meta_action_init`` injection, captured at ``on_train_start`` of a
fresh run (iteration 0) and written to ``<job.path_local>/l2sp_anchors/rank{r}_of_{W}.pt`` (one file per rank = its
shard). A resumed run (iteration > 0) reloads those ORIGINAL anchors instead of re-anchoring at the resumed weights;
resuming with a different world size is rejected.

Parameters are chosen by name substring, first matching pattern wins (the ``optimizer.lr_multipliers`` convention); a
pattern with ``alpha == 0`` is skipped and when every alpha is 0 the callback is a complete no-op. Logged every
``log_every`` steps: ``l2sp/rel_dev/<pattern>`` = ``||p - p0|| / ||p0||`` over the whole tensors of that pattern (for the
32-row action-head tables the 31 untouched rows dilute this number) and ``l2sp/rate/<pattern>`` = ``lr_t * alpha``.
Norms are summed over local shards + one all-reduce, which assumes disjoint shards (data_parallel_replicate_degree=1).
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.distributed as dist

from cosmos_framework.data.generator.action.meta.meta_action_adapter import net_param_lookup
from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback


def _local(t: torch.Tensor) -> torch.Tensor:
    try:
        from torch.distributed.tensor import DTensor

        if isinstance(t, DTensor):
            return t.to_local()
    except Exception:  # noqa: BLE001
        pass
    return t


def _optimizers(optimizer: Any) -> list[torch.optim.Optimizer]:
    return list(getattr(optimizer, "optimizers", None) or [optimizer])


def current_group_lrs(optimizer: Any) -> dict[int, float]:
    """``{id(param): lr}`` over every param group of a torch optimizer or an ``OptimizersContainer``."""
    out: dict[int, float] = {}
    for opt in _optimizers(optimizer):
        for group in opt.param_groups:
            lr = float(group["lr"])
            for p in group["params"]:
                out[id(p)] = lr
    return out


class L2SP(Callback):
    def __init__(
        self,
        alphas: dict[str, float] | None = None,
        log_every: int = 100,
        anchor_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.alphas = {str(k): float(v) for k, v in dict(alphas or {}).items()}
        for k, v in self.alphas.items():
            if v < 0:
                raise ValueError(f"L2SP: alpha for {k!r} must be >= 0, got {v}")
        self.log_every = int(log_every)
        self.anchor_dir_override = anchor_dir
        self.enabled = any(v > 0 for v in self.alphas.values())
        self._params: dict[str, torch.nn.Parameter] = {}
        self._anchors: dict[str, torch.Tensor] = {}
        self._alpha_of: dict[str, float] = {}
        self._pattern_of: dict[str, str] = {}
        self._warned_unmatched = False
        self.last_info: dict[str, float] = {}

    # ---------------------------------------------------------------- selection / anchors
    def _match(self, name: str) -> tuple[str, float] | None:
        for pattern, alpha in self.alphas.items():  # first match wins
            if pattern in name:
                return (pattern, alpha) if alpha > 0 else None
        return None

    def _anchor_path(self) -> str:
        base = self.anchor_dir_override
        if base is None:
            cfg = getattr(self, "config", None)
            if cfg is None:
                raise RuntimeError("L2SP: no anchor_dir given and no config attached (job.path_local unknown)")
            base = os.path.join(cfg.job.path_local, "l2sp_anchors")
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        return os.path.join(base, f"rank{rank}_of_{world}.pt")

    def on_train_start(self, model: Any, iteration: int = 0) -> None:
        if not self.enabled:
            return
        net = getattr(model, "net", model)
        for name, p in net_param_lookup(net).items():
            if not p.requires_grad:
                continue
            m = self._match(name)
            if m is None:
                continue
            self._params[name] = p
            self._pattern_of[name], self._alpha_of[name] = m
        if not self._params:
            raise ValueError(f"L2SP: alphas {self.alphas} matched no trainable parameter")
        path = self._anchor_path()
        if iteration == 0:
            self._anchors = {n: _local(p.data).detach().to(torch.float32).clone() for n, p in self._params.items()}
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({n: t.cpu() for n, t in self._anchors.items()}, path)
            how = f"captured at iteration 0 and saved to {path}"
        else:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"L2SP: resuming at iteration {iteration} but the anchor file {path} is missing (anchors are written by "
                    "the fresh run; a different world size needs the original run's sharding)"
                )
            saved = torch.load(path, map_location="cpu")
            missing = set(self._params) - set(saved)
            extra = set(saved) - set(self._params)
            if missing or extra:
                raise RuntimeError(
                    f"L2SP: anchor file {path} does not match the selected parameters (missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]})"
                )
            self._anchors = {}
            for n, p in self._params.items():
                loc = _local(p.data)
                if tuple(saved[n].shape) != tuple(loc.shape):
                    raise RuntimeError(
                        f"L2SP: anchor shape mismatch for {n}: {tuple(saved[n].shape)} vs local shard {tuple(loc.shape)}"
                    )
                self._anchors[n] = saved[n].to(device=loc.device, dtype=torch.float32)
            how = f"reloaded from {path}"
        by_pattern: dict[str, tuple[int, int]] = {}
        for n, p in self._params.items():
            k, c = by_pattern.get(self._pattern_of[n], (0, 0))
            by_pattern[self._pattern_of[n]] = (k + 1, c + _local(p.data).numel())
        log.critical(
            "[l2sp] anchors "
            + how
            + "; per pattern (tensors, local params, alpha): "
            + ", ".join(f"{pat}=({k}, {c}, {self.alphas[pat]:g})" for pat, (k, c) in by_pattern.items())
        )
        self._log(iteration, None)

    # ---------------------------------------------------------------- the pull
    @torch.no_grad()
    def on_before_zero_grad(self, model: Any, optimizer: Any, scheduler: Any, iteration: int = 0) -> None:
        if not self.enabled or not self._anchors:
            return
        lrs = current_group_lrs(optimizer)
        matched = 0
        for name, p in self._params.items():
            lr = lrs.get(id(p))
            if lr is None:
                continue
            matched += 1
            rate = min(1.0, lr * self._alpha_of[name])
            if rate <= 0.0:
                continue
            loc = _local(p.data)
            loc.lerp_(self._anchors[name].to(dtype=loc.dtype), rate)
        if matched == 0 and not self._warned_unmatched:
            self._warned_unmatched = True
            log.warning(
                "L2SP: none of the anchored parameters was found in the optimizer's param groups; the pull is inactive"
            )
        if self.log_every > 0 and iteration % self.log_every == 0:
            self._log(iteration, lrs)

    # ---------------------------------------------------------------- diagnostics
    def _log(self, iteration: int, lrs: dict[int, float] | None) -> None:
        patterns = sorted(set(self._pattern_of.values()))
        device = next(iter(self._anchors.values())).device if self._anchors else "cpu"
        dev = torch.zeros(len(patterns), dtype=torch.float64, device=device)
        ref = torch.zeros(len(patterns), dtype=torch.float64, device=device)
        rate: dict[str, float] = {}
        for name, p in self._params.items():
            i = patterns.index(self._pattern_of[name])
            loc = _local(p.data).to(torch.float32)
            a = self._anchors[name]
            dev[i] += (loc - a).double().pow(2).sum()
            ref[i] += a.double().pow(2).sum()
            if lrs is not None and self._pattern_of[name] not in rate and id(p) in lrs:
                rate[self._pattern_of[name]] = lrs[id(p)] * self._alpha_of[name]
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(dev, op=dist.ReduceOp.SUM)
            dist.all_reduce(ref, op=dist.ReduceOp.SUM)
        info: dict[str, float] = {}
        for i, pat in enumerate(patterns):
            r = float(ref[i].item())
            info[f"l2sp/rel_dev/{pat}"] = math.sqrt(float(dev[i].item())) / math.sqrt(r) if r > 0 else float("nan")
            if pat in rate:
                info[f"l2sp/rate/{pat}"] = rate[pat]
        self.last_info = dict(info)
        if distributed.is_rank0():
            log.info(
                f"[l2sp] iter {iteration}: " + ", ".join(f"{k.split('l2sp/')[-1]}={v:.4g}" for k, v in info.items())
            )
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(info, step=iteration)
            except Exception as e:  # noqa: BLE001
                log.warning(f"L2SP: wandb.log failed ({e})")
