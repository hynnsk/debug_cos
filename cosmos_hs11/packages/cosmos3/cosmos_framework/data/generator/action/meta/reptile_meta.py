# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Reptile meta-state for a (possibly FSDP-sharded) Cosmos3 model.

Reptile (Nichol, Achiam & Schulman 2018) has no query set and no meta-gradient. One meta-iteration is::

    theta_tilde = InnerOpt_k(theta, task data)          # k ordinary optimizer steps from theta
    theta      <- theta + eps * (theta_tilde - theta)   # move the meta-parameters toward the adapted point

so the *model's own parameters* act as the fast weights: the standard FSDP training stack (fp32 master
weights, FusedAdam, activation checkpointing) runs the inner loop, and :class:`ReptileMetaState` only keeps
one extra copy of every trainable parameter (its local shard) as ``theta``. After the inner loop it
interpolates ``theta`` toward the adapted weights and writes ``theta`` back into the model. That is what
makes FULL-parameter meta-training of ``moe_gen`` (1.4B) affordable here, where FOMAML (per-rank distinct
fast weights + a query gradient at the adapted point) was not.

theta covers exactly the parameters the optimizer trains (``optimizer.keys_to_select``) -- the downstream
post-training's trainable set:

* ``mode="full"`` : moe_gen + time_embedder + vae2llm + llm2vae + k_norm_und_for_gen + action heads
* ``mode="lora"`` : LoRA adapters + action heads + time_embedder (the cosmos_hs09 set)

Sharding: with pure FSDP (``data_parallel_replicate_degree=1``) the local shards of all ranks partition
each parameter, so norms are ``sqrt(all_reduce_sum(local sums))``. Under HSDP (replicate > 1) the group norms
would be over-counted by the replicate factor -- they are diagnostics only; the meta step itself is exact
either way because it is element-wise on each rank's own shard.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from cosmos_framework.data.generator.action.meta.meta_action_adapter import (
    DOMAIN_ROW_PARAM_NAMES,
    GROUP_ACTION_HEADS,
    GROUP_LORA,
    GROUP_MODALITY_EMBED,
    GROUP_TIME_EMBEDDER,
    MODALITY_EMBED_PARAM_NAME,
    lora_info,
    net_param_lookup,
    param_group,
    read_domain_row,
)
from cosmos_framework.utils import log

GROUP_MOE_GEN = "moe_gen"
GROUP_VAE2LLM = "vae2llm"
GROUP_LLM2VAE = "llm2vae"
GROUP_K_NORM = "k_norm_und_for_gen"

__all__ = [
    "GROUP_K_NORM",
    "GROUP_LLM2VAE",
    "GROUP_MOE_GEN",
    "GROUP_VAE2LLM",
    "ReptileMetaState",
    "capture_base_lrs",
    "local_tensor",
    "reptile_param_group",
    "reset_optimizer_state",
    "set_lr_scale",
]


def reptile_param_group(name: str) -> str:
    """theta group of a canonical parameter name (extends the hs09 groups with the full-FT modules)."""
    g = param_group(name)
    if g != "other":
        return g
    if "moe_gen" in name:
        return GROUP_MOE_GEN
    if "vae2llm" in name:
        return GROUP_VAE2LLM
    if "llm2vae" in name:
        return GROUP_LLM2VAE
    if "k_norm_und_for_gen" in name:
        return GROUP_K_NORM
    return "other"


def local_tensor(t: torch.Tensor) -> torch.Tensor:
    """The rank-local shard of a DTensor (identity for plain tensors)."""
    try:
        from torch.distributed.tensor import DTensor

        if isinstance(t, DTensor):
            return t.to_local()
    except Exception:  # noqa: BLE001
        pass
    return t


def _optimizers(container: Any) -> list[torch.optim.Optimizer]:
    return list(getattr(container, "optimizers", None) or [container])


def reset_optimizer_state(container: Any) -> None:
    """Forget the inner optimizer's moments (and FusedAdam's per-group step counter) between meta-episodes."""
    for opt in _optimizers(container):
        opt.state.clear()
        for group in opt.param_groups:
            group.pop("step", None)


def capture_base_lrs(container: Any) -> list[list[float]]:
    """Per-group base LRs (``lr * multiplier``). ``initial_lr`` when a LambdaLR already touched the groups."""
    return [[float(g.get("initial_lr", g["lr"])) for g in opt.param_groups] for opt in _optimizers(container)]


def set_lr_scale(container: Any, base_lrs: list[list[float]], scale: float) -> None:
    for opt, bases in zip(_optimizers(container), base_lrs, strict=True):
        for group, base in zip(opt.param_groups, bases, strict=True):
            group["lr"] = base * float(scale)


class ReptileMetaState:
    """Holds theta (one local-shard copy per trainable parameter) and performs the Reptile meta-step.

    Args:
        net: ``model.net`` after the optimizer was built (so ``requires_grad`` marks the trainable set).
        meta_optimizer: ``"sgd"`` (theta += eps * d, the original Reptile) or ``"adam"`` (Adam on the
            displacement ``d = theta_tilde - theta`` treated as a negative gradient; keeps 2 extra copies).
        meta_betas / meta_eps: Adam hyper-parameters for ``meta_optimizer="adam"``.
    """

    def __init__(
        self,
        net: nn.Module,
        meta_optimizer: str = "sgd",
        meta_betas: tuple[float, float] = (0.9, 0.999),
        meta_eps: float = 1e-8,
    ) -> None:
        if meta_optimizer not in ("sgd", "adam"):
            raise ValueError(f"meta_optimizer must be 'sgd' or 'adam', got {meta_optimizer!r}")
        self.net = net
        lookup = net_param_lookup(net)
        self.params: dict[str, nn.Parameter] = {n: p for n, p in lookup.items() if p.requires_grad}
        if not self.params:
            raise ValueError("no trainable parameters -- build the optimizer (keys_to_select) before ReptileMetaState")
        self.groups: dict[str, str] = {n: reptile_param_group(n) for n in self.params}
        self.theta: dict[str, torch.Tensor] = {n: local_tensor(p.data).detach().clone() for n, p in self.params.items()}
        self.meta_optimizer = meta_optimizer
        self.meta_betas = (float(meta_betas[0]), float(meta_betas[1]))
        self.meta_eps = float(meta_eps)
        self.adam_m: dict[str, torch.Tensor] | None = None
        self.adam_v: dict[str, torch.Tensor] | None = None
        self.adam_t = 0
        self.accum: dict[str, torch.Tensor] | None = None
        self.accum_count = 0
        log.info(
            f"ReptileMetaState: {len(self.params)} trainable tensors, numel by group {self.numel_by_group()}, "
            f"meta_optimizer={meta_optimizer}, lora={lora_info(net)}"
        )

    # ---- bookkeeping -----------------------------------------------------------------
    def numel_by_group(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for n, p in self.params.items():
            out[self.groups[n]] = out.get(self.groups[n], 0) + int(p.numel())  # DTensor.numel() is the global size
        return dict(sorted(out.items()))

    @property
    def numel(self) -> int:
        return int(sum(self.numel_by_group().values()))

    @staticmethod
    def _all_reduce_sum(values: dict[str, torch.Tensor]) -> dict[str, float]:
        if not values:
            return {}
        keys = sorted(values)
        t = torch.stack([values[k].float() for k in keys])
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return {k: float(v) for k, v in zip(keys, t.tolist(), strict=True)}

    @torch.no_grad()
    def displacement_norms_by_group(self) -> dict[str, float]:
        """||theta_tilde - theta||_2 per group (theta_tilde = the model's current weights)."""
        sq: dict[str, torch.Tensor] = {}
        for n, p in self.params.items():
            d = local_tensor(p.data).float() - self.theta[n].float()
            g = self.groups[n]
            sq[g] = sq.get(g, torch.zeros((), device=d.device)) + (d * d).sum()
        return {g: v**0.5 for g, v in self._all_reduce_sum(sq).items()}

    @torch.no_grad()
    def theta_norms_by_group(self) -> dict[str, float]:
        sq: dict[str, torch.Tensor] = {}
        for n in self.params:
            t = self.theta[n].float()
            g = self.groups[n]
            sq[g] = sq.get(g, torch.zeros((), device=t.device)) + (t * t).sum()
        return {g: v**0.5 for g, v in self._all_reduce_sum(sq).items()}

    # ---- fast weights <-> theta -----------------------------------------------------
    @torch.no_grad()
    def restore_to_model(self) -> None:
        """Write theta into the model (start of a meta-episode / after the meta step)."""
        for n, p in self.params.items():
            local_tensor(p.data).copy_(self.theta[n])

    @torch.no_grad()
    def accumulate_displacement(self) -> None:
        """For ``meta_batch_embodiments > 1``: add ``theta_tilde - theta`` of the finished episode to the batch."""
        if self.accum is None:
            self.accum = {n: torch.zeros_like(self.theta[n]) for n in self.params}
        for n, p in self.params.items():
            self.accum[n].add_(local_tensor(p.data) - self.theta[n])
        self.accum_count += 1

    @torch.no_grad()
    def meta_step(self, eps: float) -> float:
        """theta <- theta + eps * d, with d the (batch-averaged) displacement; then restore theta into the model.

        Returns the global L2 norm of the applied update ``eps * d`` (for SGD) or of the Adam update.
        """
        use_accum = self.accum is not None and self.accum_count > 0
        sq_update = torch.zeros((), device=next(iter(self.theta.values())).device)
        if self.meta_optimizer == "adam":
            self.adam_t += 1
            b1, b2 = self.meta_betas
            if self.adam_m is None:
                self.adam_m = {n: torch.zeros_like(self.theta[n]) for n in self.params}
                self.adam_v = {n: torch.zeros_like(self.theta[n]) for n in self.params}
        for n, p in self.params.items():
            if use_accum:
                d = self.accum[n].div_(float(self.accum_count))
            else:
                d = local_tensor(p.data) - self.theta[n]
            if self.meta_optimizer == "sgd":
                upd = d.mul_(float(eps))
            else:
                m, v = self.adam_m[n], self.adam_v[n]  # type: ignore[index]
                m.mul_(b1).add_(d, alpha=1 - b1)
                v.mul_(b2).addcmul_(d, d, value=1 - b2)
                m_hat = m / (1 - b1**self.adam_t)
                v_hat = v / (1 - b2**self.adam_t)
                upd = (m_hat / (v_hat.sqrt() + self.meta_eps)).mul_(float(eps))
            sq_update = sq_update + (upd.float() * upd.float()).sum()
            self.theta[n].add_(upd.to(self.theta[n].dtype))
        self.accum = None
        self.accum_count = 0
        self.restore_to_model()
        return self._all_reduce_sum({"u": sq_update})["u"] ** 0.5

    # ---- export ---------------------------------------------------------------------
    @torch.no_grad()
    def export_meta_action_init(
        self, scratch_domain_id: int, include_lora: bool, include_time_embedder: bool
    ) -> dict[str, torch.Tensor]:
        """theta groups that downstream cannot take from the DCP checkpoint, in the ``meta_action_init.pt``
        format of ``meta_action_adapter`` (heads = scratch row; shared groups = full tensors). Collective:
        every rank must call it (DTensor ``full_tensor``)."""
        out: dict[str, torch.Tensor] = {}
        lookup = net_param_lookup(self.net)
        for n in DOMAIN_ROW_PARAM_NAMES:
            out[n] = read_domain_row(lookup[n], int(scratch_domain_id))
        if MODALITY_EMBED_PARAM_NAME in lookup:
            out[MODALITY_EMBED_PARAM_NAME] = read_domain_row(lookup[MODALITY_EMBED_PARAM_NAME], None)
        for n, p in self.params.items():
            g = self.groups[n]
            if (g == GROUP_LORA and include_lora) or (g == GROUP_TIME_EMBEDDER and include_time_embedder):
                out[n] = read_domain_row(p, None)
        return out

    def metadata(self, mode: str, scratch_domain_id: int) -> dict[str, Any]:
        return {
            "algorithm": "reptile",
            "mode": mode,
            "scratch_domain_id": int(scratch_domain_id),
            "num_params": len(self.params),
            "numel_by_group": self.numel_by_group(),
            "meta_optimizer": self.meta_optimizer,
            "lora": lora_info(self.net),
            "include_modality_embed": MODALITY_EMBED_PARAM_NAME in self.params or True,
            "include_lora": any(g == GROUP_LORA for g in self.groups.values()),
            "include_time_embedder": any(g == GROUP_TIME_EMBEDDER for g in self.groups.values()),
        }


# keep the hs09 group names importable from here as well
__all__ += ["GROUP_ACTION_HEADS", "GROUP_LORA", "GROUP_MODALITY_EMBED", "GROUP_TIME_EMBEDDER"]
