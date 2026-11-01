# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""cosmos_hs11 v17: shared theta + one PRIVATE LoRA adapter per source embodiment for Reptile meta-training.

Motivation (docs/action_reptile_meta.md, section 11): 96 % of the v4 Reptile movement cancels across embodiments and the
downstream LIBERO fine-tune partly undoes the meta delta (cos -0.42). Hypothesis: that part is embodiment-specific.
Give every source embodiment its own LoRA adapter (``Delta_e``) that absorbs those directions, keep only the shared
part in theta:

* an episode of embodiment ``e`` runs the inner loop on ``theta + Delta_e`` (LoRA active, both train);
* the Reptile step ``theta <- theta + eps (theta_tilde - theta)`` is unchanged and touches theta ONLY
  (``ReptileMetaState(exclude_groups={GROUP_LORA})``: the adapters are never copied, interpolated or restored);
* ``Delta_e`` keeps its post-inner-loop value for the next episode of ``e`` (this bank), the other adapters are untouched.

Downstream (LIBERO) is a NEW embodiment: post-training starts from theta alone -- the DCP's LoRA tensors belong to the last
active source embodiment and are ignored by a recipe with ``model.lora_enabled=false`` -- plus the meta action heads
(``meta_action_init.pt``) exactly as before.

The bank holds one rank-local shard copy per adapter parameter and slot (FSDP2 DTensor -> ``to_local()``), like
``ReptileMetaState`` holds theta. Checkpointing is per rank (``private_lora_bank_rankXX_ofYY.pt`` next to the DCP
iteration dir), so a resume needs the same world size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
from torch import nn

from cosmos_framework.data.generator.action.meta.meta_action_adapter import GROUP_LORA, net_param_lookup
from cosmos_framework.data.generator.action.meta.reptile_meta import local_tensor, reptile_param_group
from cosmos_framework.utils import log

__all__ = ["PrivateLoraBank"]


class PrivateLoraBank:
    """Per-slot (embodiment) copies of every trainable LoRA parameter of ``net``.

    Args:
        net: ``model.net`` after the optimizer was built (``requires_grad`` marks the trainable set); it must hold
            trainable ``*.lora_A/lora_B`` parameters (``model.lora_enabled=true`` + ``"lora_"`` in ``keys_to_select``).
        slots: the embodiment names (one adapter each). Every slot starts from the network's current adapter values
            (``A`` ~ kaiming, ``B = 0``: an identity adapter, like the downstream LoRA baseline).
    """

    def __init__(self, net: nn.Module, slots: Sequence[str]) -> None:
        slots = [str(s) for s in slots]
        if not slots:
            raise ValueError("PrivateLoraBank needs at least one slot (embodiment)")
        if len(set(slots)) != len(slots):
            raise ValueError(f"duplicate slot names: {slots}")
        lookup = net_param_lookup(net)
        self.params: dict[str, nn.Parameter] = {
            n: p for n, p in lookup.items() if p.requires_grad and reptile_param_group(n) == GROUP_LORA
        }
        if not self.params:
            raise ValueError(
                "no trainable LoRA parameters: set model.lora_enabled=true and add 'lora_' to optimizer.keys_to_select"
            )
        self.slots: list[str] = slots
        self.init: dict[str, torch.Tensor] = {n: local_tensor(p.data).detach().clone() for n, p in self.params.items()}
        self.bank: dict[str, dict[str, torch.Tensor]] = {s: {n: t.clone() for n, t in self.init.items()} for s in slots}
        self.active: str | None = None
        log.info(
            f"PrivateLoraBank: {len(self.params)} LoRA tensors ({self.numel:,} params) x {len(slots)} slots {slots}"
        )

    # ---- bookkeeping -----------------------------------------------------------------------------
    @property
    def numel(self) -> int:
        """Global parameter count of ONE adapter (DTensor.numel() is the global size)."""
        return int(sum(int(p.numel()) for p in self.params.values()))

    @property
    def names(self) -> list[str]:
        return list(self.params)

    # ---- model <-> bank --------------------------------------------------------------------------
    @torch.no_grad()
    def activate(self, slot: str) -> None:
        """Write slot's adapter into the model (start of a meta-episode of that embodiment)."""
        if slot not in self.bank:
            raise KeyError(f"unknown private-LoRA slot {slot!r}; slots: {self.slots}")
        for n, p in self.params.items():
            local_tensor(p.data).copy_(self.bank[slot][n])
        self.active = slot

    @torch.no_grad()
    def store(self, slot: str | None = None) -> None:
        """Read the model's (inner-loop-updated) adapter back into ``slot`` (default: the active one)."""
        slot = self.active if slot is None else slot
        if slot is None:
            raise RuntimeError("PrivateLoraBank.store(): no active slot (call activate() first)")
        if slot not in self.bank:
            raise KeyError(f"unknown private-LoRA slot {slot!r}; slots: {self.slots}")
        for n, p in self.params.items():
            self.bank[slot][n].copy_(local_tensor(p.data))

    @torch.no_grad()
    def delta_norms(self) -> dict[str, float]:
        """Per-slot global L2 norm of (adapter - initial adapter). Collective: every rank must call it."""
        device = next(iter(self.init.values())).device
        sq = torch.zeros(len(self.slots), dtype=torch.float32, device=device)
        for i, s in enumerate(self.slots):
            for n in self.params:
                d = self.bank[s][n].float() - self.init[n].float()
                sq[i] += (d * d).sum()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sq, op=dist.ReduceOp.SUM)
        return {s: float(sq[i].sqrt()) for i, s in enumerate(self.slots)}

    # ---- checkpointing (per rank) ----------------------------------------------------------------
    @staticmethod
    def shard_filename(rank: int, world_size: int) -> str:
        return f"private_lora_bank_rank{int(rank):02d}_of{int(world_size):02d}.pt"

    def state_dict(self) -> dict[str, Any]:
        rank, world = 0, 1
        if dist.is_available() and dist.is_initialized():
            rank, world = dist.get_rank(), dist.get_world_size()
        return {
            "slots": list(self.slots),
            "names": list(self.params),
            "active": self.active,
            "rank": int(rank),
            "world_size": int(world),
            "init": {n: t.detach().cpu() for n, t in self.init.items()},
            "bank": {s: {n: t.detach().cpu() for n, t in d.items()} for s, d in self.bank.items()},
        }

    @torch.no_grad()
    def load_state_dict(self, sd: dict[str, Any]) -> None:
        if list(sd["slots"]) != list(self.slots):
            raise ValueError(f"slot mismatch: checkpoint {sd['slots']} vs config {self.slots}")
        if set(sd["names"]) != set(self.params):
            missing = sorted(set(self.params) - set(sd["names"]))
            extra = sorted(set(sd["names"]) - set(self.params))
            raise ValueError(f"LoRA parameter mismatch: missing {missing[:5]}, unexpected {extra[:5]}")
        for n, t in self.init.items():
            src = sd["init"][n]
            if tuple(src.shape) != tuple(t.shape):
                raise ValueError(f"{n}: local shard shape {tuple(t.shape)} vs checkpoint {tuple(src.shape)} (world size changed?)")
            t.copy_(src.to(t.device, t.dtype))
        for s in self.slots:
            for n, t in self.bank[s].items():
                t.copy_(sd["bank"][s][n].to(t.device, t.dtype))
        self.active = sd.get("active")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self.load_state_dict(torch.load(Path(path), map_location="cpu", weights_only=False))
