# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Meta action initialization (theta_meta) for the Cosmos3 action I/O projectors.

Cosmos3 routes actions through ``DomainAwareLinear`` layers whose weights are *per embodiment*::

    action --action2llm[domain]--> LLM hidden --backbone--> hidden --llm2action[domain]--> action

Training the five domain rows of the meta-training embodiments independently would not give a
new embodiment (LIBERO) anything to start from. The :class:`MetaActionAdapter` instead owns ONE
shared, embodiment-agnostic parameter set (fp32)::

    theta_meta = { action2llm.fc.weight    [O*I]   (flat DomainAwareLinear row, I = max_action_dim, O = hidden)
                   action2llm.bias.weight  [O]
                   llm2action.fc.weight    [O*I]   (I = hidden, O = max_action_dim)
                   llm2action.bias.weight  [O]
                   action_modality_embed   [hidden] (shared, optional) }

and implements the first-order MAML (FOMAML / ANIL-style) bookkeeping around it:

1. ``start_episode()`` copies theta_meta into *fast weights* and writes them into a scratch domain
   row of the frozen network (all samples of a meta-episode are routed to that row).
2. Support batches -> ``loss.backward()`` -> :meth:`collect_grads` reads the row gradients ->
   :meth:`inner_update` applies SGD on the fp32 fast weights and rewrites the row.
3. Query batches -> ``loss.backward()`` -> :meth:`collect_grads` -> (all-reduce across ranks) ->
   :meth:`assign_meta_grads` -> a regular optimizer steps theta_meta.

All 64-D actions are zero-padded by ``ActionTransformPipeline``, so a 10-D (RT-1) and a 20-D (YAM)
embodiment share the same projection shapes; the meta initialization is therefore defined on the
full ``max_action_dim`` projection.

Downstream, :func:`apply_meta_action_init_to_net` copies the saved ``meta_action_init.pt`` into the
LIBERO domain row (works for plain tensors and FSDP2 ``DTensor`` parameters).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from cosmos_framework.utils import log

META_ACTION_FILE_VERSION = 1
DOMAIN_ROW_PARAM_NAMES: tuple[str, ...] = (
    "action2llm.fc.weight",
    "action2llm.bias.weight",
    "llm2action.fc.weight",
    "llm2action.bias.weight",
)
MODALITY_EMBED_PARAM_NAME = "action_modality_embed"

__all__ = [
    "DOMAIN_ROW_PARAM_NAMES",
    "META_ACTION_FILE_VERSION",
    "MODALITY_EMBED_PARAM_NAME",
    "MetaActionAdapter",
    "MetaActionInitSpec",
    "apply_meta_action_init_to_net",
    "fresh_meta_action_init",
    "get_action_head_param",
    "load_meta_action_init",
    "save_meta_action_init",
    "write_domain_row",
]


# --------------------------------------------------------------------------------------------
# Parameter access helpers (plain tensors and DTensors)
# --------------------------------------------------------------------------------------------
def get_action_head_param(net: nn.Module, dotted_name: str) -> nn.Parameter:
    """Resolve ``"action2llm.fc.weight"``-style names on the VFM network."""
    obj: Any = net
    for part in dotted_name.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"{dotted_name} resolved to {type(obj).__name__}, expected a parameter")
    return obj


def _is_dtensor(t: torch.Tensor) -> bool:
    try:
        from torch.distributed.tensor import DTensor

        return isinstance(t, DTensor)
    except Exception:  # noqa: BLE001
        return False


@torch.no_grad()
def write_domain_row(param: torch.Tensor, row: int | None, value: torch.Tensor) -> None:
    """Write ``value`` into ``param[row]`` (or the whole param when ``row is None``).

    Handles FSDP2 ``DTensor`` parameters by gathering the full tensor (a collective every rank must
    join), patching the row and re-distributing with the parameter's own placements.
    """
    data = param.data
    if _is_dtensor(data):
        from torch.distributed.tensor import distribute_tensor

        full = data.full_tensor()
        if row is None:
            full.copy_(value.to(device=full.device, dtype=full.dtype))
        else:
            full[row].copy_(value.to(device=full.device, dtype=full.dtype))
        new = distribute_tensor(full, data.device_mesh, data.placements)
        try:
            data.copy_(new)
        except Exception:  # noqa: BLE001 - fall back to the local shard
            data.to_local().copy_(new.to_local())
        return
    if row is None:
        data.copy_(value.to(device=data.device, dtype=data.dtype))
    else:
        data[row].copy_(value.to(device=data.device, dtype=data.dtype))


@torch.no_grad()
def read_domain_row(param: torch.Tensor, row: int | None) -> torch.Tensor:
    """Return a fp32 copy of ``param[row]`` (or the whole param), gathering DTensors."""
    data = param.data
    if _is_dtensor(data):
        data = data.full_tensor()
    out = data if row is None else data[row]
    return out.detach().float().clone()


# --------------------------------------------------------------------------------------------
# Fresh initialization (mirrors Cosmos3VFMNetwork.init_weights for the action heads)
# --------------------------------------------------------------------------------------------
def fresh_meta_action_init(
    hidden_size: int,
    action_dim: int,
    include_modality_embed: bool = True,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Sample theta_meta with the Cosmos3 action-boundary init policy.

    ``action2llm`` (in=action_dim, out=hidden): trunc-normal std ``1/sqrt(action_dim)``;
    ``llm2action`` (in=hidden, out=action_dim): std ``1/sqrt(hidden)``; biases zero;
    ``action_modality_embed``: std ``1/sqrt(hidden)``. Identical to what the downstream recipe gets
    when it skips loading the action heads, so "baseline init" and "meta init" only differ by the
    meta-training.
    """
    device = torch.device(device)

    def _tn(shape: tuple[int, ...], std: float) -> torch.Tensor:
        t = torch.empty(shape, dtype=torch.float32, device=device)
        nn.init.trunc_normal_(t, std=std, a=-3 * std, b=3 * std, generator=generator)
        return t

    out = {
        "action2llm.fc.weight": _tn((hidden_size * action_dim,), 1.0 / math.sqrt(action_dim)),
        "action2llm.bias.weight": torch.zeros(hidden_size, dtype=torch.float32, device=device),
        "llm2action.fc.weight": _tn((hidden_size * action_dim,), 1.0 / math.sqrt(hidden_size)),
        "llm2action.bias.weight": torch.zeros(action_dim, dtype=torch.float32, device=device),
    }
    if include_modality_embed:
        out[MODALITY_EMBED_PARAM_NAME] = _tn((hidden_size,), 1.0 / math.sqrt(hidden_size))
    return out


# --------------------------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------------------------
@dataclass
class MetaActionInitSpec:
    """How theta_meta is initialized at the start of meta-training.

    ``source``:
      * ``"fresh"``           -- Cosmos3 fresh init (default; same distribution as the LIBERO baseline)
      * ``"checkpoint_row"``  -- copy one mid-trained domain row (``domain_ids[0]``)
      * ``"checkpoint_mean"`` -- average the mid-trained rows of ``domain_ids``
      * ``"zeros"``           -- all zeros (debugging)
    """

    source: str = "fresh"
    domain_ids: list[int] = field(default_factory=list)
    seed: int = 0


class MetaActionAdapter:
    """Holds theta_meta and drives fast-weight writes/reads on a (non-FSDP) network.

    Args:
        net: The ``Cosmos3VFMNetwork`` (``model.net``) with plain (non-DTensor) parameters.
        scratch_domain_id: The ``DomainAwareLinear`` row all meta-episode samples are routed to.
        include_modality_embed: Also meta-learn ``action_modality_embed``.
        init: See :class:`MetaActionInitSpec`.
        freeze_others: Set ``requires_grad=False`` on every other network parameter.
    """

    def __init__(
        self,
        net: nn.Module,
        scratch_domain_id: int,
        include_modality_embed: bool = True,
        init: MetaActionInitSpec | None = None,
        freeze_others: bool = True,
    ) -> None:
        self.net = net
        self.row = int(scratch_domain_id)
        self.include_modality_embed = bool(include_modality_embed)
        self.names: tuple[str, ...] = DOMAIN_ROW_PARAM_NAMES + (
            (MODALITY_EMBED_PARAM_NAME,) if include_modality_embed else ()
        )
        self.params: dict[str, nn.Parameter] = {n: get_action_head_param(net, n) for n in self.names}
        for n in DOMAIN_ROW_PARAM_NAMES:
            p = self.params[n]
            if _is_dtensor(p.data):
                raise TypeError(
                    "MetaActionAdapter needs plain (non-FSDP) action head parameters; build the meta model with "
                    "parallelism.enable_inference_mode=True and data_parallel_shard_degree=1."
                )
            if p.dim() != 2:
                raise ValueError(f"{n} must be a [num_domains, features] DomainAwareLinear table, got {tuple(p.shape)}")
            if not 0 <= self.row < p.shape[0]:
                raise ValueError(f"scratch_domain_id={self.row} out of range for {n} with {p.shape[0]} domains")
        self.hidden_size = int(self.params["action2llm.bias.weight"].shape[1])
        self.action_dim = int(self.params["llm2action.bias.weight"].shape[1])
        assert self.params["action2llm.fc.weight"].shape[1] == self.hidden_size * self.action_dim
        self.device = self.params["action2llm.fc.weight"].device

        if freeze_others:
            net.requires_grad_(False)
        for p in self.params.values():
            p.requires_grad_(True)

        init = init or MetaActionInitSpec()
        self.init_spec = init
        self.meta: dict[str, torch.Tensor] = self._initialize(init)
        for t in self.meta.values():
            t.requires_grad_(True)
        self.fast: dict[str, torch.Tensor] | None = None
        log.info(
            f"MetaActionAdapter: row={self.row} hidden={self.hidden_size} action_dim={self.action_dim} "
            f"params={list(self.names)} init={init.source} numel={self.numel:,}"
        )

    # ---- init --------------------------------------------------------------
    def _row_shape(self, name: str) -> tuple[int, ...]:
        p = self.params[name]
        return tuple(p.shape[1:]) if name in DOMAIN_ROW_PARAM_NAMES else tuple(p.shape)

    def _initialize(self, spec: MetaActionInitSpec) -> dict[str, torch.Tensor]:
        if spec.source == "fresh":
            gen = torch.Generator(device="cpu").manual_seed(int(spec.seed))
            fresh = fresh_meta_action_init(self.hidden_size, self.action_dim, self.include_modality_embed, gen)
            return {n: fresh[n].to(self.device) for n in self.names}
        if spec.source == "zeros":
            return {n: torch.zeros(self._row_shape(n), dtype=torch.float32, device=self.device) for n in self.names}
        if spec.source in ("checkpoint_row", "checkpoint_mean"):
            ids = list(spec.domain_ids) or [self.row]
            if spec.source == "checkpoint_row":
                ids = ids[:1]
            out: dict[str, torch.Tensor] = {}
            for n in self.names:
                if n in DOMAIN_ROW_PARAM_NAMES:
                    rows = torch.stack([read_domain_row(self.params[n], i) for i in ids], dim=0)
                    out[n] = rows.mean(dim=0)
                else:
                    out[n] = read_domain_row(self.params[n], None)
            return out
        raise ValueError(f"Unknown meta init source {spec.source!r}")

    # ---- properties --------------------------------------------------------
    @property
    def meta_parameters(self) -> list[torch.Tensor]:
        return [self.meta[n] for n in self.names]

    @property
    def numel(self) -> int:
        return int(sum(t.numel() for t in self.meta.values()))

    # ---- fast-weight plumbing ----------------------------------------------------
    @torch.no_grad()
    def _write(self, values: dict[str, torch.Tensor]) -> None:
        for n in self.names:
            row = self.row if n in DOMAIN_ROW_PARAM_NAMES else None
            write_domain_row(self.params[n], row, values[n])

    def start_episode(self) -> None:
        """Reset fast weights to theta_meta and write them into the scratch domain row."""
        self.fast = {n: self.meta[n].detach().clone() for n in self.names}
        self._write(self.fast)

    def zero_grads(self) -> None:
        for p in self.params.values():
            p.grad = None

    @torch.no_grad()
    def collect_grads(self, zero: bool = True) -> dict[str, torch.Tensor]:
        """fp32 gradients of the fast weights (rows of the network grads); missing grads -> zeros."""
        grads: dict[str, torch.Tensor] = {}
        for n in self.names:
            p = self.params[n]
            if p.grad is None:
                grads[n] = torch.zeros(self._row_shape(n), dtype=torch.float32, device=self.device)
            else:
                g = p.grad if n not in DOMAIN_ROW_PARAM_NAMES else p.grad[self.row]
                grads[n] = g.detach().float().clone()
        if zero:
            self.zero_grads()
        return grads

    @staticmethod
    def grad_norm(grads: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.sqrt(sum((g.float() ** 2).sum() for g in grads.values()))

    @staticmethod
    def clip_grads_(grads: dict[str, torch.Tensor], max_norm: float | None) -> float:
        """In-place global-norm clipping; returns the pre-clip norm."""
        norm = MetaActionAdapter.grad_norm(grads)
        if max_norm is not None and max_norm > 0:
            scale = float(max_norm) / (float(norm) + 1e-6)
            if scale < 1.0:
                for g in grads.values():
                    g.mul_(scale)
        return float(norm)

    @torch.no_grad()
    def inner_update(self, grads: dict[str, torch.Tensor], lr: float) -> None:
        """One SGD step on the fp32 fast weights, then re-write the scratch row."""
        if self.fast is None:
            raise RuntimeError("call start_episode() before inner_update()")
        for n in self.names:
            self.fast[n].sub_(grads[n], alpha=float(lr))
        self._write(self.fast)

    @torch.no_grad()
    def set_fast(self, values: dict[str, torch.Tensor]) -> None:
        """Replace the fast weights (e.g. after an Adam-style inner optimizer step) and re-write."""
        self.fast = {n: values[n].detach().clone() for n in self.names}
        self._write(self.fast)

    @torch.no_grad()
    def fast_delta_norm(self) -> float:
        """L2 distance between the current fast weights and theta_meta (adaptation magnitude)."""
        if self.fast is None:
            return 0.0
        return float(torch.sqrt(sum(((self.fast[n] - self.meta[n]) ** 2).sum() for n in self.names)))

    def assign_meta_grads(self, grads: dict[str, torch.Tensor]) -> None:
        """Set ``theta_meta.grad`` (first-order MAML: the query gradient at the adapted weights)."""
        for n in self.names:
            self.meta[n].grad = grads[n].to(self.meta[n].dtype)

    # ---- persistence --------------------------------------------------------------
    def export_meta(self) -> dict[str, torch.Tensor]:
        return {n: self.meta[n].detach().float().cpu().clone() for n in self.names}

    def load_meta(self, values: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for n in self.names:
                self.meta[n].copy_(values[n].to(self.meta[n].device, torch.float32))

    def metadata(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "action_dim": self.action_dim,
            "include_modality_embed": self.include_modality_embed,
            "scratch_domain_id": self.row,
            "param_names": list(self.names),
            "init": {"source": self.init_spec.source, "domain_ids": list(self.init_spec.domain_ids), "seed": self.init_spec.seed},
        }


# --------------------------------------------------------------------------------------------
# meta_action_init.pt I/O and downstream injection
# --------------------------------------------------------------------------------------------
def save_meta_action_init(path: str | Path, meta_params: dict[str, torch.Tensor], metadata: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "cosmos3_meta_action_init",
        "version": META_ACTION_FILE_VERSION,
        "meta_params": {k: v.detach().float().cpu().clone() for k, v in meta_params.items()},
        "metadata": dict(metadata),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_meta_action_init(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "cosmos3_meta_action_init":
        raise ValueError(f"{path} is not a cosmos3_meta_action_init file")
    return payload["meta_params"], dict(payload.get("metadata", {}))


@torch.no_grad()
def apply_meta_action_init_to_net(
    net: nn.Module,
    meta_params: dict[str, torch.Tensor],
    domain_id: int,
    include_modality_embed: bool = True,
) -> list[str]:
    """Copy theta_meta into domain row ``domain_id`` of ``net``'s action projectors.

    Returns the list of written parameter names. Shapes are validated against the network so a
    checkpoint from a different ``max_action_dim`` / hidden size fails loudly.
    """
    written: list[str] = []
    for n in DOMAIN_ROW_PARAM_NAMES:
        if n not in meta_params:
            raise KeyError(f"meta_action_init is missing {n!r}")
        p = get_action_head_param(net, n)
        expected = tuple(p.shape[1:])
        if tuple(meta_params[n].shape) != expected:
            raise ValueError(f"{n}: meta init shape {tuple(meta_params[n].shape)} != network row shape {expected}")
        if not 0 <= int(domain_id) < p.shape[0]:
            raise ValueError(f"domain_id={domain_id} out of range for {n} ({p.shape[0]} domains)")
        write_domain_row(p, int(domain_id), meta_params[n])
        written.append(n)
    if include_modality_embed and MODALITY_EMBED_PARAM_NAME in meta_params and hasattr(net, MODALITY_EMBED_PARAM_NAME):
        p = get_action_head_param(net, MODALITY_EMBED_PARAM_NAME)
        if tuple(meta_params[MODALITY_EMBED_PARAM_NAME].shape) != tuple(p.shape):
            raise ValueError(
                f"{MODALITY_EMBED_PARAM_NAME}: meta init shape {tuple(meta_params[MODALITY_EMBED_PARAM_NAME].shape)} "
                f"!= network shape {tuple(p.shape)}"
            )
        write_domain_row(p, None, meta_params[MODALITY_EMBED_PARAM_NAME])
        written.append(MODALITY_EMBED_PARAM_NAME)
    return written
