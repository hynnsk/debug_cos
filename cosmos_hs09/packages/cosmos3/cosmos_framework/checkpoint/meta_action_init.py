# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Inject a meta-learned initialization into a freshly warm-started model.

The LIBERO post-training recipes skip the action I/O projectors (and, in the LoRA recipes, the
``lora_`` adapters) when loading the base checkpoint, so those parameters start from Cosmos3's fresh
init. With ``checkpoint.meta_action_init_path`` set, the checkpointer calls
:func:`apply_meta_action_init` right after the model load and overwrites

* the target domain row (``checkpoint.meta_action_init_domain_id``, LIBERO = 5) of ``action2llm`` /
  ``llm2action``,
* ``action_modality_embed`` (``meta_action_init_include_modality_embed``),
* every ``*.lora_A/lora_B`` adapter the file carries (``meta_action_init_include_lora``),
* ``time_embedder.*`` (``meta_action_init_include_time_embedder``),

of ``net`` (and ``net_ema``) with theta_meta from ``scripts/train_action_meta.py``. Everything else
about the recipe -- trainable parameters, learning rates, data -- stays identical, so a baseline run
and a meta-initialized run differ in exactly the ``[checkpoint].meta_action_init_*`` lines of the TOML.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from cosmos_framework.data.generator.action.meta.meta_action_adapter import (
    DOMAIN_ROW_PARAM_NAMES,
    GROUP_ACTION_HEADS,
    apply_meta_action_init_to_net,
    get_action_head_param,
    load_meta_action_init,
    net_param_lookup,
    param_group,
    read_domain_row,
)
from cosmos_framework.utils import log

if TYPE_CHECKING:
    from cosmos_framework.utils.config import CheckpointConfig


def _is_dtensor(t: torch.Tensor) -> bool:
    try:
        from torch.distributed.tensor import DTensor

        return isinstance(t, DTensor)
    except Exception:  # noqa: BLE001
        return False


@torch.no_grad()
def _group_norms(net: torch.nn.Module, meta_params: dict[str, torch.Tensor], domain_id: int) -> dict[str, float]:
    """L2 norm per theta_meta group as currently stored in ``net`` (head rows gathered; shared groups
    summed over local shards + one all-reduce per group, so this stays cheap for hundreds of LoRA tensors)."""
    out: dict[str, float] = {}
    heads = sum(read_domain_row(get_action_head_param(net, n), domain_id).pow(2).sum().item() for n in DOMAIN_ROW_PARAM_NAMES)
    out[GROUP_ACTION_HEADS] = heads**0.5
    lookup = net_param_lookup(net)
    sq: dict[str, torch.Tensor] = {}
    for n in meta_params:
        g = param_group(n)
        if g == GROUP_ACTION_HEADS or n not in lookup:
            continue
        p = lookup[n].data
        local = p.to_local() if _is_dtensor(p) else p
        sq[g] = sq.get(g, torch.zeros((), device=local.device, dtype=torch.float32)) + local.float().pow(2).sum()
    for g, v in sq.items():
        # Local-shard sums of sharded (DTensor) parameters need one all-reduce per group. Assumes the
        # shards are disjoint (data_parallel_replicate_degree=1); with HSDP the norm would be overcounted,
        # which only affects this log line.
        first = next(n for n in meta_params if param_group(n) == g and n in lookup)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1 and _is_dtensor(lookup[first].data):
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
        out[g] = float(v.sqrt())
    return out


def apply_meta_action_init(model: torch.nn.Module, config_checkpoint: "CheckpointConfig") -> None:
    """Apply ``config_checkpoint.meta_action_init_*`` to ``model.net`` / ``model.net_ema``."""
    path = getattr(config_checkpoint, "meta_action_init_path", "") or ""
    if not path:
        return
    domain_id = int(getattr(config_checkpoint, "meta_action_init_domain_id", -1))
    if domain_id < 0:
        raise ValueError(
            "checkpoint.meta_action_init_domain_id must be set (>= 0) when checkpoint.meta_action_init_path is given "
            "(LIBERO uses domain id 5, see data/generator/action/utils/domain_utils.py)."
        )
    include = dict(
        include_modality_embed=bool(getattr(config_checkpoint, "meta_action_init_include_modality_embed", True)),
        include_lora=bool(getattr(config_checkpoint, "meta_action_init_include_lora", True)),
        include_time_embedder=bool(getattr(config_checkpoint, "meta_action_init_include_time_embedder", True)),
    )
    meta_params, metadata = load_meta_action_init(path)
    groups: dict[str, int] = {}
    for n in meta_params:
        groups[param_group(n)] = groups.get(param_group(n), 0) + 1
    log.critical(
        f"[meta-action-init] loading theta_meta from {path} into domain row {domain_id}; file groups (#tensors)={groups}; "
        f"{include}; metadata={metadata}"
    )

    nets = [("net", model.net)]
    if getattr(getattr(model, "config", None), "ema", None) is not None and model.config.ema.enabled:
        nets.append(("net_ema", model.net_ema))
    for tag, net in nets:
        before = _group_norms(net, meta_params, domain_id)
        written = apply_meta_action_init_to_net(net, meta_params, domain_id, **include)
        after = _group_norms(net, meta_params, domain_id)
        by_group: dict[str, int] = {}
        for n in written:
            by_group[param_group(n)] = by_group.get(param_group(n), 0) + 1
        log.info(
            f"[meta-action-init] {tag}: wrote {len(written)} tensors {by_group}; "
            f"group norms before={ {k: round(v, 4) for k, v in before.items()} } after={ {k: round(v, 4) for k, v in after.items()} }"
        )
