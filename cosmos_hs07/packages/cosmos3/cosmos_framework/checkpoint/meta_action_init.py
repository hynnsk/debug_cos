# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Inject a meta-learned action-head initialization into a freshly warm-started model.

The LIBERO post-training recipes skip the action I/O projectors when loading the base checkpoint
(``checkpoint.keys_to_skip_loading = ["action2llm", "llm2action", "action_modality_embed", ...]``),
so they start from Cosmos3's fresh init. With ``checkpoint.meta_action_init_path`` set, the
checkpointer calls :func:`apply_meta_action_init` right after the model load and overwrites the
target domain row (``checkpoint.meta_action_init_domain_id``, LIBERO = 5) of ``net`` (and
``net_ema``) with theta_meta from ``scripts/train_action_meta.py``. Everything else about the
recipe -- trainable parameters, learning rates, data -- stays identical, so a baseline run and a
meta-initialized run differ in exactly one line of the TOML.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from cosmos_framework.data.generator.action.meta.meta_action_adapter import (
    DOMAIN_ROW_PARAM_NAMES,
    apply_meta_action_init_to_net,
    get_action_head_param,
    load_meta_action_init,
    read_domain_row,
)
from cosmos_framework.utils import log

if TYPE_CHECKING:
    from cosmos_framework.utils.config import CheckpointConfig


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
    include_embed = bool(getattr(config_checkpoint, "meta_action_init_include_modality_embed", True))
    meta_params, metadata = load_meta_action_init(path)
    log.critical(
        f"[meta-action-init] loading theta_meta from {path} into domain row {domain_id} "
        f"(include_modality_embed={include_embed}); metadata={metadata}"
    )

    nets = [("net", model.net)]
    if getattr(getattr(model, "config", None), "ema", None) is not None and model.config.ema.enabled:
        nets.append(("net_ema", model.net_ema))
    for tag, net in nets:
        before = {n: read_domain_row(get_action_head_param(net, n), domain_id).norm().item() for n in DOMAIN_ROW_PARAM_NAMES}
        written = apply_meta_action_init_to_net(net, meta_params, domain_id, include_modality_embed=include_embed)
        after = {n: read_domain_row(get_action_head_param(net, n), domain_id).norm().item() for n in DOMAIN_ROW_PARAM_NAMES}
        log.info(
            f"[meta-action-init] {tag}: wrote {written}; row-{domain_id} norms before={ {k: round(v, 4) for k, v in before.items()} } "
            f"after={ {k: round(v, 4) for k, v in after.items()} }"
        )
