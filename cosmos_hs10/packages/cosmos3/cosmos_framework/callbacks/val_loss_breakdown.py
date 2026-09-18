# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Per-modality validation loss (total / action / vision) averaged over a validation pass.

``wandb_val`` (``callbacks/wandb_log_eval.py``) already logs the averaged *total* loss as ``val/loss``.
For the action-policy recipes the total is dominated by the x10-scaled vision term, so this callback
additionally averages the ``flow_matching_loss_*`` entries of ``output_batch`` and logs

    val/loss_total, val/flow_matching_loss_action, val/flow_matching_loss_vision, val/num_batches

to wandb (rank 0) and the console. Sums are reduced across ranks; non-finite values are skipped and
counted in ``val/num_nonfinite``.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback

_DEFAULT_KEYS = ("flow_matching_loss_action", "flow_matching_loss_vision")


class ValLossBreakdownCallback(Callback):
    def __init__(self, keys: tuple[str, ...] | list[str] = _DEFAULT_KEYS, prefix: str = "val") -> None:
        super().__init__()
        self.keys = tuple(keys)
        self.prefix = prefix
        self._reset()

    def _reset(self) -> None:
        self._sum: dict[str, float] = {"loss_total": 0.0, **{k: 0.0 for k in self.keys}}
        self._cnt: dict[str, int] = {"loss_total": 0, **{k: 0 for k in self.keys}}
        self._nonfinite = 0

    def on_validation_start(self, model: ImaginaireModel, dataloader_val, iteration: int = 0) -> None:
        self._reset()

    @torch.no_grad()
    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        def _add(key: str, value) -> None:
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                return
            v = float(value.detach().float())
            if v != v or v in (float("inf"), float("-inf")):
                self._nonfinite += 1
                return
            self._sum[key] += v
            self._cnt[key] += 1

        _add("loss_total", loss)
        for k in self.keys:
            _add(k, output_batch.get(k) if isinstance(output_batch, dict) else None)

    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        keys = list(self._sum)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sums = torch.tensor([self._sum[k] for k in keys], dtype=torch.float64, device=device)
        cnts = torch.tensor([float(self._cnt[k]) for k in keys], dtype=torch.float64, device=device)
        nonfinite = torch.tensor([float(self._nonfinite)], dtype=torch.float64, device=device)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(cnts, op=dist.ReduceOp.SUM)
            dist.all_reduce(nonfinite, op=dist.ReduceOp.SUM)
        info: dict[str, float] = {}
        for i, k in enumerate(keys):
            if cnts[i].item() > 0:
                info[f"{self.prefix}/{k}"] = float(sums[i] / cnts[i])
        info[f"{self.prefix}/num_batches"] = float(cnts[0].item())
        info[f"{self.prefix}/num_nonfinite"] = float(nonfinite.item())
        self.last_info = dict(info)  # for tests / inspection
        if distributed.is_rank0():
            log.info(f"[{self.prefix}] iter {iteration}: " + ", ".join(f"{k.split('/')[-1]}={v:.5f}" for k, v in info.items()))
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(info, step=iteration)
            except Exception as e:  # noqa: BLE001
                log.warning(f"ValLossBreakdownCallback: wandb.log failed ({e})")
        self._reset()
