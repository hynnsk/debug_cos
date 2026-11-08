# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import math

import torch

from cosmos_framework.callbacks.val_loss_breakdown import ValLossBreakdownCallback


def test_val_loss_breakdown_averages_and_skips_nonfinite() -> None:
    cb = ValLossBreakdownCallback()
    cb.on_validation_start(None, None, iteration=7)
    steps = [(1.0, 0.5, 2.0), (3.0, 1.5, 6.0), (math.nan, 2.5, math.inf)]
    for total, action, vision in steps:
        out = {"flow_matching_loss_action": torch.tensor(action), "flow_matching_loss_vision": torch.tensor(vision)}
        cb.on_validation_step_end(None, {}, out, torch.tensor(total), iteration=7)
    cb.on_validation_end(None, iteration=7)
    info = cb.last_info
    assert info["val/loss_total"] == 2.0  # nan skipped
    assert abs(info["val/flow_matching_loss_action"] - 1.5) < 1e-6
    assert info["val/flow_matching_loss_vision"] == 4.0  # inf skipped
    assert info["val/num_batches"] == 2 and info["val/num_nonfinite"] == 2
    # state reset for the next pass
    cb.on_validation_start(None, None, iteration=8)
    cb.on_validation_end(None, iteration=8)
    assert cb.last_info["val/num_batches"] == 0 and "val/loss_total" not in cb.last_info
