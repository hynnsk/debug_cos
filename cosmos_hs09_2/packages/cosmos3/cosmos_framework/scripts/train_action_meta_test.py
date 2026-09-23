# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for the outer-loop vision-loss variant of ``train_action_meta`` (cosmos_hs09).

Guards two promises: (1) with the default ``outer_vision_weight = 0`` the outer objective is *exactly* the original
``_select_loss`` tensor (nothing added to the graph), so the base recipe is unchanged; (2) the variant adds the RAW
vision term, not the x10-scaled one that ``loss_mode="total"`` would bring.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tomllib
import torch

from cosmos_framework.scripts.train_action_meta import (
    SAMPLER_OVERRIDE_KEYS,
    MetaTrainConfig,
    _outer_terms,
    _select_loss,
)

REPO = Path(__file__).resolve().parents[2]
BASE_TOML = REPO / "examples/toml/sft_config/action_fewshot_meta_lora_edge.toml"
VARIANT_TOML = REPO / "examples/toml/sft_config/action_fewshot_meta_lora_edge_vision_loss.toml"


def _fake_forward(vision_requires_grad: bool = True) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    a = torch.tensor(0.7, requires_grad=True)
    v = torch.tensor(0.2, requires_grad=vision_requires_grad)
    out = {"flow_matching_loss_action": a * 1.0, "flow_matching_loss_vision": v * 1.0}
    total = 10.0 * out["flow_matching_loss_vision"] + 10.0 * out["flow_matching_loss_action"]  # the model's total
    return out, total


def test_default_weight_is_zero_and_outer_is_exactly_select_loss() -> None:
    cfg = MetaTrainConfig()
    assert cfg.outer_vision_weight == 0.0
    out, total = _fake_forward()
    loss, vision = _outer_terms(out, total, cfg)
    assert vision is None
    assert loss is _select_loss(out, total, cfg.loss_mode)  # same tensor object -> identical graph, identical backward
    assert loss is out["flow_matching_loss_action"]


def test_variant_adds_raw_vision_term_not_the_scaled_total() -> None:
    cfg = MetaTrainConfig(outer_vision_weight=1.0)
    out, total = _fake_forward()
    loss, vision = _outer_terms(out, total, cfg)
    assert vision is out["flow_matching_loss_vision"]
    outer = loss + cfg.outer_vision_weight * vision
    assert outer.item() == pytest.approx(0.9)  # raw action 0.7 + 1.0 x raw vision 0.2
    assert total.item() == pytest.approx(9.0)  # what loss_mode="total" would have used (10x)
    outer.backward()
    assert out["flow_matching_loss_action"].grad_fn is not None and vision.grad_fn is not None


def test_variant_requires_a_vision_gradient() -> None:
    cfg = MetaTrainConfig(outer_vision_weight=1.0)
    out, total = _fake_forward(vision_requires_grad=False)
    with pytest.raises(RuntimeError, match="flow_matching_loss_vision"):
        _outer_terms(out, total, cfg)


def test_from_dict_validation() -> None:
    assert MetaTrainConfig.from_dict({"outer_vision_weight": 1.0}).outer_vision_weight == 1.0
    assert MetaTrainConfig.from_dict({}).outer_vision_weight == 0.0
    with pytest.raises(ValueError, match="outer_vision_weight must be >= 0"):
        MetaTrainConfig.from_dict({"outer_vision_weight": -0.5})
    with pytest.raises(ValueError, match="requires loss_mode='action'"):
        MetaTrainConfig.from_dict({"outer_vision_weight": 1.0, "loss_mode": "total"})


def test_variant_toml_is_base_toml_plus_one_key() -> None:
    base = tomllib.loads(BASE_TOML.read_text())
    var = tomllib.loads(VARIANT_TOML.read_text())
    bm, vm = base["custom"]["meta"], dict(var["custom"]["meta"])
    assert "outer_vision_weight" not in bm, "the base recipe must not carry the variant key"
    assert vm.pop("outer_vision_weight") == 1.0
    assert vm == bm, "the variant must differ from the base [custom.meta] only in outer_vision_weight"
    for section in ("model", "trainer", "checkpoint"):
        assert var[section] == base[section], section
    assert var["job"]["name"] != base["job"]["name"]
    assert {k: v for k, v in var["job"].items() if k != "name"} == {k: v for k, v in base["job"].items() if k != "name"}
    # both parse into MetaTrainConfig (sampler keys are routed elsewhere by split_custom_meta)
    strip = lambda d: {k: v for k, v in d.items() if k not in SAMPLER_OVERRIDE_KEYS}  # noqa: E731
    assert MetaTrainConfig.from_dict(strip(bm)).outer_vision_weight == 0.0
    assert MetaTrainConfig.from_dict(strip(var["custom"]["meta"])).outer_vision_weight == 1.0
