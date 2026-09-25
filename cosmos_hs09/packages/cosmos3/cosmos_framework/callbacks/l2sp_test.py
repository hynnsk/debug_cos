# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for the decoupled L2-SP callback (cosmos_hs09) and its TOML plumbing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import tomllib
import torch
from torch import nn

from cosmos_framework.callbacks.l2sp import L2SP, current_group_lrs

REPO = Path(__file__).resolve().parents[2]
BASE_TOML = REPO / "examples/toml/sft_config/action_policy_libero_10_edge_metainit.toml"
L2SP_TOML = REPO / "examples/toml/sft_config/action_policy_libero_10_edge_metainit_l2sp.toml"


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.action2llm = nn.Linear(4, 3)
        self.time_embedder = nn.Linear(2, 2)
        self.mlp_moe_gen = nn.Linear(3, 3)
        self.lora_A = nn.Linear(2, 2)  # frozen in the recipe (lr multiplier 0)


def _model() -> SimpleNamespace:
    return SimpleNamespace(net=_Net())


def _opt(net: _Net) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        [
            {"params": list(net.action2llm.parameters()), "lr": 1e-3},
            {"params": list(net.time_embedder.parameters()) + list(net.mlp_moe_gen.parameters()), "lr": 2e-3},
            {"params": list(net.lora_A.parameters()), "lr": 0.0},
        ]
    )


ALPHAS = {"action2llm": 100.0, "time_embedder": 100.0, "moe_gen": 0.0, "lora_": 100.0}


def _snapshot(net: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in net.named_parameters()}


def test_pull_is_decoupled_lerp_toward_anchor(tmp_path: Path) -> None:
    model = _model()
    cb = L2SP(alphas=ALPHAS, log_every=1, anchor_dir=str(tmp_path))
    cb.on_train_start(model, iteration=0)
    assert set(cb._anchors) == {
        "action2llm.weight",
        "action2llm.bias",
        "time_embedder.weight",
        "time_embedder.bias",
        "lora_A.weight",
        "lora_A.bias",
    }
    p0 = _snapshot(model.net)
    opt = _opt(model.net)
    with torch.no_grad():
        for p in model.net.parameters():
            p.add_(1.0)  # simulate one optimizer step that moved every weight by +1
    cb.on_before_zero_grad(model, opt, None, iteration=1)
    p1 = _snapshot(model.net)
    # action2llm: lr 1e-3 x alpha 100 = rate 0.1 -> deviation 1.0 shrinks to 0.9
    torch.testing.assert_close(p1["action2llm.weight"], p0["action2llm.weight"] + 0.9)
    # time_embedder: lr 2e-3 x 100 = 0.2 -> 0.8
    torch.testing.assert_close(p1["time_embedder.bias"], p0["time_embedder.bias"] + 0.8)
    # alpha 0 -> not anchored, untouched
    torch.testing.assert_close(p1["mlp_moe_gen.weight"], p0["mlp_moe_gen.weight"] + 1.0)
    # anchored but lr 0 (frozen group) -> untouched
    torch.testing.assert_close(p1["lora_A.weight"], p0["lora_A.weight"] + 1.0)
    # diagnostics: rel_dev = ||p - p0|| / ||p0|| over the pattern's tensors, rate = lr * alpha
    info = cb.last_info
    dev = sum(((p1[n] - p0[n]) ** 2).sum() for n in ("action2llm.weight", "action2llm.bias"))
    ref = sum((p0[n] ** 2).sum() for n in ("action2llm.weight", "action2llm.bias"))
    assert info["l2sp/rel_dev/action2llm"] == pytest.approx((dev.sqrt() / ref.sqrt()).item(), rel=1e-5)
    assert info["l2sp/rate/action2llm"] == pytest.approx(0.1) and info["l2sp/rate/time_embedder"] == pytest.approx(0.2)


def test_rate_is_clamped_to_one() -> None:
    model = _model()
    cb = L2SP(alphas={"action2llm": 1e9}, log_every=0, anchor_dir=None)
    cb.config = SimpleNamespace(job=SimpleNamespace(path_local=str(Path("/tmp") / "l2sp_clamp_test")))
    cb.on_train_start(model, iteration=0)
    p0 = _snapshot(model.net)
    with torch.no_grad():
        model.net.action2llm.weight.add_(3.0)
    cb.on_before_zero_grad(model, _opt(model.net), None, iteration=1)
    torch.testing.assert_close(model.net.action2llm.weight.detach(), p0["action2llm.weight"])  # rate 1 -> back to p0


def test_all_zero_alphas_is_a_noop(tmp_path: Path) -> None:
    model = _model()
    cb = L2SP(alphas={k: 0.0 for k in ALPHAS}, log_every=1, anchor_dir=str(tmp_path))
    assert not cb.enabled
    cb.on_train_start(model, iteration=0)
    assert cb._anchors == {} and not list(tmp_path.iterdir())
    p0 = _snapshot(model.net)
    with torch.no_grad():
        for p in model.net.parameters():
            p.add_(1.0)
    cb.on_before_zero_grad(model, _opt(model.net), None, iteration=1)
    for n, p in model.net.named_parameters():
        torch.testing.assert_close(p.detach(), p0[n] + 1.0)


def test_resume_reloads_original_anchors_not_current_weights(tmp_path: Path) -> None:
    model = _model()
    cb = L2SP(alphas=ALPHAS, log_every=0, anchor_dir=str(tmp_path))
    cb.on_train_start(model, iteration=0)
    p0 = _snapshot(model.net)
    with torch.no_grad():
        for p in model.net.parameters():
            p.add_(1.0)  # "training moved on" ... then the job restarts from a checkpoint at iteration 500
    cb2 = L2SP(alphas=ALPHAS, log_every=0, anchor_dir=str(tmp_path))
    cb2.on_train_start(model, iteration=500)
    torch.testing.assert_close(cb2._anchors["action2llm.weight"], p0["action2llm.weight"])  # original, not p0 + 1
    cb2.on_before_zero_grad(model, _opt(model.net), None, iteration=501)
    torch.testing.assert_close(model.net.action2llm.weight.detach(), p0["action2llm.weight"] + 0.9)
    # a resume without the anchor file must fail loudly rather than re-anchor silently
    cb3 = L2SP(alphas=ALPHAS, log_every=0, anchor_dir=str(tmp_path / "elsewhere"))
    with pytest.raises(FileNotFoundError):
        cb3.on_train_start(model, iteration=500)


def test_negative_alpha_and_unmatched_alphas_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        L2SP(alphas={"action2llm": -1.0})
    cb = L2SP(alphas={"does_not_exist": 1.0}, anchor_dir=str(tmp_path))
    with pytest.raises(ValueError, match="matched no trainable parameter"):
        cb.on_train_start(_model(), iteration=0)


def test_current_group_lrs_handles_containers() -> None:
    net = _Net()
    opt = _opt(net)
    container = SimpleNamespace(optimizers=[opt])
    lrs = current_group_lrs(container)
    assert (
        lrs[id(net.action2llm.weight)] == 1e-3
        and lrs[id(net.mlp_moe_gen.bias)] == 2e-3
        and lrs[id(net.lora_A.weight)] == 0.0
    )


def test_toml_variant_is_base_plus_l2sp_section() -> None:
    from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig
    from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides

    base = tomllib.loads(BASE_TOML.read_text())
    var = tomllib.loads(L2SP_TOML.read_text())
    assert "l2sp" not in base.get("trainer", {}).get("callbacks", {})
    l2 = var["trainer"]["callbacks"]["l2sp"]
    assert l2["alphas"]["action2llm"] == 100.0 and l2["alphas"]["moe_gen"] == 0.0 and l2["alphas"]["lora_"] == 0.0
    var_wo = {k: v for k, v in var.items()}
    var_wo["trainer"] = {k: v for k, v in var["trainer"].items() if k != "callbacks"}
    base_wo = dict(base)
    assert {k: v for k, v in var_wo.items() if k != "job"} == {k: v for k, v in base_wo.items() if k != "job"}
    assert var["job"]["name"] != base["job"]["name"]
    for raw in (base, var):
        SFTExperimentConfig.model_validate(raw)  # schema accepts both
    ov = build_hydra_overrides(var)
    assert "trainer.callbacks.l2sp.alphas.action2llm=100.0" in ov and "trainer.callbacks.l2sp.log_every=100" in ov
    assert not any(o.startswith("trainer.callbacks.l2sp") for o in build_hydra_overrides(base))
    with pytest.raises(Exception):
        SFTExperimentConfig.model_validate(
            {**var, "trainer": {**var["trainer"], "callbacks": {"l2sp": {"alphas": {"typo_key": 1.0}}}}}
        )
