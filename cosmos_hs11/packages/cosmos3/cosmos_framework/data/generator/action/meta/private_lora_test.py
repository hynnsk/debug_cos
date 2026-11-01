# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU tests for cosmos_hs11 v17: shared theta + private per-embodiment LoRA (``PrivateLoraBank`` and
``ReptileMetaState(exclude_groups={GROUP_LORA})``)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from cosmos_framework.data.generator.action.meta.meta_action_adapter import GROUP_LORA
from cosmos_framework.data.generator.action.meta.private_lora import PrivateLoraBank
from cosmos_framework.data.generator.action.meta.reptile_meta import GROUP_MOE_GEN, ReptileMetaState
from cosmos_framework.utils.generator.lora import LoraInjectedLinear, init_lora_weights_post_materialization

H = 8


class _Net(nn.Module):
    """A moe_gen linear with a LoRA adapter + a plain trainable module + a frozen one."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Module()
        lora = LoraInjectedLinear(nn.Linear(H, H), rank=2, alpha=4)
        lora.lora_A.to_empty(device="cpu")
        lora.lora_B.to_empty(device="cpu")
        self.layer.q_proj_moe_gen = lora
        self.vae2llm = nn.Linear(4, H)
        self.frozen = nn.Linear(H, H)
        self.frozen.requires_grad_(False)
        init_lora_weights_post_materialization(self)


def _lora_names(net: nn.Module) -> list[str]:
    return [n for n, _ in net.named_parameters() if ".lora_" in n]


def _snapshot(net: nn.Module, names: list[str]) -> dict[str, torch.Tensor]:
    params = dict(net.named_parameters())
    return {n: params[n].detach().clone() for n in names}


def _inner_step(net: nn.Module, lr: float = 0.1) -> None:
    """One SGD step on every trainable parameter with a loss that reaches the LoRA branch."""
    x = torch.randn(4, H)
    y = net.layer.q_proj_moe_gen(x).sum() + net.vae2llm(torch.randn(4, 4)).sum()
    y.backward()
    with torch.no_grad():
        for p in net.parameters():
            if p.requires_grad and p.grad is not None:
                p.add_(p.grad, alpha=-lr)
                p.grad = None


def test_theta_excludes_the_private_adapters() -> None:
    net = _Net()
    lora = _lora_names(net)
    assert lora == ["layer.q_proj_moe_gen.lora_A.weight", "layer.q_proj_moe_gen.lora_B.weight"]
    plain = ReptileMetaState(net)
    assert set(lora) <= set(plain.params) and GROUP_LORA in set(plain.groups.values())
    state = ReptileMetaState(net, exclude_groups={GROUP_LORA})
    assert not set(lora) & set(state.params)
    assert state.excluded_groups == frozenset({GROUP_LORA})
    assert set(state.groups.values()) == {GROUP_MOE_GEN, "vae2llm"}
    assert "layer.q_proj_moe_gen.weight" in state.params  # the base weight stays in theta


def test_bank_slots_are_isolated_and_persist_across_episodes() -> None:
    torch.manual_seed(0)
    net = _Net()
    lora = _lora_names(net)
    bank = PrivateLoraBank(net, ["fractal", "bridge"])
    assert bank.slots == ["fractal", "bridge"] and bank.names == lora
    init = _snapshot(net, lora)
    assert float(init["layer.q_proj_moe_gen.lora_B.weight"].abs().sum()) == 0.0  # identity adapter at the start

    bank.activate("fractal")
    _inner_step(net)
    after_fractal = _snapshot(net, lora)
    assert not torch.equal(after_fractal["layer.q_proj_moe_gen.lora_B.weight"], init["layer.q_proj_moe_gen.lora_B.weight"])
    bank.store()  # Delta_fractal keeps the update

    bank.activate("bridge")  # the other embodiment still starts from the identity adapter
    for n in lora:
        torch.testing.assert_close(dict(net.named_parameters())[n], init[n])
    _inner_step(net)
    bank.store("bridge")

    bank.activate("fractal")  # fractal's adapter comes back untouched by bridge's episode
    for n in lora:
        torch.testing.assert_close(dict(net.named_parameters())[n], after_fractal[n])
    norms = bank.delta_norms()
    assert set(norms) == {"fractal", "bridge"} and norms["fractal"] > 0.0 and norms["bridge"] > 0.0


def test_meta_step_interpolates_theta_only_and_leaves_the_active_adapter() -> None:
    torch.manual_seed(1)
    net = _Net()
    lora = _lora_names(net)
    state = ReptileMetaState(net, exclude_groups={GROUP_LORA})
    bank = PrivateLoraBank(net, ["a"])
    theta0 = {n: state.theta[n].clone() for n in state.params}
    bank.activate("a")
    _inner_step(net)
    tilde = {n: p.detach().clone() for n, p in net.named_parameters() if n in state.params}
    lora_after_inner = _snapshot(net, lora)
    bank.store()
    state.meta_step(0.5)
    params = dict(net.named_parameters())
    for n in state.params:  # theta <- theta + 0.5 (theta_tilde - theta), written back into the model
        torch.testing.assert_close(params[n].detach(), theta0[n] + 0.5 * (tilde[n] - theta0[n]))
    for n in lora:  # the adapter is not part of the interpolation and not restored
        torch.testing.assert_close(params[n].detach(), lora_after_inner[n])


def test_bank_save_and_load_roundtrip(tmp_path) -> None:
    torch.manual_seed(2)
    net = _Net()
    lora = _lora_names(net)
    bank = PrivateLoraBank(net, ["a", "b"])
    bank.activate("b")
    _inner_step(net)
    bank.store()
    stored = {n: bank.bank["b"][n].clone() for n in lora}
    path = tmp_path / PrivateLoraBank.shard_filename(0, 1)
    assert path.name == "private_lora_bank_rank00_of01.pt"
    bank.save(path)

    torch.manual_seed(3)
    other = PrivateLoraBank(_Net(), ["a", "b"])  # a fresh (differently initialised) network
    other.load(path)
    assert other.active == "b"
    for n in lora:
        torch.testing.assert_close(other.bank["b"][n], stored[n])
        torch.testing.assert_close(other.init[n], bank.init[n])
    with pytest.raises(ValueError, match="slot mismatch"):
        PrivateLoraBank(_Net(), ["a"]).load(path)


def test_bank_errors() -> None:
    plain = nn.Sequential(nn.Linear(H, H))
    with pytest.raises(ValueError, match="no trainable LoRA"):
        PrivateLoraBank(plain, ["a"])
    net = _Net()
    with pytest.raises(ValueError, match="at least one slot"):
        PrivateLoraBank(net, [])
    with pytest.raises(ValueError, match="duplicate"):
        PrivateLoraBank(net, ["a", "a"])
    bank = PrivateLoraBank(net, ["a"])
    with pytest.raises(KeyError):
        bank.activate("zzz")
    with pytest.raises(RuntimeError, match="no active slot"):
        bank.store()
