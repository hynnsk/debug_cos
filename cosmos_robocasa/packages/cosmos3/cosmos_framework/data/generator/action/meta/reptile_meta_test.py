# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for ReptileMetaState on a tiny network (plain tensors; DTensor paths are exercised on GPU smokes)."""

import torch
from torch import nn

from cosmos_framework.data.generator.action.meta.meta_action_adapter import (
    DOMAIN_ROW_PARAM_NAMES,
    GROUP_ACTION_HEADS,
    GROUP_MODALITY_EMBED,
    GROUP_TIME_EMBEDDER,
    MODALITY_EMBED_PARAM_NAME,
)
from cosmos_framework.data.generator.action.meta.reptile_meta import (
    GROUP_MOE_GEN,
    GROUP_VAE2LLM,
    ReptileMetaState,
    capture_base_lrs,
    reptile_param_group,
    reset_optimizer_state,
    set_lr_scale,
)
from cosmos_framework.model.generator.mot.domain_aware_linear import DomainAwareLinear

HIDDEN, ACTION_DIM, NUM_DOMAINS = 32, 8, 6


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action2llm = DomainAwareLinear(ACTION_DIM, HIDDEN, NUM_DOMAINS)
        self.llm2action = DomainAwareLinear(HIDDEN, ACTION_DIM, NUM_DOMAINS)
        self.action_modality_embed = nn.Parameter(torch.zeros(HIDDEN))
        self.time_embedder = nn.Sequential(nn.Linear(4, HIDDEN))
        self.layer = nn.Module()
        self.layer.q_proj_moe_gen = nn.Linear(HIDDEN, HIDDEN)
        self.vae2llm = nn.Linear(16, HIDDEN)
        self.frozen = nn.Linear(HIDDEN, HIDDEN)
        self.frozen.requires_grad_(False)  # not part of theta (like the understanding tower)


def _trainable(net: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in net.named_parameters() if p.requires_grad}


def test_groups_and_theta_copy() -> None:
    net = _TinyNet()
    state = ReptileMetaState(net)
    assert set(state.groups.values()) == {
        GROUP_ACTION_HEADS,
        GROUP_MODALITY_EMBED,
        GROUP_TIME_EMBEDDER,
        GROUP_MOE_GEN,
        GROUP_VAE2LLM,
    }
    assert "frozen.weight" not in state.params
    assert reptile_param_group("language_model.model.layers.3.mlp_moe_gen.up_proj.weight") == GROUP_MOE_GEN
    assert reptile_param_group("language_model.model.layers.3.self_attn.q_proj_moe_gen.lora_A.weight") == "lora"
    assert reptile_param_group("llm2vae.weight") == "llm2vae"
    assert (
        reptile_param_group("language_model.model.layers.0.self_attn.k_norm_und_for_gen.weight") == "k_norm_und_for_gen"
    )
    nb = state.numel_by_group()
    assert nb[GROUP_MOE_GEN] == HIDDEN * HIDDEN + HIDDEN and nb[GROUP_ACTION_HEADS] == sum(
        p.numel() for n, p in net.named_parameters() if n in DOMAIN_ROW_PARAM_NAMES
    )
    for n, p in net.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(state.theta[n], p.detach())


def test_sgd_meta_step_interpolates_and_restores_model() -> None:
    net = _TinyNet()
    state = ReptileMetaState(net, meta_optimizer="sgd")
    theta0 = _trainable(net)
    frozen0 = net.frozen.weight.detach().clone()
    with torch.no_grad():  # "inner loop": move every trainable tensor by +1
        for p in state.params.values():
            p.add_(1.0)
    disp = state.displacement_norms_by_group()
    assert all(v > 0 for v in disp.values())
    n_moe = state.numel_by_group()[GROUP_MOE_GEN]
    assert abs(disp[GROUP_MOE_GEN] - n_moe**0.5) < 1e-4  # each coordinate moved by exactly 1
    step_norm = state.meta_step(0.5)
    assert abs(step_norm - (0.5 * state.numel**0.5)) < 1e-3
    for n, p in net.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(state.theta[n], theta0[n] + 0.5)  # halfway toward theta_tilde
            torch.testing.assert_close(p.detach(), state.theta[n])  # model restored to theta
    torch.testing.assert_close(net.frozen.weight.detach(), frozen0)
    assert all(v == 0 for v in state.displacement_norms_by_group().values())


def test_batched_displacement_and_adam_outer() -> None:
    net = _TinyNet()
    state = ReptileMetaState(net, meta_optimizer="adam")
    theta0 = _trainable(net)
    for shift in (1.0, 3.0):  # two episodes from the same theta
        with torch.no_grad():
            for p in state.params.values():
                p.add_(shift)
        state.accumulate_displacement()
        state.restore_to_model()
        for n, p in net.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(p.detach(), theta0[n])
    assert state.accum_count == 2
    state.meta_step(0.1)
    # Adam's first step is ~eps * sign(mean displacement) = +0.1 per coordinate (bias-corrected)
    for n in state.params:
        torch.testing.assert_close(state.theta[n], theta0[n] + 0.1, atol=1e-5, rtol=0)
    assert state.accum is None and state.accum_count == 0


def test_export_meta_action_init_groups() -> None:
    net = _TinyNet()
    state = ReptileMetaState(net)
    with torch.no_grad():
        net.action2llm.fc.weight[3].fill_(7.0)
    heads_only = state.export_meta_action_init(3, include_lora=False, include_time_embedder=False)
    assert set(heads_only) == set(DOMAIN_ROW_PARAM_NAMES) | {MODALITY_EMBED_PARAM_NAME}
    assert torch.all(heads_only["action2llm.fc.weight"] == 7.0)  # the scratch row
    with_te = state.export_meta_action_init(3, include_lora=False, include_time_embedder=True)
    assert "time_embedder.0.weight" in with_te and "time_embedder.0.bias" in with_te
    md = state.metadata("full", 3)
    assert md["algorithm"] == "reptile" and md["numel_by_group"][GROUP_MOE_GEN] > 0 and not md["include_lora"]


def test_reset_optimizer_state_and_lr_scale() -> None:
    net = _TinyNet()
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam([{"params": params[:2], "lr": 1e-3}, {"params": params[2:], "lr": 5e-3}])
    loss = sum((p**2).sum() for p in params)
    loss.backward()
    opt.step()
    opt.param_groups[0]["step"] = torch.tensor([1])  # FusedAdam keeps its step counter in the group
    assert len(opt.state) > 0
    reset_optimizer_state(opt)
    assert len(opt.state) == 0 and all("step" not in g for g in opt.param_groups)
    base = capture_base_lrs(opt)
    assert base == [[1e-3, 5e-3]]
    set_lr_scale(opt, base, 0.5)
    assert [g["lr"] for g in opt.param_groups] == [5e-4, 2.5e-3]
    set_lr_scale(opt, base, 1.0)
    assert [g["lr"] for g in opt.param_groups] == [1e-3, 5e-3]
