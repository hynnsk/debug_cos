# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for MetaActionAdapter on a tiny network with real DomainAwareLinear projectors."""

import torch
from torch import nn

from cosmos_framework.data.generator.action.meta.meta_action_adapter import (
    DOMAIN_ROW_PARAM_NAMES,
    MetaActionAdapter,
    MetaActionInitSpec,
    apply_meta_action_init_to_net,
    fresh_meta_action_init,
    load_meta_action_init,
    save_meta_action_init,
)
from cosmos_framework.model.generator.mot.domain_aware_linear import DomainAwareLinear

HIDDEN, ACTION_DIM, NUM_DOMAINS = 32, 8, 6


class _TinyActionNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action2llm = DomainAwareLinear(ACTION_DIM, HIDDEN, NUM_DOMAINS)
        self.llm2action = DomainAwareLinear(HIDDEN, ACTION_DIM, NUM_DOMAINS)
        self.action_modality_embed = nn.Parameter(torch.zeros(HIDDEN))
        self.backbone = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, action: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        h = self.action2llm(action, domain_id) + self.action_modality_embed.view(1, -1)
        h = torch.tanh(self.backbone(h))
        return self.llm2action(h, domain_id)


def _loss(net: _TinyActionNet, row: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(16, ACTION_DIM, generator=g)
    y = torch.randn(16, ACTION_DIM, generator=g)
    dom = torch.full((16,), row, dtype=torch.long)
    return ((net(x, dom) - y) ** 2).mean()


def test_fresh_init_shapes_and_freeze() -> None:
    net = _TinyActionNet()
    adapter = MetaActionAdapter(net, scratch_domain_id=5, init=MetaActionInitSpec(source="fresh", seed=1))
    assert adapter.hidden_size == HIDDEN and adapter.action_dim == ACTION_DIM
    assert adapter.meta["action2llm.fc.weight"].shape == (HIDDEN * ACTION_DIM,)
    assert adapter.meta["llm2action.bias.weight"].shape == (ACTION_DIM,)
    assert adapter.meta["action_modality_embed"].shape == (HIDDEN,)
    assert not net.backbone.weight.requires_grad
    assert net.action2llm.fc.weight.requires_grad
    fresh = fresh_meta_action_init(HIDDEN, ACTION_DIM, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(adapter.meta["action2llm.fc.weight"].detach(), fresh["action2llm.fc.weight"])
    assert torch.count_nonzero(adapter.meta["action2llm.bias.weight"]) == 0


def test_inner_loop_reduces_support_loss_and_only_touches_scratch_row() -> None:
    torch.manual_seed(0)
    net = _TinyActionNet()
    other_rows_before = {n: getattr_chain(net, n).detach().clone() for n in DOMAIN_ROW_PARAM_NAMES}
    adapter = MetaActionAdapter(net, scratch_domain_id=2, init=MetaActionInitSpec(source="fresh", seed=3))
    adapter.start_episode()
    torch.testing.assert_close(net.action2llm.fc.weight[2].float(), adapter.meta["action2llm.fc.weight"].detach())
    loss0 = _loss(net, 2).item()
    for _ in range(20):
        loss = _loss(net, 2)
        loss.backward()
        grads = adapter.collect_grads()
        assert all(torch.isfinite(g).all() for g in grads.values())
        assert grads["action2llm.fc.weight"].abs().sum() > 0
        adapter.inner_update(grads, lr=0.05)
    loss1 = _loss(net, 2).item()
    assert loss1 < loss0, (loss0, loss1)
    assert adapter.fast_delta_norm() > 0
    # Rows other than the scratch row are untouched.
    for n in DOMAIN_ROW_PARAM_NAMES:
        p = getattr_chain(net, n).detach()
        mask = torch.ones(NUM_DOMAINS, dtype=torch.bool)
        mask[2] = False
        torch.testing.assert_close(p[mask], other_rows_before[n][mask])
    # theta_meta itself is unchanged by inner updates; only after assign_meta_grads + optimizer step.
    meta_before = adapter.export_meta()
    q = _loss(net, 2, seed=9)
    q.backward()
    adapter.assign_meta_grads(adapter.collect_grads())
    opt = torch.optim.SGD(adapter.meta_parameters, lr=0.1)
    opt.step()
    assert any(not torch.equal(meta_before[n], adapter.meta[n].detach().cpu()) for n in adapter.names)


def test_checkpoint_row_init_and_save_load_roundtrip(tmp_path) -> None:
    net = _TinyActionNet()
    with torch.no_grad():
        net.action2llm.fc.weight[1].fill_(0.25)
        net.action2llm.fc.weight[3].fill_(0.75)
    adapter = MetaActionAdapter(
        net, scratch_domain_id=0, init=MetaActionInitSpec(source="checkpoint_mean", domain_ids=[1, 3])
    )
    torch.testing.assert_close(adapter.meta["action2llm.fc.weight"].detach(), torch.full((HIDDEN * ACTION_DIM,), 0.5))
    path = save_meta_action_init(tmp_path / "meta_action_init.pt", adapter.export_meta(), adapter.metadata())
    params, meta = load_meta_action_init(path)
    assert meta["hidden_size"] == HIDDEN and meta["action_dim"] == ACTION_DIM
    target = _TinyActionNet()
    written = apply_meta_action_init_to_net(target, params, domain_id=4)
    assert set(written) == set(adapter.names)
    torch.testing.assert_close(target.action2llm.fc.weight[4].float(), params["action2llm.fc.weight"])
    torch.testing.assert_close(target.action_modality_embed.float(), params["action_modality_embed"])
    # Other rows keep their own values.
    assert not torch.equal(target.action2llm.fc.weight[0], target.action2llm.fc.weight[4])


def getattr_chain(obj, dotted: str):
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj
