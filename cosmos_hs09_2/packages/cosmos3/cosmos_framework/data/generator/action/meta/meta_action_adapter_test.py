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


# --------------------------------------------------------------------------------------------
# cosmos_hs09: shared groups (LoRA adapters + time_embedder) in theta_meta
# --------------------------------------------------------------------------------------------
class _TinyLoraNet(_TinyActionNet):
    """Heads + a LoRA-wrapped 'generation tower' linear + a time embedder, all on the forward path."""

    def __init__(self) -> None:
        super().__init__()
        from cosmos_framework.utils.generator.lora import LoraInjectedLinear, init_lora_weights_post_materialization

        self.q_proj_moe_gen = LoraInjectedLinear(nn.Linear(HIDDEN, HIDDEN), rank=4, alpha=8)
        self.q_proj_moe_gen.lora_A.to_empty(device="cpu")
        self.q_proj_moe_gen.lora_B.to_empty(device="cpu")
        init_lora_weights_post_materialization(self)
        self.time_embedder = nn.Sequential(nn.Linear(4, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN))

    def forward(self, action: torch.Tensor, domain_id: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
        t = torch.linspace(0, 1, 4).expand(action.shape[0], 4) if t is None else t
        h = self.action2llm(action, domain_id) + self.action_modality_embed.view(1, -1) + self.time_embedder(t)
        h = torch.tanh(self.backbone(h) + self.q_proj_moe_gen(h))
        return self.llm2action(h, domain_id)


def _lora_names(net: nn.Module) -> list[str]:
    return sorted(n for n, _ in net.named_parameters() if ".lora_" in n)


def test_lora_and_time_embedder_groups_inner_loop_and_apply(tmp_path) -> None:
    from cosmos_framework.data.generator.action.meta.meta_action_adapter import (
        GROUP_ACTION_HEADS,
        GROUP_LORA,
        GROUP_MODALITY_EMBED,
        GROUP_TIME_EMBEDDER,
        net_param_lookup,
    )

    torch.manual_seed(0)
    net = _TinyLoraNet()
    te_before = {n: p.detach().clone() for n, p in net.time_embedder.named_parameters()}
    adapter = MetaActionAdapter(
        net, scratch_domain_id=2, include_lora=True, include_time_embedder=True, init=MetaActionInitSpec(source="fresh", seed=3)
    )
    # groups / names
    assert set(adapter.numel_by_group()) == {GROUP_ACTION_HEADS, GROUP_MODALITY_EMBED, GROUP_LORA, GROUP_TIME_EMBEDDER}
    assert all(n in adapter.names for n in _lora_names(net))
    assert all(n in adapter.names for n in (f"time_embedder.{k}" for k in ("0.weight", "0.bias", "2.weight", "2.bias")))
    assert adapter.numel_by_group()[GROUP_LORA] == 2 * 4 * HIDDEN
    # theta_meta starts from the model for the shared groups: LoRA B = 0, time_embedder = mid-trained values
    assert torch.count_nonzero(adapter.meta["q_proj_moe_gen.lora_B.weight"]) == 0
    torch.testing.assert_close(adapter.meta["time_embedder.0.weight"].detach(), te_before["0.weight"])
    # freezing: base linear weights frozen, adapters / heads / time_embedder trainable
    assert not net.backbone.weight.requires_grad and not net.q_proj_moe_gen.weight.requires_grad
    assert net.q_proj_moe_gen.lora_A.weight.requires_grad and net.time_embedder[0].weight.requires_grad

    adapter.start_episode()
    loss0 = _loss(net, 2).item()
    for _ in range(15):
        loss = _loss(net, 2)
        loss.backward()
        grads = adapter.collect_grads()
        assert grads["q_proj_moe_gen.lora_B.weight"].abs().sum() > 0  # B gets gradient from step 1 (A is random)
        adapter.inner_update(grads, lr=0.05)
    assert _loss(net, 2).item() < loss0
    by_group = adapter.fast_delta_norms_by_group()
    assert by_group[GROUP_LORA] > 0 and by_group[GROUP_TIME_EMBEDDER] > 0 and by_group[GROUP_ACTION_HEADS] > 0
    assert torch.count_nonzero(net.q_proj_moe_gen.lora_B.weight) > 0  # fast weights were written in place
    # heads: rows other than the scratch row untouched
    for n in DOMAIN_ROW_PARAM_NAMES:
        p = getattr_chain(net, n).detach()
        assert p.shape[0] == NUM_DOMAINS

    # save -> load -> apply on a fresh network (domain row 4 + shared groups)
    path = save_meta_action_init(tmp_path / "meta_lora.pt", adapter.export_meta(), adapter.metadata())
    meta, md = load_meta_action_init(path)
    assert md["include_lora"] and md["numel_by_group"][GROUP_LORA] == 2 * 4 * HIDDEN and md["lora"]["rank"] == 4
    net2 = _TinyLoraNet()
    row3_before = net2.action2llm.fc.weight[3].detach().clone()
    written = apply_meta_action_init_to_net(net2, meta, domain_id=4)
    assert set(_lora_names(net2)) <= set(written) and "time_embedder.2.weight" in written
    torch.testing.assert_close(net2.action2llm.fc.weight[4].float(), meta["action2llm.fc.weight"])
    torch.testing.assert_close(net2.action2llm.fc.weight[3], row3_before)  # other rows untouched
    lk = net_param_lookup(net2)
    for n in _lora_names(net2) + ["time_embedder.0.weight", "action_modality_embed"]:
        torch.testing.assert_close(lk[n].detach().float(), meta[n])
    # excluding the shared groups leaves them alone; a LoRA-less target raises unless LoRA is excluded
    net3 = _TinyActionNet()
    apply_meta_action_init_to_net(net3, meta, domain_id=1, include_lora=False, include_time_embedder=False)
    torch.testing.assert_close(net3.action2llm.fc.weight[1].float(), meta["action2llm.fc.weight"])
    try:
        apply_meta_action_init_to_net(net3, meta, domain_id=1, include_lora=True, include_time_embedder=False)
    except KeyError as e:
        assert "lora" in str(e).lower()
    else:
        raise AssertionError("expected KeyError for LoRA tensors on a network without adapters")
    # a heads-only (cosmos_hs07) file still applies to a LoRA network: nothing is written for the adapters
    heads_only = {k: v for k, v in meta.items() if ".lora_" not in k and not k.startswith("time_embedder.")}
    net4 = _TinyLoraNet()
    written4 = apply_meta_action_init_to_net(net4, heads_only, domain_id=0)
    assert not any(".lora_" in n for n in written4) and torch.count_nonzero(net4.q_proj_moe_gen.lora_B.weight) == 0
