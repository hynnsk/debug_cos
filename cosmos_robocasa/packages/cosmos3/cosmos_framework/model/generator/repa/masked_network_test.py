# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Exercise real VFM forward/packing/capture/heads with small CPU decoder blocks.

This checks the two-branch autograd contract, including activation recomputation.
It does not replace a multi-GPU FSDP/CUDA smoke run with the real checkpoint.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from cosmos_framework.data.generator.sequence_packing.packers import pack_input_sequence
from cosmos_framework.data.generator.sequence_packing.runtime import from_und_gen_splits, get_gen_seq, get_und_seq
from cosmos_framework.data.generator.sequence_packing.sequence import SequencePlan
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.mot.unified_mot import _impl_forward
from cosmos_framework.model.generator.repa.alignment import RepaTeacherTokens
from cosmos_framework.model.generator.repa.masked_prediction import masked_prediction_loss
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)
        self.calls = 0
        self.fail = False

    def forward(self, pack, *args, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("injected block failure")
        und, gen = get_und_seq(pack), get_gen_seq(pack)
        gen = gen + self.linear(gen + gen.mean(0, keepdim=True)).tanh()
        und = und + self.linear(und).tanh()
        return from_und_gen_splits(und, gen, pack), {}, None


class TinyRotary(nn.Module):
    def forward(self, x, position_ids):
        z = x.new_zeros(1, position_ids.shape[-1], 8)
        return z, z


class TinyText(nn.Module):
    forward = _impl_forward  # production eager loop, capture flags, early exit and norms

    def __init__(self, checkpointed):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 16)
        self.layers = nn.ModuleList(
            [checkpoint_wrapper(TinyBlock()) if checkpointed else TinyBlock() for _ in range(10)]
        )
        self.norm = nn.LayerNorm(16)
        self.norm_moe_gen = nn.LayerNorm(16)
        self.rotary_emb = TinyRotary()


class TinyLanguage(nn.Module):
    def __init__(self, checkpointed):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=16, num_attention_heads=2, num_key_value_heads=2, head_dim=8, num_hidden_layers=10
        )
        self.model = TinyText(checkpointed)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def make_network(checkpointed=False):
    language = TinyLanguage(checkpointed)
    config = Cosmos3VFMNetworkConfig(
        vlm_config=language.config,
        action_gen=True,
        action_dim=4,
        num_embodiment_domains=1,
        latent_channel_size=3,
        latent_patch_size=2,
        repa_enabled=True,
        repa_layer_index=8,
        repa_teacher_embed_dim=8,
        repa_projector_hidden_dim=16,
        repa_teacher_grid_thw=(8, 4, 4),
        repa_target_grid_thw=(4, 2, 2),
    )
    return Cosmos3VFMNetwork(language, config)


def make_pack(actions):
    clean = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[torch.randn(1, 3, 5, 4, 8)],
        x0_tokens_action=[torch.randn(16, 4)] if actions else None,
        action_domain_id=[torch.tensor(0)] if actions else None,
    )
    return pack_input_sequence(
        [SequencePlan(has_text=True, has_vision=True, has_action=actions, condition_frame_indexes_vision=[0])],
        [[4, 5] if actions else []],
        clean,
        torch.tensor([[0.5 if actions else 0.0]]),
        {"bos_token_id": 1, "eos_token_id": 2, "start_of_generation": 3},
        latent_patch_size=2,
    )


def make_aux():
    return dict(
        packed_seq=make_pack(False),
        teacher_tokens=RepaTeacherTokens(torch.randn(1, 2, 8, 4, 4, 8)),
        mask=torch.arange(32) % 2 == 0,
    )


@pytest.mark.parametrize("checkpointed", [False, True])
def test_two_branches_preserve_main_predictions_and_only_prefix_receives_jepa_gradients(checkpointed):
    model = make_network(checkpointed)
    main, aux = make_pack(True), make_aux()
    with torch.no_grad():
        ordinary = model(main)
    outputs = model(main, repa_aux=aux)
    assert aux["teacher_tokens"].consumed
    torch.testing.assert_close(outputs["preds_action"][0], ordinary["preds_action"][0])
    torch.testing.assert_close(outputs["preds_vision"][0], ordinary["preds_vision"][0])
    blocks = model.language_model.model.layers
    assert [block.calls for block in blocks] == [3] * 8 + [2] * 2
    loss, _ = masked_prediction_loss(
        outputs["jepa_pred"], outputs["jepa_target"], outputs["jepa_mask"], outputs["jepa_counts"], 0.25
    )
    loss.backward(retain_graph=True)
    assert all(block.linear.weight.grad.abs().sum() > 0 for block in blocks[:8])
    assert all(block.linear.weight.grad is None for block in blocks[8:])
    assert model.vae2llm.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.action2llm.parameters())
    assert all(p.grad is None for p in model.llm2action.parameters())
    # The production path uses one backward for the sum of both branches.
    model.zero_grad(set_to_none=True)
    (loss + outputs["preds_action"][0].square().mean()).backward()
    assert all(block.linear.weight.grad is not None for block in blocks)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.llm2action.parameters())
    assert model.language_model.model._repa_capture_layer is None
    assert not model.language_model.model._repa_stop_after_capture


def test_capture_state_is_cleared_on_auxiliary_failure():
    model = make_network()
    text = model.language_model.model
    text.layers[1].fail = True
    with pytest.raises(RuntimeError, match="injected block failure"):
        model(make_pack(True), repa_aux=make_aux())
    assert text._repa_capture_layer is None
    assert text._repa_captured is None
    assert not text._repa_stop_after_capture
    text.layers[1].fail = False
    assert "preds_action" in model(make_pack(True))


@pytest.mark.parametrize(("step", "expected_weight"), [(0, 0.0), (100, 0.25), (200, 0.5)])
def test_model_loss_adds_jepa_with_warmup_and_preserves_gradients(step, expected_weight):
    from cosmos_framework.configs.base.defaults.model_config import RepaConfig
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    # Disable FM terms to measure the exact additional contribution through the
    # production loss assembly. The full network test above covers the FM heads.
    cfg = SimpleNamespace(
        repa=RepaConfig(objective="masked_prediction"),
        vision_gen=False,
        action_gen=False,
        sound_gen=False,
        rectified_flow_training_config=SimpleNamespace(
            normalize_loss_by_active=False, sample_level_loss_averaging=False
        ),
    )
    model = SimpleNamespace(
        config=cfg,
        repa_enabled=True,
        masked_prediction_enabled=True,
        sigreg_enabled=False,
        tensor_kwargs_fp32={"device": "cpu", "dtype": torch.float32},
        _sample_level_loss_scale=lambda *args: torch.tensor(1.0),
        _get_load_balancing_loss_meshes=lambda: (None, None),
    )
    pred = torch.randn(8, 6, requires_grad=True)
    target = torch.randn_like(pred, requires_grad=True)
    output = dict(jepa_pred=pred, jepa_target=target, jepa_mask=torch.arange(8) % 2 == 0, jepa_counts=[8])
    total, metrics = OmniMoTModel._compute_losses(model, output, None, None, torch.zeros(1), False, iteration=step)
    torch.testing.assert_close(total, expected_weight * metrics["jepa_loss"])
    assert metrics["jepa_weight"] == expected_weight
    total.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert (pred.grad.abs().sum() > 0) == (expected_weight > 0)
    assert target.grad is None
