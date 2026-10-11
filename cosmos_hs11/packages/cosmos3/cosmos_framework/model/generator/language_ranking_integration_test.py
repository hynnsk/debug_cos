# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU integration of production packing, noising, paired forward, loss and backward."""

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from cosmos_framework.configs.base.defaults.model_config import LanguageRankingConfig
from cosmos_framework.data.generator.sequence_packing import PackedSequence, SequencePlan, pack_input_sequence
from cosmos_framework.data.generator.sequence_packing.runtime import from_und_gen_splits, get_gen_seq, get_und_seq
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.repa.masked_network_test import TinyLanguage
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


class LanguageSensitiveBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def forward(self, pack, *args, **kwargs):
        und, gen = get_und_seq(pack), get_gen_seq(pack)
        gen = gen + self.linear(gen + und.mean(0, keepdim=True)).tanh()
        und = und + self.linear(und).tanh()
        return from_und_gen_splits(und, gen, pack), {}, None


class UnitWeightFlow:
    def train_time_weight(self, t, kwargs):
        return torch.ones_like(t, **kwargs)

    def get_interpolation(self, noise, clean, sigma):
        return ([x + s * (e - x) for e, x, s in zip(noise, clean, sigma)], [e - x for e, x in zip(noise, clean)])


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("independent_action_schedule", [False, True])
def test_training_step_reuses_noise_and_preserves_both_gradient_paths(
    monkeypatch, checkpointed, independent_action_schedule
):
    torch.manual_seed(7)
    language = TinyLanguage(False)
    language.model.layers = nn.ModuleList(
        [checkpoint_wrapper(LanguageSensitiveBlock()) if checkpointed else LanguageSensitiveBlock() for _ in range(10)]
    )
    net = Cosmos3VFMNetwork(
        language,
        Cosmos3VFMNetworkConfig(
            vlm_config=language.config,
            action_gen=True,
            action_dim=4,
            num_embodiment_domains=1,
            latent_channel_size=3,
            latent_patch_size=2,
            repa_enabled=False,
        ),
    )
    with torch.no_grad():
        for p in net.llm2action.parameters():
            p.normal_(std=0.1)
    monkeypatch.setattr(PackedSequence, "to_cuda", lambda self: None)
    packs, noise_calls, forwards = [], [], []

    def pack(plans, text, clean, times, **kwargs):
        result = pack_input_sequence(
            plans,
            text,
            clean,
            times,
            {"bos_token_id": 1, "eos_token_id": 2, "start_of_generation": 3},
            latent_patch_size=2,
            **kwargs,
        )
        packs.append(result)
        return result

    rf = SimpleNamespace(
        independent_action_schedule=independent_action_schedule,
        independent_sound_schedule=False,
        normalize_loss_by_active=False,
        sample_level_loss_averaging=False,
        loss_scale=10.0,
        image_loss_scale=None,
        action_loss_weight=10.0,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            language_ranking=LanguageRankingConfig(enabled=True, margin=100.0),
            rectified_flow_training_config=rf,
            vision_gen=True,
            action_gen=True,
            sound_gen=False,
            lidar_state_ch=None,
        ),
        net=net,
        parallel_dims=None,
        tensor_kwargs={"device": "cpu", "dtype": torch.float32},
        tensor_kwargs_fp32={"device": "cpu", "dtype": torch.float32},
        rectified_flow_video=UnitWeightFlow(),
        rectified_flow_image=UnitWeightFlow(),
        rectified_flow_action=UnitWeightFlow(),
        repa_enabled=False,
        masked_prediction_enabled=False,
        sigreg_enabled=False,
        sigreg_on_residual_tokens=False,
        _get_train_noise_level_vision=lambda **kw: (torch.tensor([[500.0]]), torch.tensor([[0.5]])),
        _get_train_noise_level_action=lambda **kw: (torch.tensor([[250.0]]), torch.tensor([[0.25]])),
        _pack_input_sequence=pack,
        pre_noise_memory_hook=lambda pack, clean, memory: memory,
        _resolve_repa_teacher_tokens=lambda *args: None,
        build_memory_state=lambda *args: None,
        _get_load_balancing_loss_meshes=lambda: (None, None),
    )
    for name in ("_replace_clean_with_noised", "_compute_flow_matching_loss", "_compute_losses", "denoise"):
        setattr(model, name, MethodType(getattr(OmniMoTModel, name), model))

    def add_noise(*args, **kwargs):
        result = OmniMoTModel._add_noise_to_input(model, *args, **kwargs)
        noise_calls.append(result)
        return result

    model._add_noise_to_input = add_noise
    net.register_forward_hook(lambda module, args, output: forwards.append(output))
    clean = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[torch.randn(1, 3, 5, 4, 8)],
        x0_tokens_action=[torch.randn(16, 4)],
        action_domain_id=[torch.tensor(0)],
        raw_action_dim=[torch.tensor(4)],
        action_valid_mask=[torch.ones(4)],
    )
    plans = [SequencePlan(has_text=True, has_vision=True, has_action=True, condition_frame_indexes_vision=[0])]
    memory = dict(
        skip_text=False, initial_temporal_offset=0, ranking_negative_text_indexes=[[6, 7, 8]], ranking_valid=[True]
    )
    inputs = ([[4, 5]], plans, clean, memory, None, [])
    output, loss = OmniMoTModel.training_step_from_inputs(model, inputs, 3)
    assert len(noise_calls) == 1 and len(packs) == 2 and len(forwards) == 1
    positive, negative = packs
    assert positive.text_indexes.numel() != negative.text_indexes.numel()
    for modality in ("vision", "action"):
        a, b = getattr(positive, modality), getattr(negative, modality)
        assert a.tokens[0] is b.tokens[0]
        torch.testing.assert_close(a.timesteps, b.timesteps)
        torch.testing.assert_close(a.condition_mask[0], b.condition_mask[0])
    paired = forwards[0]
    pos, neg = paired["preds_action"][0], paired["ranking_preds_action"][0]
    assert not torch.allclose(pos, neg)
    pos.retain_grad()
    neg.retain_grad()
    assert output["language_ranking_valid_fraction"].item() == 1.0
    loss.backward()
    assert pos.grad.abs().sum() > 0 and neg.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.llm2action.parameters())
    with torch.no_grad():
        ordinary = net(positive)
    torch.testing.assert_close(pos, ordinary["preds_action"][0])
