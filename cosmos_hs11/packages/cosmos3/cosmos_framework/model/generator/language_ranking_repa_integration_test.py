# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU integration: action-language ranking (paired negative branch) + REPA cosine / relation distillation in ONE
training step -- the cosmos_hs11 v105-v107 recipes (experiment ``action_policy_libero_edge_repa`` with
``[model.language_ranking]`` enabled). Same stand-in model as ``language_ranking_integration_test.py`` with the REPA
head of ``repa/masked_network_test.py``."""

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.configs.base.defaults.model_config import LanguageRankingConfig, RepaConfig
from cosmos_framework.data.generator.sequence_packing import PackedSequence, SequencePlan, pack_input_sequence
from cosmos_framework.model.generator.language_ranking_integration_test import LanguageSensitiveBlock, UnitWeightFlow
from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.repa.alignment import RepaTeacherTokens
from cosmos_framework.model.generator.repa.masked_network_test import TinyLanguage
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

TEACHER = (1, 2, 8, 4, 4, 8)  # [B, views, T_t, H_t, W_t, D_t] -> adapted onto 4 x 2 x 2 MoT tokens per view


@pytest.mark.parametrize("relation_loss_weight", [0.0, 0.5])
def test_ranking_and_repa_terms_share_one_training_step(monkeypatch, relation_loss_weight):
    torch.manual_seed(7)
    language = TinyLanguage(False)
    language.model.layers = nn.ModuleList([LanguageSensitiveBlock() for _ in range(10)])
    net = Cosmos3VFMNetwork(
        language,
        Cosmos3VFMNetworkConfig(
            vlm_config=language.config,
            action_gen=True,
            action_dim=4,
            num_embodiment_domains=1,
            latent_channel_size=3,
            latent_patch_size=2,
            repa_enabled=True,
            repa_layer_index=8,
            repa_teacher_embed_dim=TEACHER[-1],
            repa_projector_hidden_dim=16,
            repa_teacher_grid_thw=TEACHER[2:5],
            repa_target_grid_thw=(4, 2, 2),
        ),
    )
    with torch.no_grad():
        for p in net.llm2action.parameters():
            p.normal_(std=0.1)
    monkeypatch.setattr(PackedSequence, "to_cuda", lambda self: None)
    packs, forwards, teacher_calls = [], [], []

    def pack(plans, text, clean, times, **kwargs):
        result = pack_input_sequence(
            plans, text, clean, times, {"bos_token_id": 1, "eos_token_id": 2, "start_of_generation": 3},
            latent_patch_size=2, **kwargs,
        )
        packs.append(result)
        return result

    def teacher(data_batch, plans, precomputed=None):
        teacher_calls.append(1)
        return RepaTeacherTokens(torch.randn(*TEACHER))

    rf = SimpleNamespace(
        independent_action_schedule=False, independent_sound_schedule=False, normalize_loss_by_active=False,
        sample_level_loss_averaging=False, loss_scale=10.0, image_loss_scale=None, action_loss_weight=10.0,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            language_ranking=LanguageRankingConfig(enabled=True, margin=100.0),
            repa=RepaConfig(loss_weight=0.5, relation_loss_weight=relation_loss_weight, relation_distance="l1"),
            rectified_flow_training_config=rf,
            rectified_flow_inference_config=SimpleNamespace(num_train_timesteps=1000),
            vision_gen=True, action_gen=True, sound_gen=False, lidar_state_ch=None,
        ),
        net=net,
        parallel_dims=None,
        tensor_kwargs={"device": "cpu", "dtype": torch.float32},
        tensor_kwargs_fp32={"device": "cpu", "dtype": torch.float32},
        rectified_flow_video=UnitWeightFlow(),
        rectified_flow_image=UnitWeightFlow(),
        rectified_flow_action=UnitWeightFlow(),
        repa_enabled=True,
        masked_prediction_enabled=False,
        sigreg_enabled=False,
        sigreg_on_residual_tokens=False,
        _get_train_noise_level_vision=lambda **kw: (torch.tensor([[500.0]]), torch.tensor([[0.5]])),
        _get_train_noise_level_action=lambda **kw: (torch.tensor([[250.0]]), torch.tensor([[0.25]])),
        _pack_input_sequence=pack,
        pre_noise_memory_hook=lambda pack, clean, memory: memory,
        _resolve_repa_teacher_tokens=teacher,
        build_memory_state=lambda *args: None,
        _get_load_balancing_loss_meshes=lambda: (None, None),
    )
    for name in (
        "_replace_clean_with_noised", "_compute_flow_matching_loss", "_compute_losses", "_repa_loss_terms", "denoise",
        "_add_noise_to_input",
    ):
        setattr(model, name, MethodType(getattr(OmniMoTModel, name), model))
    net.register_forward_hook(lambda module, args, output: forwards.append(output))
    clean = GenerationDataClean(
        batch_size=1, is_image_batch=False,
        x0_tokens_vision=[torch.randn(1, 3, 5, 4, 8)], x0_tokens_action=[torch.randn(16, 4)],
        action_domain_id=[torch.tensor(0)], raw_action_dim=[torch.tensor(4)], action_valid_mask=[torch.ones(4)],
    )
    plans = [SequencePlan(has_text=True, has_vision=True, has_action=True, condition_frame_indexes_vision=[0])]
    memory = dict(
        skip_text=False, initial_temporal_offset=0, ranking_negative_text_indexes=[[6, 7, 8]], ranking_valid=[True]
    )
    output, loss = OmniMoTModel.training_step_from_inputs(model, ([[4, 5]], plans, clean, memory, None, []), 3)

    # One teacher call, two packs (positive + negative instruction), ONE root forward carrying both branches.
    assert len(teacher_calls) == 1 and len(packs) == 2 and len(forwards) == 1
    paired = forwards[0]
    assert "ranking_preds_action" in paired and "repa_pred" in paired and not paired.get("repa_empty", False)
    assert paired["repa_pred"].shape == (32, TEACHER[-1])  # 4 predicted latent frames x 2x2 tokens x 2 views
    assert not torch.allclose(paired["preds_action"][0], paired["ranking_preds_action"][0])
    for key in (
        "language_ranking_loss", "language_ranking_weighted_loss", "repa_loss", "repa_cos_sim", "repa_rel_loss",
        "repa_weighted_loss", "flow_matching_loss_action", "flow_matching_loss_vision",
    ):
        assert key in output and torch.isfinite(torch.as_tensor(output[key])).all(), key
    assert output["language_ranking_valid_fraction"].item() == 1.0
    assert float(output["repa_weight"]) == 0.5
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.repa_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.llm2action.parameters())
    assert net.language_model.model._repa_capture_layer is None  # capture state cleared after the paired forward
    # The negative branch ran without teacher targets: a plain REPA forward of the positive pack yields the same rows.
    with torch.no_grad():
        plain = net(packs[0], repa_teacher_tokens=RepaTeacherTokens(torch.randn(*TEACHER)))
    assert plain["repa_pred"].shape == paired["repa_pred"].shape
    torch.testing.assert_close(plain["preds_action"][0], paired["preds_action"][0])
