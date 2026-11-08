# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os
from pathlib import Path

import pytest
import torch

from cosmos_framework.model.generator.repa.dinov2_teacher import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    DINOv2Teacher,
    resolve_dinov2_spec,
)
from cosmos_framework.model.generator.repa.teachers import resolve_repa_teacher_spec


def _tiny_dinov2():
    from transformers import Dinov2Config, Dinov2Model

    cfg = Dinov2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, mlp_ratio=2, patch_size=14, image_size=28)
    return Dinov2Model(cfg)


def test_specs_and_registry():
    assert resolve_dinov2_spec("dinov2_vitb14").embed_dim == 768
    assert resolve_dinov2_spec("dinov2-base").hf_repo == "facebook/dinov2-base"
    spec = resolve_repa_teacher_spec("dinov2_vitb14")
    assert (spec.family, spec.patch_size, spec.tubelet_size, spec.default_input_size) == ("dinov2", 14, 1, 224)
    spec = resolve_repa_teacher_spec("vjepa2_1_vit_base_384")
    assert (spec.family, spec.patch_size, spec.tubelet_size, spec.default_input_size) == ("vjepa2_1", 16, 2, 256)
    with pytest.raises(ValueError):
        resolve_repa_teacher_spec("clip_vitb16")


def test_tiny_model_forward_layout_and_preprocess():
    teacher = DINOv2Teacher("dinov2_vitb14", input_size=28, num_frames=4, device="cpu", chunk_size=2, model=_tiny_dinov2())
    assert teacher.grid_thw == (4, 2, 2) and teacher.embed_dim == 32
    assert not any(p.requires_grad for p in teacher.parameters())
    teacher.train()
    assert not teacher.training
    clips = torch.randint(0, 256, (3, 3, 4, 40, 40), dtype=torch.uint8)  # resized 40 -> 28
    tokens = teacher(clips)
    assert tokens.shape == (3, 4, 2, 2, 32) and torch.isfinite(tokens).all()
    # per-frame independence: frame 2 of clip 1 only depends on that frame
    clips2 = clips.clone()
    clips2[1, :, 2] = 0
    tokens2 = teacher(clips2)
    assert torch.allclose(tokens[0], tokens2[0]) and torch.allclose(tokens[1, :2], tokens2[1, :2])
    assert not torch.allclose(tokens[1, 2], tokens2[1, 2])
    white = torch.full((2, 3, 28, 28), 255, dtype=torch.uint8)
    x = teacher.preprocess_frames(white)
    for c in range(3):
        torch.testing.assert_close(x[:, c], torch.full_like(x[:, c], (1.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]))
    with pytest.raises(ValueError):
        DINOv2Teacher("dinov2_vitb14", input_size=256, model=_tiny_dinov2(), device="cpu")


def _cached() -> bool:
    hub = os.environ.get("HF_HUB_CACHE") or (os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None)
    return bool(hub) and Path(hub, "models--facebook--dinov2-base").exists()


@pytest.mark.skipif(not _cached(), reason="facebook/dinov2-base not in the local HF cache")
def test_real_dinov2_base_from_cache_encodes_224_clip():
    teacher = DINOv2Teacher("dinov2_vitb14", input_size=224, num_frames=16, device="cpu")
    assert teacher.grid_thw == (16, 16, 16) and teacher.embed_dim == 768
    clip = torch.randint(0, 256, (1, 3, 16, 256, 256), dtype=torch.uint8)
    tokens = teacher(clip)
    assert tokens.shape == (1, 16, 16, 16, 768) and torch.isfinite(tokens).all()
