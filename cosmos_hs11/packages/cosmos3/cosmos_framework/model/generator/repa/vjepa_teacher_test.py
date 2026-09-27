# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os
from pathlib import Path

import pytest
import torch

from cosmos_framework.model.generator.repa.vjepa_teacher import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    VJEPA21Teacher,
    clean_backbone_state_dict,
    resolve_teacher_spec,
)


def test_spec_resolution_and_aliases():
    assert resolve_teacher_spec("vjepa2_1_vit_base_384").embed_dim == 768
    assert resolve_teacher_spec("vit-l/16").embed_dim == 1024
    assert resolve_teacher_spec("vitb").filename == "vjepa2_1_vitb_dist_vitG_384.pt"
    with pytest.raises(ValueError):
        resolve_teacher_spec("vit_huge")


def test_clean_backbone_keys():
    sd = {"module.backbone.blocks.0.attn.qkv.weight": torch.zeros(1), "module.backbone.norms_block.3.bias": torch.zeros(1)}
    assert set(clean_backbone_state_dict(sd)) == {"blocks.0.attn.qkv.weight", "norms_block.3.bias"}


def test_preprocess_normalizes_and_resizes():
    teacher = VJEPA21Teacher("vjepa2_1_vit_base_384", input_size=32, num_frames=4, load_weights=False, device="cpu")
    clips = torch.full((2, 3, 4, 48, 48), 255, dtype=torch.uint8)
    x = teacher.preprocess(clips)
    assert x.shape == (2, 3, 4, 32, 32) and x.dtype == torch.float32
    for c in range(3):
        expected = (1.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        torch.testing.assert_close(x[:, c], torch.full_like(x[:, c], expected))
    with pytest.raises(ValueError):
        teacher.preprocess(clips[:, :, :3])  # wrong frame count


def test_random_init_forward_shape_and_frozen():
    teacher = VJEPA21Teacher("vjepa2_1_vit_base_384", input_size=32, num_frames=4, load_weights=False, device="cpu", chunk_size=1)
    assert teacher.grid_thw == (2, 2, 2)
    assert not any(p.requires_grad for p in teacher.parameters())
    teacher.train()  # must stay in eval mode
    assert not teacher.training and not teacher.encoder.training
    clips = torch.randint(0, 256, (3, 3, 4, 32, 32), dtype=torch.uint8)
    tokens = teacher(clips)
    assert tokens.shape == (3, 2, 2, 2, 768)
    assert torch.isfinite(tokens).all()


def _release_checkpoint() -> Path | None:
    storage = os.environ.get("COSMOS_STORAGE")
    if not storage:
        return None
    p = Path(storage) / "checkpoints" / "vjepa2_1" / "vjepa2_1_vitb_dist_vitG_384.pt"
    return p if p.exists() else None


@pytest.mark.skipif(_release_checkpoint() is None, reason="V-JEPA 2.1 ViT-B checkpoint not available locally")
def test_release_checkpoint_loads_strict_and_encodes_256px_clip():
    teacher = VJEPA21Teacher("vjepa2_1_vit_base_384", input_size=256, num_frames=16, device="cpu", allow_download=False)
    assert teacher.grid_thw == (8, 16, 16)
    clip = torch.randint(0, 256, (1, 3, 16, 256, 256), dtype=torch.uint8)
    tokens = teacher(clip)
    assert tokens.shape == (1, 8, 16, 16, 768)
    assert torch.isfinite(tokens).all()
    # last per-layer LayerNorm applied -> roughly unit-scale features
    assert 0.1 < tokens.float().std().item() < 10.0
