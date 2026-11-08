# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Frozen DINOv2 image encoder as an alternative REPA teacher (the teacher the REPA paper itself uses).

DINOv2 is an image model, so every frame of every camera view is encoded independently and the result is laid out
like the V-JEPA tokens, ``[B, T, H_p, W_p, D]`` with ``T`` = number of raw frames (no tubelets). The downstream
target adapters are unchanged: the ``avgpool`` adapter pools ``T=16 -> 4`` with bins of 4 consecutive frames, which
is exactly the Wan VAE mapping of latent frame ``t`` to raw frames ``4t-3..4t``, and ``16x16 -> 5x5`` in space.

Weights come from the Hugging Face hub (``facebook/dinov2-{small,base,large}``, ``transformers.Dinov2Model``); the
local HF cache (``$HF_HOME``) is tried first so GPU nodes without internet work once the snapshot is present
(``huggingface_hub.snapshot_download("facebook/dinov2-base")``). Preprocessing follows the REPA reference: RGB in
``[0,1]``, resized to ``input_size`` (224 -> 16x16 patches of 14 px, the same 16x16 grid as V-JEPA at 256), ImageNet
mean/std. Patch tokens are taken after the final LayerNorm (``last_hidden_state`` minus CLS/register tokens), i.e.
DINOv2's ``x_norm_patchtokens``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_framework.utils import log

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DINOv2Spec:
    name: str
    hf_repo: str
    embed_dim: int
    depth: int
    num_heads: int
    patch_size: int = 14


DINOV2_TEACHERS: dict[str, DINOv2Spec] = {
    "dinov2_vits14": DINOv2Spec("dinov2_vits14", "facebook/dinov2-small", 384, 12, 6),
    "dinov2_vitb14": DINOv2Spec("dinov2_vitb14", "facebook/dinov2-base", 768, 12, 12),
    "dinov2_vitl14": DINOv2Spec("dinov2_vitl14", "facebook/dinov2-large", 1024, 24, 16),
}
DINOV2_ALIASES: dict[str, str] = {
    "dinov2_base": "dinov2_vitb14",
    "dinov2-base": "dinov2_vitb14",
    "dinov2_small": "dinov2_vits14",
    "dinov2_large": "dinov2_vitl14",
}


def is_dinov2_teacher(name: str) -> bool:
    key = DINOV2_ALIASES.get(name.lower(), name.lower())
    return key in DINOV2_TEACHERS


def resolve_dinov2_spec(name: str) -> DINOv2Spec:
    key = DINOV2_ALIASES.get(name.lower(), name.lower())
    if key not in DINOV2_TEACHERS:
        raise ValueError(f"Unknown DINOv2 teacher {name!r}; expected one of {sorted(DINOV2_TEACHERS)} or an alias.")
    return DINOV2_TEACHERS[key]


def _load_dinov2_model(spec: DINOv2Spec, checkpoint_path: str | None, load_weights: bool) -> nn.Module:
    from transformers import Dinov2Config, Dinov2Model

    source = checkpoint_path or spec.hf_repo
    if not load_weights:
        # cpu / meta builds (checkpoint conversion, smoke paths): architecture only, random weights.
        cfg = Dinov2Config(
            hidden_size=spec.embed_dim,
            num_hidden_layers=spec.depth,
            num_attention_heads=spec.num_heads,
            mlp_ratio=4,
            patch_size=spec.patch_size,
        )
        return Dinov2Model(cfg)
    try:
        model = Dinov2Model.from_pretrained(source, local_files_only=True)
        log.info(f"DINOv2Teacher: loaded {source} from the local Hugging Face cache")
    except Exception as local_err:  # noqa: BLE001 - fall back to the hub
        log.warning(f"DINOv2Teacher: {source} not in the local HF cache ({type(local_err).__name__}); downloading")
        model = Dinov2Model.from_pretrained(source)
    return model


class DINOv2Teacher(nn.Module):
    """Frozen DINOv2 returning per-frame patch tokens ``[B, T, H_p, W_p, D]`` (same interface as ``VJEPA21Teacher``)."""

    def __init__(
        self,
        name: str = "dinov2_vitb14",
        *,
        checkpoint_path: str | None = None,
        input_size: int = 224,
        num_frames: int = 16,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
        chunk_size: int = 32,
        load_weights: bool = True,
        model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.spec = resolve_dinov2_spec(name)
        self.patch_size = int(self.spec.patch_size)
        if input_size % self.patch_size != 0:
            raise ValueError(
                f"teacher_input_size={input_size} must be a multiple of the DINOv2 patch size {self.patch_size} "
                "(use 224 for a 16x16 grid, 252 for 18x18)."
            )
        self.input_size = int(input_size)
        self.num_frames = int(num_frames)
        self.dtype = dtype
        self.chunk_size = max(1, int(chunk_size))
        self.model = model if model is not None else _load_dinov2_model(self.spec, checkpoint_path, load_weights)
        self._embed_dim = int(self.model.config.hidden_size)
        self._num_prefix_tokens = 1 + int(getattr(self.model.config, "num_register_tokens", 0) or 0)  # CLS (+registers)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.model.requires_grad_(False)
        if device is not None:
            self.to(device)
        self.eval()
        log.info(
            f"DINOv2Teacher: {self.spec.name} ({sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M params, "
            f"D={self._embed_dim}, grid {self.grid_thw}, input {self.input_size}px per frame)"
        )

    def train(self, mode: bool = True) -> "DINOv2Teacher":  # noqa: D401 - always frozen / eval
        return super().train(False)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def grid_thw(self) -> tuple[int, int, int]:
        side = self.input_size // self.patch_size
        return (self.num_frames, side, side)

    def preprocess_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """``[N,3,H,W]`` uint8 (0..255) or float (0..1) RGB -> normalized float32 at ``input_size``."""
        x = frames.float()
        if frames.dtype == torch.uint8 or x.max() > 1.5:
            x = x / 255.0
        if tuple(x.shape[-2:]) != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False, antialias=True)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """``[B,3,T,H,W]`` -> ``[B,T,H_p,W_p,D]`` post-LayerNorm patch tokens of every frame, in ``self.dtype``."""
        if clips.ndim != 5 or clips.shape[1] != 3:
            raise ValueError(f"Expected [B,3,T,H,W] RGB clips, got {tuple(clips.shape)}")
        if clips.shape[2] != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames per clip, got {clips.shape[2]}")
        t_frames, hp, wp = self.grid_thw
        use_autocast = clips.is_cuda and self.dtype in (torch.bfloat16, torch.float16)
        outs = []
        for start in range(0, clips.shape[0], self.chunk_size):
            chunk = clips[start : start + self.chunk_size]  # [b,3,T,H,W]
            b = chunk.shape[0]
            frames = chunk.permute(0, 2, 1, 3, 4).reshape(b * t_frames, 3, *chunk.shape[-2:])  # [b*T,3,H,W]
            x = self.preprocess_frames(frames)
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    hidden = self.model(pixel_values=x).last_hidden_state  # [b*T, 1(+reg)+H_p*W_p, D]
            else:
                hidden = self.model(pixel_values=x).last_hidden_state
            tokens = hidden[:, self._num_prefix_tokens :]  # drop CLS / registers -> [b*T, H_p*W_p, D]
            if tokens.shape[1] != hp * wp:
                raise RuntimeError(f"DINOv2 returned {tokens.shape[1]} patch tokens, expected {hp}*{wp}")
            tokens = tokens.reshape(b, t_frames, hp, wp, tokens.shape[-1])
            outs.append(tokens.to(self.dtype if clips.is_cuda else tokens.dtype))
        return torch.cat(outs, dim=0)
