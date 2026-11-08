# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Frozen V-JEPA 2.1 ``ema_encoder`` used as the REPA teacher.

Reproduces ``src/hub/backbones.py::_make_vjepa2_1_model`` of facebookresearch/vjepa2 (encoder only):
``patch_size=16, tubelet_size=2, use_rope=True, use_sdpa=True, img_temporal_dim_size=1, interpolate_rope=True``,
weights = the ``ema_encoder`` entry of the released checkpoint, loaded with ``strict=True`` after stripping
the ``module.`` / ``backbone.`` prefixes.

Preprocessing follows the upstream eval transform (``evals/video_classification_frozen/utils.py`` /
``app/vjepa_2_1/transforms.py``): RGB in ``[0,1]`` normalized with the ImageNet mean/std
``(0.485,0.456,0.406) / (0.229,0.224,0.225)``, square ``input_size`` frames, an even number of frames.

Input resolution. The 2.1 checkpoints were released "at 384px", but their RoPE positions are rescaled onto the
pretraining grid of ``256/16 = 16`` patches per side (``interpolate_rope=True``, see ``RoPEAttention``), so a
256x256 clip -- the native LIBERO resolution -- is fed on exactly the pretrained grid (scale factor 1) and is
the default here (``input_size=256``). Other sizes are resized bilinearly and handled by the same RoPE
interpolation (384 costs ~2.3x the tokens).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_framework.model.generator.repa.vjepa2_1 import vision_transformer as vjepa_vit
from cosmos_framework.utils import log

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"


@dataclass(frozen=True)
class VJEPA21Spec:
    name: str
    arch: str  # constructor name in vision_transformer.py
    embed_dim: int
    depth: int
    filename: str  # released checkpoint file (contains encoder / ema_encoder / predictor / opt ...)

    @property
    def url(self) -> str:
        return f"{_VJEPA_BASE_URL}/{self.filename}"

    @property
    def slim_filename(self) -> str:
        """Encoder-only export written by ``scripts/export_vjepa2_1_encoder.py`` (much smaller than the original)."""
        return self.filename.replace(".pt", ".ema_encoder.pt")


VJEPA21_TEACHERS: dict[str, VJEPA21Spec] = {
    "vjepa2_1_vit_base_384": VJEPA21Spec(
        name="vjepa2_1_vit_base_384", arch="vit_base", embed_dim=768, depth=12, filename="vjepa2_1_vitb_dist_vitG_384.pt"
    ),
    "vjepa2_1_vit_large_384": VJEPA21Spec(
        name="vjepa2_1_vit_large_384",
        arch="vit_large",
        embed_dim=1024,
        depth=24,
        filename="vjepa2_1_vitl_dist_vitG_384.pt",
    ),
}
# Short aliases accepted by the config.
VJEPA21_ALIASES: dict[str, str] = {
    "vit_base": "vjepa2_1_vit_base_384",
    "vit-b/16": "vjepa2_1_vit_base_384",
    "vitb": "vjepa2_1_vit_base_384",
    "vit_large": "vjepa2_1_vit_large_384",
    "vit-l/16": "vjepa2_1_vit_large_384",
    "vitl": "vjepa2_1_vit_large_384",
}


def resolve_teacher_spec(name: str) -> VJEPA21Spec:
    key = VJEPA21_ALIASES.get(name.lower(), name)
    if key not in VJEPA21_TEACHERS:
        raise ValueError(f"Unknown V-JEPA 2.1 teacher {name!r}; expected one of {sorted(VJEPA21_TEACHERS)} or an alias.")
    return VJEPA21_TEACHERS[key]


def default_checkpoint_dir() -> Path:
    storage = os.environ.get("COSMOS_STORAGE")
    if storage:
        return Path(storage) / "checkpoints" / "vjepa2_1"
    torch_home = os.environ.get("TORCH_HOME") or str(Path.home() / ".cache" / "torch")
    return Path(torch_home) / "hub" / "checkpoints"


def resolve_checkpoint_path(spec: VJEPA21Spec, checkpoint_path: str | None, allow_download: bool = True) -> Path:
    """Explicit path > ``<default dir>/<slim>`` > ``<default dir>/<full>`` > download of the full file."""
    if checkpoint_path:
        p = Path(checkpoint_path).expanduser()
        if p.is_dir():
            for candidate in (p / spec.slim_filename, p / spec.filename):
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(f"Neither {spec.slim_filename} nor {spec.filename} found under {p}")
        if not p.exists():
            raise FileNotFoundError(f"V-JEPA 2.1 checkpoint not found: {p}")
        return p
    ckpt_dir = default_checkpoint_dir()
    for candidate in (ckpt_dir / spec.slim_filename, ckpt_dir / spec.filename):
        if candidate.exists():
            return candidate
    if not allow_download:
        raise FileNotFoundError(f"V-JEPA 2.1 checkpoint {spec.filename} not found under {ckpt_dir}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    target = ckpt_dir / spec.filename
    log.warning(f"Downloading {spec.url} -> {target} (pre-download this file on shared storage for multi-rank runs)")
    torch.hub.download_url_to_file(spec.url, str(target), progress=False)
    return target


def clean_backbone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip the ``module.`` / ``backbone.`` prefixes like upstream ``_clean_backbone_key``."""
    cleaned = {}
    for key, val in state_dict.items():
        key = key.replace("module.", "").replace("backbone.", "")
        cleaned[key] = val
    return cleaned


def load_encoder_state_dict(path: Path, checkpoint_key: str = "ema_encoder") -> dict[str, torch.Tensor]:
    """Return the cleaned encoder state dict from a full release checkpoint or an encoder-only export."""
    obj = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"Unexpected checkpoint content in {path}: {type(obj).__name__}")
    if checkpoint_key in obj and isinstance(obj[checkpoint_key], dict):
        state_dict = obj[checkpoint_key]
    elif "encoder" in obj and isinstance(obj["encoder"], dict) and any(torch.is_tensor(v) for v in obj["encoder"].values()):
        log.warning(f"{path} has no {checkpoint_key!r} entry; falling back to 'encoder' (non-EMA weights)")
        state_dict = obj["encoder"]
    elif all(torch.is_tensor(v) for v in obj.values()):
        state_dict = obj  # encoder-only export
    else:
        raise KeyError(f"{path} does not contain {checkpoint_key!r}, 'encoder' or a plain state dict (keys: {list(obj)[:8]})")
    return clean_backbone_state_dict(state_dict)


def build_vjepa21_encoder(
    spec: VJEPA21Spec,
    *,
    input_size: int = 256,
    num_frames: int = 16,
    patch_size: int = 16,
    tubelet_size: int = 2,
) -> nn.Module:
    """Encoder construction mirroring ``_make_vjepa2_1_model`` (encoder part) of the upstream hub loader."""
    ctor = getattr(vjepa_vit, spec.arch)
    return ctor(
        patch_size=patch_size,
        img_size=(input_size, input_size),
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        use_sdpa=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
        # The hub loader passes ``n_output_distillation=1`` for B/L; it only selects which per-layer norms are
        # concatenated in *training* mode. Feature extraction returns ``norms_block[-1](x)`` regardless.
        n_output_distillation=1,
    )


class VJEPA21Teacher(nn.Module):
    """Frozen V-JEPA 2.1 encoder returning per-tubelet patch tokens ``[B, T_t, H_p, W_p, D]``.

    Always in eval mode, never trained, never checkpointed (it is not part of ``OmniMoTModel.state_dict``).
    Runs under ``torch.no_grad`` + autocast(``dtype``); long inputs are processed in ``chunk_size`` clips.
    """

    def __init__(
        self,
        name: str = "vjepa2_1_vit_base_384",
        *,
        checkpoint_path: str | None = None,
        input_size: int = 256,
        num_frames: int = 16,
        patch_size: int = 16,
        tubelet_size: int = 2,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
        chunk_size: int = 32,
        load_weights: bool = True,
        allow_download: bool = True,
    ) -> None:
        super().__init__()
        if input_size % patch_size != 0:
            raise ValueError(f"input_size={input_size} must be a multiple of patch_size={patch_size}")
        if num_frames % tubelet_size != 0:
            raise ValueError(f"num_frames={num_frames} must be a multiple of tubelet_size={tubelet_size}")
        self.spec = resolve_teacher_spec(name)
        self.input_size = int(input_size)
        self.num_frames = int(num_frames)
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.dtype = dtype
        self.chunk_size = max(1, int(chunk_size))
        self.checkpoint_path: Path | None = None

        self.encoder = build_vjepa21_encoder(
            self.spec,
            input_size=self.input_size,
            num_frames=self.num_frames,
            patch_size=self.patch_size,
            tubelet_size=self.tubelet_size,
        )
        if load_weights:
            self.checkpoint_path = resolve_checkpoint_path(self.spec, checkpoint_path, allow_download=allow_download)
            state_dict = load_encoder_state_dict(self.checkpoint_path)
            missing, unexpected = self.encoder.load_state_dict(state_dict, strict=True)
            assert not missing and not unexpected
            log.info(
                f"VJEPA21Teacher: loaded {self.spec.name} ema_encoder from {self.checkpoint_path} "
                f"({sum(p.numel() for p in self.encoder.parameters()) / 1e6:.1f}M params, grid {self.grid_thw})"
            )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)
        self.encoder.requires_grad_(False)
        if device is not None:
            self.to(device)
        self.eval()

    # -- module protocol -------------------------------------------------------------------------
    def train(self, mode: bool = True) -> "VJEPA21Teacher":  # noqa: D401 - always frozen / eval
        return super().train(False)

    @property
    def embed_dim(self) -> int:
        return self.spec.embed_dim

    @property
    def grid_thw(self) -> tuple[int, int, int]:
        side = self.input_size // self.patch_size
        return (self.num_frames // self.tubelet_size, side, side)

    # -- preprocessing -----------------------------------------------------------------------------
    def preprocess(self, clips: torch.Tensor) -> torch.Tensor:
        """``[B,3,T,H,W]`` uint8 (0..255) or float (0..1) RGB -> normalized float32 at ``input_size``."""
        if clips.ndim != 5 or clips.shape[1] != 3:
            raise ValueError(f"Expected [B,3,T,H,W] RGB clips, got {tuple(clips.shape)}")
        if clips.shape[2] != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames per clip, got {clips.shape[2]}")
        x = clips.float()
        if clips.dtype == torch.uint8:
            x = x / 255.0
        elif x.max() > 1.5:  # float tensor still in 0..255 levels
            x = x / 255.0
        b, c, t, h, w = x.shape
        if (h, w) != (self.input_size, self.input_size):
            x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False, antialias=True)
            x = x.view(b, t, c, self.input_size, self.input_size).permute(0, 2, 1, 3, 4)
        return (x - self.mean) / self.std

    # -- feature extraction --------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """``[B,3,T,H,W]`` -> ``[B,T_t,H_p,W_p,D]`` last-layer (per-layer-normed) tokens, in ``self.dtype``.

        Preprocessing (uint8 -> normalized float32) and encoding run per ``chunk_size`` clips so the float copy
        of a 256-clip batch (~3 GB at 16x256x256) is never materialized at once.
        """
        if clips.ndim != 5:
            raise ValueError(f"Expected [B,3,T,H,W] clips, got {tuple(clips.shape)}")
        tt, hp, wp = self.grid_thw
        outs = []
        use_autocast = clips.is_cuda and self.dtype in (torch.bfloat16, torch.float16)
        for start in range(0, clips.shape[0], self.chunk_size):
            chunk = self.preprocess(clips[start : start + self.chunk_size])
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    tokens = self.encoder(chunk)  # [b, T_t*H_p*W_p, D]
            else:
                tokens = self.encoder(chunk)
            outs.append(tokens.to(self.dtype if clips.is_cuda else tokens.dtype))
        tokens = torch.cat(outs, dim=0)
        if tokens.shape[1] != tt * hp * wp:
            raise RuntimeError(f"Teacher returned {tokens.shape[1]} tokens, expected {tt}*{hp}*{wp}")
        return tokens.view(tokens.shape[0], tt, hp, wp, tokens.shape[-1])
