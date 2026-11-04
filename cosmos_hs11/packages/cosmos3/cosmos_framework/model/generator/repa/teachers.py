# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Registry of REPA teachers: V-JEPA 2.1 video encoders (default) and DINOv2 image encoders (``dinov2_*``).

``resolve_repa_teacher_spec`` gives the static facts the network config needs before any weights are loaded
(feature dim, patch size, temporal tubelet), ``build_repa_teacher`` instantiates the frozen module. Both families
expose the same runtime interface: ``forward([B,3,T,H,W]) -> [B,T_t,H_p,W_p,D]``, ``embed_dim``, ``grid_thw``,
``input_size``, ``spec.name``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from cosmos_framework.model.generator.repa.dinov2_teacher import DINOv2Teacher, is_dinov2_teacher, resolve_dinov2_spec
from cosmos_framework.model.generator.repa.vjepa_teacher import VJEPA21Teacher, resolve_teacher_spec as resolve_vjepa_spec


@dataclass(frozen=True)
class RepaTeacherSpec:
    name: str
    family: str  # "vjepa2_1" | "dinov2"
    embed_dim: int
    patch_size: int
    tubelet_size: int  # frames per temporal token (2 for V-JEPA, 1 for image models)
    default_input_size: int


# cosmos_hs11 v115+: INTERNAL teachers -- no external encoder. The target rows are built by ``OmniMoTModel`` in the MoT
# token order itself (``RepaTokenTargets``): ``self_init`` = the gen-tower residual stream after MoT block
# ``repa.teacher_layer_index`` of a FROZEN copy of the network as loaded at start (self-distillation / feature anchoring);
# ``wan_vae`` = the clean Wan2.2 VAE latent patches (``vae2llm`` input, C * p * p per token) of the predicted frames.
INTERNAL_REPA_TEACHERS = ("self_init", "wan_vae")


def is_internal_repa_teacher(name: str) -> bool:
    return str(name).strip().lower() in INTERNAL_REPA_TEACHERS


class InternalRepaTeacher:
    """Stand-in for the frozen encoder of the internal teachers (``self_init`` / ``wan_vae``): carries the facts the
    logs read (``spec`` / ``embed_dim`` / ``grid_thw`` / ``input_size``) and nothing else. Never called on frames."""

    def __init__(self, name: str, embed_dim: int, grid_thw: tuple[int, int, int], layer_index: int | None) -> None:
        self.spec = RepaTeacherSpec(str(name), "internal", int(embed_dim), 1, 1, 0)
        self.embed_dim = int(embed_dim)
        self.grid_thw = tuple(int(v) for v in grid_thw)
        self.input_size = 0
        self.layer_index = None if layer_index is None else int(layer_index)


def resolve_repa_teacher_spec(name: str) -> RepaTeacherSpec:
    if is_internal_repa_teacher(name):
        # embed_dim 0 = resolved by the network/model at build time (hidden size, or C * p * p for the VAE patches)
        return RepaTeacherSpec(str(name).strip().lower(), "internal", 0, 1, 1, 0)
    if is_dinov2_teacher(name):
        d = resolve_dinov2_spec(name)
        return RepaTeacherSpec(d.name, "dinov2", d.embed_dim, d.patch_size, 1, 224)
    v = resolve_vjepa_spec(name)
    return RepaTeacherSpec(v.name, "vjepa2_1", v.embed_dim, 16, 2, 256)


def build_repa_teacher(
    name: str,
    *,
    checkpoint_path: str | None,
    input_size: int,
    num_frames: int,
    dtype: torch.dtype,
    device: torch.device | str | None,
    chunk_size: int,
    load_weights: bool = True,
) -> nn.Module:
    spec = resolve_repa_teacher_spec(name)
    if spec.family == "internal":
        raise ValueError(
            f"REPA teacher {spec.name!r} is internal (no encoder to build); OmniMoTModel builds its targets itself"
        )
    if spec.family == "dinov2":
        return DINOv2Teacher(
            name,
            checkpoint_path=checkpoint_path,
            input_size=input_size,
            num_frames=num_frames,
            dtype=dtype,
            device=device,
            chunk_size=chunk_size,
            load_weights=load_weights,
        )
    return VJEPA21Teacher(
        name,
        checkpoint_path=checkpoint_path,
        input_size=input_size,
        num_frames=num_frames,
        dtype=dtype,
        device=device,
        chunk_size=chunk_size,
        load_weights=load_weights,
    )
