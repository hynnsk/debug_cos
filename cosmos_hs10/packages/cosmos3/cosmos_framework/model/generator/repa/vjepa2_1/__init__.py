# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Vendored V-JEPA 2.1 encoder (facebookresearch/vjepa2, ``app/vjepa_2_1/models``).

Only the encoder (``VisionTransformer`` + its blocks / patch embeddings) is vendored; the predictor,
masking and training code are not needed for feature extraction. See README.md for provenance.
"""

from cosmos_framework.model.generator.repa.vjepa2_1.vision_transformer import (
    VIT_EMBED_DIMS,
    VisionTransformer,
    vit_base,
    vit_large,
)

__all__ = ["VIT_EMBED_DIMS", "VisionTransformer", "vit_base", "vit_large"]
