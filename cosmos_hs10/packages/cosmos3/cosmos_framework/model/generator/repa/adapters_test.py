# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.model.generator.repa.adapters import (
    AvgPoolConvTargetAdapter,
    AvgPoolTargetAdapter,
    RepaProjector,
    StridedConvTargetAdapter,
    build_target_adapter,
    concat_views_along_width,
    repa_cosine_loss,
    split_flat_tokens,
    strided_conv_geometry,
)

TEACHER_GRID = (8, 16, 16)
TARGET_GRID = (4, 5, 5)


def _teacher_tokens(b: int = 2, d: int = 6, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, d, *TEACHER_GRID, generator=g)


def test_avgpool_temporal_pairs_and_spatial_bins():
    x = _teacher_tokens()
    y = AvgPoolTargetAdapter()(x, TARGET_GRID)
    assert y.shape == (2, 6, 4, 5, 5)
    # temporal: consecutive tubelet pairs; spatial: adaptive bins [0,4),[3,7),[6,10),[9,13),[12,16)
    ref_t = x.view(2, 6, 4, 2, 16, 16).mean(3)
    bins = [(0, 4), (3, 7), (6, 10), (9, 13), (12, 16)]
    ref = torch.stack(
        [torch.stack([ref_t[..., h0:h1, w0:w1].mean(dim=(-2, -1)) for (w0, w1) in bins], dim=-1) for (h0, h1) in bins],
        dim=-2,
    )
    torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)


def test_strided_conv_geometry_matches_adaptive_bins():
    assert strided_conv_geometry(8, 4) == (2, 2)
    assert strided_conv_geometry(16, 5) == (4, 3)
    assert strided_conv_geometry(16, 16) == (1, 1)
    with pytest.raises(ValueError):
        strided_conv_geometry(4, 8)


@pytest.mark.parametrize("variant", ["avgpool_conv", "strided_conv"])
def test_learnable_adapters_start_as_avgpool(variant):
    x = _teacher_tokens()
    ref = AvgPoolTargetAdapter()(x, TARGET_GRID)
    adapter = build_target_adapter(variant, 6, in_grid=TEACHER_GRID, out_grid=TARGET_GRID)
    adapter.reset_parameters()
    y = adapter(x, TARGET_GRID)
    assert y.shape == ref.shape
    torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)
    assert sum(p.numel() for p in adapter.parameters()) > 0
    # gradient reaches every adapter parameter (targets are learnable for variants 2/3)
    y.sum().backward()
    assert all(p.grad is not None for p in adapter.parameters())


def test_strided_conv_full_variant_and_shape_checks():
    x = _teacher_tokens()
    ref = AvgPoolTargetAdapter()(x, TARGET_GRID)
    adapter = StridedConvTargetAdapter(6, in_grid=TEACHER_GRID, out_grid=TARGET_GRID, depthwise=False)
    adapter.reset_parameters()
    torch.testing.assert_close(adapter(x, TARGET_GRID), ref, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError):
        adapter(x, (4, 6, 5))  # target grid mismatch
    with pytest.raises(ValueError):
        adapter(x[..., :8, :], TARGET_GRID)  # teacher grid mismatch


def test_avgpool_conv_kernel_must_be_odd():
    with pytest.raises(ValueError):
        AvgPoolConvTargetAdapter(6, kernel_size=2)


def test_projector_shapes_and_init():
    proj = RepaProjector(12, 32, 8)
    proj.reset_parameters()
    out = proj(torch.randn(5, 12))
    assert out.shape == (5, 8)


def test_cosine_loss_extremes():
    a = torch.randn(7, 5)
    loss, cos = repa_cosine_loss(a, a)
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(cos, torch.tensor(1.0), atol=1e-6, rtol=0)
    loss, cos = repa_cosine_loss(a, -a)
    torch.testing.assert_close(loss, torch.tensor(2.0), atol=1e-6, rtol=0)
    with pytest.raises(ValueError):
        repa_cosine_loss(a, a[:3])


def test_concat_views_along_width_layout():
    # view-major batch of 2 samples x 2 views, distinguishable by constant values
    b, v, d, t, h, wv = 2, 2, 3, 4, 5, 5
    y = torch.zeros(b * v, d, t, h, wv)
    for i in range(b * v):
        y[i] = i + 1  # sample s, view k -> value 2*s + k + 1
    out = concat_views_along_width(y, v)
    assert out.shape == (b, t, h, v * wv, d)
    assert torch.all(out[0, :, :, :wv] == 1) and torch.all(out[0, :, :, wv:] == 2)
    assert torch.all(out[1, :, :, :wv] == 3) and torch.all(out[1, :, :, wv:] == 4)


def test_split_flat_tokens_packing_order():
    shapes = [(5, 5, 10), (3, 2, 4)]
    n = sum(t * h * w for t, h, w in shapes)
    flat = torch.arange(n).unsqueeze(-1).float()
    grids = split_flat_tokens(flat, shapes)
    assert [tuple(g.shape) for g in grids] == [(5, 5, 10, 1), (3, 2, 4, 1)]
    # token (t,h,w) of sample 0 sits at t*50 + h*10 + w
    assert grids[0][2, 3, 4, 0].item() == 2 * 50 + 3 * 10 + 4
    assert grids[1][0, 0, 0, 0].item() == 250
    with pytest.raises(ValueError):
        split_flat_tokens(flat, [(5, 5, 10)])
