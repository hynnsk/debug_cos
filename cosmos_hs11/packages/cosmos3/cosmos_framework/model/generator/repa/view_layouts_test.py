# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests of the composite-canvas view layouts of the REPA teacher (cosmos_hs11 v11)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.model.generator.repa.view_layouts import (
    VIEW_LAYOUTS,
    compose_view_grids,
    crop_and_resize_views,
    crop_view,
    layout_canvas_grid,
    resolve_view_layout,
    validate_repa_view_layouts,
    validate_view_layout,
)

P2 = VIEW_LAYOUTS["primary_over_two"]


def test_resolve_names_and_explicit_boxes():
    assert resolve_view_layout(None) is None and resolve_view_layout("") is None and resolve_view_layout("canvas") is None
    assert resolve_view_layout("primary_over_two") == P2
    assert resolve_view_layout([[0, 1, 0, 0.5], [0, 1, 0.5, 1]]) == ((0.0, 1.0, 0.0, 0.5), (0.0, 1.0, 0.5, 1.0))
    with pytest.raises(ValueError, match="unknown"):
        resolve_view_layout("nope")


def test_validate_rejects_gaps_overlaps_and_bad_boxes():
    validate_view_layout(P2)
    with pytest.raises(ValueError, match="tile"):
        validate_view_layout([(0, 0.5, 0, 1)])
    with pytest.raises(ValueError, match="overlap"):
        validate_view_layout([(0, 1, 0, 0.6), (0, 1, 0.4, 0.8)])  # area 1.0 but the boxes overlap on x in [0.4, 0.6]
    with pytest.raises(ValueError, match="rectangle"):
        validate_view_layout([(0.5, 0.5, 0, 1), (0, 1, 0, 1)])


def test_canvas_grid_of_primary_over_two_is_24x16_at_16_patches():
    assert layout_canvas_grid(P2, (16, 16)) == (24, 16)
    assert layout_canvas_grid([(0, 1, 0, 0.5), (0, 1, 0.5, 1)], (16, 16)) == (16, 32)  # LIBERO-like width split
    with pytest.raises(ValueError, match="whole"):
        layout_canvas_grid([(0, 0.3, 0, 1), (0.3, 1, 0, 1)], (16, 16))  # 0.3 of a 53-cell grid is not whole


def test_crop_view_pixels_match_compose_multiview_layout():
    frames = torch.zeros(3, 4, 540, 640, dtype=torch.uint8)
    top, left, right = (crop_view(frames, b) for b in P2)
    assert tuple(top.shape[-2:]) == (360, 640)
    assert tuple(left.shape[-2:]) == (180, 320) and tuple(right.shape[-2:]) == (180, 320)
    views = crop_and_resize_views(frames, P2, 32)
    assert [tuple(v.shape) for v in views] == [(3, 4, 32, 32)] * 3 and all(v.dtype == torch.uint8 for v in views)


def test_compose_places_each_view_in_its_cells_and_pools_the_small_ones():
    t, d = 2, 3
    grids = [torch.full((t, 16, 16, d), float(v + 1)) for v in range(3)]  # top = 1, left wrist = 2, right wrist = 3
    canvas = compose_view_grids(grids, P2)
    assert canvas.shape == (t, 24, 16, d)
    assert torch.equal(canvas[:, :16, :], torch.ones(t, 16, 16, d))
    assert torch.equal(canvas[:, 16:, :8], torch.full((t, 8, 8, d), 2.0))
    assert torch.equal(canvas[:, 16:, 8:], torch.full((t, 8, 8, d), 3.0))
    # pooling is an exact area average: a 2x2 checkerboard wrist grid pools to its mean
    wrist = torch.zeros(t, 16, 16, d)
    wrist[:, ::2, ::2] = 4.0
    canvas = compose_view_grids([grids[0], wrist, grids[2]], P2)
    assert torch.allclose(canvas[:, 16:, :8], torch.full((t, 8, 8, d), 1.0))
    with pytest.raises(ValueError, match="view grids"):
        compose_view_grids(grids[:2], P2)


def test_validate_repa_view_layouts_prerequisites():
    def cfg(**kw):
        base = dict(view_layouts={"molmoact2_yam": "primary_over_two", "fractal": "canvas"}, num_views=1, target_adapter="avgpool", student_upsampler="none", objective="token")
        base.update(kw)
        return SimpleNamespace(**base)

    assert validate_repa_view_layouts(cfg()) == {"molmoact2_yam": "primary_over_two"}
    assert validate_repa_view_layouts(cfg(view_layouts={})) == {} and validate_repa_view_layouts(cfg(view_layouts={"a": "canvas"})) == {}
    assert validate_repa_view_layouts(SimpleNamespace(num_views=2)) == {}  # old configs without the field
    for bad in (dict(num_views=2), dict(target_adapter="strided_conv"), dict(student_upsampler="trilinear"), dict(objective="masked_prediction")):
        with pytest.raises(ValueError):
            validate_repa_view_layouts(cfg(**bad))
