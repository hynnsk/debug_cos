# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Per-embodiment camera layouts of a composite canvas for the REPA teacher (cosmos_hs11 v11).

The stock REPA path splits the native canvas into ``num_views`` equal slices along the WIDTH (the LIBERO
third-person | wrist layout). Some source embodiments ship a 2 x 2 composite instead -- ``compose_multiview``
puts the primary camera on top and two half-sized side / wrist cameras below it (MolmoAct2-YAM, RoboMIND
``concat_view``) -- which a width split cannot separate. A *layout* lists the rectangles the views occupy as
fractions of the canvas ``(y0, y1, x0, x1)``; the teacher then encodes every view separately at its own input
resolution (a full patch grid per view) and the per-view grids are placed back into one canvas-shaped teacher
grid (:func:`compose_view_grids`), which the ordinary ``avgpool`` adapter pools onto the MoT token grid of the
whole canvas. For the alignment head the canvas therefore stays ONE view (``repa.num_views = 1``).

Canvas grid: the largest view keeps the teacher's native patch grid (16 x 16 at 224 px), smaller views are
area-pooled into their (smaller) cells -- e.g. ``primary_over_two`` -> 24 x 16 cells: the primary view fills the
top 16 x 16, each wrist view the bottom 8 x 8 (the wrists are also a quarter of the canvas area for the MoT).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

Box = tuple[float, float, float, float]  # (y0, y1, x0, x1) as fractions of the canvas height / width

_TOL = 1e-3

# Named layouts usable from the TOML (``[model.repa.view_layouts] <embodiment> = "<name>"``).
VIEW_LAYOUTS: dict[str, tuple[Box, ...]] = {
    # the whole canvas is one teacher image (identical to num_views = 1 without a layout)
    "canvas": (),
    # cosmos_framework.data.generator.action.utils.viewpoint_utils.compose_multiview: primary view [T,C,H,W] above two
    # half-sized views [T,C,H/2,W/2] side by side -> canvas 3H/2 x W. MolmoAct2-YAM (top | left wrist, right wrist),
    # RoboMIND concat_view (top | left, right).
    "primary_over_two": ((0.0, 2.0 / 3.0, 0.0, 1.0), (2.0 / 3.0, 1.0, 0.0, 0.5), (2.0 / 3.0, 1.0, 0.5, 1.0)),
}


def resolve_view_layout(spec: str | Sequence[Sequence[float]] | None) -> tuple[Box, ...] | None:
    """A layout name (``VIEW_LAYOUTS``) or an explicit box sequence -> validated boxes; ``None`` / ``""`` /
    ``"canvas"`` -> ``None`` (= the stock path)."""
    if spec is None:
        return None
    if isinstance(spec, str):
        name = spec.strip()
        if name == "" or name == "canvas":
            return None
        if name not in VIEW_LAYOUTS:
            raise ValueError(f"unknown REPA view layout {spec!r}; known: {sorted(VIEW_LAYOUTS)}")
        boxes = VIEW_LAYOUTS[name]
    else:
        boxes = tuple(tuple(float(v) for v in b) for b in spec)  # type: ignore[assignment]
    if not boxes:
        return None
    validate_view_layout(boxes)
    return boxes


def validate_view_layout(boxes: Sequence[Box]) -> None:
    """Every box inside [0, 1], non-empty, and the boxes tile the canvas exactly (no gap / overlap)."""
    if len(boxes) == 0:
        raise ValueError("a view layout needs at least one box")
    area = 0.0
    for b in boxes:
        if len(b) != 4:
            raise ValueError(f"a view box is (y0, y1, x0, x1), got {b}")
        y0, y1, x0, x1 = (float(v) for v in b)
        if not (0.0 <= y0 < y1 <= 1.0 + _TOL and 0.0 <= x0 < x1 <= 1.0 + _TOL):
            raise ValueError(f"view box {b} is not a non-empty rectangle inside the unit canvas")
        area += (y1 - y0) * (x1 - x0)
    if abs(area - 1.0) > _TOL:
        raise ValueError(f"view boxes must tile the whole canvas (total area {area:.4f} != 1)")
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            oy = min(a[1], b[1]) - max(a[0], b[0])
            ox = min(a[3], b[3]) - max(a[2], b[2])
            if oy > _TOL and ox > _TOL:
                raise ValueError(f"view boxes {a} and {b} overlap")


def _cells(frac0: float, frac1: float, n: int, what: str) -> tuple[int, int]:
    a, b = frac0 * n, frac1 * n
    ia, ib = int(round(a)), int(round(b))
    if abs(a - ia) > _TOL or abs(b - ib) > _TOL:
        raise ValueError(f"view box edge {frac0:.4f}-{frac1:.4f} does not fall on whole {what} cells of a {n}-cell grid")
    if ib <= ia:
        raise ValueError(f"view box {frac0:.4f}-{frac1:.4f} covers no {what} cell of a {n}-cell grid")
    return ia, ib


def layout_canvas_grid(boxes: Sequence[Box], patches: tuple[int, int]) -> tuple[int, int]:
    """Canvas teacher grid ``(Hc, Wc)`` such that the largest view keeps the teacher's ``patches = (Hp, Wp)`` and every
    box maps onto whole cells (``primary_over_two`` at 16 x 16 -> 24 x 16)."""
    hp, wp = int(patches[0]), int(patches[1])
    max_h = max(float(b[1]) - float(b[0]) for b in boxes)
    max_w = max(float(b[3]) - float(b[2]) for b in boxes)
    hc, wc = int(round(hp / max_h)), int(round(wp / max_w))
    for b in boxes:  # raises when an edge is not a whole cell
        _cells(b[0], b[1], hc, "row")
        _cells(b[2], b[3], wc, "column")
    return hc, wc


def crop_view(frames: torch.Tensor, box: Box) -> torch.Tensor:
    """``[..., H, W]`` canvas pixels -> the view rectangle (edges rounded to whole pixels)."""
    h, w = int(frames.shape[-2]), int(frames.shape[-1])
    y0, y1 = int(round(float(box[0]) * h)), int(round(float(box[1]) * h))
    x0, x1 = int(round(float(box[2]) * w)), int(round(float(box[3]) * w))
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"view box {box} is empty on a {h}x{w} canvas")
    return frames[..., y0:y1, x0:x1]


def crop_and_resize_views(frames: torch.Tensor, boxes: Sequence[Box], size: int) -> list[torch.Tensor]:
    """``[C,T,H,W]`` uint8 canvas frames -> one ``[C,T,size,size]`` uint8 clip per box (antialiased bilinear, the
    teacher's own resize; views of different pixel sizes become stackable teacher inputs)."""
    if frames.ndim != 4:
        raise ValueError(f"expected [C,T,H,W] frames, got {tuple(frames.shape)}")
    out: list[torch.Tensor] = []
    size = int(size)
    for box in boxes:
        view = crop_view(frames, box)  # [C,T,h,w]
        if tuple(view.shape[-2:]) == (size, size):
            out.append(view.contiguous())
            continue
        x = view.permute(1, 0, 2, 3).float()  # [T,C,h,w]
        y = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        if frames.dtype == torch.uint8:
            y = y.round_().clamp_(0, 255).to(torch.uint8)
        else:
            y = y.to(frames.dtype)
        out.append(y.permute(1, 0, 2, 3).contiguous())
    return out


def validate_repa_view_layouts(repa_cfg: Any) -> dict[str, str]:
    """The non-trivial entries of ``repa.view_layouts`` (validated names) and the prerequisites of the composite path:
    one canvas view for the alignment head, an avgpool-type adapter (the canvas grid is not the teacher grid) and no
    student upsampler / masked prediction. Empty dict = stock path, nothing checked."""
    layouts = getattr(repa_cfg, "view_layouts", None) or {}
    active = {str(k): str(v) for k, v in dict(layouts).items() if resolve_view_layout(v) is not None}
    if not active:
        return {}
    problems = []
    if int(repa_cfg.num_views) != 1:
        problems.append(f"num_views must be 1 (the composite canvas is one view for the head), got {repa_cfg.num_views}")
    if str(repa_cfg.target_adapter) == "strided_conv":
        problems.append("target_adapter='strided_conv' needs the fixed teacher grid; use 'avgpool' / 'avgpool_conv'")
    if str(getattr(repa_cfg, "student_upsampler", "none")) != "none":
        problems.append("student_upsampler expects the raw teacher grid; set it to 'none'")
    if str(repa_cfg.objective) == "masked_prediction":
        problems.append("objective='masked_prediction' builds its own targets; view layouts are not applied there")
    if problems:
        raise ValueError("repa.view_layouts " + str(active) + ": " + "; ".join(problems))
    return active


def _resize_grid(grid: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """``[T, Hp, Wp, D]`` -> ``[T, bh, bw, D]``: area pooling when shrinking (or equal), bilinear when enlarging."""
    t, hp, wp, d = grid.shape
    if (hp, wp) == tuple(size):
        return grid
    x = grid.permute(0, 3, 1, 2)  # [T, D, Hp, Wp]
    orig_dtype = x.dtype
    if not x.is_cuda:
        x = x.float()
    if size[0] <= hp and size[1] <= wp:
        y = F.adaptive_avg_pool2d(x, tuple(int(v) for v in size))
    else:
        y = F.interpolate(x.float(), size=tuple(int(v) for v in size), mode="bilinear", align_corners=False)
    return y.to(orig_dtype).permute(0, 2, 3, 1)


def compose_view_grids(
    grids: Sequence[torch.Tensor], boxes: Sequence[Box], canvas_grid: tuple[int, int] | None = None
) -> torch.Tensor:
    """Per-view teacher grids ``[T, Hp, Wp, D]`` (one per box, same order) -> one canvas grid ``[T, Hc, Wc, D]`` with
    every view resized into its box of cells."""
    if len(grids) != len(boxes):
        raise ValueError(f"{len(grids)} view grids for {len(boxes)} boxes")
    if any(g.ndim != 4 for g in grids):
        raise ValueError("every view grid must be [T, Hp, Wp, D]")
    patches = (int(grids[0].shape[1]), int(grids[0].shape[2]))
    if any((int(g.shape[1]), int(g.shape[2])) != patches or int(g.shape[0]) != int(grids[0].shape[0]) for g in grids):
        raise ValueError("all view grids of a sample must share the teacher grid [T, Hp, Wp]")
    hc, wc = canvas_grid if canvas_grid is not None else layout_canvas_grid(boxes, patches)
    t, d = int(grids[0].shape[0]), int(grids[0].shape[-1])
    out = grids[0].new_zeros((t, hc, wc, d))
    filled = torch.zeros((hc, wc), dtype=torch.bool)
    for grid, box in zip(grids, boxes):
        r0, r1 = _cells(box[0], box[1], hc, "row")
        c0, c1 = _cells(box[2], box[3], wc, "column")
        out[:, r0:r1, c0:c1] = _resize_grid(grid, (r1 - r0, c1 - c0)).to(out.dtype)
        filled[r0:r1, c0:c1] = True
    if not bool(filled.all()):
        raise ValueError("view boxes leave canvas cells uncovered")
    return out
