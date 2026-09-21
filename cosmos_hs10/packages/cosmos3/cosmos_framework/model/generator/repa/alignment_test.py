# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.model.generator.repa.adapters import (
    RepaLinearProjector,
    RepaProjector,
    repa_cosine_loss,
    repa_relation_loss,
    token_relation_matrix,
)
from cosmos_framework.model.generator.repa.alignment import RepaAlignmentHead

D_MOT, D_T = 16, 6
TEACHER_GRID = (8, 16, 16)
TOKEN_SHAPE = (5, 5, 10)  # LIBERO-10 concat_view: 5 latent frames x 5 x 10 tokens (two 5x5 views)


def _head(variant: str = "avgpool", projector_type: str = "mlp") -> RepaAlignmentHead:
    head = RepaAlignmentHead(
        hidden_size=D_MOT,
        teacher_embed_dim=D_T,
        projector_hidden_dim=32,
        projector_type=projector_type,
        target_adapter=variant,
        teacher_grid_thw=TEACHER_GRID,
        target_grid_thw=(4, 5, 5),
        num_views=2,
    )
    head.reset_parameters()
    return head


def _inputs(batch: int = 3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    t, h, w = TOKEN_SHAPE
    hidden = torch.randn(batch * t * h * w, D_MOT, generator=g, requires_grad=True)
    teacher = [torch.randn(2, *TEACHER_GRID, D_T, generator=g) for _ in range(batch)]
    nfi = [torch.arange(1, t) for _ in range(batch)]
    shapes = [TOKEN_SHAPE] * batch
    return hidden, shapes, nfi, teacher


@pytest.mark.parametrize("variant", ["avgpool", "avgpool_conv", "strided_conv"])
def test_forward_shapes_and_counts(variant):
    head = _head(variant)
    hidden, shapes, nfi, teacher = _inputs()
    out = head(hidden, shapes, nfi, teacher)
    n = 3 * 4 * 5 * 10  # 3 samples x 4 predicted frames x 5 x 10 tokens
    assert out["pred"].shape == (n, D_T) and out["target"].shape == (n, D_T)
    assert out["num_tokens_per_sample"] == [200, 200, 200]
    assert out["empty"] is False
    loss, _ = repa_cosine_loss(out["pred"], out["target"])
    loss.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert all(p.grad is not None for p in head.projector.parameters())
    if variant != "avgpool":
        assert all(p.grad is not None for p in head.target_adapter.parameters())


def test_projector_type_selects_mlp_or_linear_head():
    assert isinstance(_head().projector, RepaProjector)  # default = REPA MLP
    head = _head("avgpool", projector_type="linear")
    assert isinstance(head.projector, RepaLinearProjector) and head.projector_type == "linear"
    hidden, shapes, nfi, teacher = _inputs()
    out = head(hidden, shapes, nfi, teacher)
    n = 3 * 4 * 5 * 10
    assert out["pred"].shape == (n, D_T) and out["target"].shape == (n, D_T)
    # pred is exactly the affine image of the selected tokens (no hidden layer / nonlinearity in between)
    rows = torch.cat([g.index_select(0, i).reshape(-1, D_MOT) for g, i in zip(hidden.view(3, 5, 5, 10, D_MOT), nfi)])
    torch.testing.assert_close(out["pred"], head.projector.fc(rows))
    loss, _ = repa_cosine_loss(out["pred"], out["target"])
    loss.backward()
    assert all(p.grad is not None for p in head.projector.parameters())
    # the probe path (no alignable tokens) still touches the linear projector's parameters
    probe = head.probe(hidden.device, hidden.dtype)
    assert probe.requires_grad and float(probe) == 0.0
    with pytest.raises(ValueError, match="projector_type"):
        _head("avgpool", projector_type="conv")


def test_relation_loss_consumes_head_outputs_per_sample():
    head = _head("avgpool")
    hidden, shapes, nfi, teacher = _inputs()
    out = head(hidden, shapes, nfi, teacher)
    counts = out["num_tokens_per_sample"]
    loss = repa_relation_loss(out["pred"], out["target"], counts)
    # relation maps are formed within each sample: 3 windows x (4*5*10)^2 entries
    per_sample = [
        (token_relation_matrix(p) - token_relation_matrix(t)).square().mean()
        for p, t in zip(torch.split(out["pred"], counts), torch.split(out["target"], counts))
    ]
    torch.testing.assert_close(loss, torch.stack(per_sample).mean(), atol=1e-6, rtol=0)
    # a teacher-matching student (pred == target) gives 0 for the relation term
    torch.testing.assert_close(
        repa_relation_loss(out["target"], out["target"], counts), torch.tensor(0.0), atol=1e-6, rtol=0
    )
    loss.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert all(p.grad is not None for p in head.projector.parameters())


def test_target_geometry_views_frames_and_noisy_selection():
    head = _head("avgpool")
    t, h, w = TOKEN_SHAPE
    # teacher tokens encode (sample, view, tubelet-pair index) so the adapted target is checkable exactly
    teacher = []
    for s in range(2):
        tt = torch.zeros(2, *TEACHER_GRID, D_T)
        for v in range(2):
            for tub in range(8):
                tt[v, tub, ..., 0] = s  # sample id
                tt[v, tub, ..., 1] = v  # view id
                tt[v, tub, ..., 2] = tub // 2  # latent frame index - 1 (pairs of tubelets)
        teacher.append(tt)
    hidden = torch.randn(2 * t * h * w, D_MOT)
    nfi = [torch.tensor([1, 2, 3, 4]), torch.tensor([2, 4])]  # sample 1: only some frames noised
    out = head(hidden, [TOKEN_SHAPE] * 2, nfi, teacher)
    assert out["num_tokens_per_sample"] == [4 * 50, 2 * 50]
    target = out["target"]
    # sample 0: rows ordered (frame, h, w); frame k (k=1..4) -> channel2 == k-1; left half view 0, right half view 1
    t0 = target[: 4 * 50].view(4, h, w, D_T)
    assert torch.all(t0[..., 0] == 0)
    assert torch.all(t0[:, :, :5, 1] == 0) and torch.all(t0[:, :, 5:, 1] == 1)
    for k in range(4):
        assert torch.all(t0[k, ..., 2] == k)
    # sample 1: frames 2 and 4 -> channel2 == 1 and 3
    t1 = target[4 * 50 :].view(2, h, w, D_T)
    assert torch.all(t1[..., 0] == 1)
    assert torch.all(t1[0, ..., 2] == 1) and torch.all(t1[1, ..., 2] == 3)
    # prediction rows are the projector applied to the selected hidden tokens (frame 2 of sample 1 = grid[2])
    grid1 = hidden[t * h * w :].view(t, h, w, D_MOT)
    expected = head.projector(grid1[2].reshape(-1, D_MOT))
    torch.testing.assert_close(out["pred"][4 * 50 : 4 * 50 + 50], expected)


def test_non_uniform_shapes_fall_back_to_per_sample_path():
    head = _head("avgpool")
    shapes = [(5, 5, 10), (3, 4, 8)]
    hidden = torch.randn(sum(t * h * w for t, h, w in shapes), D_MOT)
    teacher = [torch.randn(2, *TEACHER_GRID, D_T) for _ in shapes]
    nfi = [torch.arange(1, 5), torch.arange(1, 3)]
    out = head(hidden, shapes, nfi, teacher)
    assert out["num_tokens_per_sample"] == [4 * 50, 2 * 32]
    assert out["pred"].shape == (4 * 50 + 2 * 32, D_T)


def test_rejects_conditioning_frame_and_bad_widths():
    head = _head("avgpool")
    hidden, shapes, nfi, teacher = _inputs(batch=1)
    with pytest.raises(ValueError, match="conditioning frame"):
        head(hidden, shapes, [torch.tensor([0, 1])], teacher)
    with pytest.raises(ValueError, match="num_views"):
        head(hidden[: 5 * 5 * 9], [(5, 5, 9)], nfi, teacher)


def test_empty_batch_returns_probe_with_grad():
    head = _head("avgpool_conv")
    hidden, shapes, _, teacher = _inputs(batch=1)
    out = head(hidden, shapes, [torch.tensor([], dtype=torch.long)], teacher)
    assert out["empty"] is True
    (0.0 * out["pred"].sum()).backward()
    assert all(p.grad is not None for p in head.parameters())


def test_probe_touches_every_parameter():
    head = _head("strided_conv")
    probe = head.probe(torch.device("cpu"), torch.float32)
    probe.backward()
    assert all(p.grad is not None for p in head.parameters())


def test_batched_teacher_tensor_matches_list_input():
    head = _head("avgpool")
    hidden, shapes, nfi, teacher = _inputs(batch=2, seed=3)
    out_list = head(hidden, shapes, nfi, teacher)
    out_batched = head(hidden, shapes, nfi, torch.stack(teacher, dim=0))
    torch.testing.assert_close(out_list["target"], out_batched["target"])
    torch.testing.assert_close(out_list["pred"], out_batched["pred"])
