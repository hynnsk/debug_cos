# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.model.generator.repa.adapters import (
    AvgPoolConvTargetAdapter,
    AvgPoolTargetAdapter,
    RepaLinearProjector,
    RepaProjector,
    StridedConvTargetAdapter,
    build_projector,
    build_target_adapter,
    concat_views_along_width,
    repa_cosine_loss,
    repa_relation_loss,
    repa_spatial_normalized_cosine_loss,
    repa_temporal_difference_cosine_loss,
    split_flat_tokens,
    strided_conv_geometry,
    token_relation_matrix,
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


def test_linear_projector_is_a_single_affine_map():
    proj = RepaLinearProjector(12, 8)
    proj.reset_parameters()
    x = torch.randn(5, 12)
    out = proj(x)
    assert out.shape == (5, 8)
    torch.testing.assert_close(out, x @ proj.fc.weight.T + proj.fc.bias)
    assert sum(p.numel() for p in proj.parameters()) == 12 * 8 + 8
    # exactly one Linear, no activation
    assert [type(m) for m in proj.modules()] == [RepaLinearProjector, torch.nn.Linear]


def test_build_projector_selects_variant_and_rejects_unknown():
    mlp = build_projector("mlp", 12, 32, 8)
    lin = build_projector("linear", 12, 32, 8)  # hidden_dim ignored
    assert isinstance(mlp, RepaProjector) and isinstance(lin, RepaLinearProjector)
    assert mlp(torch.randn(3, 12)).shape == lin(torch.randn(3, 12)).shape == (3, 8)
    assert sum(p.numel() for p in mlp.parameters()) > sum(p.numel() for p in lin.parameters())
    with pytest.raises(ValueError, match="projector_type"):
        build_projector("conv", 12, 32, 8)


def test_cosine_loss_extremes():
    a = torch.randn(7, 5)
    loss, cos = repa_cosine_loss(a, a)
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(cos, torch.tensor(1.0), atol=1e-6, rtol=0)
    loss, cos = repa_cosine_loss(a, -a)
    torch.testing.assert_close(loss, torch.tensor(2.0), atol=1e-6, rtol=0)
    with pytest.raises(ValueError):
        repa_cosine_loss(a, a[:3])


def test_temporal_difference_loss_matches_same_patch_transitions_only():
    g = torch.Generator().manual_seed(4)
    frames, patches, dim = 3, 6, 8
    target = torch.randn(frames, patches, dim, generator=g)
    # A time-constant, patch-specific offset ruins absolute alignment but cancels in every temporal difference.
    pred = target + 5.0 * torch.randn(1, patches, dim, generator=g)
    frame_indexes = [torch.tensor([1, 2, 4])]
    # Frame 4 is not adjacent to frame 2 and must not be bridged; changing it cannot affect the transition loss.
    pred[2].mul_(100.0)
    pred.requires_grad_(True)
    loss, cos = repa_temporal_difference_cosine_loss(
        pred.flatten(0, 1), target.flatten(0, 1), [frames * patches], frame_indexes
    )
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(cos, torch.tensor(1.0), atol=1e-6, rtol=0)
    assert repa_cosine_loss(pred.flatten(0, 1), target.flatten(0, 1))[0] > 0.1
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_spatial_normalized_loss_uses_independent_per_frame_statistics():
    g = torch.Generator().manual_seed(5)
    frames, patches, dim = 4, 7, 6
    target = torch.randn(frames, patches, dim, generator=g)
    scale = torch.rand(frames, 1, dim, generator=g) + 0.2
    shift = 10.0 * torch.randn(frames, 1, dim, generator=g)
    pred = (target * scale + shift).requires_grad_(True)
    loss, cos = repa_spatial_normalized_cosine_loss(
        pred.flatten(0, 1),
        target.flatten(0, 1),
        [frames * patches],
        [torch.arange(1, frames + 1)],
    )
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=2e-6, rtol=0)
    torch.testing.assert_close(cos, torch.tensor(1.0), atol=2e-6, rtol=0)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def _relation_inputs(counts=(7, 7, 7), d=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = sum(counts)
    return torch.randn(n, d, generator=g, requires_grad=True), torch.randn(n, d, generator=g), list(counts)


def test_token_relation_matrix_is_cosine_similarity():
    x = torch.randn(4, 6, 3)
    rel = token_relation_matrix(x)
    assert rel.shape == (4, 6, 6)
    torch.testing.assert_close(rel.diagonal(dim1=-2, dim2=-1), torch.ones(4, 6), atol=1e-5, rtol=0)
    ref = torch.nn.functional.cosine_similarity(x[0, :, None], x[0, None, :], dim=-1)
    torch.testing.assert_close(rel[0], ref, atol=1e-5, rtol=0)
    assert token_relation_matrix(x.to(torch.bfloat16)).dtype == torch.float32  # computed in fp32


def test_relation_loss_extremes_and_invariances():
    pred, target, counts = _relation_inputs()
    zero = repa_relation_loss(target, target, counts)
    torch.testing.assert_close(zero, torch.tensor(0.0), atol=1e-6, rtol=0)
    # per-token rescaling does not change cosine relations; a global rotation of one side does not either,
    # whereas the direct cosine loss is NOT rotation invariant.
    q, _ = torch.linalg.qr(torch.randn(5, 5))
    scale = torch.rand(pred.shape[0], 1) * 3 + 0.1
    loss = repa_relation_loss(pred, target, counts)
    torch.testing.assert_close(repa_relation_loss((pred * scale) @ q, target, counts), loss, atol=1e-5, rtol=0)
    assert not torch.allclose(repa_cosine_loss(pred @ q, target)[0], repa_cosine_loss(pred, target)[0])
    # relation entries live in [-1, 1], so every entry-wise gap |d| <= 2: l2 = mean d^2 <= 2 * mean |d| = 2 * l1
    l1 = repa_relation_loss(pred, target, counts, distance="l1")
    l2 = repa_relation_loss(pred, target, counts, distance="l2")
    assert 0 < float(l2) <= 4 and 0 < float(l1) <= 2 and float(l2) <= 2 * float(l1)
    # explicit worst case on one 2-token sample: relations +1 vs -1 -> off-diagonal gaps of 2 -> l2 = (0+4+4+0)/4
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    b = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    torch.testing.assert_close(repa_relation_loss(a, b, [2]), torch.tensor(2.0), atol=1e-5, rtol=0)
    torch.testing.assert_close(repa_relation_loss(a, b, [2], distance="l1"), torch.tensor(1.0), atol=1e-5, rtol=0)


def test_relation_loss_batched_and_ragged_paths_agree_and_backprop():
    pred, target, counts = _relation_inputs(counts=(7, 7, 7))
    fast = repa_relation_loss(pred, target, counts)
    # ragged path: reference computed sample by sample
    parts = [
        (token_relation_matrix(p) - token_relation_matrix(t)).square().sum()
        for p, t in zip(torch.split(pred, counts), torch.split(target, counts))
    ]
    torch.testing.assert_close(fast, sum(parts) / (3 * 49), atol=1e-6, rtol=0)
    pred2, target2, counts2 = _relation_inputs(counts=(4, 0, 9), seed=1)  # zero-token sample is skipped
    ragged = repa_relation_loss(pred2, target2, counts2)
    parts2 = [
        (token_relation_matrix(p) - token_relation_matrix(t)).square().sum()
        for p, t in zip(torch.split(pred2, counts2), torch.split(target2, counts2))
        if p.shape[0] > 0
    ]
    torch.testing.assert_close(ragged, sum(parts2) / (16 + 81), atol=1e-6, rtol=0)
    (fast + ragged).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred2.grad is not None and torch.isfinite(pred2.grad).all()
    assert torch.count_nonzero(pred2.grad) > 0


def test_relation_loss_rejects_bad_inputs():
    pred, target, counts = _relation_inputs()
    with pytest.raises(ValueError, match="same shape"):
        repa_relation_loss(pred, target[:-1], counts)
    with pytest.raises(ValueError, match="sums to"):
        repa_relation_loss(pred, target, [7, 7])
    with pytest.raises(ValueError, match="distance"):
        repa_relation_loss(pred, target, counts, distance="cosine")
    with pytest.raises(ValueError, match="at least one"):
        repa_relation_loss(pred[:0], target[:0], [0, 0])


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


def test_centered_cosine_detects_constant_shortcut():
    from cosmos_framework.model.generator.repa.adapters import centered_cosine_similarity

    g = torch.Generator().manual_seed(0)
    shared = torch.randn(1, 16, generator=g) * 10
    target = shared + torch.randn(64, 16, generator=g)  # big common component + small token-specific part
    constant_pred = shared.expand(64, 16)
    _, raw = repa_cosine_loss(constant_pred, target)
    assert raw > 0.9  # the raw cosine is fooled by the shared direction
    assert abs(centered_cosine_similarity(constant_pred, target).item()) < 1e-2
    assert centered_cosine_similarity(target, target).item() > 0.999


def test_centered_cosine_loss_ignores_shared_direction_and_has_grad():
    from cosmos_framework.model.generator.repa.adapters import repa_centered_cosine_loss

    g = torch.Generator().manual_seed(1)
    shared = torch.randn(1, 16, generator=g) * 10
    dev = torch.randn(64, 16, generator=g)
    target = shared + dev
    # constant prediction: raw loss ~0.1 but centered loss = 1 (no token-specific structure matched)
    const = shared.expand(64, 16).clone().requires_grad_(True)
    loss, cos = repa_centered_cosine_loss(const, target)
    assert abs(loss.item() - 1.0) < 1e-2 and abs(cos.item()) < 1e-2
    # a near-constant prediction receives a finite gradient towards the deviations
    noisy = (shared + 1e-2 * torch.randn(64, 16, generator=g)).requires_grad_(True)
    loss, _ = repa_centered_cosine_loss(noisy, target)
    loss.backward()
    assert noisy.grad is not None and torch.isfinite(noisy.grad).all() and noisy.grad.abs().sum() > 0
    # matching the deviations with any offset / scale is perfect
    perfect = 2.5 * dev + 3.0
    loss, cos = repa_centered_cosine_loss(perfect, target)
    assert loss.item() < 1e-4 and cos.item() > 0.999
