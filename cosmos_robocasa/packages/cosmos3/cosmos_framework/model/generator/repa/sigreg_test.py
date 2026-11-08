# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.model.generator.repa.sigreg import sigreg_loss


def test_sigreg_prefers_isotropic_gaussian_to_collapsed_tokens():
    g = torch.Generator().manual_seed(0)
    gaussian = torch.randn(4096, 16, generator=g)
    collapsed = torch.zeros_like(gaussian)
    kwargs = dict(num_slices=64, num_points=17, slice_batch_size=16, seed=7)
    assert sigreg_loss(gaussian, **kwargs) < sigreg_loss(collapsed, **kwargs)


def test_sigreg_is_deterministic_for_a_seed_and_backpropagates():
    x = torch.randn(256, 12, generator=torch.Generator().manual_seed(1), requires_grad=True)
    kwargs = dict(num_slices=24, num_points=9, integration_max=3.0, slice_batch_size=8, seed=11)
    loss = sigreg_loss(x, **kwargs)
    torch.testing.assert_close(loss, sigreg_loss(x, **kwargs))
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad) > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_slices": 0},
        {"num_points": 1},
        {"integration_max": 0.0},
        {"slice_batch_size": 0},
    ],
)
def test_sigreg_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError, match="SIGReg requires"):
        sigreg_loss(torch.randn(8, 4), **kwargs)


def test_sigreg_rejects_non_matrix_input():
    with pytest.raises(ValueError, match=r"\[N,D\]"):
        sigreg_loss(torch.randn(2, 3, 4))


def test_sigreg_normalize_by_count_is_bounded_and_batch_size_invariant():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(512, 32, generator=g) * 5.0 + 3.0  # far from N(0, 1)
    raw = sigreg_loss(x, num_slices=64, slice_batch_size=64, seed=1)
    per_token = sigreg_loss(x, num_slices=64, slice_batch_size=64, seed=1, normalize_by_count=True)
    torch.testing.assert_close(per_token * x.shape[0], raw)
    assert 0.0 < per_token.item() < 1.8  # integral of |ecf - cf|^2 w on [-5, 5] is bounded by ~sqrt(pi)
    # duplicating the tokens leaves the per-token integral unchanged but doubles the N-scaled statistic
    xx = torch.cat([x, x], dim=0)
    torch.testing.assert_close(sigreg_loss(xx, num_slices=64, slice_batch_size=64, seed=1, normalize_by_count=True), per_token)
    torch.testing.assert_close(sigreg_loss(xx, num_slices=64, slice_batch_size=64, seed=1), 2 * raw)
    # Gaussian tokens score far lower than the shifted/scaled ones
    gauss = sigreg_loss(torch.randn(512, 32, generator=g), num_slices=64, slice_batch_size=64, seed=1, normalize_by_count=True)
    assert gauss.item() < 0.1 * per_token.item()
