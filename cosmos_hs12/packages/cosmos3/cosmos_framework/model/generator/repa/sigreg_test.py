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
