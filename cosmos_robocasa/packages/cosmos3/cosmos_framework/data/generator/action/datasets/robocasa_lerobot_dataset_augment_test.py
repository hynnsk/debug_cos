# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""cosmos_hs11: train-time image augmentation of RoboCasaLeRobotDataset (the DROID / GR00T crop + ColorJitter recipe)."""

import pytest
import torch

from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import RoboCasaLeRobotDataset

_IMAGE_FEATURES = {"wrist": "wrist_key", "left": "left_key", "right": "right_key"}


def _dataset(use_image_augmentation: bool, jitter_after_compose: bool = False) -> RoboCasaLeRobotDataset:
    """Bare instance for the compose methods only (``__init__`` opens LeRobot shards, which this test does not need)."""
    ds = object.__new__(RoboCasaLeRobotDataset)
    ds._image_features = _IMAGE_FEATURES
    ds._use_image_augmentation = use_image_augmentation
    ds._jitter_after_compose = jitter_after_compose
    ds._image_augmentor = None
    ds._color_augmentor = None
    ds._augment_crop_scale = 0.95
    ds._augment_rotation_deg = 0.0
    ds._augment_hue = 0.08
    return ds


def _gradient(t: int = 3, h: int = 32, w: int = 32) -> torch.Tensor:
    """A non-uniform frame so a crop offset is visible: value = x / w, same for every frame and channel."""
    ramp = torch.linspace(0.0, 1.0, w).view(1, 1, 1, w).expand(t, 3, h, w)
    return ramp.clone()


@pytest.mark.L0
def test_left_wrist_without_augmentation_is_the_plain_concat() -> None:
    sample = {"left_key": torch.full((2, 3, 16, 16), 0.2), "wrist_key": torch.full((2, 3, 16, 16), 0.8)}
    out = _dataset(use_image_augmentation=False)._compose_left_wrist(sample)
    assert out.shape == (2, 3, 16, 32)
    torch.testing.assert_close(out[..., :16], torch.full((2, 3, 16, 16), 0.2))
    torch.testing.assert_close(out[..., 16:], torch.full((2, 3, 16, 16), 0.8))


@pytest.mark.L0
@pytest.mark.parametrize("jitter_after_compose", [False, True])
def test_left_wrist_augmentation_keeps_shape_and_uses_one_crop_for_both_views(jitter_after_compose: bool) -> None:
    torch.manual_seed(0)
    ds = _dataset(use_image_augmentation=True, jitter_after_compose=jitter_after_compose)
    g = _gradient()
    out = ds._compose_left_wrist({"left_key": g.clone(), "wrist_key": g.clone()})
    assert out.shape == (3, 3, 32, 64)
    # identical inputs -> identical outputs: the crop window and the colour factors are drawn ONCE per window
    torch.testing.assert_close(out[..., :32], out[..., 32:])
    # every frame of a view got the same crop too
    torch.testing.assert_close(out[0], out[1])
    # and it is not the identity (crop 95 % + resize moves the ramp; jitter changes values)
    assert not torch.allclose(out[..., :32], g)


@pytest.mark.L0
def test_multi_view_augmentation_keeps_tiling() -> None:
    torch.manual_seed(1)
    ds = _dataset(use_image_augmentation=True)
    sample = {
        "wrist_key": torch.full((2, 3, 16, 32), 0.1),
        "left_key": torch.full((2, 3, 16, 32), 0.5),
        "right_key": torch.full((2, 3, 16, 32), 0.9),
    }
    out = ds._compose_multi_view(sample)
    assert out.shape == (2, 3, 24, 32)
    for region in (out[:, :, :16, :], out[:, :, 16:, :16], out[:, :, 16:, 16:]):
        assert region.std() < 1e-4  # uniform tiles stay uniform: no bleeding across the seams


@pytest.mark.L0
def test_augmentor_is_built_once_and_off_path_is_untouched() -> None:
    ds = _dataset(use_image_augmentation=False)
    ds._compose_left_wrist({"left_key": torch.zeros(1, 3, 8, 8), "wrist_key": torch.zeros(1, 3, 8, 8)})
    assert ds._image_augmentor is None and ds._color_augmentor is None
    ds = _dataset(use_image_augmentation=True)
    ds._compose_left_wrist({"left_key": torch.zeros(1, 3, 8, 8), "wrist_key": torch.zeros(1, 3, 8, 8)})
    first = ds._image_augmentor
    ds._compose_left_wrist({"left_key": torch.zeros(1, 3, 8, 8), "wrist_key": torch.zeros(1, 3, 8, 8)})
    assert ds._image_augmentor is first


@pytest.mark.L0
def test_cosmos_policy_stronger_aug_adds_one_rotation_per_window() -> None:
    """cosmos_robocasa: augment_rotation_deg > 0 appends a RandomRotation (one angle per window, applied to every frame of
    every view); the default 0.0 keeps the transform list byte-identical to the cosmos_hs11 recipe."""
    import torchvision.transforms.v2 as T

    torch.manual_seed(2)
    ds = _dataset(use_image_augmentation=True)
    ds._augment_rotation_deg = 5.0
    ds._augment_hue = 0.05
    g = _gradient()
    out = ds._compose_left_wrist({"left_key": g.clone(), "wrist_key": g.clone()})
    assert out.shape == (3, 3, 32, 64)
    kinds = [type(t) for t in ds._image_augmentor.transforms]
    assert kinds == [T.RandomCrop, T.Resize, T.RandomRotation, T.ColorJitter]
    assert ds._image_augmentor.transforms[3].hue == (-0.05, 0.05)
    torch.testing.assert_close(out[..., :32], out[..., 32:])  # same angle / crop / colour for both views ...
    torch.testing.assert_close(out[0], out[1])  # ... and for every frame
    plain = _dataset(use_image_augmentation=True)
    plain._compose_left_wrist({"left_key": g.clone(), "wrist_key": g.clone()})
    assert [type(t) for t in plain._image_augmentor.transforms] == [T.RandomCrop, T.Resize, T.ColorJitter]
