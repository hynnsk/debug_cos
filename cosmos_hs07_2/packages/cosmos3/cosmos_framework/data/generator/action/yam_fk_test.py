# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests: URDF chain FK for the YAM arm vs. MuJoCo on the matching arm-only MJCF."""

import numpy as np
import pytest

from cosmos_framework.data.generator.action.yam_fk import (
    YAM_JOINT_DIM,
    YAM_MJCF_PATH,
    YAM_MOUNT_TO_TCP,
    bimanual_yam_ee_poses,
    get_yam_arm_fk,
    yam_ee_poses,
    yam_mount_poses,
)


def test_chain_and_home_pose() -> None:
    fk = get_yam_arm_fk()
    assert fk.num_joints == 6
    home = yam_mount_poses(np.zeros((1, 6)))[0]
    # Home pose M from the i2rt README (product-of-exponentials section).
    expected = np.array(
        [
            [0.0, 0.0, -1.0, 0.110597],
            [0.0, -1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.173502],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(home, expected, atol=2e-5)


def test_ee_pose_is_rigid_and_z_is_approach() -> None:
    rng = np.random.default_rng(0)
    q = rng.uniform(-1.0, 1.0, size=(8, 6))
    ee = yam_ee_poses(q)
    assert ee.shape == (8, 4, 4)
    rots = ee[:, :3, :3].astype(np.float64)
    np.testing.assert_allclose(np.einsum("nij,nik->njk", rots, rots), np.tile(np.eye(3), (8, 1, 1)), atol=1e-5)
    np.testing.assert_allclose(np.linalg.det(rots), 1.0, atol=1e-5)
    # The tool centre sits 0.14465 m in front of the mount frame along the fingers, and the
    # re-referenced +z axis points from the mount towards the fingertips.
    mount = yam_mount_poses(q)
    offset = ee[:, :3, 3] - mount[:, :3, 3]
    np.testing.assert_allclose(np.linalg.norm(offset, axis=-1), np.linalg.norm(YAM_MOUNT_TO_TCP[:3, 3]), atol=1e-5)
    cos = np.einsum("ni,ni->n", offset / np.linalg.norm(offset, axis=-1, keepdims=True), ee[:, :3, 2])
    assert np.all(cos > 0.999)


def test_bimanual_split() -> None:
    rng = np.random.default_rng(1)
    q14 = rng.uniform(-1.0, 1.0, size=(5, YAM_JOINT_DIM))
    left, right = bimanual_yam_ee_poses(q14)
    np.testing.assert_allclose(left, yam_ee_poses(q14[:, 0:6]))
    np.testing.assert_allclose(right, yam_ee_poses(q14[:, 7:13]))


def test_matches_mujoco_mjcf() -> None:
    mujoco = pytest.importorskip("mujoco")
    spec = mujoco.MjSpec.from_file(str(YAM_MJCF_PATH))

    def _delete(element) -> None:
        if hasattr(spec, "delete"):
            spec.delete(element)
        else:
            element.delete()

    def _strip(body) -> None:
        for g in list(body.geoms):
            _delete(g)
        for child in body.bodies:
            _strip(child)

    _strip(spec.worldbody)
    for m in list(spec.meshes):
        _delete(m)
    model = spec.compile()
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    rng = np.random.default_rng(2)
    q = rng.uniform(-1.2, 1.2, size=(16, 6))
    ours = yam_mount_poses(q)
    for i in range(q.shape[0]):
        data.qpos[:6] = q[i]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(ours[i, :3, 3], data.xpos[body_id], atol=1e-5)
        np.testing.assert_allclose(ours[i, :3, :3], data.xmat[body_id].reshape(3, 3), atol=1e-5)
