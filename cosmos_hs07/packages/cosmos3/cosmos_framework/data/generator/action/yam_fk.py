# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Forward kinematics for the i2rt YAM arm (NumPy, URDF-driven) and the bimanual YAM action contract.

The MolmoAct2 Bimanual-YAM LeRobot repos store 14-D absolute joint targets::

    [left_joint_0..5, left_gripper, right_joint_0..5, right_gripper]

Cosmos does not consume joint space directly for this embodiment; the ``molmoact2_yam`` domain is
the 20-D dual-arm end-effector contract ``[L pos(3), L rot6d(6), L grip(1), R pos(3), R rot6d(6),
R grip(1)]`` (see ``domain_utils.EMBODIMENT_TO_RAW_ACTION_DIM``). This module turns joint angles
into end-effector poses with a small URDF chain walker so no MuJoCo/Pinocchio build is needed at
data-loading time (MuJoCo is only used by the cross-check test).

Assets: ``robot_assets/yam.urdf`` and ``robot_assets/yam.xml`` are the arm-only models published by
i2rt (``i2rt/robot_models/arm/yam/v1``, Apache-2.0). Both arms of the bimanual rig are identical
YAM arms; since Cosmos actions are *frame-wise relative* poses expressed in the end-effector frame,
the (unknown) rigid placement of each arm base cancels out and an identity base suffices.

End-effector frame: the URDF ``gripper`` mount frame (child of ``joint6``) is re-referenced to the
i2rt ``linear_4310`` gripper's ``grasp_site`` -- the tool centre between the fingertips, at
``(0, 0, -0.14465)`` m in the mount frame, rotated 180 degrees about x -- so that the resulting
frame has ``+z`` pointing along the fingers (approach direction), matching the OpenCV-style
``z = approach`` convention used by the other Cosmos action datasets.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

YAM_URDF_PATH = Path(__file__).resolve().parent / "robot_assets" / "yam.urdf"
YAM_MJCF_PATH = Path(__file__).resolve().parent / "robot_assets" / "yam.xml"

YAM_BASE_LINK = "base"
YAM_EE_LINK = "gripper"  # end-effector mount frame (child of joint6)
YAM_ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
YAM_ARM_JOINTS = 6

# 14-D MolmoAct2 Bimanual-YAM joint layout.
YAM_LEFT_ARM_SLICE = slice(0, 6)
YAM_LEFT_GRIPPER_IDX = 6
YAM_RIGHT_ARM_SLICE = slice(7, 13)
YAM_RIGHT_GRIPPER_IDX = 13
YAM_JOINT_DIM = 14

# ``grasp_site`` of i2rt/robot_models/gripper/linear_4310/linear_4310.xml, expressed in the
# ``gripper`` mount frame: tool centre between the fingertips (same point at every opening),
# quaternion (w,x,y,z) = (0, 1, 0, 0) == 180 deg about x so +z points along the fingers.
_GRASP_SITE_POS = np.array([9.68103407e-05, 3.88982673e-05, -0.144650259], dtype=np.float64)
_GRASP_SITE_ROT = R.from_quat([1.0, 0.0, 0.0, 0.0]).as_matrix()  # scipy (x,y,z,w) == 180 deg about x

YAM_MOUNT_TO_TCP: np.ndarray = np.eye(4, dtype=np.float64)
YAM_MOUNT_TO_TCP[:3, :3] = _GRASP_SITE_ROT
YAM_MOUNT_TO_TCP[:3, 3] = _GRASP_SITE_POS

# Additional TCP -> OpenCV alignment. Identity: after the grasp-site re-referencing the frame
# already has +z along the approach direction; x/y only fix which lateral axis is "right"/"down",
# a convention that cancels within an embodiment for relative actions.
YAM_TCP_TO_OPENCV: np.ndarray = np.eye(3, dtype=np.float32)


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF rpy -> rotation matrix, ``R = Rz(yaw) Ry(pitch) Rx(roll)``."""
    return R.from_euler("xyz", rpy, degrees=False).as_matrix()


@dataclass(frozen=True)
class _ChainJoint:
    name: str
    joint_type: str
    origin: np.ndarray  # [4,4] parent-link -> joint frame
    axis: np.ndarray  # [3] unit axis in the joint frame (revolute) -- zeros for fixed joints


class URDFChainFK:
    """Minimal serial-chain forward kinematics parsed from a URDF file.

    Only the joints on the path ``base_link -> tip_link`` are used; every other joint (e.g. the
    prismatic finger joints of the YAM gripper) is ignored. Revolute/continuous joints consume one
    entry of ``q`` each (in chain order); fixed joints consume none.
    """

    def __init__(self, urdf_path: str | Path, base_link: str, tip_link: str) -> None:
        tree = ET.parse(str(urdf_path))
        robot = tree.getroot()
        joints_by_child: dict[str, ET.Element] = {}
        for joint in robot.findall("joint"):
            child = joint.find("child").attrib["link"]
            joints_by_child[child] = joint

        chain: list[_ChainJoint] = []
        link = tip_link
        while link != base_link:
            if link not in joints_by_child:
                raise ValueError(f"Link {link!r} is not connected to base link {base_link!r} in {urdf_path}.")
            joint = joints_by_child[link]
            origin_el = joint.find("origin")
            xyz = np.zeros(3)
            rpy = np.zeros(3)
            if origin_el is not None:
                xyz = np.array([float(v) for v in origin_el.attrib.get("xyz", "0 0 0").split()], dtype=np.float64)
                rpy = np.array([float(v) for v in origin_el.attrib.get("rpy", "0 0 0").split()], dtype=np.float64)
            origin = np.eye(4, dtype=np.float64)
            origin[:3, :3] = _rpy_to_matrix(rpy)
            origin[:3, 3] = xyz
            joint_type = joint.attrib.get("type", "fixed")
            axis = np.zeros(3, dtype=np.float64)
            if joint_type in ("revolute", "continuous", "prismatic"):
                axis_el = joint.find("axis")
                axis = np.array(
                    [float(v) for v in (axis_el.attrib["xyz"] if axis_el is not None else "1 0 0").split()],
                    dtype=np.float64,
                )
                norm = np.linalg.norm(axis)
                if norm <= 0:
                    raise ValueError(f"Joint {joint.attrib['name']!r} has a zero axis.")
                axis = axis / norm
            chain.append(_ChainJoint(joint.attrib["name"], joint_type, origin, axis))
            link = joint.find("parent").attrib["link"]
        chain.reverse()
        self._chain = chain
        self.joint_names = tuple(j.name for j in chain if j.joint_type != "fixed")
        self.num_joints = len(self.joint_names)

    def forward(self, q: np.ndarray) -> np.ndarray:
        """Batched FK: ``q`` of shape ``[N, num_joints]`` -> tip poses ``[N, 4, 4]`` in the base frame."""
        q = np.asarray(q, dtype=np.float64)
        if q.ndim == 1:
            q = q[None]
        if q.shape[-1] != self.num_joints:
            raise ValueError(f"Expected q with {self.num_joints} joints, got shape {q.shape}.")
        n = q.shape[0]
        poses = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))  # [N,4,4]
        qi = 0
        for joint in self._chain:
            poses = poses @ joint.origin  # [N,4,4]
            if joint.joint_type == "fixed":
                continue
            motion = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
            if joint.joint_type == "prismatic":
                motion[:, :3, 3] = q[:, qi : qi + 1] * joint.axis[None, :]
            else:
                motion[:, :3, :3] = R.from_rotvec(q[:, qi : qi + 1] * joint.axis[None, :]).as_matrix()
            poses = poses @ motion
            qi += 1
        return poses


@lru_cache(maxsize=1)
def get_yam_arm_fk() -> URDFChainFK:
    """Cached FK of the 6-DOF YAM arm (``base`` -> ``gripper`` mount frame)."""
    fk = URDFChainFK(YAM_URDF_PATH, YAM_BASE_LINK, YAM_EE_LINK)
    if fk.joint_names != YAM_ARM_JOINT_NAMES:
        raise RuntimeError(f"Unexpected YAM joint chain {fk.joint_names}; expected {YAM_ARM_JOINT_NAMES}.")
    return fk


def yam_mount_poses(q_arm: np.ndarray) -> np.ndarray:
    """``[N, 6]`` arm joint angles (rad) -> ``[N, 4, 4]`` gripper *mount* poses in the arm base frame."""
    return get_yam_arm_fk().forward(q_arm)


def yam_ee_poses(q_arm: np.ndarray) -> np.ndarray:
    """``[N, 6]`` arm joint angles (rad) -> ``[N, 4, 4]`` tool-centre poses (``+z`` = approach).

    Applies :data:`YAM_MOUNT_TO_TCP` (grasp-site re-referencing) and :data:`YAM_TCP_TO_OPENCV`.
    """
    poses = yam_mount_poses(q_arm) @ YAM_MOUNT_TO_TCP  # [N,4,4]
    poses[:, :3, :3] = poses[:, :3, :3] @ YAM_TCP_TO_OPENCV.astype(np.float64)
    return poses.astype(np.float32)


def bimanual_yam_ee_poses(q14: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``[N, 14]`` bimanual joint vector -> ``(left [N,4,4], right [N,4,4])`` tool-centre poses."""
    q14 = np.asarray(q14, dtype=np.float64)
    if q14.ndim != 2 or q14.shape[1] != YAM_JOINT_DIM:
        raise ValueError(f"Expected [N, {YAM_JOINT_DIM}] bimanual joints, got {q14.shape}.")
    left = yam_ee_poses(q14[:, YAM_LEFT_ARM_SLICE])
    right = yam_ee_poses(q14[:, YAM_RIGHT_ARM_SLICE])
    return left, right


__all__ = [
    "URDFChainFK",
    "YAM_ARM_JOINTS",
    "YAM_ARM_JOINT_NAMES",
    "YAM_JOINT_DIM",
    "YAM_LEFT_ARM_SLICE",
    "YAM_LEFT_GRIPPER_IDX",
    "YAM_MJCF_PATH",
    "YAM_MOUNT_TO_TCP",
    "YAM_RIGHT_ARM_SLICE",
    "YAM_RIGHT_GRIPPER_IDX",
    "YAM_TCP_TO_OPENCV",
    "YAM_URDF_PATH",
    "bimanual_yam_ee_poses",
    "get_yam_arm_fk",
    "yam_ee_poses",
    "yam_mount_poses",
]
