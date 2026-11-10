# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Pure-python pieces of the ORIGINAL-RoboCasa (v0.2) closed-loop protocol: no simulator import, so they are unit-testable
in the training env and shared by ``closed_loop_eval.py`` (simulator env).

The protocol is Cosmos Policy's RoboCasa evaluation (cosmos_policy/experiments/robot/robocasa/run_robocasa_eval.py,
``--deterministic True --seed 195 --num_trials_per_task 50``): 24 atomic tasks, PandaMobile + OSC_POSE, object split
"B" (held-out instances), the 5 official test scenes ``((1,1),(2,2),(4,4),(6,9),(7,10))`` with episodes 0-9 in scene 0,
10-19 in scene 1, ..., one fresh env per episode seeded ``seed * episode_idx * 256``, 10 zero-action settling steps,
the registry horizon per task, first ``_check_success()`` ends the episode. Images are the three 224x224 cameras
flipped upright; proprio = ``[gripper_qpos(2), eef_pos(3), eef_quat(4)]``.

What the Cosmos3 policy consumes instead of Cosmos Policy's latent frames (must match ``action_policy_robocasa_edge``):
``[agentview_left | eye_in_hand]`` 224x448 composite, the 10D state token ``[eef_pos, rot6d(eef_quat), qpos0 - qpos1]``
min/max-normalized with the ``state`` block of the bundled stats JSON, and 16 ``[pos, rot6d, gripper]`` actions per query
(the server already inverts ``quantile_rot``), decoded to the env's 12D action with the base block zero / mode -1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROBOCASA_V02_TASKS: tuple[str, ...] = (
    "PnPCounterToCab", "PnPCabToCounter", "PnPCounterToSink", "PnPSinkToCounter", "PnPCounterToMicrowave",
    "PnPMicrowaveToCounter", "PnPCounterToStove", "PnPStoveToCounter", "OpenSingleDoor", "CloseSingleDoor",
    "OpenDoubleDoor", "CloseDoubleDoor", "OpenDrawer", "CloseDrawer", "TurnOnSinkFaucet", "TurnOffSinkFaucet",
    "TurnSinkSpout", "TurnOnStove", "TurnOffStove", "CoffeeSetupMug", "CoffeeServeMug", "CoffeePressButton",
    "TurnOnMicrowave", "TurnOffMicrowave",
)

# Horizons of robocasa v0.2 ``dataset_registry.SINGLE_STAGE_TASK_DATASETS`` (= Cosmos Policy's TASK_MAX_STEPS).
TASK_MAX_STEPS: dict[str, int] = {
    "PnPCounterToCab": 500, "PnPCabToCounter": 500, "PnPCounterToSink": 700, "PnPSinkToCounter": 500,
    "PnPCounterToMicrowave": 600, "PnPMicrowaveToCounter": 500, "PnPCounterToStove": 500, "PnPStoveToCounter": 500,
    "OpenSingleDoor": 500, "CloseSingleDoor": 500, "OpenDoubleDoor": 1000, "CloseDoubleDoor": 700,
    "OpenDrawer": 500, "CloseDrawer": 500, "TurnOnSinkFaucet": 500, "TurnOffSinkFaucet": 500, "TurnSinkSpout": 500,
    "TurnOnStove": 500, "TurnOffStove": 500, "CoffeeSetupMug": 600, "CoffeeServeMug": 600, "CoffeePressButton": 300,
    "TurnOnMicrowave": 500, "TurnOffMicrowave": 500,
}

TEST_SCENES: tuple[tuple[int, int], ...] = ((1, 1), (2, 2), (4, 4), (6, 9), (7, 10))  # (layout, style)
EPISODES_PER_SCENE = 10
OBJ_INSTANCE_SPLIT = "B"
ROBOTS = "PandaMobile"
CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
IMAGE_RES = 224
NUM_SETTLE_STEPS = 10
DEFAULT_SEED = 195  # Cosmos Policy reports the mean over seeds 195 / 196 / 197
CONTROLLER_CONFIG_PATH = Path(__file__).with_name("osc_pose_controller_config.json")
MOBILE_BASE_TAIL = np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)  # base_motion(3) + torso + mode(-1 = arm)


def scene_for_episode(episode_idx: int) -> tuple[int, int]:
    """Cosmos Policy: ``scene_index = (episode_idx // 10) % len(scenes)``."""
    return TEST_SCENES[(int(episode_idx) // EPISODES_PER_SCENE) % len(TEST_SCENES)]


def env_seed_for_episode(base_seed: int, episode_idx: int, deterministic: bool = True) -> int | None:
    """Cosmos Policy: ``seed = cfg.seed * episode_idx * 256`` when deterministic (episode 0 -> 0), else unseeded."""
    return int(base_seed) * int(episode_idx) * 256 if deterministic else None


def load_controller_config(path: Path | str = CONTROLLER_CONFIG_PATH) -> dict:
    cfg = json.loads(Path(path).read_text())
    cfg.pop("_comment", None)
    return cfg


def env_kwargs_for_episode(task: str, episode_idx: int, base_seed: int, deterministic: bool, image_res: int = IMAGE_RES) -> dict:
    """``robosuite.make`` kwargs of Cosmos Policy's ``create_robocasa_env`` for one episode (scene + seed resolved)."""
    if task not in TASK_MAX_STEPS:
        raise ValueError(f"unknown RoboCasa v0.2 task {task!r}; one of {ROBOCASA_V02_TASKS}")
    return dict(
        env_name=task,
        robots=ROBOTS,
        controller_configs=load_controller_config(),
        camera_names=list(CAMERAS),
        camera_widths=int(image_res),
        camera_heights=int(image_res),
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=env_seed_for_episode(base_seed, episode_idx, deterministic),
        obj_instance_split=OBJ_INSTANCE_SPLIT,
        generative_textures=None,
        randomize_cameras=False,
        layout_and_style_ids=(scene_for_episode(episode_idx),),
        translucent_robot=False,
    )


# ----------------------------- observations -----------------------------

def upright(img: np.ndarray) -> np.ndarray:
    """MuJoCo renders bottom-up; the training videos (and Cosmos Policy, ``flip_images=True``) are upright."""
    return np.ascontiguousarray(img[::-1])


def compose_left_wrist(obs: dict) -> np.ndarray:
    """``[agentview_left | eye_in_hand]`` -> (224, 448, 3) uint8, mirroring ``RoboCasaLeRobotDataset._compose_left_wrist``."""
    left = upright(obs["robot0_agentview_left_image"])
    wrist = upright(obs["robot0_eye_in_hand_image"])
    return np.concatenate([left, wrist], axis=1).astype(np.uint8)


def compose_wrist_lr(obs: dict) -> np.ndarray:
    """``wrist`` over ``[left | right]`` at half size (``camera_set="wrist_lr"``), mirroring ``_compose_multi_view``."""
    from PIL import Image

    wrist = upright(obs["robot0_eye_in_hand_image"])
    left = upright(obs["robot0_agentview_left_image"])
    right = upright(obs["robot0_agentview_right_image"])
    h, w = wrist.shape[:2]
    lh = np.asarray(Image.fromarray(left).resize((w // 2, h // 2), Image.BILINEAR))
    rh = np.asarray(Image.fromarray(right).resize((w // 2, h // 2), Image.BILINEAR))
    return np.concatenate([wrist, np.concatenate([lh, rh], axis=1)], axis=0).astype(np.uint8)


COMPOSERS = {"left_wrist": compose_left_wrist, "wrist_lr": compose_wrist_lr}


def quat_xyzw_to_rot6d(quat_xyzw: np.ndarray) -> np.ndarray:
    """Unit quaternion (x, y, z, w) -> the first two COLUMNS of its rotation matrix, as ``convert_rotation(...,
    "quat_xyzw" -> "matrix" -> "rot6d")`` in the reader (verified against training samples)."""
    x, y, z, w = (float(v) for v in np.asarray(quat_xyzw, dtype=np.float64).reshape(4))
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    m = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return np.concatenate([m[:, 0], m[:, 1]]).astype(np.float32)


def raw_state_token(obs: dict) -> np.ndarray:
    """``[eef_pos(3), rot6d(eef_quat)(6), qpos0 - qpos1]`` = ``RoboCasaLeRobotDataset._build_initial_state`` on the
    converted state ``[.., eef_pos(7:10), eef_quat(10:14), gripper_qpos(14:16)]`` (the converter wrote
    ``robot_states`` = ``[gripper_qpos, eef_pos, eef_quat]`` into those columns)."""
    pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(3)
    rot6d = quat_xyzw_to_rot6d(np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(4))
    qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    grip = np.array([qpos[0] - qpos[1]], dtype=np.float32)
    return np.concatenate([pos, rot6d, grip]).astype(np.float32)  # [10]


class StateNormalizer:
    """Forward ``state_normalization`` of the training reader (``resolve_action_normalization`` on the ``state`` block):
    minmax -> ``(x - (max+min)/2) / ((max-min)/2)``; quantile -> the same with q01/q99; meanstd -> ``(x - mean) / std``."""

    def __init__(self, stats_path: Path | str, method: str = "minmax") -> None:
        raw = json.loads(Path(stats_path).read_text())
        if "state" not in raw:
            raise KeyError(f"{stats_path} has no 'state' block (regenerate it with the converter --stats-out)")
        st = {k: np.asarray(v, dtype=np.float32) for k, v in raw["state"].items()}
        if method == "minmax":
            lo, hi = st["min"], st["max"]
        elif method == "quantile":
            lo, hi = st["q01"], st["q99"]
        elif method == "meanstd":
            self.offset, self.scale = st["mean"], np.maximum(st["std"], 1e-8)
            self.method = method
            return
        else:
            raise ValueError(f"unsupported state normalization {method!r}")
        self.offset = (hi + lo) / 2.0
        self.scale = np.maximum((hi - lo) / 2.0, 1e-8)
        self.method = method

    def __call__(self, token: np.ndarray) -> np.ndarray:
        return ((np.asarray(token, dtype=np.float32) - self.offset) / self.scale).astype(np.float32)


# ----------------------------- actions -----------------------------

def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """6D rotation (two columns) -> 3x3 via Gram-Schmidt (the inverse of ``quat_xyzw_to_rot6d`` / the reader's rot6d)."""
    a1, a2 = np.asarray(r6[:3], dtype=np.float64), np.asarray(r6[3:6], dtype=np.float64)
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def matrix_to_rotvec(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector (the OSC_POSE rotation command the demos were recorded with)."""
    m = np.asarray(m, dtype=np.float64)
    cos = np.clip((np.trace(m) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos))
    if angle < 1e-8:
        return np.zeros(3)
    if abs(np.pi - angle) < 1e-6:  # near pi: axis from the symmetric part
        axis = np.sqrt(np.maximum(np.diag(m) + 1.0, 0.0) / 2.0)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis * angle
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]) / (2.0 * np.sin(angle))
    return axis * angle


def decode_action10_to_env12(a10: np.ndarray) -> np.ndarray:
    """``[pos(3), rot6d(6), gripper(1)]`` (denormalized by the server) -> v0.2 env action
    ``[pos(3), rotvec(3), gripper(1), base_motion(3)=0, torso=0, mode=-1]`` (Cosmos Policy appends ``[0,0,0,0,-1]``)."""
    a10 = np.asarray(a10, dtype=np.float64).reshape(-1)
    if a10.shape[0] != 10:
        raise ValueError(f"expected a 10D action, got {a10.shape[0]}D (check the server's raw_action_dim / stats)")
    env = np.zeros(12, dtype=np.float32)
    env[0:3] = np.clip(a10[0:3], -1.0, 1.0)
    env[3:6] = np.clip(matrix_to_rotvec(rot6d_to_matrix(a10[3:9])), -1.0, 1.0)
    env[6] = float(np.clip(a10[9], -1.0, 1.0))
    env[7:12] = MOBILE_BASE_TAIL
    return env
