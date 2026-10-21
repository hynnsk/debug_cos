"""NVIDIA LIBERO_LeRobot_v3 (20 fps, AV1, LeRobot v3.0) for pi05: dataset spec, policy transforms, eval helpers.

Conventions of the v3 release (checked against ``meta/info.json`` + ``meta/stats.json`` and the cosmos_hs11 eval):
* ``observation.state`` [8] = eef xyz, eef axis-angle (3), gripper qpos (2) -- exactly what ``examples/libero/main.py``
  builds from the simulator (``robot0_eef_pos``, quat -> axis-angle, ``robot0_gripper_qpos``).
* ``action`` [7] = the LIBERO OSC controller command (xyz in [-0.94, 0.94], rotation in [-0.375, 0.375]) except that the
  gripper is stored in **[0, 1] with 1 = open** instead of the env's [-1, 1] with -1 = open. Training uses the raw v3
  actions (quantile-normalized); :func:`v3_action_to_env` maps a predicted chunk back to env commands at evaluation.
* frames are stored upright, i.e. already rotated by 180 degrees relative to the raw simulator render, which is what the
  openpi client does before sending images (``img[::-1, ::-1]``).
"""

from __future__ import annotations

import dataclasses
import os
import pathlib

import einops
import numpy as np

from openpi import transforms as _transforms
from openpi.hs11 import lerobot_v3
from openpi.models import model as _model

LIBERO_SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")

# The FINAL 3-demo-per-task subsets of cosmos_hs11 (copied to assets/episode_subsets).
FINAL_3EP_SUBSETS = {
    "libero_10": "libero_10_3ep_per_task_seed42.json",
    "libero_goal": "libero_goal_3ep_per_task_v2.json",
    "libero_object": "libero_object_3ep_per_task_seed42.json",
    "libero_spatial": "libero_spatial_3ep_per_task_v3.json",
}
FINAL_1EP_SUBSETS = {suite: f"{suite}_1ep_per_task_v1.json" for suite in LIBERO_SUITES}

# Parent directory holding libero_10/ libero_goal/ libero_object/ libero_spatial/ (LeRobot v3.0 dirs).
ENV_LIBERO_V3_ROOT = "LIBERO_V3_ROOT"
# Max env steps per suite (openpi examples/libero/main.py; identical to the cosmos_hs11 eval).
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def libero_v3_root(suite: str) -> str:
    """``$LIBERO_V3_ROOT/<suite>`` (default ``$COSMOS_STORAGE/data/LIBERO_LeRobot_v3/<suite>``)."""
    if suite not in LIBERO_SUITES:
        raise ValueError(f"unknown LIBERO suite {suite!r}; choose from {LIBERO_SUITES}")
    base = os.environ.get(ENV_LIBERO_V3_ROOT)
    if not base:
        storage = os.environ.get("COSMOS_STORAGE", os.path.expanduser("~/project/cosmos_storage"))
        base = os.path.join(storage, "data", "LIBERO_LeRobot_v3")
    return os.path.join(base, suite)


def make_libero_v3_spec(suite: str, episode_subset: str | None, *, root: str | None = None) -> lerobot_v3.V3DatasetSpec:
    return lerobot_v3.V3DatasetSpec(
        roots=(root or libero_v3_root(suite),),
        image_keys={"base": "observation.images.image", "wrist": "observation.images.wrist_image"},
        state_keys=("observation.state",),
        action_keys=("action",),
        episode_subset=episode_subset,
        name=suite if episode_subset is None else f"{suite}[{pathlib.Path(episode_subset).stem}]",
    )


def build_libero_v3_dataset(spec: lerobot_v3.V3DatasetSpec, action_horizon: int) -> lerobot_v3.LeRobotV3Dataset:
    """Dataset builder used by the data config (``HS11_SKIP_IMAGES=1`` returns dummy frames: fast norm stats)."""
    skip_images = os.environ.get("HS11_SKIP_IMAGES", "0") == "1"
    return lerobot_v3.build_dataset(spec, action_horizon, skip_images=skip_images)


# Dataset keys -> the keys the LIBERO policy server receives from the client (identical to openpi's LIBERO example).
REPACK = {
    "observation/image": "image/base",
    "observation/wrist_image": "image/wrist",
    "observation/state": "state",
    "actions": "actions",
    "prompt": "prompt",
}


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoV3Inputs(_transforms.DataTransformFn):
    """Same slots as ``openpi.policies.libero_policy.LiberoInputs`` (base + left wrist, right wrist zero-masked)."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class PromptDropout(_transforms.DataTransformFn):
    """Replace the prompt by ``uncond_prompt`` with probability ``p`` (training only, cf. cfg_dropout_rate 0.1 in Cosmos).

    Gives the policy a trained unconditional branch so classifier-free guidance at evaluation (``GUIDANCE=...``) is
    well defined. Put it in ``repack_transforms`` (dataset side) so inference is unaffected.
    """

    p: float = 0.0
    uncond_prompt: str = ""

    def __call__(self, data: dict) -> dict:
        if self.p > 0.0 and "prompt" in data:
            import random

            if random.random() < self.p:
                data = {**data, "prompt": self.uncond_prompt}
        return data


@dataclasses.dataclass(frozen=True)
class LiberoV3Outputs(_transforms.DataTransformFn):
    """Un-pad to the 7 LIBERO dims. Actions stay in the v3 convention (gripper in [0, 1]); see ``v3_action_to_env``."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :7])}


# ---------------------------------------------------------------------------------------------------------------
# Evaluation helpers (shared with examples/libero_v3/main_v3.py)
# ---------------------------------------------------------------------------------------------------------------
def gripper_v3_to_env(g: float, *, binarize: bool = True) -> float:
    """v3 gripper in [0, 1] (1 = open) -> LIBERO env command in [-1, 1] (negative = open).

    ``binarize`` (default) snaps to {-1, +1} at 0.5, which is what the NVIDIA cosmos-rl reference does and what a
    weak policy needs (a continuous ``1 - 2g`` near 0.5 never actuates the gripper).
    """
    if binarize:
        return -1.0 if g > 0.5 else 1.0
    return float(np.clip(-(2.0 * g - 1.0), -1.0, 1.0))


def v3_action_to_env(action: np.ndarray, *, binarize_gripper: bool = True) -> np.ndarray:
    """[..., 7] v3 action chunk -> env commands (xyz / rotation pass through, gripper remapped)."""
    out = np.asarray(action, dtype=np.float32).copy()
    g = out[..., 6]
    if binarize_gripper:
        out[..., 6] = np.where(g > 0.5, -1.0, 1.0)
    else:
        out[..., 6] = np.clip(-(2.0 * g - 1.0), -1.0, 1.0)
    return out
