"""The five robot_fewshot embodiments of cosmos_hs11 (LeRobot v3.0) as openpi data configs for Reptile meta-training.

Native actions are used as-is (no delta/absolute conversion), zero-padded to the model's 32 dims and quantile-normalized
with per-embodiment statistics (``assets/<reptile config>/<embodiment>/norm_stats.json``, see
``scripts/hs11/compute_norm_stats_reptile.py``). Camera slots follow openpi: one third-person ``base_0_rgb`` and up to two
wrist views; missing views are zero images with ``image_mask=False``.

| name             | root subdir           | base camera                      | wrists        | state                             | action                         | fps |
| fractal          | google_robot_rt1      | observation.images.image         | -             | observation.state [8]             | action [7] (world deltas+grip) | 3   |
| bridge           | bridge_v2             | observation.images.image_0       | -             | observation.state [7]             | action [7]                     | 5   |
| robomind_ur      | robomind/ur_1rgb      | observation.images.camera_top    | -             | observation.states.joint_position [7] | actions.joint_position [7] | 30  |
| robomind_franka  | robomind/franka_1rgb  | observation.images.camera_top    | -             | observation.states.joint_position [8] | actions.joint_position [8] | 30  |
| molmoact2_yam    | yam/repos (100 repos) | observation.images.top           | left, right   | observation.state [14]            | action [14]                    | 30  |
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
from collections.abc import Mapping

import numpy as np

from openpi import transforms as _transforms
from openpi.hs11 import lerobot_v3
from openpi.models import model as _model

ENV_ROBOT_FEWSHOT_ROOT = "ROBOT_FEWSHOT_ROOT"
YAM_MANIFEST = "yam_100_repos_seed42.txt"


@dataclasses.dataclass(frozen=True)
class EmbodimentSpec:
    name: str
    root_subdir: str
    image_keys: Mapping[str, str]  # alias in {"base", "wrist_left", "wrist_right"} -> feature key
    state_keys: tuple[str, ...]
    action_keys: tuple[str, ...]
    fps: float
    description: str = ""
    # For repo collections (YAM): a manifest listing the repo names under root_subdir.
    manifest: str | None = None

    def roots(self, data_root: str | None = None) -> tuple[str, ...]:
        base = pathlib.Path(resolve_data_root(data_root)) / self.root_subdir
        if self.manifest is None:
            return (str(base),)
        manifest = base.parent / self.manifest
        if manifest.is_file():
            names = [ln.strip().split("/")[-1] for ln in manifest.read_text().splitlines() if ln.strip()]
            roots = [base / n for n in names]
        else:
            roots = sorted(p for p in base.iterdir() if (p / "meta" / "info.json").is_file())
        return tuple(str(r) for r in roots)

    def dataset_spec(self, data_root: str | None = None, *, min_frames: int = 1) -> lerobot_v3.V3DatasetSpec:
        return lerobot_v3.V3DatasetSpec(
            roots=self.roots(data_root),
            image_keys=dict(self.image_keys),
            state_keys=self.state_keys,
            action_keys=self.action_keys,
            min_frames=min_frames,
            name=self.name,
        )


def resolve_data_root(data_root: str | None = None) -> str:
    if data_root:
        return data_root
    if os.environ.get(ENV_ROBOT_FEWSHOT_ROOT):
        return os.environ[ENV_ROBOT_FEWSHOT_ROOT]
    storage = os.environ.get("COSMOS_STORAGE", os.path.expanduser("~/project/cosmos_storage"))
    return os.path.join(storage, "data", "robot_fewshot")


EMBODIMENTS: dict[str, EmbodimentSpec] = {
    "fractal": EmbodimentSpec(
        name="fractal",
        root_subdir="google_robot_rt1",
        image_keys={"base": "observation.images.image"},
        state_keys=("observation.state",),
        action_keys=("action",),
        fps=3.0,
        description="Google Robot RT-1 (fractal20220817), 256x320, 3 fps, 7-D world-frame delta + gripper",
    ),
    "bridge": EmbodimentSpec(
        name="bridge",
        root_subdir="bridge_v2",
        image_keys={"base": "observation.images.image_0"},
        state_keys=("observation.state",),
        action_keys=("action",),
        fps=5.0,
        description="BridgeData V2 WidowX, image_0 480x640, 5 fps, 7-D delta + gripper",
    ),
    "robomind_ur": EmbodimentSpec(
        name="robomind_ur",
        root_subdir="robomind/ur_1rgb",
        image_keys={"base": "observation.images.camera_top"},
        state_keys=("observation.states.joint_position",),
        action_keys=("actions.joint_position",),
        fps=30.0,
        description="RoboMIND UR5e, camera_top 480x640, 30 fps, 7-D joint-position targets (6 joints + gripper)",
    ),
    "robomind_franka": EmbodimentSpec(
        name="robomind_franka",
        root_subdir="robomind/franka_1rgb",
        image_keys={"base": "observation.images.camera_top"},
        state_keys=("observation.states.joint_position",),
        action_keys=("actions.joint_position",),
        fps=30.0,
        description="RoboMIND single-arm Franka, camera_top 720x1280, 30 fps, 8-D joint-position targets",
    ),
    "molmoact2_yam": EmbodimentSpec(
        name="molmoact2_yam",
        root_subdir="yam/repos",
        image_keys={"base": "observation.images.top", "wrist_left": "observation.images.left", "wrist_right": "observation.images.right"},
        state_keys=("observation.state",),
        action_keys=("action",),
        fps=30.0,
        manifest=YAM_MANIFEST,
        description="MolmoAct2 bimanual YAM (100 LeRobot repos), top + two wrist cameras 360x640, 30 fps, 14-D joint targets",
    ),
}
DEFAULT_EMBODIMENTS = tuple(EMBODIMENTS)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


@dataclasses.dataclass(frozen=True)
class EmbodimentInputs(_transforms.DataTransformFn):
    """``image/<alias>`` + state/actions/prompt -> openpi model inputs (missing wrists are zero images, masked out)."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base = _parse_image(data["image/base"])
        images = {"base_0_rgb": base}
        masks = {"base_0_rgb": np.True_}
        pad_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
        for slot, alias in (("left_wrist_0_rgb", "image/wrist_left"), ("right_wrist_0_rgb", "image/wrist_right")):
            if alias in data:
                images[slot] = _parse_image(data[alias])
                masks[slot] = np.True_
            else:
                images[slot] = np.zeros_like(base)
                masks[slot] = pad_mask
        inputs = {"state": np.asarray(data["state"], dtype=np.float32), "image": images, "image_mask": masks}
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


def build_embodiment_dataset(spec: lerobot_v3.V3DatasetSpec, action_horizon: int) -> lerobot_v3.LeRobotV3Dataset:
    skip_images = os.environ.get("HS11_SKIP_IMAGES", "0") == "1"
    return lerobot_v3.build_dataset(spec, action_horizon, skip_images=skip_images)


def make_embodiment_data_config(
    name: str,
    model_config: _model.BaseModelConfig,
    assets_dir: str | pathlib.Path,
    *,
    data_root: str | None = None,
    require_norm_stats: bool = True,
):
    """``openpi.training.config.DataConfig`` for one embodiment (norm stats from ``<assets_dir>/<name>``)."""
    import functools

    import openpi.shared.normalize as _normalize
    from openpi.training import config as _config  # lazy: config.py imports this package lazily too

    spec = EMBODIMENTS[name]
    stats_dir = pathlib.Path(assets_dir) / name
    norm_stats = None
    if (stats_dir / "norm_stats.json").is_file():
        norm_stats = _normalize.load(stats_dir)
    elif require_norm_stats:
        raise FileNotFoundError(
            f"norm stats for embodiment {name!r} not found at {stats_dir}; run scripts/hs11/compute_norm_stats_reptile.py"
        )
    dataset_spec = spec.dataset_spec(data_root)
    return _config.DataConfig(
        repo_id=name,
        asset_id=name,
        norm_stats=norm_stats,
        repack_transforms=_transforms.Group(),
        data_transforms=_transforms.Group(inputs=[EmbodimentInputs(model_type=model_config.model_type)]),
        model_transforms=_config.ModelTransformFactory()(model_config),
        use_quantile_norm=model_config.model_type != _model.ModelType.PI0,
        prompt_from_task=False,
        dataset_builder=functools.partial(build_embodiment_dataset, dataset_spec),
    )
