# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Meta-ready embodiment datasets for cross-embodiment few-shot training of the action heads.

Two things are added on top of the stock Cosmos action readers:

1. **Memory-light row storage.** ``FractalLeRobotDataset`` / ``BridgeOrigLeRobotDataset`` /
   ``RoboMINDURDataset`` / ``RoboMINDFrankaDataset`` all read frames through
   ``ActionBaseDataset._rows`` (a sorted ``list[dict]`` of every frame). The subclasses below swap
   that for a :class:`~cosmos_framework.data.generator.action.meta.lazy_rows.LazyRowTable` so the
   3.8M-frame RT-1 / UR indexes cost a few hundred MB instead of tens of GB per process. Sample
   construction (FK, frame conventions, normalization) is untouched -- it still runs the upstream
   ``__getitem__``.
2. **Episode bookkeeping.** Every meta-ready dataset exposes ``episode_table()`` (flat-index window
   ranges per demonstration + task id) and ``get_shuffle_blocks()``, which the episodic sampler
   (K demonstrations first, then windows inside them) and ``ActionIterableShuffleDataset`` need.

:func:`build_embodiment_sft_dataset` wraps a reader into ``ActionSFTDataset`` with the same
``ActionTransformPipeline`` settings the LIBERO post-training recipe uses, so meta-training samples
have exactly the model-facing format (padded 64-D actions, JSON prompts, sequence plans) that the
downstream few-shot post-training sees.

Registry keys (``EMBODIMENT_REGISTRY``): ``fractal`` (Google Robot RT-1), ``bridge`` (WidowX
BridgeData V2), ``robomind_ur`` (UR5e), ``robomind_franka`` (single-arm Franka), ``molmoact2_yam``
(bimanual YAM, 20-D).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionSFTDataset
from cosmos_framework.data.generator.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.generator.action.datasets.bridge_orig_lerobot_dataset import BridgeOrigLeRobotDataset
from cosmos_framework.data.generator.action.datasets.fractal_lerobot_dataset import (
    _SKIPPED_EPISODE_IDS as _FRACTAL_SKIPPED_EPISODE_IDS,
)
from cosmos_framework.data.generator.action.datasets.fractal_lerobot_dataset import FractalLeRobotDataset
from cosmos_framework.data.generator.action.datasets.robomind_franka_dataset import RoboMINDFrankaDataset
from cosmos_framework.data.generator.action.datasets.robomind_ur_dataset import RoboMINDURDataset
from cosmos_framework.data.generator.action.meta.lazy_rows import EpisodeTable, LazyRowTable
from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
from cosmos_framework.utils import log

__all__ = [
    "EMBODIMENT_REGISTRY",
    "BridgeMetaDataset",
    "EmbodimentSpec",
    "FractalMetaDataset",
    "RoboMINDFrankaMetaDataset",
    "RoboMINDURMetaDataset",
    "build_embodiment_raw_dataset",
    "build_embodiment_sft_dataset",
    "build_meta_transform",
    "get_episode_table",
    "resolve_embodiment_root",
]


# --------------------------------------------------------------------------------------------
# LazyRowTable mixin
# --------------------------------------------------------------------------------------------
class _LazyRowsMixin:
    """Install a :class:`LazyRowTable` as ``_rows`` and derive ``episode_table()`` from it."""

    _rows_cache: LazyRowTable | None
    _chunk_length: int
    _sample_stride: int
    _root: Path

    def _install_lazy_rows(self, exclude_episodes=(), columns: list[str] | None = None) -> None:
        if self._sample_stride != 1:
            raise NotImplementedError("Meta-ready readers assume sample_stride=1 (flat index == row index).")
        self._rows_cache = LazyRowTable.from_lerobot_root(self._root, columns=columns, exclude_episodes=exclude_episodes)
        self._episode_table_cache = self._rows_cache.episode_table(self._chunk_length)
        log.info(
            f"{type(self).__name__}: root={self._root} frames={len(self._rows_cache)} "
            f"episodes={len(self._episode_table_cache)} windows={self._episode_table_cache.total_windows}"
        )

    def episode_table(self) -> EpisodeTable:
        return self._episode_table_cache

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        return self._episode_table_cache.blocks()

    def __len__(self) -> int:
        # Only windows that lie fully inside one episode are addressable through the episodic
        # sampler, but keep the base semantics (row-indexed length) for generic callers.
        return max(0, len(self._rows_cache) - self._chunk_length)


class FractalMetaDataset(_LazyRowsMixin, FractalLeRobotDataset):
    """RT-1 with lazy rows. Re-implements ``__init__`` because the upstream one materializes
    ``_rows`` to drop the base-motion episodes (and assigns to the read-only ``_rows`` property)."""

    def __init__(
        self,
        root: str,
        fps: float = 3.0,
        chunk_length: int = 16,
        mode: str = "wam",
        pose_convention: str = "backward_framewise",
        tolerance_s: float = 1e-4,
        viewpoint: str = "ego_view",
        action_normalization: str | None = "quantile",
        sample_stride: int = 1,
    ) -> None:
        if viewpoint != "ego_view":
            raise NotImplementedError("FractalLeRobotDataset only supports ego_view.")
        ActionBaseDataset.__init__(
            self,
            root=root,
            domain_name="fractal",
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention=pose_convention,
            tolerance_s=tolerance_s,
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            sample_stride=sample_stride,
        )
        self._install_lazy_rows(exclude_episodes=_FRACTAL_SKIPPED_EPISODE_IDS)


class BridgeMetaDataset(_LazyRowsMixin, BridgeOrigLeRobotDataset):
    """BridgeData V2 (WidowX-250) with lazy rows."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "wam")
        super().__init__(*args, **kwargs)
        self._install_lazy_rows()


class RoboMINDURMetaDataset(_LazyRowsMixin, RoboMINDURDataset):
    """RoboMIND UR5e (MuJoCo FK on ``actions.joint_position``) with lazy rows."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "wam")
        super().__init__(*args, **kwargs)
        self._install_lazy_rows()


class RoboMINDFrankaMetaDataset(_LazyRowsMixin, RoboMINDFrankaDataset):
    """RoboMIND single-arm Franka (10-D) with lazy rows."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "wam")
        kwargs.setdefault("embodiment_type", "robomind-franka")
        kwargs.setdefault("viewpoint", "third_person_view")
        super().__init__(*args, **kwargs)
        self._install_lazy_rows()


# --------------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------------
def _build_yam(root: str, **kwargs: Any) -> Dataset:
    from cosmos_framework.data.generator.action.datasets.yam_repo_collection_dataset import YAMRepoCollectionDataset

    kwargs.setdefault("mode", "wam")
    return YAMRepoCollectionDataset(root=root, **kwargs)


@dataclass(frozen=True)
class EmbodimentSpec:
    """How to build one embodiment's raw reader.

    ``root_subdir`` is joined onto the shared ``ROBOT_FEWSHOT_ROOT`` unless an absolute ``root``
    override is passed. ``dataset_kwargs`` are the reader's defaults for the meta stage.
    """

    name: str
    domain_name: str
    action_dim: int
    root_subdir: str
    factory: Callable[..., Dataset]
    fps: float
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    description: str = ""


EMBODIMENT_REGISTRY: dict[str, EmbodimentSpec] = {
    "fractal": EmbodimentSpec(
        name="fractal",
        domain_name="fractal",
        action_dim=10,
        root_subdir="google_robot_rt1",
        factory=FractalMetaDataset,
        fps=3.0,
        dataset_kwargs=dict(viewpoint="ego_view", action_normalization="quantile"),
        description="Google Robot RT-1 (fractal), ego view 256x320, 3 FPS, 10-D EE deltas",
    ),
    "bridge": EmbodimentSpec(
        name="bridge",
        domain_name="bridge_orig_lerobot",
        action_dim=10,
        root_subdir="bridge_v2",
        factory=BridgeMetaDataset,
        fps=5.0,
        dataset_kwargs=dict(viewpoint="ego_view", action_normalization="quantile"),
        description="BridgeData V2 (WidowX-250), image_0 480x640, 5 FPS, 10-D EE deltas",
    ),
    "robomind_ur": EmbodimentSpec(
        name="robomind_ur",
        domain_name="robomind-ur",
        action_dim=10,
        root_subdir="robomind/ur_1rgb",
        factory=RoboMINDURMetaDataset,
        fps=30.0,
        dataset_kwargs=dict(viewpoint="third_person_view", action_normalization="quantile"),
        description="RoboMIND UR5e (MuJoCo FK on joint targets), camera_top 480x640, 30 FPS, 10-D EE deltas",
    ),
    "robomind_franka": EmbodimentSpec(
        name="robomind_franka",
        domain_name="robomind-franka",
        action_dim=10,
        root_subdir="robomind/franka_1rgb",
        factory=RoboMINDFrankaMetaDataset,
        fps=30.0,
        dataset_kwargs=dict(
            embodiment_type="robomind-franka", viewpoint="third_person_view", action_normalization="quantile"
        ),
        description="RoboMIND single-arm Franka, camera_top 720x1280, 30 FPS, 10-D EE deltas",
    ),
    "molmoact2_yam": EmbodimentSpec(
        name="molmoact2_yam",
        domain_name="molmoact2_yam",
        action_dim=20,
        root_subdir="yam/repos",
        factory=_build_yam,
        fps=30.0,
        dataset_kwargs=dict(viewpoint="concat_view", action_normalization="quantile", fk_source="action"),
        description="MolmoAct2 Bimanual-YAM (100 repos), top+wrists composite, 30 FPS, 20-D dual-arm EE deltas",
    ),
}


def resolve_embodiment_root(spec: EmbodimentSpec, data_root: str | Path | None, root_override: str | None = None) -> Path:
    if root_override:
        return Path(root_override)
    if data_root is None:
        raise ValueError(f"Embodiment {spec.name!r}: provide data_root or an explicit root.")
    return Path(data_root) / spec.root_subdir


def build_embodiment_raw_dataset(
    name: str,
    data_root: str | Path | None,
    chunk_length: int = 16,
    mode: str = "wam",
    root_override: str | None = None,
    dataset_kwargs: dict[str, Any] | None = None,
) -> Dataset:
    """Instantiate the raw (un-transformed) reader for ``name`` with the registry defaults."""
    if name not in EMBODIMENT_REGISTRY:
        raise KeyError(f"Unknown embodiment {name!r}; available: {sorted(EMBODIMENT_REGISTRY)}")
    spec = EMBODIMENT_REGISTRY[name]
    root = resolve_embodiment_root(spec, data_root, root_override)
    kwargs: dict[str, Any] = dict(spec.dataset_kwargs)
    kwargs.update(dataset_kwargs or {})
    kwargs.setdefault("fps", spec.fps)
    kwargs["chunk_length"] = chunk_length
    kwargs["mode"] = mode
    log.info(f"Building embodiment dataset {name!r} from {root} ({spec.description}) kwargs={kwargs}")
    return spec.factory(root=str(root), **kwargs)


def build_meta_transform(
    *,
    tokenizer_config: dict | None,
    max_action_dim: int = 64,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = True,
    format_prompt_as_json: bool = True,
) -> ActionTransformPipeline:
    """One ``ActionTransformPipeline`` shared by every embodiment (mirrors the LIBERO recipe)."""
    return ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
        format_prompt_as_json=format_prompt_as_json,
    )


def build_embodiment_sft_dataset(
    name: str,
    data_root: str | Path | None,
    transform: ActionTransformPipeline,
    resolution: str | int | None = "256",
    chunk_length: int = 16,
    mode: str = "wam",
    root_override: str | None = None,
    dataset_kwargs: dict[str, Any] | None = None,
) -> ActionSFTDataset:
    """Raw reader + shared transform -> ``ActionSFTDataset`` yielding model-ready samples."""
    raw = build_embodiment_raw_dataset(
        name,
        data_root,
        chunk_length=chunk_length,
        mode=mode,
        root_override=root_override,
        dataset_kwargs=dataset_kwargs,
    )
    return ActionSFTDataset(raw, transform, resolution)


def get_episode_table(dataset: Dataset) -> EpisodeTable:
    """Return the ``EpisodeTable`` of a raw or ``ActionSFTDataset``-wrapped meta-ready dataset."""
    inner = getattr(dataset, "_dataset", dataset)
    fn = getattr(inner, "episode_table", None)
    if fn is None:
        raise TypeError(
            f"{type(inner).__name__} has no episode_table(); use the meta-ready readers in "
            "cosmos_framework.data.generator.action.meta.embodiments."
        )
    return fn()
