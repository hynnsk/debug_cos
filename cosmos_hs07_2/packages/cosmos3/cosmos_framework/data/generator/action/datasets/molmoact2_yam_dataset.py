# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""MolmoAct2 Bimanual-YAM LeRobot dataset (one ``allenai/<date>-<task>-<nn>`` repo).

Robot: ``bi_yam_follower`` (two i2rt YAM arms), 30 FPS, three 360x640 cameras
(``observation.images.top`` / ``left`` / ``right``), 14-D absolute joint ``action`` and
``observation.state``::

    [left_joint_0..5, left_gripper, right_joint_0..5, right_gripper]

Cosmos consumes the 20-D dual-arm end-effector contract of the ``molmoact2_yam`` domain::

    14-D joint trajectory -> YAM forward kinematics (yam_fk) -> left/right tool-centre poses
    -> frame-wise relative pose (backward_framewise) -> rot6d -> [L pos3, L rot6d, L grip,
    R pos3, R rot6d, R grip]

Gripper values are kept as stored (``1`` = open, ``0`` = closed -- episodes start at ``~1.0``),
which already matches the Cosmos ``0=closed, 1=open`` convention.

Like ``LIBEROLeRobotDataset`` this reader keeps a compact NumPy index (episode / task / timestamp /
joints per row) instead of ``ActionBaseDataset._rows`` dicts, only ever yields windows that lie
inside one episode, and exposes ``get_shuffle_blocks()`` / ``episode_table()`` for the streaming
and episodic samplers. Use :class:`YAMRepoCollectionDataset` to concatenate many repos.
"""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from cosmos_framework.data.generator.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.generator.action.meta.lazy_rows import EpisodeEntry, EpisodeTable
from cosmos_framework.data.generator.action.utils.action_spec import ActionSpec, Gripper, Pos, Rot, build_action_spec
from cosmos_framework.data.generator.action.utils.pose_utils import pose_abs_to_rel
from cosmos_framework.data.generator.action.utils.viewpoint_utils import compose_multiview
from cosmos_framework.data.generator.action.yam_fk import (
    YAM_JOINT_DIM,
    YAM_LEFT_GRIPPER_IDX,
    YAM_RIGHT_GRIPPER_IDX,
    bimanual_yam_ee_poses,
)
from cosmos_framework.utils import log

PoseConvention = Literal["backward_framewise"]
Viewpoint = Literal["concat_view", "third_person_view"]
FKSource = Literal["action", "observation.state"]

_IMAGE_FEATURES = {
    "top": "observation.images.top",
    "left": "observation.images.left",
    "right": "observation.images.right",
}
_NORMALIZER_PATH = Path(__file__).parent.parent / "normalizer_stats/molmoact2_yam_stats.json"

_CONCAT_VIEW_DESCRIPTION = (
    "The top row shows a third-person perspective looking down at the bimanual YAM robot from above. "
    "The bottom-left view is the left wrist camera, and the bottom-right view is the right wrist camera."
)


def dual_arm_yam_action_spec() -> ActionSpec:
    return build_action_spec(
        Pos(prefix="left"),
        Rot("rot6d", prefix="left"),
        Gripper(prefix="left"),
        Pos(prefix="right"),
        Rot("rot6d", prefix="right"),
        Gripper(prefix="right"),
    )


@lru_cache(maxsize=1)
def _load_yam_stats_cached(path: str) -> dict[str, torch.Tensor]:
    from cosmos_framework.data.generator.action.action_normalization import load_action_stats

    return {key: torch.from_numpy(value).float() for key, value in load_action_stats(path).items()}


class MolmoAct2YAMDataset(ActionBaseDataset):
    """One MolmoAct2 Bimanual-YAM repo -> 20-D dual-arm frame-wise-relative rot6d actions."""

    def __init__(
        self,
        root: str,
        fps: float = 30.0,
        chunk_length: int = 16,
        mode: str = "wam",
        pose_convention: PoseConvention = "backward_framewise",
        tolerance_s: float | None = None,
        viewpoint: Viewpoint = "concat_view",
        action_normalization: str | None = "quantile",
        sample_stride: int = 1,
        fk_source: FKSource = "action",
        use_annotated_tasks: bool = True,
        embodiment_type: str = "molmoact2_yam",
    ) -> None:
        if viewpoint not in ("concat_view", "third_person_view"):
            raise NotImplementedError(f"MolmoAct2YAMDataset does not support viewpoint={viewpoint!r}.")
        if sample_stride != 1:
            raise NotImplementedError("MolmoAct2YAMDataset indexes valid within-episode windows; sample_stride must be 1.")
        if fk_source not in ("action", "observation.state"):
            raise ValueError(f"fk_source must be 'action' or 'observation.state', got {fk_source!r}.")
        if tolerance_s is None:
            # LeRobot v3 repos concatenate many episodes into one mp4, so absolute pts run into the
            # thousands of seconds where float32 (used inside lerobot's decode check) only resolves
            # ~1.2e-4 s -- the 1e-4 default then rejects perfectly aligned frames. Half a frame period
            # still guarantees we get the nearest frame.
            tolerance_s = 0.5 / float(fps)
        super().__init__(
            root=root,
            domain_name=embodiment_type,
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention=pose_convention,
            tolerance_s=tolerance_s,
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            sample_stride=sample_stride,
        )
        # Trust the repo's native fps (30) for conditioning_fps / prompt duration.
        info_fps = self._info.get("fps")
        if info_fps:
            if float(info_fps) != float(fps):
                log.info(f"[{self._root.name}] Using dataset native fps={info_fps} for conditioning (requested {fps}).")
            self._fps = float(info_fps)
            self._dt = 1.0 / self._fps
        self._fk_source = fk_source
        self._video_keys = list(_IMAGE_FEATURES.values()) if viewpoint == "concat_view" else [_IMAGE_FEATURES["top"]]

        # ---- compact frame index (rows sorted by global frame index) ----
        index_parts, episode_parts, task_parts, ts_parts, joint_parts = [], [], [], [], []
        for path in sorted((self._root / "data").glob("chunk-*/file-*.parquet")):
            table = pq.read_table(path, columns=["index", "episode_index", "task_index", "timestamp", fk_source])
            index_parts.append(table["index"].to_numpy())
            episode_parts.append(table["episode_index"].to_numpy())
            task_parts.append(table["task_index"].to_numpy())
            ts_parts.append(table["timestamp"].to_numpy())
            joints = np.asarray(table[fk_source].to_pylist(), dtype=np.float32)
            if joints.ndim != 2 or joints.shape[1] != YAM_JOINT_DIM:
                raise ValueError(f"{path}: expected [N, {YAM_JOINT_DIM}] '{fk_source}', got {joints.shape}.")
            joint_parts.append(joints)
        if not index_parts:
            raise FileNotFoundError(f"No data parquet found under {self._root / 'data'}.")
        order = np.argsort(np.concatenate(index_parts).astype(np.int64), kind="stable")
        self._row_episode = np.concatenate(episode_parts).astype(np.int64)[order]
        self._row_task = np.concatenate(task_parts).astype(np.int64)[order]
        self._row_timestamp = np.concatenate(ts_parts).astype(np.float64)[order]
        self._row_joints = np.concatenate(joint_parts, axis=0).astype(np.float32)[order]
        if not np.all(np.diff(self._row_episode) >= 0):
            raise ValueError(f"{self._root}: episode_index not contiguous after sorting by frame index.")

        ep_vals, ep_starts, ep_counts = np.unique(self._row_episode, return_index=True, return_counts=True)
        self._ep_vals = ep_vals.astype(np.int64)
        self._ep_starts = ep_starts.astype(np.int64)
        self._ep_counts = ep_counts.astype(np.int64)
        # Within-episode windows need chunk_length + 1 frames -> count - chunk_length windows.
        self._ep_windows = np.maximum(0, self._ep_counts - self._chunk_length)
        self._valid_cum = np.cumsum(self._ep_windows).astype(np.int64)

        # Per-episode annotated instruction (MolmoAct2 ships meta/tasks_annotated.parquet).
        self._annotated_tasks: dict[int, str] = {}
        annotated = self._root / "meta" / "tasks_annotated.parquet"
        if use_annotated_tasks and annotated.exists():
            try:
                df = pd.read_parquet(annotated)
                col = "task" if "task" in df.columns else df.columns[0]
                for ep, text in zip(df.index.tolist(), df[col].tolist()):
                    if isinstance(text, str) and text.strip():
                        self._annotated_tasks[int(ep)] = text.strip()
            except Exception as e:  # noqa: BLE001 - annotations are optional
                log.warning(f"[{self._root.name}] failed to read tasks_annotated.parquet ({e}); using tasks.parquet.")

        log.info(
            f"Loaded MolmoAct2 YAM repo {self._root.name}: episodes={len(self._ep_vals)} frames={len(self._row_episode)} "
            f"windows={int(self._valid_cum[-1]) if self._valid_cum.size else 0} fps={self._fps} viewpoint={viewpoint} "
            f"fk_source={fk_source} annotated_tasks={len(self._annotated_tasks)}"
        )

    # ---- spec / dims -------------------------------------------------------
    @property
    def action_dim(self) -> int:
        return 20

    def _action_spec(self) -> ActionSpec:
        return dual_arm_yam_action_spec()

    @classmethod
    def _stats_path(cls) -> Path:
        return _NORMALIZER_PATH

    @classmethod
    def load_action_stats(cls) -> dict[str, torch.Tensor]:
        # Shared by all 100 repo instances (one json read + one log line per process, not per repo).
        return {k: v.clone() for k, v in _load_yam_stats_cached(str(cls._stats_path())).items()}

    @property
    def fk_source(self) -> str:
        return self._fk_source

    # ---- index helpers ------------------------------------------------------
    def __len__(self) -> int:
        return int(self._valid_cum[-1]) if self._valid_cum.size else 0

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        """Per-episode ``(start, length)`` flat-index blocks for ``ActionIterableShuffleDataset``."""
        return self.episode_table().blocks()

    def episode_table(self) -> EpisodeTable:
        entries: list[EpisodeEntry] = []
        prev = 0
        for ep, start, count, nwin, cum in zip(
            self._ep_vals.tolist(),
            self._ep_starts.tolist(),
            self._ep_counts.tolist(),
            self._ep_windows.tolist(),
            self._valid_cum.tolist(),
        ):
            entries.append(
                EpisodeEntry(
                    episode_index=int(ep),
                    task_index=int(self._row_task[start]),
                    first_flat_index=int(prev),
                    num_windows=int(nwin),
                    num_frames=int(count),
                )
            )
            prev = cum
        return EpisodeTable(entries)

    def task_text(self, task_index: int) -> str:
        return self._tasks[int(task_index)]

    def episode_raw_actions(self, ep_pos: int) -> np.ndarray:
        """Raw (un-normalized) ``[n_frames - 1, 20]`` frame-wise actions over one whole episode.

        Identical per-frame values to what windows of this episode produce (frame-wise relative poses
        only depend on consecutive frames), without decoding any video -- used to compute the
        normalization statistics (``tools/compute_action_stats_from_dataset.py``).
        """
        start = int(self._ep_starts[ep_pos])
        count = int(self._ep_counts[ep_pos])
        if count < 2:
            return np.zeros((0, self.action_dim), dtype=np.float32)
        q = self._row_joints[start : start + count]  # [n,14]
        left_abs, right_abs = bimanual_yam_ee_poses(q)
        left_rel = pose_abs_to_rel(left_abs, rotation_format="rot6d", pose_convention=self._pose_convention)  # [n-1,9]
        right_rel = pose_abs_to_rel(right_abs, rotation_format="rot6d", pose_convention=self._pose_convention)
        n_rel = left_rel.shape[0]
        return np.concatenate(
            [
                left_rel,
                q[:n_rel, YAM_LEFT_GRIPPER_IDX : YAM_LEFT_GRIPPER_IDX + 1],
                right_rel,
                q[:n_rel, YAM_RIGHT_GRIPPER_IDX : YAM_RIGHT_GRIPPER_IDX + 1],
            ],
            axis=-1,
        ).astype(np.float32)

    def iter_episode_raw_actions(self):
        """Yield ``episode_raw_actions`` for every episode of this repo."""
        for ep_pos in range(len(self._ep_vals)):
            yield self.episode_raw_actions(ep_pos)

    def _locate(self, idx: int) -> tuple[int, int]:
        """Flat index -> (episode position, first row of the window)."""
        ep = int(np.searchsorted(self._valid_cum, idx, side="right"))
        prev = int(self._valid_cum[ep - 1]) if ep > 0 else 0
        start = int(self._ep_starts[ep]) + (idx - prev)
        return ep, start

    # ---- sample build --------------------------------------------------------
    def __getitem__(self, idx: int) -> dict[str, Any]:
        n = len(self)
        if not 0 <= idx < n:
            raise IndexError(f"index {idx} out of range for {n} windows")
        last_err: Exception | None = None
        for _attempt in range(8):
            try:
                return self._build_item(idx)
            except Exception as e:  # noqa: BLE001 - skip past undecodable frames
                last_err = e
                log.warning(f"YAM[{self._root.name}]: sample idx={idx} failed ({type(e).__name__}: {e}); resampling")
                idx = random.randint(0, n - 1)
        raise RuntimeError(f"YAM[{self._root.name}]: failed to load a sample after 8 resamples; last error: {last_err}")

    def _build_item(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        ep, start = self._locate(int(idx))
        episode_index = int(self._ep_vals[ep])
        episode = self._episodes[episode_index]
        stop = start + self._chunk_length + 1  # T+1 rows: current frame + T future frames

        timestamps = [float(self._row_timestamp[j]) for j in range(start, stop)]
        video = self._load_video(episode, timestamps)
        action, initial_pose_left, initial_pose_right, joint_configs = self._build_raw_action(
            self._row_joints[start:stop]
        )

        caption = self._annotated_tasks.get(episode_index) or self._tasks[int(self._row_task[start])]
        ai_caption = random.choice([part.strip() for part in caption.split(" | ") if part.strip()] or [caption])

        extras: dict[str, Any] = {
            "initial_pose": initial_pose_left,
            "initial_pose_right": initial_pose_right,
            "joint_configs": joint_configs,
        }
        if self._viewpoint == "concat_view":
            extras["additional_view_description"] = _CONCAT_VIEW_DESCRIPTION
        return self._build_result(mode=mode, video=video, action=action, ai_caption=ai_caption, **extras)

    def _load_video(self, episode: dict[str, Any], timestamps: list[float]) -> torch.Tensor:
        # lerobot is a heavy, optional ("train" extra) dependency; import lazily.
        from lerobot.datasets.video_utils import decode_video_frames

        frames_by_key: dict[str, torch.Tensor] = {}
        for key in self._video_keys:
            from_ts = float(episode.get(f"videos/{key}/from_timestamp", 0.0))
            frames_by_key[key] = decode_video_frames(
                self._video_path(episode, key),
                [from_ts + ts for ts in timestamps],
                self._tolerance_s,
            )  # [T,C,H,W] in [0,1]
        if self._viewpoint == "concat_view":
            return compose_multiview(
                frames_by_key[_IMAGE_FEATURES["top"]],
                frames_by_key[_IMAGE_FEATURES["left"]],
                frames_by_key[_IMAGE_FEATURES["right"]],
            )  # [T,C,3H/2,W]
        return frames_by_key[_IMAGE_FEATURES["top"]]

    def _build_raw_action(
        self, q: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``q``: ``[T+1, 14]`` absolute bimanual joints -> (``[T, 20]`` action, init poses, joint_configs)."""
        T = q.shape[0] - 1
        left_abs, right_abs = bimanual_yam_ee_poses(q)  # [T+1,4,4] each
        left_rel = pose_abs_to_rel(left_abs, rotation_format="rot6d", pose_convention=self._pose_convention)  # [T,9]
        right_rel = pose_abs_to_rel(right_abs, rotation_format="rot6d", pose_convention=self._pose_convention)
        # Gripper command for the T predicted steps; stored 1 = open, 0 = closed (Cosmos convention).
        left_grip = q[:T, YAM_LEFT_GRIPPER_IDX : YAM_LEFT_GRIPPER_IDX + 1]
        right_grip = q[:T, YAM_RIGHT_GRIPPER_IDX : YAM_RIGHT_GRIPPER_IDX + 1]
        action = np.concatenate(
            [left_rel[-T:], left_grip, right_rel[-T:], right_grip], axis=-1
        ).astype(np.float32)  # [T,20]
        initial_pose_left = torch.from_numpy(left_abs[0].copy()).float()
        initial_pose_right = torch.from_numpy(right_abs[0].copy()).float()
        joint_configs = torch.from_numpy(q[1 : 1 + T].copy()).float()  # [T,14] post-action joint targets
        return torch.from_numpy(action).float(), initial_pose_left, initial_pose_right, joint_configs
