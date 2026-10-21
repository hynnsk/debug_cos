"""Minimal LeRobot **v3.0** dataset reader (parquet + PyAV video) for openpi.

openpi pins LeRobot at a pre-v3 commit whose ``LeRobotDataset`` cannot open v3.0 datasets (``meta/episodes/chunk-*/
file-*.parquet``, ``data/chunk-*/file-*.parquet`` holding MANY episodes per file, ``videos/<key>/chunk-*/file-*.mp4``
with per-episode ``from_timestamp`` offsets). Every dataset used by cosmos_hs11 (NVIDIA LIBERO_LeRobot_v3 and the five
robot_fewshot embodiments) is v3.0, so this module reads them directly and never imports ``lerobot``.

A sample is a dict with openpi-style keys::

    {"image/<alias>": uint8 [H, W, 3] (one per requested camera), "state": float32 [Ds], "actions": float32 [H, Da],
     "prompt": str, "task_index": int, "episode_index": int, "frame_index": int}

``actions`` is the action-horizon chunk ``rows[t : t + H]`` of the concatenated action columns, clipped at the episode end
by repeating the last action (LeRobot's ``delta_timestamps`` padding; openpi ignores the pad mask). Frames are decoded
at their real timestamps (``from_timestamp + frame_index / fps``) with PyAV, so AV1-encoded datasets work wherever PyAV
bundles dav1d. Heavy tables load lazily so the object pickles cheaply into spawned DataLoader workers.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import pathlib
import threading
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class V3DatasetSpec:
    """Which LeRobot v3.0 directory to read and which columns become the sample fields."""

    # One LeRobot root (contains meta/, data/, videos/) or several roots concatenated (e.g. the 100 YAM repos).
    roots: tuple[str, ...]
    # Output image alias -> video feature key, e.g. {"base": "observation.images.image", "wrist": "observation.images.wrist_image"}.
    # Stored as a tuple of (alias, key) pairs so the spec stays hashable (a dict is accepted and converted).
    image_keys: tuple[tuple[str, str], ...]
    # Data columns concatenated (in order) into the state / action vectors.
    state_keys: tuple[str, ...]
    action_keys: tuple[str, ...]
    # Optional fixed episode subset: json ``{"tasks": {"<task_index>": {"episodes": [...]}}}`` (cosmos_hs11 format);
    # bare file names resolve under ``assets/episode_subsets`` of the repo. Only meaningful for a single root.
    episode_subset: str | None = None
    # Episodes with fewer frames are dropped.
    min_frames: int = 1
    # Human-readable name (logging, assets).
    name: str = "v3"

    def __post_init__(self):
        pairs = self.image_keys.items() if isinstance(self.image_keys, Mapping) else self.image_keys
        object.__setattr__(self, "image_keys", tuple((str(k), str(v)) for k, v in pairs))
        object.__setattr__(self, "roots", tuple(str(r) for r in self.roots))
        if not self.roots:
            raise ValueError("V3DatasetSpec.roots must not be empty")
        if self.episode_subset is not None and len(self.roots) != 1:
            raise ValueError("episode_subset is only supported with a single root")

    @property
    def image_key_map(self) -> dict[str, str]:
        return dict(self.image_keys)


_EPISODE_SUBSETS_DIR = pathlib.Path(__file__).resolve().parents[3] / "assets" / "episode_subsets"


def resolve_episode_subset_path(path: str) -> pathlib.Path:
    p = pathlib.Path(path).expanduser()
    if not p.is_absolute() and not p.exists():
        p = _EPISODE_SUBSETS_DIR / p.name
    if not p.exists():
        raise FileNotFoundError(f"episode subset not found: {path!r} (also looked in {_EPISODE_SUBSETS_DIR})")
    return p


def load_episode_subset(path: str | pathlib.Path) -> dict[int, list[int]]:
    """``{task_index: [episode_index, ...]}`` from a cosmos_hs11 subset json."""
    spec = json.loads(pathlib.Path(path).read_text())
    tasks = spec.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError(f"Malformed episode subset {path}: expected a non-empty 'tasks' mapping.")
    out: dict[int, list[int]] = {}
    for task_key, entry in tasks.items():
        episodes = entry["episodes"] if isinstance(entry, dict) else entry
        out[int(task_key)] = [int(e) for e in episodes]
    return out


# ---------------------------------------------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------------------------------------------
def _column_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    """List<float> columns -> [n, d] float32 (flatten + reshape; every row must have the same length)."""
    arr = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if pa.types.is_list(arr.type) or pa.types.is_fixed_size_list(arr.type) or pa.types.is_large_list(arr.type):
        if pa.types.is_fixed_size_list(arr.type):
            width = arr.type.list_size
            flat = arr.values
        else:
            offsets = np.asarray(arr.offsets)
            lengths = np.diff(offsets)
            if lengths.size == 0:
                return np.zeros((0, 0), dtype=np.float32)
            if not np.all(lengths == lengths[0]):
                return np.stack([np.asarray(x, dtype=np.float32) for x in arr.to_pylist()], axis=0)
            width = int(lengths[0])
            flat = arr.flatten()
        values = np.asarray(flat.to_numpy(zero_copy_only=False), dtype=np.float32)
        return np.ascontiguousarray(values.reshape(len(arr), width))
    values = arr.to_numpy(zero_copy_only=False)
    if values.dtype == object:  # e.g. numpy arrays stored as objects
        return np.stack([np.asarray(x, dtype=np.float32) for x in values], axis=0)
    return np.asarray(values)


def _read_tasks(root: pathlib.Path) -> dict[int, str]:
    table = pq.read_table(root / "meta" / "tasks.parquet")
    columns = table.column_names
    if "task" in columns:  # LeRobot v2.x layout
        texts = table["task"].to_pylist()
    elif "__index_level_0__" in columns:  # pandas index stored by v3.0 (task text is the DataFrame index)
        texts = table["__index_level_0__"].to_pylist()
    else:
        # Fall back to pandas, which restores the (unnamed) index.
        df = table.to_pandas()
        texts = [str(t) for t in df.index]
    task_index = table["task_index"].to_pylist()
    return {int(i): str(t) for t, i in zip(texts, task_index, strict=True)}


@dataclasses.dataclass(frozen=True)
class EpisodeInfo:
    episode_index: int
    length: int
    task_index: int
    from_index: int  # first row of the episode in the index-sorted frame table
    videos: Mapping[str, tuple[pathlib.Path, float]]  # feature key -> (mp4 path, from_timestamp)


# ---------------------------------------------------------------------------------------------------------------
# Video decoding
# ---------------------------------------------------------------------------------------------------------------
def decode_frame(path: pathlib.Path | str, timestamp: float, *, tolerance_s: float) -> np.ndarray:
    """Decode the frame whose presentation time is closest to ``timestamp`` (seconds) as uint8 [H, W, 3]."""
    import av  # imported lazily: only DataLoader workers need it

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = float(stream.time_base) if stream.time_base is not None else None
        if time_base:
            # Seek to the last keyframe at or before the target, then decode forward.
            offset = int(max(timestamp - tolerance_s, 0.0) / time_base)
            container.seek(offset, stream=stream, backward=True, any_frame=False)
        best, best_dt = None, float("inf")
        for frame in container.decode(stream):
            t = frame.time
            if t is None:
                continue
            dt = abs(t - timestamp)
            if dt < best_dt:
                best, best_dt = frame, dt
            if t >= timestamp - 1e-6 and dt <= tolerance_s:
                return frame.to_ndarray(format="rgb24")
            if t > timestamp + tolerance_s:
                break
        if best is not None and best_dt <= 2 * tolerance_s:
            return best.to_ndarray(format="rgb24")
    raise RuntimeError(f"No frame within {tolerance_s:.4f}s of t={timestamp:.4f}s in {path}")


# ---------------------------------------------------------------------------------------------------------------
# Single-root dataset
# ---------------------------------------------------------------------------------------------------------------
class LeRobotV3Root:
    """One LeRobot v3.0 directory. Episode metadata is read eagerly (small); frame tables lazily (large)."""

    def __init__(self, root: str | pathlib.Path, spec: V3DatasetSpec, action_horizon: int, *, skip_images: bool = False):
        self.root = pathlib.Path(root)
        self.spec = spec
        self.action_horizon = int(action_horizon)
        self.skip_images = skip_images
        if not (self.root / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"{self.root} is not a LeRobot dataset dir (meta/info.json missing)")
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.fps = float(self.info["fps"])
        self.tasks = _read_tasks(self.root)
        self._tolerance_s = 0.5 / self.fps
        self._episodes: dict[int, EpisodeInfo] = self._read_episodes()
        self._lock = threading.Lock()
        self._frames: dict[str, np.ndarray] | None = None  # lazily loaded, index-sorted columns
        self._row_episode: np.ndarray | None = None
        self._row_frame: np.ndarray | None = None
        self._row_task: np.ndarray | None = None
        self._row_timestamp: np.ndarray | None = None
        self._state: np.ndarray | None = None
        self._actions: np.ndarray | None = None

    # ---- pickling (spawned DataLoader workers): drop the lock and the lazily loaded tables ---------------------
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_lock"] = None
        for key in ("_frames", "_row_episode", "_row_frame", "_row_task", "_row_timestamp", "_state", "_actions"):
            state[key] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    # ---- metadata -------------------------------------------------------------------------------------------
    def _video_path(self, key: str, chunk_index: int, file_index: int) -> pathlib.Path:
        template = self.info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
        rel = template.format(
            video_key=key, chunk_index=chunk_index, file_index=file_index, episode_chunk=chunk_index, episode_file=file_index
        )
        return self.root / rel

    def _read_episodes(self) -> dict[int, EpisodeInfo]:
        files = sorted((self.root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No episode metadata under {self.root / 'meta' / 'episodes'} (LeRobot v3.0 expected)")
        wanted = ["episode_index", "length", "dataset_from_index", "tasks"]
        for key in self.spec.image_key_map.values():
            wanted += [f"videos/{key}/chunk_index", f"videos/{key}/file_index", f"videos/{key}/from_timestamp"]
        episodes: dict[int, EpisodeInfo] = {}
        for f in files:
            schema = pq.read_schema(f)
            cols = [c for c in wanted if c in schema.names]
            table = pq.read_table(f, columns=cols)
            for row in table.to_pylist():
                ep = int(row["episode_index"])
                videos = {}
                for key in self.spec.image_key_map.values():
                    ci = int(row.get(f"videos/{key}/chunk_index", 0) or 0)
                    fi = int(row.get(f"videos/{key}/file_index", 0) or 0)
                    ts0 = float(row.get(f"videos/{key}/from_timestamp", 0.0) or 0.0)
                    videos[key] = (self._video_path(key, ci, fi), ts0)
                episodes[ep] = EpisodeInfo(
                    episode_index=ep,
                    length=int(row["length"]),
                    task_index=-1,  # filled from the frame table (tasks column is a repr string in v3.0)
                    from_index=int(row.get("dataset_from_index", -1) if row.get("dataset_from_index") is not None else -1),
                    videos=videos,
                )
        return episodes

    @property
    def episodes(self) -> dict[int, EpisodeInfo]:
        return self._episodes

    # ---- frame tables (lazy) ----------------------------------------------------------------------------------
    def _ensure_frames(self) -> None:
        if self._frames is not None:
            return
        with self._lock:
            if self._frames is not None:
                return
            files = sorted((self.root / "data").glob("chunk-*/file-*.parquet"))
            if not files:
                raise FileNotFoundError(f"No data parquet under {self.root / 'data'}")
            needed = ["index", "episode_index", "frame_index", "timestamp", "task_index", *self.spec.state_keys, *self.spec.action_keys]
            tables = []
            for f in files:
                schema = pq.read_schema(f)
                missing = [c for c in needed if c not in schema.names and c != "task_index"]
                if missing:
                    raise KeyError(f"{f}: missing columns {missing}; available: {schema.names}")
                cols = [c for c in needed if c in schema.names]
                tables.append(pq.read_table(f, columns=cols))
            table = pa.concat_tables(tables, promote_options="default")
            order = np.argsort(np.asarray(table["index"].to_numpy()).astype(np.int64), kind="stable")
            frames: dict[str, np.ndarray] = {}
            for name in table.column_names:
                frames[name] = np.ascontiguousarray(_column_to_numpy(table[name])[order])
            self._row_episode = frames["episode_index"].astype(np.int64).reshape(-1)
            self._row_frame = frames["frame_index"].astype(np.int64).reshape(-1)
            self._row_timestamp = frames["timestamp"].astype(np.float64).reshape(-1)
            if "task_index" in frames:
                self._row_task = frames["task_index"].astype(np.int64).reshape(-1)
            else:
                self._row_task = np.zeros_like(self._row_episode)
            self._state = np.concatenate([frames[k].reshape(len(order), -1) for k in self.spec.state_keys], axis=-1).astype(np.float32)
            self._actions = np.concatenate([frames[k].reshape(len(order), -1) for k in self.spec.action_keys], axis=-1).astype(np.float32)
            # Episode -> first row / task (robust to missing dataset_from_index).
            ep_vals, ep_starts = np.unique(self._row_episode, return_index=True)
            starts = {int(e): int(s) for e, s in zip(ep_vals, ep_starts, strict=True)}
            for ep, info in list(self._episodes.items()):
                if ep in starts:
                    s = starts[ep]
                    self._episodes[ep] = dataclasses.replace(info, from_index=s, task_index=int(self._row_task[s]))
            self._frames = frames

    def episode_task_index(self, episode_index: int) -> int:
        self._ensure_frames()
        return self._episodes[episode_index].task_index

    @property
    def state_dim(self) -> int:
        self._ensure_frames()
        return int(self._state.shape[-1])

    @property
    def action_dim(self) -> int:
        self._ensure_frames()
        return int(self._actions.shape[-1])

    # ---- samples ---------------------------------------------------------------------------------------------
    def frame(self, episode_index: int, t: int) -> dict[str, Any]:
        """Sample at local frame ``t`` of an episode (actions = chunk starting at t, clipped to the episode end)."""
        self._ensure_frames()
        info = self._episodes[episode_index]
        if not 0 <= t < info.length:
            raise IndexError(f"frame {t} outside episode {episode_index} of length {info.length}")
        start = info.from_index + t
        stop = info.from_index + info.length
        rows = np.arange(start, min(start + self.action_horizon, stop))
        actions = self._actions[rows]
        if actions.shape[0] < self.action_horizon:
            pad = np.repeat(actions[-1:], self.action_horizon - actions.shape[0], axis=0)
            actions = np.concatenate([actions, pad], axis=0)
        sample: dict[str, Any] = {
            "state": self._state[start].copy(),
            "actions": np.ascontiguousarray(actions, dtype=np.float32),
            "task_index": int(self._row_task[start]),
            "prompt": self.tasks.get(int(self._row_task[start]), ""),
            "episode_index": int(episode_index),
            "frame_index": int(t),
        }
        for alias, key in self.spec.image_key_map.items():
            if self.skip_images:
                sample[f"image/{alias}"] = np.zeros((1, 1, 3), dtype=np.uint8)
                continue
            path, ts0 = info.videos[key]
            timestamp = ts0 + float(self._row_timestamp[start] - self._row_timestamp[info.from_index])
            sample[f"image/{alias}"] = decode_frame(path, timestamp, tolerance_s=self._tolerance_s)
        return sample


# ---------------------------------------------------------------------------------------------------------------
# Public dataset: one or many roots, optional episode subset, flat frame index
# ---------------------------------------------------------------------------------------------------------------
class LeRobotV3Dataset:
    """Map-style dataset over the kept episodes of one or several v3.0 roots (openpi ``Dataset`` protocol)."""

    def __init__(self, spec: V3DatasetSpec, action_horizon: int, *, skip_images: bool = False):
        self.spec = spec
        self.action_horizon = int(action_horizon)
        self.roots = [LeRobotV3Root(r, spec, action_horizon, skip_images=skip_images) for r in spec.roots]
        self.fps = self.roots[0].fps
        # (root_id, episode_index) -> global episode id; global task ids by text when several roots are merged.
        self._episodes: list[tuple[int, int, int]] = []  # (root_id, episode_index, length)
        keep_sets: list[set[int] | None] = [None] * len(self.roots)
        if spec.episode_subset is not None:
            subset_path = resolve_episode_subset_path(spec.episode_subset)
            subset = load_episode_subset(subset_path)
            root = self.roots[0]
            keep: set[int] = set()
            for task_index, episodes in subset.items():
                for e in episodes:
                    if e not in root.episodes:
                        raise ValueError(f"Episode subset {subset_path} lists episode {e} which is not in {root.root}")
                    actual = root.episode_task_index(e)
                    if actual != task_index:
                        raise ValueError(
                            f"Episode subset {subset_path} lists episode {e} under task {task_index}, but the dataset maps it to task {actual}"
                        )
                    if e in keep:
                        raise ValueError(f"Episode subset {subset_path} lists episode {e} more than once")
                    keep.add(e)
            keep_sets[0] = keep
            logger.info(f"[{spec.name}] fixed episode subset {subset_path.name}: {len(keep)} episodes across {len(subset)} tasks")
        for rid, root in enumerate(self.roots):
            for ep, info in sorted(root.episodes.items()):
                if keep_sets[rid] is not None and ep not in keep_sets[rid]:
                    continue
                if info.length < spec.min_frames:
                    continue
                self._episodes.append((rid, ep, info.length))
        if not self._episodes:
            raise ValueError(f"[{spec.name}] no episodes kept")
        lengths = np.asarray([e[2] for e in self._episodes], dtype=np.int64)
        self._starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        self._lengths = lengths
        self._total = int(lengths.sum())
        logger.info(
            f"[{spec.name}] {len(self.roots)} root(s), {len(self._episodes)} episodes, {self._total} frames, fps {self.fps}, "
            f"horizon {self.action_horizon}"
        )

    # ---- episode bookkeeping (used by the Reptile sampler) ------------------------------------------------------
    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    def episode_length(self, episode_id: int) -> int:
        return int(self._lengths[episode_id])

    def episode_task_text(self, episode_id: int) -> str:
        rid, ep, _ = self._episodes[episode_id]
        root = self.roots[rid]
        return root.tasks.get(root.episode_task_index(ep), "")

    def episode_task_key(self, episode_id: int) -> str:
        """Task identity usable across roots (text, falling back to root-local index)."""
        rid, ep, _ = self._episodes[episode_id]
        root = self.roots[rid]
        ti = root.episode_task_index(ep)
        return root.tasks.get(ti) or f"{rid}:{ti}"

    def flat_index(self, episode_id: int, t: int) -> int:
        return int(self._starts[episode_id] + t)

    def locate(self, index: int) -> tuple[int, int]:
        """flat index -> (episode_id, local frame)."""
        if not 0 <= index < self._total:
            raise IndexError(index)
        ep = int(np.searchsorted(self._starts, index, side="right") - 1)
        return ep, int(index - self._starts[ep])

    @property
    def tasks(self) -> dict[int, str]:
        """Task index -> text of the FIRST root (what ``PromptFromLeRobotTask`` would use)."""
        return dict(self.roots[0].tasks)

    # ---- Dataset protocol -----------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return self._total

    def __getitem__(self, index) -> dict[str, Any]:
        ep, t = self.locate(int(index))
        return self.get_frame(ep, t)

    def get_frame(self, episode_id: int, t: int) -> dict[str, Any]:
        rid, ep, _ = self._episodes[episode_id]
        sample = self.roots[rid].frame(ep, t)
        sample["episode_id"] = int(episode_id)
        return sample


def episode_frame_pairs(dataset: LeRobotV3Dataset, episode_ids: Iterable[int], frames: Mapping[int, Sequence[int]]) -> list[int]:
    """Flat indices for ``{episode_id: [t, ...]}``."""
    return [dataset.flat_index(e, t) for e in episode_ids for t in frames[e]]


@functools.lru_cache(maxsize=None)
def _cached_dataset(spec: V3DatasetSpec, action_horizon: int, skip_images: bool) -> LeRobotV3Dataset:
    return LeRobotV3Dataset(spec, action_horizon, skip_images=skip_images)


def build_dataset(spec: V3DatasetSpec, action_horizon: int, *, skip_images: bool = False) -> LeRobotV3Dataset:
    """Construct (and cache per process) the dataset for a spec."""
    return _cached_dataset(spec, int(action_horizon), bool(skip_images))
