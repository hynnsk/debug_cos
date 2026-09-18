# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Column-store view of a LeRobot ``data/`` directory that behaves like ``ActionBaseDataset._rows``.

``ActionBaseDataset._rows`` materializes every frame of a LeRobot dataset as a Python dict
(``pq.read_table(...).to_pylist()``) and sorts the list. For the few-shot meta-training mix
(RT-1 3.8M frames, RoboMIND-UR 3.8M, YAM 10M, ...) that costs tens of GB per process and every
DataLoader worker (one per rank x worker) pays it again through copy-on-write page faults.

:class:`LazyRowTable` keeps the same columns as contiguous NumPy arrays (one array per parquet
column, rows sorted by the LeRobot ``index`` column) and builds the per-row dict only when a row
is accessed. It supports exactly the operations the readers use on ``self._rows``:

* ``len(rows)``
* ``rows[i]`` -> ``dict`` (Bridge: ``first_row = self._rows[idx]``)
* ``rows[a:b]`` -> ``list[dict]`` (every reader: ``self._rows[row_idx : row_idx + T + 1]``)
* iteration

Row values are NumPy scalars / 1-D arrays, which the readers already coerce with ``int()``,
``float()`` and ``np.asarray(...)``.

:class:`EpisodeTable` is the per-episode window bookkeeping the episodic few-shot sampler needs:
for every episode the first *flat dataset index* that starts a full ``chunk_length + 1`` frame
window inside the episode and how many such windows exist. For the row-indexed readers the flat
index is the row position; datasets with their own compact index (LIBERO, YAM) expose their own
``episode_table()`` with the same schema.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "EpisodeEntry",
    "EpisodeTable",
    "LazyRowTable",
    "build_episode_table_from_rows",
]


def _column_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    """Convert one parquet column to a NumPy array; fixed-width list columns become ``[N, D]``."""
    array = column.combine_chunks()
    col_type = array.type
    if pa.types.is_fixed_size_list(col_type):
        width = int(col_type.list_size)
        flat = array.flatten().to_numpy(zero_copy_only=False)
        return np.ascontiguousarray(flat.reshape(len(array), width))
    if pa.types.is_list(col_type) or pa.types.is_large_list(col_type):
        offsets = np.asarray(array.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
        widths = np.diff(offsets)
        if len(widths) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        if not np.all(widths == widths[0]):
            raise ValueError(f"Ragged list column (widths {np.unique(widths)[:5]}...) is not supported.")
        width = int(widths[0])
        flat = array.flatten().to_numpy(zero_copy_only=False)
        return np.ascontiguousarray(flat.reshape(len(array), width))
    if pa.types.is_struct(col_type) or pa.types.is_dictionary(col_type) or pa.types.is_string(col_type):
        return np.asarray(array.to_pylist(), dtype=object)
    return array.to_numpy(zero_copy_only=False)


class LazyRowTable:
    """Dict-on-access row view over column arrays (drop-in for ``ActionBaseDataset._rows``)."""

    def __init__(self, columns: dict[str, np.ndarray]) -> None:
        if not columns:
            raise ValueError("LazyRowTable needs at least one column.")
        lengths = {len(v) for v in columns.values()}
        if len(lengths) != 1:
            raise ValueError(f"All columns must have the same length, got {lengths}.")
        self._columns = columns
        self._n = int(lengths.pop())

    # ---- construction --------------------------------------------------------
    @classmethod
    def from_lerobot_root(
        cls,
        root: str | Path,
        columns: list[str] | None = None,
        exclude_episodes: Iterable[int] = (),
        sort_by: str | None = "index",
    ) -> "LazyRowTable":
        """Read ``<root>/data/chunk-*/file-*.parquet`` into column arrays sorted by ``sort_by``.

        Args:
            root: LeRobot dataset root (contains ``data/`` and ``meta/``).
            columns: Optional parquet column subset; ``None`` reads every column.
            exclude_episodes: ``episode_index`` values whose rows are dropped (e.g. the Google
                Robot base-motion outliers).
            sort_by: Column used to order rows (LeRobot's global frame ``index``). ``None`` keeps
                file order.
        """
        root = Path(root)
        files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No data parquet found under {root / 'data'}.")
        table = pa.concat_tables([pq.read_table(f, columns=columns) for f in files], promote_options="default")
        cols = {name: _column_to_numpy(table[name]) for name in table.column_names}
        n = table.num_rows
        order = np.arange(n)
        if sort_by is not None and sort_by in cols:
            order = np.argsort(cols[sort_by].astype(np.int64), kind="stable")
        exclude = sorted({int(e) for e in exclude_episodes})
        if exclude and "episode_index" in cols:
            keep = ~np.isin(cols["episode_index"][order], np.asarray(exclude, dtype=np.int64))
            order = order[keep]
        return cls({k: np.ascontiguousarray(v[order]) for k, v in cols.items()})

    # ---- rows interface -----------------------------------------------------
    def __len__(self) -> int:
        return self._n

    def _row(self, i: int) -> dict[str, Any]:
        return {k: v[i] for k, v in self._columns.items()}

    def __getitem__(self, idx: int | slice | np.integer) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(idx, slice):
            return [self._row(i) for i in range(*idx.indices(self._n))]
        i = int(idx)
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError(f"row index {idx} out of range for {self._n} rows")
        return self._row(i)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(self._n):
            yield self._row(i)

    @property
    def columns(self) -> list[str]:
        return list(self._columns.keys())

    def column(self, name: str) -> np.ndarray:
        """Return the full column array (no copy)."""
        return self._columns[name]

    # ---- episode bookkeeping --------------------------------------------------
    def episode_table(self, chunk_length: int) -> "EpisodeTable":
        """Per-episode window table with the row position as flat dataset index."""
        return build_episode_table_from_rows(
            episode_index=self.column("episode_index"),
            task_index=self.column("task_index") if "task_index" in self._columns else None,
            chunk_length=chunk_length,
        )


@dataclass(frozen=True)
class EpisodeEntry:
    """One demonstration: where its full-length windows live in the dataset's flat index space."""

    episode_index: int  # dataset-local episode id
    task_index: int  # dataset-local task id (-1 when unknown)
    first_flat_index: int  # first flat dataset index whose window lies entirely inside this episode
    num_windows: int  # number of consecutive valid windows (0 when the episode is too short)
    num_frames: int  # episode length in frames

    @property
    def flat_indices(self) -> range:
        return range(self.first_flat_index, self.first_flat_index + self.num_windows)


@dataclass
class EpisodeTable:
    """Episode -> window bookkeeping consumed by the episodic few-shot sampler."""

    entries: list[EpisodeEntry]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[EpisodeEntry]:
        return iter(self.entries)

    def __getitem__(self, i: int) -> EpisodeEntry:
        return self.entries[i]

    @property
    def total_windows(self) -> int:
        return int(sum(e.num_windows for e in self.entries))

    @property
    def task_ids(self) -> list[int]:
        return sorted({e.task_index for e in self.entries})

    def usable(self, min_windows: int = 1) -> "EpisodeTable":
        """Drop episodes shorter than ``min_windows`` full windows."""
        return EpisodeTable([e for e in self.entries if e.num_windows >= min_windows])

    def by_task(self) -> dict[int, list[int]]:
        """Map task id -> positions (into ``entries``) of the episodes with that task."""
        out: dict[int, list[int]] = {}
        for pos, e in enumerate(self.entries):
            out.setdefault(e.task_index, []).append(pos)
        return out

    def shifted(self, offset: int, episode_offset: int = 0, task_remap: dict[int, int] | None = None) -> "EpisodeTable":
        """Return a copy with flat indices (and optionally ids) shifted -- used by multi-root wrappers."""
        entries = []
        for e in self.entries:
            entries.append(
                EpisodeEntry(
                    episode_index=e.episode_index + episode_offset,
                    task_index=task_remap[e.task_index] if task_remap is not None else e.task_index,
                    first_flat_index=e.first_flat_index + offset,
                    num_windows=e.num_windows,
                    num_frames=e.num_frames,
                )
            )
        return EpisodeTable(entries)

    def blocks(self) -> list[tuple[int, int]]:
        """``(start, length)`` flat-index blocks for ``ActionIterableShuffleDataset.get_shuffle_blocks``."""
        return [(e.first_flat_index, e.num_windows) for e in self.entries if e.num_windows > 0]


def build_episode_table_from_rows(
    episode_index: np.ndarray,
    task_index: np.ndarray | None,
    chunk_length: int,
) -> EpisodeTable:
    """Build an :class:`EpisodeTable` for row-indexed readers (flat index == row position).

    The readers slice ``rows[r : r + chunk_length + 1]`` (current frame + ``chunk_length``
    future frames), so a window starting at row ``r`` is fully inside an episode occupying rows
    ``[start, start + count)`` iff ``r + chunk_length <= start + count - 1``. Episodes must be
    contiguous in row order (true for LeRobot data sorted by ``index``).
    """
    episode_index = np.asarray(episode_index).astype(np.int64)
    if episode_index.size == 0:
        return EpisodeTable([])
    if not np.all(np.diff(episode_index) >= 0):
        raise ValueError("episode_index must be non-decreasing (rows sorted by LeRobot index) to build episode blocks.")
    ep_vals, ep_starts, ep_counts = np.unique(episode_index, return_index=True, return_counts=True)
    task = None if task_index is None else np.asarray(task_index).astype(np.int64)
    entries: list[EpisodeEntry] = []
    for ep, start, count in zip(ep_vals.tolist(), ep_starts.tolist(), ep_counts.tolist()):
        entries.append(
            EpisodeEntry(
                episode_index=int(ep),
                task_index=int(task[start]) if task is not None else -1,
                first_flat_index=int(start),
                num_windows=max(0, int(count) - int(chunk_length)),
                num_frames=int(count),
            )
        )
    return EpisodeTable(entries)
