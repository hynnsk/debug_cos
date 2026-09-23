# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Concatenate many MolmoAct2 Bimanual-YAM repos into one dataset with a shared flat index.

The MolmoAct2-BimanualYAM collection is published as hundreds of small LeRobot repos
(``allenai/<date>-<task>-<nn>``). This wrapper owns one :class:`MolmoAct2YAMDataset` per repo and
maps a global flat index onto ``(child, local index)``::

    0 ... len(repo_0)-1        -> repo_0
    len(repo_0) ... +len(repo_1) -> repo_1
    ...

It also merges the per-repo ``get_shuffle_blocks()`` / ``episode_table()`` with the right offsets
(the streaming sampler shuffles episode blocks; the episodic sampler picks demonstrations), and
assigns *global* task ids by instruction text because every repo numbers its tasks from 0.
"""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.datasets.molmoact2_yam_dataset import (
    MolmoAct2YAMDataset,
    dual_arm_yam_action_spec,
)
from cosmos_framework.data.generator.action.meta.lazy_rows import EpisodeEntry, EpisodeTable
from cosmos_framework.data.generator.action.utils.action_spec import ActionSpec
from cosmos_framework.utils import log


def discover_yam_repo_roots(
    root: str | Path | None = None,
    roots: list[str] | tuple[str, ...] | None = None,
    manifest: str | Path | None = None,
    max_repos: int | None = None,
) -> list[Path]:
    """Resolve the list of LeRobot repo roots.

    Priority: explicit ``roots`` > ``manifest`` (one ``allenai/<name>`` or ``<name>`` per line, resolved
    under ``root``) > every ``<root>/*/meta/info.json`` subdirectory (sorted). ``max_repos`` truncates
    the sorted list (handy for smoke tests).
    """
    if roots:
        resolved = [Path(r) for r in roots]
    elif manifest is not None:
        if root is None:
            raise ValueError("manifest requires root (the directory holding the repo subdirectories).")
        names = [ln.strip() for ln in Path(manifest).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        resolved = [Path(root) / n.split("/")[-1] for n in names]
    else:
        if root is None:
            raise ValueError("Provide root, roots or manifest.")
        resolved = sorted(p for p in Path(root).iterdir() if (p / "meta" / "info.json").is_file())
    missing = [str(p) for p in resolved if not (p / "meta" / "info.json").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} YAM repo(s) are missing meta/info.json, e.g. {missing[:3]}")
    if max_repos is not None:
        resolved = resolved[: int(max_repos)]
    if not resolved:
        raise FileNotFoundError("No YAM repos found.")
    return resolved


class YAMRepoCollectionDataset(Dataset):
    """Offset-aware concatenation of :class:`MolmoAct2YAMDataset` children (see module docstring)."""

    def __init__(
        self,
        root: str | None = None,
        roots: list[str] | None = None,
        manifest: str | None = None,
        max_repos: int | None = None,
        **child_kwargs: Any,
    ) -> None:
        super().__init__()
        repo_roots = discover_yam_repo_roots(root=root, roots=roots, manifest=manifest, max_repos=max_repos)
        self._children: list[MolmoAct2YAMDataset] = []
        self._child_roots: list[Path] = []
        for repo in repo_roots:
            child = MolmoAct2YAMDataset(root=str(repo), **child_kwargs)
            if len(child) == 0:
                log.warning(f"Skipping YAM repo {repo.name}: no window of chunk_length+1 frames fits in any episode.")
                continue
            self._children.append(child)
            self._child_roots.append(repo)
        if not self._children:
            raise ValueError("YAMRepoCollectionDataset: every repo was empty.")
        self._offsets = [0]
        for child in self._children:
            self._offsets.append(self._offsets[-1] + len(child))
        self._length = self._offsets[-1]

        # Global task ids by instruction text (each repo numbers its own tasks from 0).
        self._task_texts: list[str] = []
        self._task_id_by_text: dict[str, int] = {}
        self._child_task_remap: list[dict[int, int]] = []
        for child in self._children:
            remap: dict[int, int] = {}
            for local_id, text in child._tasks.items():
                key = text.strip()
                if key not in self._task_id_by_text:
                    self._task_id_by_text[key] = len(self._task_texts)
                    self._task_texts.append(key)
                remap[int(local_id)] = self._task_id_by_text[key]
            self._child_task_remap.append(remap)
        table = self.episode_table()
        log.info(
            f"YAMRepoCollectionDataset: repos={len(self._children)} episodes={len(table)} "
            f"windows={self._length} unique_tasks={len(self._task_texts)}"
        )

    # ---- pass-through metadata (ActionBaseDataset-compatible surface) --------------
    @property
    def children(self) -> list[MolmoAct2YAMDataset]:
        return self._children

    @property
    def repo_roots(self) -> list[Path]:
        return list(self._child_roots)

    @property
    def task_texts(self) -> list[str]:
        return list(self._task_texts)

    def _first(self) -> MolmoAct2YAMDataset:
        return self._children[0]

    @property
    def fps(self) -> float:
        return self._first().fps

    @property
    def chunk_length(self) -> int:
        return self._first().chunk_length

    @property
    def mode(self) -> str:
        return self._first().mode

    @mode.setter
    def mode(self, value: str) -> None:
        for child in self._children:
            child.mode = value

    @property
    def domain_name(self) -> str:
        return self._first().domain_name

    @property
    def domain_id(self) -> int:
        return self._first().domain_id

    @property
    def viewpoint(self) -> str:
        return self._first().viewpoint

    @property
    def action_dim(self) -> int:
        return 20

    @property
    def action_normalization(self) -> str | None:
        return self._first().action_normalization

    @property
    def action_names(self) -> list[str]:
        return dual_arm_yam_action_spec().names

    def _action_spec(self) -> ActionSpec:
        return dual_arm_yam_action_spec()

    @classmethod
    def load_action_stats(cls) -> dict[str, torch.Tensor]:
        return MolmoAct2YAMDataset.load_action_stats()

    # ---- indexing --------------------------------------------------------------
    def __len__(self) -> int:
        return self._length

    def _locate(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += self._length
        if not 0 <= idx < self._length:
            raise IndexError(f"index {idx} out of range for {self._length} windows")
        child_idx = bisect.bisect_right(self._offsets, idx) - 1
        return child_idx, idx - self._offsets[child_idx]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        child_idx, local = self._locate(int(idx))
        return self._children[child_idx][local]

    def iter_episode_raw_actions(self):
        """Yield raw ``[n-1, 20]`` per-episode actions across every repo (for normalization stats)."""
        for child in self._children:
            yield from child.iter_episode_raw_actions()

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        blocks: list[tuple[int, int]] = []
        for child, offset in zip(self._children, self._offsets[:-1]):
            blocks.extend((start + offset, length) for start, length in child.get_shuffle_blocks())
        return blocks

    def episode_table(self) -> EpisodeTable:
        """Global episode table: flat indices offset per repo, episode ids made unique, tasks remapped."""
        entries: list[EpisodeEntry] = []
        episode_offset = 0
        for child, offset, remap in zip(self._children, self._offsets[:-1], self._child_task_remap):
            table = child.episode_table()
            for e in table.entries:
                entries.append(
                    EpisodeEntry(
                        episode_index=episode_offset + e.episode_index,
                        task_index=remap.get(e.task_index, -1),
                        first_flat_index=offset + e.first_flat_index,
                        num_windows=e.num_windows,
                        num_frames=e.num_frames,
                    )
                )
            episode_offset += (max(ep.episode_index for ep in table.entries) + 1) if len(table) else 0
        return EpisodeTable(entries)
