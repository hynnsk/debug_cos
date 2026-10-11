# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU test of the action-similarity hard negatives inside ``LIBEROLeRobotDataset`` (cosmos_hs11 v27) on a
synthetic 10-task LeRobot v3 layout (parquet only; the constructor never decodes video)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import LIBEROLeRobotDataset

_FRAMES = 40
_DEMOS = 3


def _write_fake_libero(root, rng):
    """10 tasks x 3 demos. Tasks 0 and 1 share the same start motion; task 9's demos are unique."""
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 20, "codebase_version": "v3.0"}))
    names = [f"task number {t}" for t in range(10)]
    pd.DataFrame({"task_index": list(range(10))}, index=names).to_parquet(root / "meta" / "tasks.parquet")
    patterns = {t: rng.normal(size=(_FRAMES, 7)) * 0.05 for t in range(10)}
    patterns[1] = patterns[0] + 0.001 * rng.normal(size=(_FRAMES, 7))
    rows, episodes, index = [], [], 0
    for ep in range(10 * _DEMOS):
        task = ep // _DEMOS
        episodes.append({"episode_index": ep, "task_index": task, "length": _FRAMES})
        for f in range(_FRAMES):
            action = patterns[task][f] + 0.0005 * rng.normal(size=7)
            # gripper channel: task-dependent, but identical for tasks 0 and 1 (they must stay near-duplicates)
            action[6] = 1.0 if ((task // 2) % 2 == 0) == (f < _FRAMES // 2) else -1.0
            rows.append(
                {
                    "index": index,
                    "episode_index": ep,
                    "task_index": task,
                    "timestamp": f / 20.0,
                    "action": action.astype(np.float32).tolist(),
                }
            )
            index += 1
    pq.write_table(pa.Table.from_pylist(episodes), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.Table.from_pylist(rows), root / "data" / "chunk-000" / "file-000.parquet")


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("fake_libero_goal")
    _write_fake_libero(root, np.random.default_rng(0))
    return root


def _dataset(root, **kwargs):
    return LIBEROLeRobotDataset(root=str(root), split="full", **kwargs)


def test_action_similarity_negatives_are_built_from_the_kept_demonstrations(fake_root):
    ds = _dataset(
        fake_root,
        language_ranking=True,
        language_ranking_negative_sampling="action_similarity",
        language_ranking_hard_fraction=1.0,
        language_ranking_hard_temperature=0.1,  # sharp: p(1|0) = 1 / (1 + 8 e^-10) ~ 1
        language_ranking_hard_prefix_frames=16,
    )
    assert len(ds) == 10 * _DEMOS * (_FRAMES - 16)
    negatives = ds.instruction_negatives
    assert negatives is not None and negatives.weights is not None
    order, z = ds._ranking_negative_distances
    assert order == list(range(10)) and z.shape == (10, 10) and np.isclose(np.diag(z).sum(), 0.0)
    assert z[0, 1] < 0.1 < z[0, 9]
    w0 = dict(zip([t for t in order if t != 0], negatives.weights[0]))
    assert max(w0, key=w0.get) == 1 and w0[1] > 0.8
    report = ds.ranking_negative_report
    assert report and "task number 1" in report and "t0" in report
    # samples honour the weights: task 0 windows essentially always get task 1's caption
    assert {negatives.sample(0) for _ in range(100)} == {"task number 1"}


def test_uniform_mode_and_defaults_are_the_v23_behaviour(fake_root):
    ds = _dataset(fake_root, language_ranking=True)
    assert ds.instruction_negatives is not None and ds.instruction_negatives.weights is None
    assert ds.ranking_negative_report is None and ds._ranking_negative_distances is None
    ds2 = _dataset(fake_root, language_ranking=True, language_ranking_negative_sampling="uniform",
                   language_ranking_hard_fraction=0.9)
    assert ds2.instruction_negatives.weights is None  # the hard knobs are ignored in uniform mode
    assert _dataset(fake_root, language_ranking=False).instruction_negatives is None


def test_episode_subset_restricts_the_trajectories_used_for_the_weights(fake_root, tmp_path):
    # Keep only the FIRST demo of every task and make task 9's kept demo a copy of task 0's pattern:
    # the weights must follow the kept subset, not the whole dataset.
    subset = {"tasks": {str(t): {"episodes": [t * _DEMOS]} for t in range(10)}, "n_episodes": 10}
    path = tmp_path / "subset.json"
    path.write_text(json.dumps(subset))
    ds = _dataset(
        fake_root,
        episode_subset_path=str(path),
        language_ranking=True,
        language_ranking_negative_sampling="action_similarity",
        language_ranking_hard_fraction=1.0,
        language_ranking_hard_temperature=0.5,
        language_ranking_hard_prefix_frames=64,  # longer than the 40-frame demos -> clipped per demo
    )
    assert len(ds) == 10 * (_FRAMES - 16)
    order, z = ds._ranking_negative_distances
    assert z[0, 1] < 0.1
    assert len(ds.instruction_negatives.weights) == 10


def test_bad_hard_negative_settings_fail_at_construction(fake_root):
    with pytest.raises(ValueError, match="not one of"):
        _dataset(fake_root, language_ranking=True, language_ranking_negative_sampling="text")
    with pytest.raises(ValueError, match="prefix_frames"):
        _dataset(fake_root, language_ranking=True, language_ranking_negative_sampling="action_similarity",
                 language_ranking_hard_prefix_frames=0)
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        _dataset(fake_root, language_ranking=True, language_ranking_negative_sampling="action_similarity",
                 language_ranking_hard_fraction=2.0)
    with pytest.raises(ValueError, match="positive and finite"):
        _dataset(fake_root, language_ranking=True, language_ranking_negative_sampling="action_similarity",
                 language_ranking_hard_temperature=0.0)
