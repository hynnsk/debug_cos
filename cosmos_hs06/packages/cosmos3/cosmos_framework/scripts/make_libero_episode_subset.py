# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Sample a FIXED few-shot episode subset of a LIBERO LeRobot suite (N demos per task).

Writes a JSON that ``LIBEROLeRobotDataset(episode_subset_path=...)`` consumes, so the
same episodes are used on every rank and every run. Standalone: needs only
pyarrow + numpy (no torch / cosmos_framework import).

    python -m cosmos_framework.scripts.make_libero_episode_subset \\
        --root $LIBERO_ROOT --episodes-per-task 10 --seed 42 \\
        -o cosmos_framework/data/generator/action/episode_subsets/libero_10_10ep_per_task_seed42.json

The bundled subsets were produced with exactly the command above against the 20 FPS
``nvidia/LIBERO_LeRobot_v3`` ``libero_10`` suite (379 episodes / 10 tasks):

* ``libero_10_10ep_per_task_seed42.json`` — ``--episodes-per-task 10`` (100 eps, cosmos_hs03)
* ``libero_10_3ep_per_task_seed42.json``  — ``--episodes-per-task 3``  (30 eps, cosmos_hs04)
* ``libero_10_1ep_per_task_seed42.json``  — ``--episodes-per-task 1``  (10 eps, cosmos_hs05)

Re-running reproduces the same file; do NOT regenerate with a different seed unless
you intend to change the training set. Note the two subsets are sampled independently
(same seed, different N), so the 1-/3-shot sets are NOT subsets of the 10-shot set.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _load_episode_tasks(root: Path) -> tuple[dict[int, int], dict[int, str]]:
    """Return ``{episode_index: task_index}`` and ``{task_index: task_text}``."""
    tasks_df = pd.read_parquet(root / "meta" / "tasks.parquet")
    # LeRobot v2.x: "task" column; v3.0: task text is the (unnamed) index.
    task_texts = tasks_df["task"] if "task" in tasks_df.columns else tasks_df.index
    tasks = {int(ti): str(t) for t, ti in zip(task_texts, tasks_df["task_index"])}

    ep_parts, task_parts = [], []
    for path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "task_index"])
        ep_parts.append(table["episode_index"].to_numpy())
        task_parts.append(table["task_index"].to_numpy())
    if not ep_parts:
        raise FileNotFoundError(f"No data parquet found under {root / 'data'}.")
    ep = np.concatenate(ep_parts).astype(np.int64)
    tk = np.concatenate(task_parts).astype(np.int64)
    ep_vals, first = np.unique(ep, return_index=True)
    ep_task = {int(e): int(t) for e, t in zip(ep_vals, tk[first])}
    # Sanity: an episode must map to exactly one task.
    for e in ep_vals:
        assert len(np.unique(tk[ep == e])) == 1, f"episode {int(e)} spans multiple tasks"
    return ep_task, tasks


def sample_subset(ep_task: dict[int, int], tasks: dict[int, str], episodes_per_task: int, seed: int) -> dict:
    rng = random.Random(seed)  # one RNG, tasks visited in ascending task_index order
    out_tasks: dict[str, dict] = {}
    for task_index in sorted(tasks):
        candidates = sorted(e for e, t in ep_task.items() if t == task_index)
        if len(candidates) < episodes_per_task:
            raise ValueError(
                f"task {task_index} has only {len(candidates)} episodes (< {episodes_per_task})."
            )
        chosen = sorted(rng.sample(candidates, episodes_per_task))
        out_tasks[str(task_index)] = {
            "task": tasks[task_index],
            "n_available": len(candidates),
            "episodes": chosen,
        }
    n_total = sum(len(v["episodes"]) for v in out_tasks.values())
    return {
        "description": (
            f"Fixed few-shot LIBERO subset: {episodes_per_task} randomly sampled demonstration "
            f"episodes per task (random.Random(seed={seed}), tasks in ascending task_index order, "
            f"candidates sorted by episode_index). Consumed by LIBEROLeRobotDataset(episode_subset_path=...)."
        ),
        "seed": seed,
        "episodes_per_task": episodes_per_task,
        "n_tasks": len(out_tasks),
        "n_episodes_total_in_dataset": len(ep_task),
        "n_episodes": n_total,
        "tasks": out_tasks,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Local LeRobot suite dir (e.g. $LIBERO_ROOT = <dir>/libero_10).")
    ap.add_argument("--episodes-per-task", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-o", "--output", required=True, help="Output JSON path.")
    args = ap.parse_args()

    root = Path(args.root)
    ep_task, tasks = _load_episode_tasks(root)
    subset = sample_subset(ep_task, tasks, args.episodes_per_task, args.seed)
    subset["source_root_basename"] = root.name

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subset, indent=2) + "\n")
    print(
        f"Wrote {out}: {subset['n_episodes']} episodes "
        f"({args.episodes_per_task}/task x {subset['n_tasks']} tasks, seed={args.seed}) "
        f"out of {subset['n_episodes_total_in_dataset']}."
    )


if __name__ == "__main__":
    main()
