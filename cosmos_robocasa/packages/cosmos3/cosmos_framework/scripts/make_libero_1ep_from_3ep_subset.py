# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Derive a 1-demo-per-task LIBERO subset from an existing 3-demo-per-task subset JSON (cosmos_hs11 1ep recipes).

The single episode of every task is drawn with ``random.Random(seed)`` from that task's THREE episodes listed in the
parent subset (tasks in ascending task_index order, candidates sorted by episode_index), never from the whole dataset,
so the 1-demo training set is a strict subset of the 3-demo one. The output keeps the parent's bookkeeping fields and
records the choice under ``v1_selection`` so it can be audited::

    PYTHONPATH=. python cosmos_framework/scripts/make_libero_1ep_from_3ep_subset.py \
        --root $COSMOS_STORAGE/data/LIBERO_LeRobot_v3/libero_goal \
        --parent libero_goal_3ep_per_task_v2.json -o cosmos_framework/data/generator/action/episode_subsets/libero_goal_1ep_per_task_v1.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from cosmos_framework.scripts.make_libero_episode_subset import _load_episode_tasks

_SUBSETS_DIR = Path(__file__).resolve().parents[1] / "data" / "generator" / "action" / "episode_subsets"


def derive_1ep_subset(parent: dict, ep_task: dict[int, int], tasks: dict[int, str], seed: int, parent_name: str) -> dict:
    rng = random.Random(seed)
    out_tasks: dict[str, dict] = {}
    selection: dict[str, dict] = {}
    for task_key in sorted(parent["tasks"], key=int):
        entry = parent["tasks"][task_key]
        candidates = sorted(int(e) for e in (entry["episodes"] if isinstance(entry, dict) else entry))
        if len(candidates) != 3:
            raise ValueError(f"task {task_key}: expected 3 parent episodes, got {candidates}")
        for e in candidates:
            if ep_task.get(e) != int(task_key):
                raise ValueError(f"task {task_key}: parent episode {e} maps to task {ep_task.get(e)} in the dataset")
        chosen = rng.choice(candidates)
        n_available = sum(1 for t in ep_task.values() if t == int(task_key))
        out_tasks[task_key] = {"task": tasks[int(task_key)], "n_available": n_available, "episodes": [chosen]}
        selection[task_key] = {"task": tasks[int(task_key)], "candidates": candidates, "chosen": chosen}
    return {
        "description": (
            f"1 demonstration per task, drawn with random.Random(seed={seed}) from the 3 episodes each task has in "
            f"{parent_name} (tasks in ascending task_index order, candidates sorted by episode_index); NOT sampled from the "
            "whole dataset, so this is a strict subset of the parent 3-demo subset. Consumed by "
            "LIBEROLeRobotDataset(episode_subset_path=...)."
        ),
        "seed": seed,
        "episodes_per_task": 1,
        "n_excluded": 0,
        "n_tasks": len(out_tasks),
        "n_episodes_total_in_dataset": len(ep_task),
        "n_episodes": len(out_tasks),
        "tasks": out_tasks,
        "excluded_from": [],
        "source_root_basename": parent["source_root_basename"],
        "based_on": parent_name,
        "v1_seed": seed,
        "v1_selection": selection,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Local LeRobot suite dir (<dir>/libero_<suite>)")
    ap.add_argument("--parent", required=True, help="3-demo subset JSON (bundled file name or path)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-o", "--output", required=True, help="Output JSON path")
    args = ap.parse_args()

    parent_path = Path(args.parent)
    if not parent_path.exists():
        parent_path = _SUBSETS_DIR / parent_path.name
    parent = json.loads(parent_path.read_text())
    root = Path(args.root)
    if parent.get("source_root_basename") != root.name:
        raise ValueError(f"parent subset is for {parent.get('source_root_basename')!r} but --root is {root.name!r}")
    ep_task, tasks = _load_episode_tasks(root)
    out = derive_1ep_subset(parent, ep_task, tasks, args.seed, parent_path.name)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    picks = ", ".join(f"t{k}:{v['chosen']}" for k, v in out["v1_selection"].items())
    print(f"{Path(args.output).name}: {out['n_episodes']} episodes from {parent_path.name} -> {picks}")


if __name__ == "__main__":
    main()
