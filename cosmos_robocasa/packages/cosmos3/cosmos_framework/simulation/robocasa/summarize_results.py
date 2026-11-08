# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Aggregate per-task ``results.json`` files of ``closed_loop_eval.py`` into one ``summary.json`` + a table.

    python cosmos_framework/simulation/robocasa/summarize_results.py <OUT_DIR> [--tasks T1 T2 ...]

Reads ``<OUT_DIR>/<task>/results.json`` (one record per rollout, ``"policy": true|false``), writes
``<OUT_DIR>/summary.json`` with per-task and mean (task-averaged, the RoboCasa365 protocol) success rates and prints
the table. Standard library only, so it runs in either Python env. Missing tasks are listed, not silently dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_TASKS = (
    "CloseBlenderLid", "CloseFridge", "CloseToasterOvenDoor", "CoffeeSetupMug", "NavigateKitchen", "OpenCabinet",
    "OpenDrawer", "OpenStandMixerHead", "PickPlaceCounterToCabinet", "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter", "PickPlaceSinkToCounter", "PickPlaceToasterToCounter", "SlideDishwasherRack",
    "TurnOffStove", "TurnOnElectricKettle", "TurnOnMicrowave", "TurnOnSinkFaucet",
)


def summarize(out_dir: Path, tasks: list[str]) -> dict:
    per_task: dict[str, dict] = {}
    missing: list[str] = []
    for task in tasks:
        f = out_dir / task / "results.json"
        if not f.is_file():
            missing.append(task)
            continue
        records = json.loads(f.read_text())
        n = len(records)
        ok = sum(1 for r in records if r.get("policy"))
        per_task[task] = {
            "successes": ok,
            "episodes": n,
            "success_rate": (ok / n) if n else 0.0,
            "mean_steps": (sum(int(r.get("steps", 0)) for r in records) / n) if n else None,
        }
    rates = [v["success_rate"] for v in per_task.values()]
    total_ok = sum(v["successes"] for v in per_task.values())
    total_n = sum(v["episodes"] for v in per_task.values())
    return {
        "num_tasks_evaluated": len(per_task),
        "num_tasks_missing": len(missing),
        "missing_tasks": missing,
        "mean_task_success_rate": (sum(rates) / len(rates)) if rates else None,  # RoboCasa365 protocol: average over tasks
        "pooled_success_rate": (total_ok / total_n) if total_n else None,
        "total_successes": total_ok,
        "total_episodes": total_n,
        "per_task": per_task,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    args = ap.parse_args(argv)

    summary = summarize(args.out_dir, args.tasks)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    width = max(len(t) for t in args.tasks)
    print(f"{'task':<{width}}  succ/ep   rate")
    for task, v in summary["per_task"].items():
        print(f"{task:<{width}}  {v['successes']:>3}/{v['episodes']:<3}  {100 * v['success_rate']:5.1f}%")
    for task in summary["missing_tasks"]:
        print(f"{task:<{width}}  MISSING (no results.json)")
    if summary["mean_task_success_rate"] is not None:
        print(
            f"\nmean over {summary['num_tasks_evaluated']} tasks: {100 * summary['mean_task_success_rate']:.1f}%   "
            f"(pooled {summary['total_successes']}/{summary['total_episodes']} = {100 * summary['pooled_success_rate']:.1f}%)"
        )
    print(f"summary: {args.out_dir / 'summary.json'}")
    return 1 if summary["missing_tasks"] else 0


if __name__ == "__main__":
    sys.exit(main())
