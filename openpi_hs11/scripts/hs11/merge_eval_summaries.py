"""Merge several eval summary.json files (e.g. task subsets run on different GPUs) into one.

    python scripts/hs11/merge_eval_summaries.py --out <dir>/summary.json <dir_a>/summary.json <dir_b>/summary.json ...
"""

from __future__ import annotations

import argparse
import json
import pathlib


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    merged = None
    tasks: dict[int, dict] = {}
    for p in args.inputs:
        d = json.loads(pathlib.Path(p).read_text())
        if merged is None:
            merged = {k: v for k, v in d.items() if k not in ("task_results", "selected_task_ids", "total_episodes", "total_successes", "overall_success_rate")}
        elif d["task_suite"] != merged["task_suite"]:
            raise ValueError(f"{p}: suite {d['task_suite']} != {merged['task_suite']}")
        for t in d["task_results"]:
            if t["task_id"] in tasks:
                raise ValueError(f"task {t['task_id']} appears twice ({p})")
            tasks[t["task_id"]] = t
    results = [tasks[k] for k in sorted(tasks)]
    merged["task_results"] = results
    merged["selected_task_ids"] = sorted(tasks)
    merged["total_episodes"] = sum(t["episodes"] for t in results)
    merged["total_successes"] = sum(t["successes"] for t in results)
    merged["overall_success_rate"] = merged["total_successes"] / max(1, merged["total_episodes"])
    merged["merged_from"] = args.inputs
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2))
    print(f"{merged['task_suite']}: {len(results)} tasks, SR {merged['overall_success_rate']:.3f} over {merged['total_episodes']} episodes -> {out}")
    for t in results:
        print(f"  t{t['task_id']} {t['success_rate']:.2f}  {t['task_description']}")


if __name__ == "__main__":
    main()
