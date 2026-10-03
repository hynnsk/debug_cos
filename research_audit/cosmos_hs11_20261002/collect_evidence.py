"""CPU-only, read-only audit of existing LIBERO/Reptile artifacts.

Run with the workspace's default Python (requires pyarrow, PyYAML).
Only audit_summary.json next to this script is written. No training is launched.
"""

from collections import Counter
import json
import math
from pathlib import Path
import random
import re
import statistics

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "cosmos_hs11/packages/cosmos3"
OUT = ROOT / "cosmos_storage/outputs"
SUBSETS = CODE / "cosmos_framework/data/generator/action/episode_subsets"
DATA = ROOT / "cosmos_storage/data/LIBERO_LeRobot_v3/libero_10"
META = OUT / "cosmos3_action_meta/reptile_meta/edge_reptile_full_k8w16_s10_eps0.5_seed42_v2"
POST = OUT / "cosmos3_action_libero/edge_libero10_fewshot"
REPTILE_NAME = "edge_libero10_3ep_fullft_reptileinit"
REPA_NAME = REPTILE_NAME + "_repa_dinov2b_l24_avgpool_w5.0"


def read_json(path):
    return json.loads(path.read_text())


def relative(path):
    return str(path.relative_to(ROOT))


train = read_json(SUBSETS / "libero_10_3ep_per_task_seed42.json")
val = read_json(SUBSETS / "libero_10_val_5ep_per_task_seed42_excl3ep.json")
train_ids, val_ids = [
    {e for task in subset["tasks"].values() for e in task["episodes"]}
    for subset in (train, val)
]
assert len(train_ids) == 30 and len(val_ids) == 50 and not train_ids & val_ids
episode_tasks, lengths = {}, Counter()
for file in sorted((DATA / "data").rglob("*.parquet")):
    columns = pq.read_table(file, columns=["episode_index", "task_index"]).to_pydict()
    for episode, task in zip(columns["episode_index"], columns["task_index"]):
        assert episode not in episode_tasks or episode_tasks[episode] == task
        episode_tasks[episode] = task
        lengths[episode] += 1
rng = random.Random(42)
task_data = []
for key, task in train["tasks"].items():
    candidates = sorted(e for e, t in episode_tasks.items() if t == int(key))
    assert sorted(rng.sample(candidates, 3)) == task["episodes"]
    task_data.append({
        "dataset_task_id": int(key), "description": task["task"],
        "train_episodes": task["episodes"],
        "train_windows": sum(lengths[e] - 16 for e in task["episodes"]),
    })
for key, task in val["tasks"].items():
    assert all(episode_tasks[e] == int(key) for e in task["episodes"])
data_summary = {
    "episodes": len(lengths), "frames": sum(lengths.values()),
    "train_episodes": len(train_ids), "val_episodes": len(val_ids), "overlap": [],
    "seed42_sampling_reproduced": True,
    "train_windows": sum(lengths[e] - 16 for e in train_ids),
    "val_windows": sum(lengths[e] - 16 for e in val_ids),
    "all_windows": sum(lengths[e] - 16 for e in lengths), "per_task": task_data,
}

names = {
    "hs08_v2_50demo_2000", "hs10_repa_dinov2b_l24_avgpool_w5.0_50demo_2000",
    REPTILE_NAME + "_50demo_2000", REPA_NAME + "_50demo_2000",
    REPTILE_NAME + "_repa_dinov2b_l8_avgpool_w5.0_50demo_2000",
    REPTILE_NAME + "_repa_dinov2b_l8_avgpool_w5.0_ramp200_50demo_2000",
    REPTILE_NAME + "_repa_dinov2b_l8_avgpool_w1.0_ramp200_50demo_2000",
    REPTILE_NAME + "_repa_dinov2b_l24_subgrid122_w5.0_50demo_2000",
    "edge_libero10_all379_fullft_reptileinit_50demo_2000",
}
evals, raw_evals = {}, {}
for file in sorted((OUT / "eval").rglob("summary.json")):
    name = file.parent.name
    if name not in names:
        continue
    assert name not in evals, f"Ambiguous run name: {name}"
    raw = read_json(file)
    raw_evals[name] = raw
    evals[name] = {
        "path": relative(file), "total_episodes": raw["total_episodes"],
        "total_successes": raw["total_successes"], "success_rate": raw["overall_success_rate"],
        "per_task": [{k: t[k] for k in ("task_id", "task_description", "episodes", "successes")}
                     for t in raw["task_results"]],
        "metadata_limitation": "Summary does not record checkpoint, eval seed, initial-state hashes, or server settings.",
    }
a = raw_evals[REPTILE_NAME + "_50demo_2000"]
b = raw_evals[REPA_NAME + "_50demo_2000"]
before = {(t["task_description"], e["episode"]): e["success"]
          for t in a["task_results"] for e in t["episode_results"]}
after = {(t["task_description"], e["episode"]): e["success"]
         for t in b["task_results"] for e in t["episode_results"]}
assert before.keys() == after.keys()
wins = sum(after[k] and not before[k] for k in before)
losses = sum(before[k] and not after[k] for k in before)
discordant = wins + losses
paired = {
    "conditional_only": "Episode IDs match; identical historical rollout conditions are not proven by the summaries.",
    "wins": wins, "losses": losses,
    "exact_mcnemar_two_sided_p": min(1, 2 * sum(math.comb(discordant, k)
        for k in range(min(wins, losses) + 1)) / 2**discordant),
}

rows = [json.loads(line) for line in (META / "reptile_train_log.jsonl").read_text().splitlines()]
buckets = []
for lo, hi in ((1, 250), (251, 500), (501, 750), (751, 1000)):
    selected = [r for r in rows if lo <= r["iteration"] <= hi and math.isfinite(r["query_loss"])]
    buckets.append({
        "iteration_range": [lo, hi], "query_evaluations": len(selected),
        "zero_shot_mean": statistics.mean(r["query_loss_zero_shot"] for r in selected),
        "adapted_mean": statistics.mean(r["query_loss"] for r in selected),
        "negative_gain_count": sum(r["query_loss_gain"] < 0 for r in selected),
    })
meta_summary = {
    "path": relative(META / "reptile_train_log.jsonl"), "logged_iterations": len(rows),
    "last_iteration": rows[-1]["iteration"], "query_buckets": buckets,
    "embodiment_counts": dict(Counter(e for r in rows for e in r["embodiments"])),
    "summed_hours": {k: sum(r[k] for r in rows) / 3600
                     for k in ("iter_time", "t_encode", "t_inner", "t_eval", "t_data")},
}
configs, val_logs = {}, {}
for folder in (META, POST / REPTILE_NAME, POST / REPA_NAME, POST / "hs08_v2"):
    cfg = yaml.safe_load((folder / "config.yaml").read_text())
    configs[folder.name] = {
        "path": relative(folder / "config.yaml"),
        "checkpoint": {k: cfg["checkpoint"].get(k) for k in ("load_path", "meta_action_init_path", "load_ema_to_reg")},
        "lr": cfg["optimizer"]["lr"], "lr_multipliers": cfg["optimizer"].get("lr_multipliers"),
        "cycle_lengths": cfg["scheduler"]["cycle_lengths"],
        "validation": {k: v for k, v in cfg["trainer"].items() if "val" in k},
        "repa": cfg["model"]["config"].get("repa", {}),
    }
    val_logs[folder.name] = []
    for file in sorted((folder / "wandb").glob("run-*/files/output.log")):
        for number, line in enumerate(file.read_text().splitlines(), 1):
            match = re.search(r"\[val\] iter (\d+): (.*)", line)
            if match:
                vals = {k: float(v) for k, v in re.findall(r"(\w+)=(-?[\d.]+)", match[2])}
                val_logs[folder.name].append({"iteration": int(match[1]), "path": relative(file), "line": number, **vals})

output = {
    "data": data_summary, "evaluation": evals, "conditional_paired_comparison": paired,
    "meta": meta_summary, "saved_configs": configs, "validation_logs": val_logs,
}
destination = Path(__file__).with_name("audit_summary.json")
destination.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(f"Wrote {destination}")
print(f"Checked {len(lengths)} episodes; train={len(train_ids)}, val={len(val_ids)}, disjoint; {len(evals)} evaluations.")
