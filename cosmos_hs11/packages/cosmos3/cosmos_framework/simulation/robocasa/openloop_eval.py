# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Open-loop action accuracy of a RoboCasa policy server on held-out demo windows (no simulator) -- cosmos_hs11.

Why: the closed-loop success rate of the RoboCasa365 recipes sits near the floor (~7 % mean) while the validation
flow-matching loss looks fine. The FM loss is a velocity-regression error at random noise levels; it says nothing about
how accurate the actually SAMPLED 16-step action chunk is. This script measures that directly, with the same inputs the
closed-loop eval sends (first frame of the window, language, EEF state token) and the same decoder
(``decode_15d_armfirst_to_env12``), against the demonstrator's actions of the same window:

  * small errors + high correlation  -> the policy fits the demos; closed-loop failure = compounding / precision / scenes
  * large errors / ~0 correlation    -> the policy never fit the demos; look at the recipe (normalization, state token, ...)

Windows come from ``RoboCasaLeRobotDataset`` (the training reader, ``split="val"`` = the seeded 1 % hold-out, 5 demos per
task, never trained on), so encoding, alignment and the state token are exactly the training ones. Metrics are reported in
the env's 12D action space ``[eef_pos(3), eef_rotvec(3), gripper(1), base_motion(4), control_mode(1)]`` (OSC inputs in
[-1, 1]), per dim group, together with two trivial baselines (zero action, per-task mean action) and the error as a
function of the step inside the chunk. The policy server is the normal ``action_policy_server_robocasa`` (one or several,
round-robin); see examples/eval_robocasa_openloop.sh.

    python cosmos_framework/simulation/robocasa/openloop_eval.py --server-urls http://127.0.0.1:8901 \
        --robocasa-root $ROBOCASA_ROOT --out-dir <dir> --windows-per-task 12
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import (
    DEFAULT_ALL_ATOMIC_TASKS,
    RoboCasaLeRobotDataset,
)
from cosmos_framework.simulation.robocasa.eval_utils import decode_15d_armfirst_to_env12, predict

ENV_GROUPS = {"pos": slice(0, 3), "rot": slice(3, 6), "grip": slice(6, 7), "base": slice(7, 11), "mode": slice(11, 12)}


def _window_dataset(root: str, task: str, episodes_per_task: int | None, split: str) -> RoboCasaLeRobotDataset:
    # Same kwargs as action_policy_robocasa_edge._robocasa_packing_loader (val loader: no cfg dropout, all val demos).
    return RoboCasaLeRobotDataset(
        root=root,
        fps=20,
        chunk_length=16,
        mode="wam",
        viewpoint="concat_view",
        camera_set="left_wrist",
        task_names=[task],
        use_state=True,
        use_base_action=True,
        base_encoding="raw_armfirst",
        episodes_per_task=episodes_per_task,
        episode_subset_seed=42,
        action_normalization=None,
        split=split,
        split_val_ratio=0.01,
        split_seed=42,
    )


def _sample_windows(ds: RoboCasaLeRobotDataset, n: int, rng: random.Random, min_motion: float) -> list[dict]:
    """``n`` windows whose demo chunk is not fully idle (the server is always told idle_frame=0, like the closed loop)."""
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    out, skipped = [], 0
    for i in idxs:
        s = ds[i]
        a = s["action"].float().numpy()  # [17, 15]: row 0 = state token, rows 1..16 = the chunk
        gt15 = a[1:]
        gt_env = np.stack([decode_15d_armfirst_to_env12(r, False) for r in gt15])  # [16, 12]
        if np.abs(gt_env[:, 0:6]).max() < min_motion:
            skipped += 1
            continue
        frame0 = s["video"][:, 0].permute(1, 2, 0).contiguous().numpy()  # [H, W, 3] uint8, what the eval composes
        out.append(
            dict(idx=i, frame=frame0, prompt=str(s["ai_caption"]), state=a[0].tolist(), gt15=gt15, gt_env=gt_env)
        )
        if len(out) >= n:
            break
    return out, skipped


def _query(server_url: str, w: dict, timeout: float, image_size: int) -> np.ndarray:
    res = predict(server_url, w["frame"], w["prompt"], image_size, timeout, state=w["state"])
    pred15 = np.asarray(res["action"], dtype=np.float32)[:16]
    if pred15.shape != (16, 15):
        raise RuntimeError(f"unexpected prediction shape {pred15.shape}")
    return pred15


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-urls", nargs="+", required=True, help="one or more policy servers (round-robin)")
    ap.add_argument("--robocasa-root", required=True, help="<ROBOCASA_DATA>/v30/target/atomic")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks", nargs="*", default=list(DEFAULT_ALL_ATOMIC_TASKS))
    ap.add_argument(
        "--split", default="val", choices=["val", "train"], help="val = held-out 1 %% demos; train = the training demos"
    )
    ap.add_argument(
        "--episodes-per-task",
        type=int,
        default=None,
        help="with --split train: the run's data fraction (50 / 150 / None)",
    )
    ap.add_argument("--windows-per-task", type=int, default=12)
    ap.add_argument("--repeat-first", type=int, default=2, help="windows per task queried twice (sampling noise)")
    ap.add_argument("--min-motion", type=float, default=0.02, help="skip windows whose demo arm command is below this")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    # 1. windows
    windows: list[dict] = []
    for task in args.tasks:
        t0 = time.time()
        ds = _window_dataset(
            args.robocasa_root, task, args.episodes_per_task if args.split == "train" else None, args.split
        )
        ws, skipped = _sample_windows(ds, args.windows_per_task, rng, args.min_motion)
        for w in ws:
            w["task"] = task
        windows.extend(ws)
        print(
            f"[openloop] {task}: {len(ds)} {args.split} windows, sampled {len(ws)} (skipped {skipped} idle) in {time.time() - t0:.0f}s",
            flush=True,
        )
    print(f"[openloop] {len(windows)} windows total; querying {args.server_urls}", flush=True)

    # 2. predictions (one worker per server)
    def run(i_w):
        i, w = i_w
        url = args.server_urls[i % len(args.server_urls)]
        pred = _query(url, w, args.timeout, args.image_size)
        pred2 = None
        if (i % args.windows_per_task) < args.repeat_first:
            pred2 = _query(url, w, args.timeout, args.image_size)
        return i, pred, pred2

    t0 = time.time()
    results: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
    with ThreadPoolExecutor(max_workers=len(args.server_urls)) as ex:
        for i, pred, pred2 in ex.map(run, list(enumerate(windows))):
            results[i] = (pred, pred2)
            if len(results) % 20 == 0:
                print(f"[openloop] {len(results)}/{len(windows)} done, {time.time() - t0:.0f}s", flush=True)
    print(f"[openloop] inference done in {time.time() - t0:.0f}s", flush=True)

    # 3. metrics in env space
    gt = np.stack([w["gt_env"] for w in windows])  # [N, 16, 12]
    pred_env = np.stack(
        [np.stack([decode_15d_armfirst_to_env12(r, False) for r in results[i][0]]) for i in range(len(windows))]
    )
    gt15 = np.stack([w["gt15"] for w in windows])
    pred15 = np.stack([results[i][0] for i in range(len(windows))])
    tasks = np.array([w["task"] for w in windows])
    task_mean = {t: gt[tasks == t].reshape(-1, 12).mean(0) for t in set(tasks)}
    base_mean = np.stack([np.broadcast_to(task_mean[t], (16, 12)) for t in tasks])
    summary: dict = {
        "n_windows": int(len(windows)),
        "windows_per_task": args.windows_per_task,
        "split": args.split,
        "episodes_per_task": args.episodes_per_task,
        "tasks": sorted(set(tasks)),
    }
    for g, sl in ENV_GROUPS.items():
        e = np.abs(pred_env[..., sl] - gt[..., sl])
        summary[g] = {
            "mae": float(e.mean()),
            "rmse": float(np.sqrt(((pred_env[..., sl] - gt[..., sl]) ** 2).mean())),
            "mae_zero_baseline": float(np.abs(gt[..., sl]).mean()),
            "mae_taskmean_baseline": float(np.abs(base_mean[..., sl] - gt[..., sl]).mean()),
            "corr": _corr(pred_env[..., sl], gt[..., sl]),
            "mean_abs_pred": float(np.abs(pred_env[..., sl]).mean()),
            "mean_abs_gt": float(np.abs(gt[..., sl]).mean()),
            "mae_per_dim": [float(x) for x in e.reshape(-1, e.shape[-1]).mean(0)],
            "corr_per_dim": [_corr(pred_env[..., sl][..., d], gt[..., sl][..., d]) for d in range(e.shape[-1])],
        }
    # gripper / mode sign agreement
    summary["grip"]["sign_acc"] = float((np.sign(pred_env[..., 6]) == np.sign(gt[..., 6])).mean())
    trans = np.array([len(set(np.sign(w["gt_env"][:, 6]))) > 1 for w in windows])
    summary["grip"]["n_windows_with_gripper_transition"] = int(trans.sum())
    if trans.any():
        summary["grip"]["sign_acc_transition_windows"] = float(
            (np.sign(pred_env[trans][..., 6]) == np.sign(gt[trans][..., 6])).mean()
        )
    summary["mode"]["sign_acc"] = float((np.sign(pred_env[..., 11]) == np.sign(gt[..., 11])).mean())
    # error vs step inside the chunk (pos MAE), and per task
    summary["pos_mae_per_step"] = [float(x) for x in np.abs(pred_env[..., 0:3] - gt[..., 0:3]).mean(axis=(0, 2))]
    summary["per_task"] = {
        t: {
            "pos_mae": float(np.abs(pred_env[tasks == t][..., 0:3] - gt[tasks == t][..., 0:3]).mean()),
            "pos_corr": _corr(pred_env[tasks == t][..., 0:3], gt[tasks == t][..., 0:3]),
            "rot_mae": float(np.abs(pred_env[tasks == t][..., 3:6] - gt[tasks == t][..., 3:6]).mean()),
            "grip_sign_acc": float((np.sign(pred_env[tasks == t][..., 6]) == np.sign(gt[tasks == t][..., 6])).mean()),
        }
        for t in sorted(set(tasks))
    }
    # raw 15D space (what the loss sees) + sampling noise
    summary["raw15_mae_per_dim"] = [float(x) for x in np.abs(pred15 - gt15).reshape(-1, 15).mean(0)]
    reps = [(results[i][0], results[i][1]) for i in range(len(windows)) if results[i][1] is not None]
    if reps:
        d = np.stack(
            [
                np.abs(
                    np.stack([decode_15d_armfirst_to_env12(r, False) for r in a])
                    - np.stack([decode_15d_armfirst_to_env12(r, False) for r in b])
                )
                for a, b in reps
            ]
        )
        summary["sampling_noise"] = {
            "n": len(reps),
            "pos_mad_between_two_samples": float(d[..., 0:3].mean()),
            "rot_mad": float(d[..., 3:6].mean()),
        }

    with open(os.path.join(args.out_dir, "openloop_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    np.savez_compressed(
        os.path.join(args.out_dir, "openloop_predictions.npz"),
        gt_env=gt,
        pred_env=pred_env,
        gt15=gt15,
        pred15=pred15,
        tasks=tasks,
        idx=np.array([w["idx"] for w in windows]),
    )
    # 4. report
    print("\n[openloop] env-space accuracy over", len(windows), "held-out windows x 16 steps")
    print(f"{'group':6s} {'MAE':>8s} {'zero-bl':>8s} {'mean-bl':>8s} {'corr':>7s} {'|pred|':>7s} {'|gt|':>7s}")
    for g in ENV_GROUPS:
        m = summary[g]
        print(
            f"{g:6s} {m['mae']:8.4f} {m['mae_zero_baseline']:8.4f} {m['mae_taskmean_baseline']:8.4f} {m['corr']:7.3f} {m['mean_abs_pred']:7.4f} {m['mean_abs_gt']:7.4f}"
        )
    print(
        f"gripper sign acc {summary['grip']['sign_acc']:.3f} (transition windows: {summary['grip'].get('sign_acc_transition_windows', float('nan')):.3f}, n={summary['grip']['n_windows_with_gripper_transition']}) | mode sign acc {summary['mode']['sign_acc']:.3f}"
    )
    print("pos MAE per chunk step:", " ".join(f"{x:.3f}" for x in summary["pos_mae_per_step"]))
    if "sampling_noise" in summary:
        sn = summary["sampling_noise"]
        print(
            f"sampling noise (two samples, same input): pos {sn['pos_mad_between_two_samples']:.4f} rot {sn['rot_mad']:.4f} (n={sn['n']})"
        )
    print(f"{'task':28s} {'pos_mae':>8s} {'pos_corr':>9s} {'rot_mae':>8s} {'grip_acc':>9s}")
    for t, m in summary["per_task"].items():
        print(f"{t:28s} {m['pos_mae']:8.4f} {m['pos_corr']:9.3f} {m['rot_mae']:8.4f} {m['grip_sign_acc']:9.3f}")
    print("[openloop] wrote", os.path.join(args.out_dir, "openloop_summary.json"))


if __name__ == "__main__":
    main()
