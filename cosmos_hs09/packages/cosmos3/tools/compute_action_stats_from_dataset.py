# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compute action normalization statistics (q01/q99/mean/std/min/max) for a Cosmos action dataset.

Unlike ``tools/compute_action_stats.py`` (raw LeRobot ``action`` column) this runs the dataset's own
*conversion* (e.g. joint -> FK -> frame-wise relative rot6d) so the stats match what the reader
emits with ``action_normalization=None``. Datasets that implement ``iter_episode_raw_actions()``
(MolmoAct2 YAM) are processed episode-by-episode without decoding video; other readers fall back to
sampling windows through ``__getitem__``.

Example (bundled YAM stats)::

    python tools/compute_action_stats_from_dataset.py --embodiment molmoact2_yam \\
        --data-root $ROBOT_FEWSHOT_ROOT \\
        -o cosmos_framework/data/generator/action/normalizer_stats/molmoact2_yam_stats.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cosmos_framework.data.generator.action.meta.embodiments import EMBODIMENT_REGISTRY, build_embodiment_raw_dataset


def _collect_episode_actions(dataset, frame_stride: int, max_frames: int | None) -> np.ndarray:
    parts: list[np.ndarray] = []
    total = 0
    t0 = time.time()
    for i, ep in enumerate(dataset.iter_episode_raw_actions()):
        if ep.shape[0] == 0:
            continue
        ep = ep[::frame_stride]
        parts.append(ep)
        total += ep.shape[0]
        if (i + 1) % 200 == 0:
            print(f"  episodes={i + 1} frames={total} elapsed={time.time() - t0:.0f}s", flush=True)
        if max_frames is not None and total >= max_frames:
            break
    if not parts:
        raise RuntimeError("No actions collected.")
    return np.concatenate(parts, axis=0)


def _collect_window_actions(dataset, num_windows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(dataset)
    idx = rng.choice(n, size=min(num_windows, n), replace=False)
    parts = []
    for j, i in enumerate(idx):
        parts.append(dataset[int(i)]["action"].numpy())
        if (j + 1) % 100 == 0:
            print(f"  windows={j + 1}", flush=True)
    return np.concatenate(parts, axis=0)


def compute_stats(actions: np.ndarray) -> dict[str, list[float]]:
    finite = np.isfinite(actions).all(axis=-1)
    dropped = int((~finite).sum())
    if dropped:
        print(f"WARNING: dropping {dropped} non-finite rows")
    a = actions[finite].astype(np.float64)
    return {
        "q01": np.quantile(a, 0.01, axis=0).round(6).tolist(),
        "q99": np.quantile(a, 0.99, axis=0).round(6).tolist(),
        "mean": a.mean(axis=0).round(6).tolist(),
        "std": a.std(axis=0).round(6).tolist(),
        "min": a.min(axis=0).round(6).tolist(),
        "max": a.max(axis=0).round(6).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embodiment", required=True, choices=sorted(EMBODIMENT_REGISTRY))
    ap.add_argument("--data-root", required=True, help="ROBOT_FEWSHOT_ROOT (parent of the per-embodiment dirs).")
    ap.add_argument("--root", default=None, help="Explicit dataset root (overrides --data-root/<subdir>).")
    ap.add_argument("--chunk-length", type=int, default=16)
    ap.add_argument("--frame-stride", type=int, default=1, help="Episode path: keep every k-th frame.")
    ap.add_argument("--max-frames", type=int, default=None, help="Episode path: stop after this many frames.")
    ap.add_argument("--num-windows", type=int, default=2000, help="Window fallback path: windows to sample.")
    ap.add_argument("--max-repos", type=int, default=None, help="YAM only: limit the number of repos.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    dataset_kwargs: dict = {"action_normalization": None}
    if args.embodiment == "molmoact2_yam" and args.max_repos is not None:
        dataset_kwargs["max_repos"] = args.max_repos
    dataset = build_embodiment_raw_dataset(
        args.embodiment,
        args.data_root,
        chunk_length=args.chunk_length,
        mode="wam",
        root_override=args.root,
        dataset_kwargs=dataset_kwargs,
    )
    t0 = time.time()
    if hasattr(dataset, "iter_episode_raw_actions"):
        actions = _collect_episode_actions(dataset, args.frame_stride, args.max_frames)
        source = "episodes"
    else:
        actions = _collect_window_actions(dataset, args.num_windows, args.seed)
        source = "windows"
    print(f"collected {actions.shape} raw actions ({source}) in {time.time() - t0:.0f}s")
    stats = compute_stats(actions)
    stats["_meta"] = {
        "embodiment": args.embodiment,
        "action_dim": int(actions.shape[1]),
        "action_names": list(getattr(dataset, "action_names", [])),
        "num_rows": int(actions.shape[0]),
        "source": source,
        "frame_stride": args.frame_stride,
        "chunk_length": args.chunk_length,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"Wrote {out}")
    for k in ("q01", "q99"):
        print(k, np.round(stats[k], 4).tolist())


if __name__ == "__main__":
    main()
