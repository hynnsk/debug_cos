"""Per-embodiment quantile normalization statistics for Reptile meta-training (state + actions only, no video decode).

Writes ``<assets_dir>/<embodiment>/norm_stats.json`` in openpi's format (mean/std/q01/q99), computed over up to
``max_frames`` uniformly spread frames of each embodiment::

    uv run scripts/hs11/compute_norm_stats_reptile.py --assets-dir ./assets/pi05_reptile --max-frames 200000
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib

import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as normalize
from openpi.hs11 import embodiments as _emb
from openpi.hs11 import lerobot_v3


@dataclasses.dataclass
class Args:
    assets_dir: str = "./assets/pi05_reptile"
    embodiments: tuple[str, ...] = _emb.DEFAULT_EMBODIMENTS
    data_root: str | None = None
    action_horizon: int = 10
    max_frames: int = 200_000  # per embodiment; frames are spread uniformly over the dataset
    overwrite: bool = False


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    for name in args.embodiments:
        out_dir = pathlib.Path(args.assets_dir) / name
        if (out_dir / "norm_stats.json").is_file() and not args.overwrite:
            logging.info(f"{name}: {out_dir / 'norm_stats.json'} exists, skipping (use --overwrite)")
            continue
        spec = _emb.EMBODIMENTS[name].dataset_spec(args.data_root)
        dataset = lerobot_v3.LeRobotV3Dataset(spec, args.action_horizon, skip_images=True)
        n = len(dataset)
        indices = np.linspace(0, n - 1, num=min(n, args.max_frames)).astype(np.int64)
        stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
        buf_state, buf_actions = [], []
        for i in tqdm.tqdm(indices, desc=f"{name} ({n} frames)"):
            sample = dataset[int(i)]
            buf_state.append(sample["state"])
            buf_actions.append(sample["actions"])
            if len(buf_state) >= 4096:
                stats["state"].update(np.stack(buf_state))
                stats["actions"].update(np.stack(buf_actions))
                buf_state, buf_actions = [], []
        if buf_state:
            stats["state"].update(np.stack(buf_state))
            stats["actions"].update(np.stack(buf_actions))
        norm_stats = {k: v.get_statistics() for k, v in stats.items()}
        normalize.save(out_dir, norm_stats)
        logging.info(f"{name}: wrote {out_dir / 'norm_stats.json'} (state dim {norm_stats['state'].mean.shape}, action dim {norm_stats['actions'].mean.shape})")


if __name__ == "__main__":
    main(tyro.cli(Args))
