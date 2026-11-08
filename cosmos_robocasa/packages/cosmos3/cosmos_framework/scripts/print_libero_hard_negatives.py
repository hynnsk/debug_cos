# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Print the action-similarity hard-negative distribution a LIBERO few-shot recipe would train with (cosmos_hs11 v27).

Builds the LIBERO dataset exactly as the recipe does (parquet only, no video decoding) and prints the normalized
task-distance matrix plus the top negatives per task, so the ``[model.language_ranking]`` knobs can be inspected
before a GPU job is launched::

    PYTHONPATH=. python cosmos_framework/scripts/print_libero_hard_negatives.py \
        --root $COSMOS_STORAGE/data/LIBERO_LeRobot_v3/libero_goal --episode-subset libero_goal_3ep_per_task_v2.json \
        --fraction 0.7 --temperature 0.5 --prefix-frames 64
"""

from __future__ import annotations

import argparse

from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import LIBEROLeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="Local LeRobot dir of ONE suite, e.g. <dir>/libero_goal")
    parser.add_argument(
        "--episode-subset",
        default=None,
        help="Fixed few-shot subset json (bundled filename or path), e.g. libero_goal_3ep_per_task_v2.json; "
        "omitted = every episode minus the seeded 1%% val split (split='train').",
    )
    parser.add_argument("--fraction", type=float, default=0.7, help="hard_negative_fraction")
    parser.add_argument("--temperature", type=float, default=0.5, help="hard_negative_temperature")
    parser.add_argument("--prefix-frames", type=int, default=64, help="hard_negative_prefix_frames")
    parser.add_argument("--rotation-space", default="6d", help="Recipe rotation_space (6d in every LIBERO recipe)")
    args = parser.parse_args()

    dataset = LIBEROLeRobotDataset(
        root=args.root,
        rotation_space=args.rotation_space,
        episode_subset_path=args.episode_subset,
        language_ranking=True,
        language_ranking_negative_sampling="action_similarity",
        language_ranking_hard_fraction=args.fraction,
        language_ranking_hard_temperature=args.temperature,
        language_ranking_hard_prefix_frames=args.prefix_frames,
    )
    print(
        f"root={args.root} subset={args.episode_subset} fraction={args.fraction} temperature={args.temperature} "
        f"prefix_frames={args.prefix_frames} kept_windows={len(dataset)}"
    )
    print(dataset.ranking_negative_report)


if __name__ == "__main__":
    main()
