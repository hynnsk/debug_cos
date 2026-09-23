#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for action_policy_libero_nano — Cosmos3-Nano LIBERO-10
# action-policy SFT (FSDP, full SFT). Drives cosmos_framework.scripts.train
# against examples/toml/sft_config/action_policy_libero_10_nano.toml.
#
# FEW-SHOT variant (cosmos_hs08): the Nano counterpart of
# launch_sft_action_policy_libero_10_edge.sh — trains on the same fixed
# 30-demonstration subset of libero_10 (3 demos/task, seed 42) with the same
# held-out 50-demo validation set, LR, schedule and global batch (256). Only the
# topology differs: Nano (Qwen3-VL-8B MoT) needs FSDP shard 8 x 32 windows/rank
# on one 8 x 48 GB node instead of Edge's shard 2 x 128. See the TOML header.
#
# Point LIBERO_ROOT at the libero_10 suite ONLY. Use the 20 FPS
# nvidia/LIBERO_LeRobot_v3. See docs/action_policy_libero_posttrain.md.
#
# Required env vars:
#   LIBERO_ROOT           local LIBERO-10 LeRobot dataset dir, e.g. <dir>/libero_10 (no default)
# Optional env vars (defaults below; override to relocate data/checkpoints):
#   BASE_CHECKPOINT_PATH  default: examples/checkpoints/Cosmos3-Nano (the Cosmos3-Nano DCP dir)
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   HF_TOKEN              if any tokenizer download requires gated HF access
#   OUTPUT_ROOT           default: outputs/train
#   NPROC_PER_NODE        ranks per node; must equal the TOML's data_parallel_shard_degree (8)
#
# Pre-sync the 20 FPS suite once:
#   hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset --include 'libero_10/**' --local-dir <dir>
#   export LIBERO_ROOT=<dir>/libero_10
#
# Usage (single 8-GPU node, e.g. via the site SLURM wrapper):
#   NPROC_PER_NODE=8 BASE_CHECKPOINT_PATH=$COSMOS_STORAGE/checkpoints/Cosmos3-Nano \
#     sr 8 48 bash examples/launch_sft_action_policy_libero_10_nano.sh
# Smoke (2 iters, no wandb):
#   EXTRA_TAIL_OVERRIDES="trainer.max_iter=2 trainer.run_validation_on_start=false job.wandb_mode=disabled" \
#     NPROC_PER_NODE=8 sr 8 48 bash examples/launch_sft_action_policy_libero_10_nano.sh

TOML_FILE="examples/toml/sft_config/action_policy_libero_10_nano_v3.toml"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

# LIBEROLeRobotDataset reads ${oc.env:LIBERO_ROOT} directly (a LOCAL LeRobot dir);
# export it so torchrun (launched in this shell) inherits it.
export LIBERO_ROOT="${LIBERO_ROOT:-}"

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$LIBERO_ROOT'\''). Pre-sync: hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset --include '\''libero_10/**'\'' --local-dir <dir> (then LIBERO_ROOT=<dir>/libero_10). See docs/action_policy_libero_posttrain.md" >&2; exit 1; }'

# Extra Hydra overrides from the environment: a space-separated string word-split into
# the TAIL_OVERRIDES array. An exported string survives `bash <wrapper>` (a child
# process), unlike a TAIL_OVERRIDES array set in your shell. Use it for smoke runs,
# e.g. EXTRA_TAIL_OVERRIDES="trainer.max_iter=5 job.wandb_mode=offline".
TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
