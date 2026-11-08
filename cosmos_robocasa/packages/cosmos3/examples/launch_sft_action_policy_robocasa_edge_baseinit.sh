#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# cosmos_robocasa: ORIGINAL RoboCasa benchmark (24 atomic tasks x ~50 human demos, the Cosmos Policy export) FULL fine-tune
# from the MID-TRAINED Cosmos3-Edge checkpoint = the fresh-init BASELINE of launch_sft_action_policy_robocasa_edge_reptileinit.sh.
# Drives examples/toml/sft_config/action_policy_robocasa_edge_baseinit.toml (Cosmos Policy's RoboCasa configuration mapped
# onto Cosmos3-Edge). See docs/action_policy_robocasa_posttrain.md.
#
# Required env vars:
#   ROBOCASA_ROOT          converted LeRobot v3 shards <ROBOCASA_ROOT>/<Task>/<date>/lerobot (default
#                          $COSMOS_STORAGE/data/robocasa_cosmos_policy_lerobot_v3; produced by examples/setup_robocasa.sh)
# Optional env vars (defaults below; override to relocate checkpoints):
#   BASE_CHECKPOINT_PATH   mid-trained Cosmos3-Edge DCP dir (default examples/checkpoints/Cosmos3-Edge)
#   WAN_VAE_PATH           default examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   TOML_FILE, IMAGINAIRE_OUTPUT_ROOT, NPROC_PER_NODE, MASTER_PORT, EXTRA_TAIL_OVERRIDES
# Usage (4 GPUs: shard 4 x 128 windows = global batch 512):
#   export BASE_CHECKPOINT_PATH=$COSMOS_STORAGE/checkpoints/Cosmos3-Edge WAN_VAE_PATH=$COSMOS_STORAGE/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   NPROC_PER_NODE=4 sr 4 48 --exclude haring examples/launch_sft_action_policy_robocasa_edge_baseinit.sh
# Smoke: EXTRA_TAIL_OVERRIDES="trainer.max_iter=5 job.wandb_mode=disabled" ... (a final DCP is still written: use a
# throw-away IMAGINAIRE_OUTPUT_ROOT and job.name).

: "${COSMOS_STORAGE:=$HOME/project/cosmos_storage}"
export COSMOS_STORAGE
TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_robocasa_edge_baseinit.toml}"
export ROBOCASA_ROOT="${ROBOCASA_ROOT:-$COSMOS_STORAGE/data/robocasa_cosmos_policy_lerobot_v3}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"
: "${MASTER_PORT:=50900}"   # distinct per recipe so the two inits can share a node (reptileinit: 50910)
export MASTER_PORT
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# 24 converted task shards + the bundled normalization stats the recipe reads (both produced by examples/setup_robocasa.sh).
EXTRA_DATASET_CHECK='_n=$(ls -d "$ROBOCASA_ROOT"/*/*/lerobot/meta/info.json 2>/dev/null | wc -l); [[ "$_n" -ge 24 ]] || { echo "ERROR: ROBOCASA_ROOT must hold the 24 converted LeRobot v3 task shards <Task>/<date>/lerobot/meta/info.json (found $_n under '\''$ROBOCASA_ROOT'\''). Run examples/setup_robocasa.sh (STEPS=\"download convert stats\")." >&2; exit 1; }; _stats="$WORKDIR/cosmos_framework/data/generator/action/normalizer_stats/robocasa_cosmos_policy_frame_wise_relative_rot6d.json"; [[ -f "$_stats" ]] || { echo "ERROR: normalization stats missing: $_stats (run STEPS=stats examples/setup_robocasa.sh after the conversion)." >&2; exit 1; }'

# Extra Hydra overrides from the environment (space-separated string), e.g. EXTRA_TAIL_OVERRIDES="trainer.max_iter=5".
TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
