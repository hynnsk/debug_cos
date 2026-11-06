#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# cosmos_hs11: RoboCasa365 target/atomic (18 tasks) FULL fine-tune from the MID-TRAINED Cosmos3-Edge checkpoint = the
# "Target only" setting of RoboCasa365 Table 2 and the fresh-init BASELINE of launch_sft_action_policy_robocasa_edge_reptileinit.sh.
# DATA_FRACTION selects the recipe: 10 -> 50 demos/task, 30 -> 150, 100 -> every remaining train demo (~500/task).
# Drives examples/toml/sft_config/action_policy_robocasa_edge_baseinit_${DATA_FRACTION}pct.toml. See docs/action_policy_robocasa_posttrain.md.
#
# Required env vars:
#   ROBOCASA_ROOT          converted v3.0 shards, i.e. <ROBOCASA_DATA>/v30/target/atomic (default
#                          $COSMOS_STORAGE/data/robocasa365/v30/target/atomic; produced by examples/setup_robocasa365.sh)
# Optional env vars (defaults below; override to relocate checkpoints):
#   DATA_FRACTION          10 | 30 | 100 (default 10)
#   BASE_CHECKPOINT_PATH   mid-trained Cosmos3-Edge DCP dir (default examples/checkpoints/Cosmos3-Edge)
#   WAN_VAE_PATH           default examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   IMAGINAIRE_OUTPUT_ROOT, NPROC_PER_NODE, MASTER_PORT, EXTRA_TAIL_OVERRIDES
# Usage (4 GPUs: shard 4 x 128 windows = global batch 512):
#   export BASE_CHECKPOINT_PATH=$COSMOS_STORAGE/checkpoints/Cosmos3-Edge WAN_VAE_PATH=$COSMOS_STORAGE/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   DATA_FRACTION=10 NPROC_PER_NODE=4 sr 4 48 examples/launch_sft_action_policy_robocasa_edge_baseinit.sh

: "${COSMOS_STORAGE:=$HOME/project/cosmos_storage}"
export COSMOS_STORAGE
DATA_FRACTION="${DATA_FRACTION:-10}"
case "$DATA_FRACTION" in
    10)  _PORT=50800 ;;
    30)  _PORT=50801 ;;
    100) _PORT=50802 ;;
    *)   echo "ERROR: DATA_FRACTION must be 10, 30 or 100 (got '$DATA_FRACTION')" >&2; exit 1 ;;
esac
TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_robocasa_edge_baseinit_${DATA_FRACTION}pct_v3.toml}"
export ROBOCASA_ROOT="${ROBOCASA_ROOT:-$COSMOS_STORAGE/data/robocasa365/v30/target/atomic}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"
: "${MASTER_PORT:=$_PORT}"   # distinct per fraction / init so concurrent jobs can share a node (reptileinit: 50810-50812)
export MASTER_PORT
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

EXTRA_DATASET_CHECK='_n=$(ls -d "$ROBOCASA_ROOT"/*/*/lerobot/meta/tasks.parquet 2>/dev/null | wc -l); [[ "$_n" -ge 18 ]] || { echo "ERROR: ROBOCASA_ROOT must hold the 18 converted v3.0 task shards <task>/<date>/lerobot/meta/tasks.parquet (found $_n under '\''$ROBOCASA_ROOT'\''). Run examples/setup_robocasa365.sh (steps data + convert)." >&2; exit 1; }'

# Extra Hydra overrides from the environment (space-separated string), e.g. EXTRA_TAIL_OVERRIDES="trainer.max_iter=5".
TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
