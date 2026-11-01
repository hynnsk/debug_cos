#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# cosmos_hs11: RoboCasa365 target/atomic (18 tasks) FULL fine-tune warm-started from a full-mode REPTILE checkpoint = the
# "Target only" setting of RoboCasa365 Table 2 with the Reptile meta-init (compare launch_sft_action_policy_robocasa_edge_baseinit.sh).
# DATA_FRACTION selects the recipe: 10 -> 50 demos/task, 30 -> 150, 100 -> every remaining train demo (~500/task).
# Drives examples/toml/sft_config/action_policy_robocasa_edge_reptileinit_${DATA_FRACTION}pct.toml. See docs/action_policy_robocasa_posttrain.md.
#
# Required env vars:
#   ROBOCASA_ROOT          converted v3.0 shards, i.e. <ROBOCASA_DATA>/v30/target/atomic (default
#                          $COSMOS_STORAGE/data/robocasa365/v30/target/atomic; produced by examples/setup_robocasa365.sh)
#   REPTILE_CKPT_PATH      <reptile job>/checkpoints/iter_XXXXXXXXX  (the directory that contains model/)
#   META_ACTION_INIT_PATH  <reptile job>/meta_action_init_iter_XXXXXX.pt (the matching action heads)
# Optional: DATA_FRACTION (10 | 30 | 100, default 10), WAN_VAE_PATH, IMAGINAIRE_OUTPUT_ROOT, NPROC_PER_NODE, MASTER_PORT,
#           EXTRA_TAIL_OVERRIDES.
# Usage (4 GPUs: shard 4 x 128 windows = global batch 512):
#   JOB=$COSMOS_STORAGE/outputs/cosmos3_action_meta/reptile_meta/edge_reptile_v4
#   export REPTILE_CKPT_PATH=$JOB/checkpoints/iter_000001000 META_ACTION_INIT_PATH=$JOB/meta_action_init_iter_001000.pt
#   DATA_FRACTION=10 NPROC_PER_NODE=4 sr 4 48 examples/launch_sft_action_policy_robocasa_edge_reptileinit.sh

: "${COSMOS_STORAGE:=$HOME/project/cosmos_storage}"
export COSMOS_STORAGE
DATA_FRACTION="${DATA_FRACTION:-10}"
case "$DATA_FRACTION" in
    10)  _PORT=50810 ;;
    30)  _PORT=50811 ;;
    100) _PORT=50812 ;;
    *)   echo "ERROR: DATA_FRACTION must be 10, 30 or 100 (got '$DATA_FRACTION')" >&2; exit 1 ;;
esac
TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_robocasa_edge_reptileinit_${DATA_FRACTION}pct.toml}"
export ROBOCASA_ROOT="${ROBOCASA_ROOT:-$COSMOS_STORAGE/data/robocasa365/v30/target/atomic}"
export REPTILE_CKPT_PATH="${REPTILE_CKPT_PATH:-}"
export META_ACTION_INIT_PATH="${META_ACTION_INIT_PATH:-}"
: "${MASTER_PORT:=$_PORT}"   # distinct per fraction / init so concurrent jobs can share a node (baseinit: 50800-50802, reptileinit_freshheads: 50820-50822)
export MASTER_PORT
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
# _sft_launcher_common.sh validates BASE_CHECKPOINT_PATH / defaults WAN_VAE_PATH when this is set; the TOML itself
# reads REPTILE_CKPT_PATH.
: "${BASE_CHECKPOINT_PATH:=${REPTILE_CKPT_PATH}}"

EXTRA_DATASET_CHECK='_n=$(ls -d "$ROBOCASA_ROOT"/*/*/lerobot/meta/tasks.parquet 2>/dev/null | wc -l); [[ "$_n" -ge 18 ]] || { echo "ERROR: ROBOCASA_ROOT must hold the 18 converted v3.0 task shards <task>/<date>/lerobot/meta/tasks.parquet (found $_n under '\''$ROBOCASA_ROOT'\''). Run examples/setup_robocasa365.sh (steps data + convert)." >&2; exit 1; }; [[ -d "$REPTILE_CKPT_PATH/model" ]] || { echo "ERROR: REPTILE_CKPT_PATH must be a Reptile DCP iteration dir containing model/ (got: '\''$REPTILE_CKPT_PATH'\''). Run examples/launch_reptile_meta_edge.sh first." >&2; exit 1; }; [[ -f "$META_ACTION_INIT_PATH" ]] || { echo "ERROR: META_ACTION_INIT_PATH must point at the Reptile meta_action_init*.pt (got: '\''$META_ACTION_INIT_PATH'\'')." >&2; exit 1; }'

# Extra Hydra overrides from the environment (space-separated string), e.g. EXTRA_TAIL_OVERRIDES="trainer.max_iter=5".
TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
