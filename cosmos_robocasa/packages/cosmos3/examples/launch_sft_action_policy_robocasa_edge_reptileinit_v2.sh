#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# cosmos_robocasa: ORIGINAL RoboCasa benchmark (24 atomic tasks x ~50 human demos, the Cosmos Policy export) FULL fine-tune
# warm-started from a full-mode REPTILE checkpoint (compare launch_sft_action_policy_robocasa_edge_baseinit.sh: same recipe,
# mid-trained init + 5x fresh heads). Drives examples/toml/sft_config/action_policy_robocasa_edge_reptileinit.toml.
# See docs/action_policy_robocasa_posttrain.md.
#
# Required env vars:
#   ROBOCASA_ROOT          converted LeRobot v3 shards <ROBOCASA_ROOT>/<Task>/<date>/lerobot (default
#                          $COSMOS_STORAGE/data/robocasa_cosmos_policy_lerobot_v3; produced by examples/setup_robocasa.sh)
#   REPTILE_CKPT_PATH      <reptile job>/checkpoints/iter_XXXXXXXXX  (the directory that contains model/)
#   META_ACTION_INIT_PATH  <reptile job>/meta_action_init_iter_XXXXXX.pt (the matching action heads)
# Optional: TOML_FILE, WAN_VAE_PATH, IMAGINAIRE_OUTPUT_ROOT, NPROC_PER_NODE, MASTER_PORT, EXTRA_TAIL_OVERRIDES.
# Usage (4 GPUs: shard 4 x 128 windows = global batch 512):
#   JOB=$COSMOS_STORAGE/outputs/cosmos3_action_meta/reptile_meta/edge_reptile_v4      # examples/launch_reptile_meta_edge_v4.sh
#   export REPTILE_CKPT_PATH=$JOB/checkpoints/iter_000001000 META_ACTION_INIT_PATH=$JOB/meta_action_init_iter_001000.pt
#   NPROC_PER_NODE=4 sr 4 48 --exclude haring examples/launch_sft_action_policy_robocasa_edge_reptileinit.sh

: "${COSMOS_STORAGE:=$HOME/project/cosmos_storage}"
export COSMOS_STORAGE
TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_robocasa_edge_reptileinit_v2.toml}"
export ROBOCASA_ROOT="${ROBOCASA_ROOT:-$COSMOS_STORAGE/data/robocasa_cosmos_policy_lerobot_v3}"
export REPTILE_CKPT_PATH="${REPTILE_CKPT_PATH:-}"
export META_ACTION_INIT_PATH="${META_ACTION_INIT_PATH:-}"
: "${MASTER_PORT:=50910}"   # distinct per recipe so the two inits can share a node (baseinit: 50900)
export MASTER_PORT
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
# _sft_launcher_common.sh validates BASE_CHECKPOINT_PATH / defaults WAN_VAE_PATH when this is set; the TOML itself
# reads REPTILE_CKPT_PATH.
: "${BASE_CHECKPOINT_PATH:=${REPTILE_CKPT_PATH}}"

EXTRA_DATASET_CHECK='_n=$(ls -d "$ROBOCASA_ROOT"/*/*/lerobot/meta/info.json 2>/dev/null | wc -l); [[ "$_n" -ge 24 ]] || { echo "ERROR: ROBOCASA_ROOT must hold the 24 converted LeRobot v3 task shards <Task>/<date>/lerobot/meta/info.json (found $_n under '\''$ROBOCASA_ROOT'\''). Run examples/setup_robocasa.sh (STEPS=\"download convert stats\")." >&2; exit 1; }; _stats="$WORKDIR/cosmos_framework/data/generator/action/normalizer_stats/robocasa_cosmos_policy_frame_wise_relative_rot6d.json"; [[ -f "$_stats" ]] || { echo "ERROR: normalization stats missing: $_stats (run STEPS=stats examples/setup_robocasa.sh after the conversion)." >&2; exit 1; }; [[ -d "$REPTILE_CKPT_PATH/model" ]] || { echo "ERROR: REPTILE_CKPT_PATH must be a Reptile DCP iteration dir containing model/ (got: '\''$REPTILE_CKPT_PATH'\''). Run examples/launch_reptile_meta_edge_v4.sh first." >&2; exit 1; }; [[ -f "$META_ACTION_INIT_PATH" ]] || { echo "ERROR: META_ACTION_INIT_PATH must point at the Reptile meta_action_init*.pt (got: '\''$META_ACTION_INIT_PATH'\'')." >&2; exit 1; }'

# Extra Hydra overrides from the environment (space-separated string), e.g. EXTRA_TAIL_OVERRIDES="trainer.max_iter=5".
TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
