#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# cosmos_hs11: LIBERO-10 few-shot (3 demos/task) FULL fine-tune warm-started from a full-mode REPTILE checkpoint,
# + action-language ranking with ACTION-SIMILARITY HARD NEGATIVES (v64: mirrors LIBERO-goal v28 (loss_weight 5.0: the strength arm)).
# Copied from launch_sft_action_policy_libero_10_edge_reptileinit_v3.sh; drives action_policy_libero_10_edge_reptileinit_v64.toml.
# See docs/action_policy_libero_language_ranking.md. Inspect the negative distribution first with
#   PYTHONPATH=. python cosmos_framework/scripts/print_libero_hard_negatives.py --root $LIBERO_ROOT --episode-subset libero_10_3ep_per_task_seed42.json --fraction 0.7 --temperature 0.5
#
# Required env vars:
#   LIBERO_ROOT            local LIBERO-10 LeRobot dataset dir, e.g. <dir>/libero_10
#   REPTILE_CKPT_PATH      <reptile job>/checkpoints/iter_XXXXXXXXX  (the directory that contains model/)
#   META_ACTION_INIT_PATH  <reptile job>/meta_action_init_iter_XXXXXX.pt (the matching action heads)
# Optional: WAN_VAE_PATH, IMAGINAIRE_OUTPUT_ROOT, NPROC_PER_NODE, MASTER_PORT, EXTRA_TAIL_OVERRIDES.
# Usage:
#   NPROC_PER_NODE=4 sr 4 48 examples/launch_sft_action_policy_libero_10_edge_reptileinit_v64.sh

TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_libero_10_edge_reptileinit_v64.toml}"
export LIBERO_ROOT="${LIBERO_ROOT:-}"
export REPTILE_CKPT_PATH="${REPTILE_CKPT_PATH:-}"
export META_ACTION_INIT_PATH="${META_ACTION_INIT_PATH:-}"
: "${MASTER_PORT:=50131}"
export MASTER_PORT
# _sft_launcher_common.sh validates BASE_CHECKPOINT_PATH / defaults WAN_VAE_PATH when this is set; the TOML itself
# reads REPTILE_CKPT_PATH.
: "${BASE_CHECKPOINT_PATH:=${REPTILE_CKPT_PATH}}"

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$LIBERO_ROOT'\'')." >&2; exit 1; }; [[ -d "$REPTILE_CKPT_PATH/model" ]] || { echo "ERROR: REPTILE_CKPT_PATH must be a Reptile DCP iteration dir containing model/ (got: '\''$REPTILE_CKPT_PATH'\''). Run examples/launch_reptile_meta_edge.sh first." >&2; exit 1; }; [[ -f "$META_ACTION_INIT_PATH" ]] || { echo "ERROR: META_ACTION_INIT_PATH must point at the Reptile meta_action_init*.pt (got: '\''$META_ACTION_INIT_PATH'\'')." >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
