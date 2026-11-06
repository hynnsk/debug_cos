#!/usr/bin/env bash
# v4.9.5: action_policy_libero_goal_edge_reptileinit_v4.9.5.toml = Reptile theta from REPTILE_CKPT_PATH (point it at the action_reptile_meta_edge_v4.9 run) + FRESH action heads (keys_to_skip_loading, no meta_action_init), head LR x5. PLUS the action-language ranking hinge (relative margin 0.2, full gradient through E_neg (negative_grad_scale 1.0), weight 1.0; = v4.9.4 + [model.language_ranking]).
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# cosmos_hs11: LIBERO-goal few-shot (3 demos/task) FULL fine-tune warm-started from a full-mode REPTILE checkpoint.
# Drives examples/toml/sft_config/action_policy_libero_goal_edge_reptileinit_v4.9.5.toml. See docs/action_reptile_meta.md.
#
# Required env vars:
#   LIBERO_ROOT            local LIBERO-goal LeRobot dataset dir, e.g. <dir>/libero_goal
#   REPTILE_CKPT_PATH      <reptile job>/checkpoints/iter_XXXXXXXXX  (the directory that contains model/)
# Optional: WAN_VAE_PATH, IMAGINAIRE_OUTPUT_ROOT, NPROC_PER_NODE, MASTER_PORT, EXTRA_TAIL_OVERRIDES.
# Usage:
#   NPROC_PER_NODE=2 sr 2 48 examples/launch_sft_action_policy_libero_goal_edge_reptileinit_v4.9.5.sh

TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_libero_goal_edge_reptileinit_v4.9.5.toml}"
export LIBERO_ROOT="${LIBERO_ROOT:-}"
export REPTILE_CKPT_PATH="${REPTILE_CKPT_PATH:-}"
: "${MASTER_PORT:=50874}"
export MASTER_PORT
# _sft_launcher_common.sh validates BASE_CHECKPOINT_PATH / defaults WAN_VAE_PATH when this is set; the TOML itself
# reads REPTILE_CKPT_PATH.
: "${BASE_CHECKPOINT_PATH:=${REPTILE_CKPT_PATH}}"

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$LIBERO_ROOT'\'')." >&2; exit 1; }; [[ -d "$REPTILE_CKPT_PATH/model" ]] || { echo "ERROR: REPTILE_CKPT_PATH must be a Reptile DCP iteration dir containing model/ (got: '\''$REPTILE_CKPT_PATH'\''). Run examples/launch_reptile_meta_edge.sh first." >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
