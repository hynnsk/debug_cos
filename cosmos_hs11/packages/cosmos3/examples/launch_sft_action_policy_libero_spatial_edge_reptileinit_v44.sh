#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# LIBERO-spatial Reptile post-training with action-language ranking (v23).
# Copied from launch_sft_action_policy_libero_object_edge_reptileinit_v7.sh; source TOML is action_policy_libero_object_edge_reptileinit_v7.1.toml.
# Required: LIBERO_ROOT=<path>/libero_spatial, REPTILE_CKPT_PATH, META_ACTION_INIT_PATH.
# Optional: WAN_VAE_PATH, NPROC_PER_NODE, MASTER_PORT, IMAGINAIRE_OUTPUT_ROOT, EXTRA_TAIL_OVERRIDES.
# See docs/action_policy_libero_language_ranking.md.

TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_libero_spatial_edge_reptileinit_v44.toml}"
export LIBERO_ROOT="${LIBERO_ROOT:-}"
export REPTILE_CKPT_PATH="${REPTILE_CKPT_PATH:-}"
export META_ACTION_INIT_PATH="${META_ACTION_INIT_PATH:-}"
: "${MASTER_PORT:=50144}"
export MASTER_PORT
# _sft_launcher_common.sh validates BASE_CHECKPOINT_PATH / defaults WAN_VAE_PATH when this is set; the TOML itself
# reads REPTILE_CKPT_PATH.
: "${BASE_CHECKPOINT_PATH:=${REPTILE_CKPT_PATH}}"

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$LIBERO_ROOT'\'')." >&2; exit 1; }; [[ -d "$REPTILE_CKPT_PATH/model" ]] || { echo "ERROR: REPTILE_CKPT_PATH must be a Reptile DCP iteration dir containing model/ (got: '\''$REPTILE_CKPT_PATH'\''). Run examples/launch_reptile_meta_edge.sh first." >&2; exit 1; }; [[ -f "$META_ACTION_INIT_PATH" ]] || { echo "ERROR: META_ACTION_INIT_PATH must point at the Reptile meta_action_init*.pt (got: '\''$META_ACTION_INIT_PATH'\'')." >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
