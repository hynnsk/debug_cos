#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# LIBERO-object Reptile post-training, v107: fresh action heads on the Reptile theta (no META_ACTION_INIT_PATH needed),
# action-language ranking hinge only (no REPA): relative margin 0.2, full gradient through E_neg (negative_grad_scale 1.0), weight 2.0.
# Drives action_policy_libero_object_edge_reptileinit_v107.toml.
# Launch (2 GPUs): LIBERO_ROOT=<path>/libero_object REPTILE_CKPT_PATH=<reptile job>/checkpoints/iter_XXXXXXXXX \
#                  NPROC_PER_NODE=2 sr 2 48 bash examples/launch_sft_action_policy_libero_object_edge_reptileinit_v107.sh
# Required: LIBERO_ROOT=<path>/libero_object, REPTILE_CKPT_PATH.
# Optional: WAN_VAE_PATH, NPROC_PER_NODE, MASTER_PORT, IMAGINAIRE_OUTPUT_ROOT, EXTRA_TAIL_OVERRIDES.

TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_libero_object_edge_reptileinit_v107.toml}"
: "${COSMOS_STORAGE:=$HOME/project/cosmos_storage}"
export COSMOS_STORAGE
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export LIBERO_ROOT="${LIBERO_ROOT:-}"
export REPTILE_CKPT_PATH="${REPTILE_CKPT_PATH:-}"
: "${MASTER_PORT:=50868}"
export MASTER_PORT
# _sft_launcher_common.sh validates BASE_CHECKPOINT_PATH / defaults WAN_VAE_PATH when this is set; the TOML itself
# reads REPTILE_CKPT_PATH.
: "${BASE_CHECKPOINT_PATH:=${REPTILE_CKPT_PATH}}"

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$LIBERO_ROOT'\'')." >&2; exit 1; }; [[ "$LIBERO_ROOT" == *libero_object* ]] || echo ">>> WARNING: LIBERO_ROOT does not look like the libero_object suite (got: $LIBERO_ROOT)." >&2; [[ -d "$REPTILE_CKPT_PATH/model" ]] || { echo "ERROR: REPTILE_CKPT_PATH must be a Reptile DCP iteration dir containing model/ (got: '\''$REPTILE_CKPT_PATH'\''). Run examples/launch_reptile_meta_edge.sh first." >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
