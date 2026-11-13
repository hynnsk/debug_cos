#!/usr/bin/env bash
# v29.5: action_policy_libero_object_edge_reptileinit_v29.5.toml = Reptile theta from REPTILE_CKPT_PATH (point it at the matching meta run) + FRESH action heads; recipe knobs (head LR, batch, ranking loss): see the TOML. Launcher copied from v29.4.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# LIBERO-object Reptile post-training with uniform-negative action-language ranking, v27 = asymmetric gradient + relative margin + self-paced hinge.
# Copied from launch_sft_action_policy_libero_object_edge_reptileinit_v23.sh; drives action_policy_libero_object_edge_reptileinit_v27.toml.
# Required: LIBERO_ROOT=<path>/libero_object, REPTILE_CKPT_PATH (no META_ACTION_INIT_PATH needed: fresh heads).
# Optional: WAN_VAE_PATH, NPROC_PER_NODE, MASTER_PORT, IMAGINAIRE_OUTPUT_ROOT, EXTRA_TAIL_OVERRIDES.
# See docs/action_policy_libero_language_ranking.md.

TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_libero_object_edge_reptileinit_v29.5.toml}"
export LIBERO_ROOT="${LIBERO_ROOT:-}"
export REPTILE_CKPT_PATH="${REPTILE_CKPT_PATH:-}"
: "${MASTER_PORT:=51059}"
export MASTER_PORT
# _sft_launcher_common.sh validates BASE_CHECKPOINT_PATH / defaults WAN_VAE_PATH when this is set; the TOML itself
# reads REPTILE_CKPT_PATH.
: "${BASE_CHECKPOINT_PATH:=${REPTILE_CKPT_PATH}}"

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$LIBERO_ROOT'\'')." >&2; exit 1; }; [[ -d "$REPTILE_CKPT_PATH/model" ]] || { echo "ERROR: REPTILE_CKPT_PATH must be a Reptile DCP iteration dir containing model/ (got: '\''$REPTILE_CKPT_PATH'\''). Run examples/launch_reptile_meta_edge.sh first." >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
