#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# cosmos_hs09 BASELINE: LIBERO-10 few-shot (3 demos/task) LoRA post-training of the mid-trained Cosmos3-Edge
# (LoRA B=0 + fresh action heads + mid-trained time_embedder). Drives
# examples/toml/sft_config/action_policy_libero_10_lora_edge.toml. See docs/action_fewshot_meta_lora.md.
#
# Required env vars:
#   LIBERO_ROOT            local LIBERO-10 LeRobot dataset dir, e.g. <dir>/libero_10
# Optional: BASE_CHECKPOINT_PATH (default examples/checkpoints/Cosmos3-Edge), WAN_VAE_PATH,
#           IMAGINAIRE_OUTPUT_ROOT, NPROC_PER_NODE, MASTER_PORT, EXTRA_TAIL_OVERRIDES.
# Usage:
#   NPROC_PER_NODE=4 sr 4 48 examples/launch_sft_action_policy_libero_10_lora_edge.sh

TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_libero_10_lora_edge.toml}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"

export LIBERO_ROOT="${LIBERO_ROOT:-}"

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$LIBERO_ROOT'\''). See docs/action_policy_libero_posttrain.md" >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
