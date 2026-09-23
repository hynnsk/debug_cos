#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# cosmos_hs09 OURS: LIBERO-10 few-shot (3 demos/task) LoRA post-training from the META-LEARNED init
# (LoRA + action heads + time_embedder = theta_meta of action_fewshot_meta_lora_edge). Same recipe as
# launch_sft_action_policy_libero_10_lora_edge.sh (the baseline); drives
# examples/toml/sft_config/action_policy_libero_10_lora_edge_metainit.toml. See docs/action_fewshot_meta_lora.md.
#
# Required env vars:
#   LIBERO_ROOT            local LIBERO-10 LeRobot dataset dir, e.g. <dir>/libero_10
#   META_ACTION_INIT_PATH  meta_action_init[_iter_XXXXXX].pt written by launch_meta_action_fewshot_lora_edge.sh
# Optional: BASE_CHECKPOINT_PATH, WAN_VAE_PATH, IMAGINAIRE_OUTPUT_ROOT, NPROC_PER_NODE, MASTER_PORT, EXTRA_TAIL_OVERRIDES.
# Usage:
#   export META_ACTION_INIT_PATH=$IMAGINAIRE_OUTPUT_ROOT/cosmos3_action_meta/fewshot_meta_lora/<name>/meta_action_init.pt
#   NPROC_PER_NODE=4 sr 4 48 examples/launch_sft_action_policy_libero_10_lora_edge_metainit.sh

TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_libero_10_lora_edge_metainit.toml}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"

export LIBERO_ROOT="${LIBERO_ROOT:-}"
export META_ACTION_INIT_PATH="${META_ACTION_INIT_PATH:-}"

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$LIBERO_ROOT'\'')." >&2; exit 1; }; [[ -f "$META_ACTION_INIT_PATH" ]] || { echo "ERROR: META_ACTION_INIT_PATH must point at a meta_action_init.pt (got: '\''$META_ACTION_INIT_PATH'\''). Run examples/launch_meta_action_fewshot_lora_edge.sh first." >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
