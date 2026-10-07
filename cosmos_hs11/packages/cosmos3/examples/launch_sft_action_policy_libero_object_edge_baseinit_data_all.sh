#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# cosmos_hs11: LIBERO-Object FULL-DATA (ALL 454 episodes, validation off) FULL fine-tune from the mid-trained Cosmos3-Edge
# checkpoint = the full-data UPPER BOUND of the 3-demo recipes of this suite (fresh-init baseline and reptileinit v3).
# Drives examples/toml/sft_config/action_policy_libero_object_edge_baseinit_data_all.toml. See docs/action_reptile_meta.md, 10b.
#
# Required env vars:
#   LIBERO_ROOT            local LIBERO-Object LeRobot dataset dir, i.e. <dir>/libero_object (the basename is checked: the recipe's
#                          all-episodes json is suite specific)
# Optional env vars (defaults below; override to relocate checkpoints):
#   BASE_CHECKPOINT_PATH   mid-trained Cosmos3-Edge DCP dir (default examples/checkpoints/Cosmos3-Edge)
#   WAN_VAE_PATH           default examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   IMAGINAIRE_OUTPUT_ROOT, NPROC_PER_NODE, MASTER_PORT, EXTRA_TAIL_OVERRIDES
# Usage (8 GPUs: shard 8 x 256 windows/rank = global batch 2048, 2000 iterations, no validation):
#   LIBERO_ROOT=$COSMOS_STORAGE/data/LIBERO_LeRobot_v3/libero_object \
#   BASE_CHECKPOINT_PATH=$COSMOS_STORAGE/checkpoints/Cosmos3-Edge WAN_VAE_PATH=$COSMOS_STORAGE/checkpoints/wan22_vae/Wan2.2_VAE.pth \
#   NPROC_PER_NODE=8 sr 8 48 examples/launch_sft_action_policy_libero_object_edge_baseinit_data_all.sh
# OOM fallback with the same global batch (128 windows/rank x grad_accum 2):
#   EXTRA_TAIL_OVERRIDES="dataloader_train.max_samples_per_batch=128 trainer.grad_accum_iter=2" NPROC_PER_NODE=8 sr 8 48 examples/launch_sft_action_policy_libero_object_edge_baseinit_data_all.sh

TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_libero_object_edge_baseinit_data_all.toml}"
export LIBERO_ROOT="${LIBERO_ROOT:-}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"
: "${MASTER_PORT:=50034}"   # distinct per suite/recipe so concurrent jobs can share a node (3-demo baseinit suites: 50030-50032)
export MASTER_PORT

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$LIBERO_ROOT'\'')." >&2; exit 1; }; [[ "$(basename "${LIBERO_ROOT%/}")" == "libero_object" ]] || { echo "ERROR: this launcher trains the libero_object suite (its all-episodes json is suite specific) but LIBERO_ROOT is '\''$LIBERO_ROOT'\''. Point LIBERO_ROOT at <dir>/libero_object." >&2; exit 1; }'

# Extra Hydra overrides from the environment (space-separated string), e.g. the OOM fallback in the header.
TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
