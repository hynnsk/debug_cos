#!/usr/bin/env bash
# v19: 1-GPU baseline run of action_policy_libero_object_edge_baseinit_v19.toml (lr 1e-05, 128 windows/rank, shard 1).
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# cosmos_hs11: LIBERO-Object few-shot (3 demos/task) FULL fine-tune from the mid-trained Cosmos3-Edge checkpoint = the
# fresh-init BASELINE of launch_sft_action_policy_libero_object_edge_reptileinit_v3.sh (same recipe, no Reptile warm start).
# Drives examples/toml/sft_config/action_policy_libero_object_edge_baseinit.toml. See docs/action_reptile_meta.md, section 10b.
#
# Required env vars:
#   LIBERO_ROOT            local LIBERO-Object LeRobot dataset dir, i.e. <dir>/libero_object (the basename is checked: the recipe's
#                          episode-subset jsons are suite specific)
# Optional env vars (defaults below; override to relocate checkpoints):
#   BASE_CHECKPOINT_PATH   mid-trained Cosmos3-Edge DCP dir (default examples/checkpoints/Cosmos3-Edge)
#   WAN_VAE_PATH           default examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   IMAGINAIRE_OUTPUT_ROOT, NPROC_PER_NODE, MASTER_PORT, EXTRA_TAIL_OVERRIDES
# Usage (1 GPU: shard 1 x 128 windows = global batch 128):
#   LIBERO_ROOT=$COSMOS_STORAGE/data/LIBERO_LeRobot_v3/libero_object \
#   BASE_CHECKPOINT_PATH=$COSMOS_STORAGE/checkpoints/Cosmos3-Edge WAN_VAE_PATH=$COSMOS_STORAGE/checkpoints/wan22_vae/Wan2.2_VAE.pth \
#   NPROC_PER_NODE=1 sr 1 48 examples/launch_sft_action_policy_libero_object_edge_baseinit_v19.sh

TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_libero_object_edge_baseinit_v19.toml}"
export LIBERO_ROOT="${LIBERO_ROOT:-}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"
: "${MASTER_PORT:=50654}"   # distinct per suite/recipe so concurrent jobs can share a node (reptileinit v3 suites: 50006-50008)
export MASTER_PORT
: "${NPROC_PER_NODE:=1}"   # 1-GPU recipe (TOML data_parallel_shard_degree = 1)
export NPROC_PER_NODE

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$LIBERO_ROOT'\'')." >&2; exit 1; }; [[ "$(basename "${LIBERO_ROOT%/}")" == "libero_object" ]] || { echo "ERROR: this launcher trains the libero_object suite (its episode-subset jsons are suite specific) but LIBERO_ROOT is '\''$LIBERO_ROOT'\''. Point LIBERO_ROOT at <dir>/libero_object." >&2; exit 1; }'

# Extra Hydra overrides from the environment (space-separated string), e.g. EXTRA_TAIL_OVERRIDES="trainer.max_iter=5".
TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
