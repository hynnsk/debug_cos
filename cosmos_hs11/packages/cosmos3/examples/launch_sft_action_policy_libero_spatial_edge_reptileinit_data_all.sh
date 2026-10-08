#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# cosmos_hs11: LIBERO-Spatial FULL-DATA (ALL 432 episodes, validation off) FULL fine-tune warm-started from a full-mode REPTILE
# checkpoint -- the all-data counterpart of the 3-demo reptileinit recipes of this suite.
# Drives examples/toml/sft_config/action_policy_libero_spatial_edge_reptileinit_data_all.toml. See docs/action_reptile_meta.md.
#
# Required env vars:
#   LIBERO_ROOT            local LIBERO-Spatial LeRobot dataset dir, i.e. <dir>/libero_spatial (the basename is checked: the recipe's
#                          all-episodes json is suite specific)
#   REPTILE_CKPT_PATH      <reptile job>/checkpoints/iter_XXXXXXXXX  (the directory that contains model/)
#   META_ACTION_INIT_PATH  <reptile job>/meta_action_init_iter_XXXXXX.pt (the matching action heads)
# Optional: WAN_VAE_PATH, IMAGINAIRE_OUTPUT_ROOT, NPROC_PER_NODE, MASTER_PORT, EXTRA_TAIL_OVERRIDES.
# Usage (2 GPUs: shard 2 x 128 windows/rank = global batch 256, 2000 iterations, no validation, ~1 day):
#   JOB=$COSMOS_STORAGE/outputs/cosmos3_action_meta/reptile_meta/edge_reptile_v4
#   export LIBERO_ROOT=$COSMOS_STORAGE/data/LIBERO_LeRobot_v3/libero_spatial
#   export REPTILE_CKPT_PATH=$JOB/checkpoints/iter_000001000
#   export META_ACTION_INIT_PATH=$JOB/meta_action_init_iter_001000.pt
#   NPROC_PER_NODE=2 sr 2 48 --exclude haring examples/launch_sft_action_policy_libero_spatial_edge_reptileinit_data_all.sh

TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_libero_spatial_edge_reptileinit_data_all.toml}"
export LIBERO_ROOT="${LIBERO_ROOT:-}"
export REPTILE_CKPT_PATH="${REPTILE_CKPT_PATH:-}"
export META_ACTION_INIT_PATH="${META_ACTION_INIT_PATH:-}"
: "${MASTER_PORT:=50038}"   # distinct per suite/recipe so concurrent jobs can share a node (reptileinit data_all suites: 50036-50038)
export MASTER_PORT
# _sft_launcher_common.sh validates BASE_CHECKPOINT_PATH / defaults WAN_VAE_PATH when this is set; the TOML itself
# reads REPTILE_CKPT_PATH.
: "${BASE_CHECKPOINT_PATH:=${REPTILE_CKPT_PATH}}"

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$LIBERO_ROOT'\'')." >&2; exit 1; }; [[ "$(basename "${LIBERO_ROOT%/}")" == "libero_spatial" ]] || { echo "ERROR: this launcher trains the libero_spatial suite (its all-episodes json is suite specific) but LIBERO_ROOT is '\''$LIBERO_ROOT'\''. Point LIBERO_ROOT at <dir>/libero_spatial." >&2; exit 1; }; [[ -d "$REPTILE_CKPT_PATH/model" ]] || { echo "ERROR: REPTILE_CKPT_PATH must be a Reptile DCP iteration dir containing model/ (got: '\''$REPTILE_CKPT_PATH'\''). Run examples/launch_reptile_meta_edge.sh first." >&2; exit 1; }; [[ -f "$META_ACTION_INIT_PATH" ]] || { echo "ERROR: META_ACTION_INIT_PATH must point at the Reptile meta_action_init*.pt (got: '\''$META_ACTION_INIT_PATH'\'')." >&2; exit 1; }'

# Extra Hydra overrides from the environment (space-separated string).
TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
