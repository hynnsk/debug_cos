#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# LIBERO-spatial Reptile post-training, v115: fresh action heads on the Reptile theta (no META_ACTION_INIT_PATH needed),
# action-language ranking hinge (relative margin 0.2, full gradient through E_neg (negative_grad_scale 1.0), weight 1)
# PLUS DINOv2 ViT-B/14 REPA on CLEAN raw predicted frames (avgpool to [4, 5, 5]): student block 24, feature cosine (w 5) + relation loss (w 5, l1, margin 0.1) (experiment action_policy_libero_edge_repa, [model.repa] + [model.language_ranking] in the TOML).
# Drives action_policy_libero_spatial_edge_reptileinit_v115.toml.
# Launch (2 GPUs): LIBERO_ROOT=<path>/libero_spatial REPTILE_CKPT_PATH=<reptile job>/checkpoints/iter_XXXXXXXXX \
#                  NPROC_PER_NODE=2 sr 2 48 bash examples/launch_sft_action_policy_libero_spatial_edge_reptileinit_v115.sh
# Required: LIBERO_ROOT=<path>/libero_spatial, REPTILE_CKPT_PATH.
# Optional: WAN_VAE_PATH, NPROC_PER_NODE, MASTER_PORT, IMAGINAIRE_OUTPUT_ROOT, EXTRA_TAIL_OVERRIDES,
#           HF_HOME (the frozen teacher facebook/dinov2-base is read from $HF_HOME/hub; default $COSMOS_STORAGE/hf_cache).

TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_libero_spatial_edge_reptileinit_v115.toml}"
: "${COSMOS_STORAGE:=$HOME/project/cosmos_storage}"
export COSMOS_STORAGE
# Frozen DINOv2 teacher (facebook/dinov2-base) is loaded from the HF hub cache; GPU nodes have no internet.
export HF_HOME="${HF_HOME:-$COSMOS_STORAGE/hf_cache}" HF_HUB_CACHE="${HF_HUB_CACHE:-$COSMOS_STORAGE/hf_cache/hub}"
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export LIBERO_ROOT="${LIBERO_ROOT:-}"
export REPTILE_CKPT_PATH="${REPTILE_CKPT_PATH:-}"
: "${MASTER_PORT:=50902}"
export MASTER_PORT
# _sft_launcher_common.sh validates BASE_CHECKPOINT_PATH / defaults WAN_VAE_PATH when this is set; the TOML itself
# reads REPTILE_CKPT_PATH.
: "${BASE_CHECKPOINT_PATH:=${REPTILE_CKPT_PATH}}"

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$LIBERO_ROOT'\'')." >&2; exit 1; }; [[ "$LIBERO_ROOT" == *libero_spatial* ]] || echo ">>> WARNING: LIBERO_ROOT does not look like the libero_spatial suite (got: $LIBERO_ROOT)." >&2; [[ -d "$REPTILE_CKPT_PATH/model" ]] || { echo "ERROR: REPTILE_CKPT_PATH must be a Reptile DCP iteration dir containing model/ (got: '\''$REPTILE_CKPT_PATH'\''). Run examples/launch_reptile_meta_edge.sh first." >&2; exit 1; }; [[ -d "$HF_HUB_CACHE/models--facebook--dinov2-base" ]] || echo ">>> WARNING: facebook/dinov2-base not in $HF_HUB_CACHE; it will be downloaded (needs internet on this node)." >&2'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
