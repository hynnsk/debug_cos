#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# cosmos_hs11: LIBERO-GOAL few-shot (3 demos/task) FULL fine-tune from a full-mode REPTILE checkpoint + DINOv2 ViT-B/14 REPA
# distillation -- LIBERO-Goal port of launch_sft_action_policy_libero_10_edge_reptileinit_repa_dinov2_v49.sh (student-side trilinear upsampler to the full 224 px grid, MoT block 24).
# = launch_sft_action_policy_libero_goal_edge_reptileinit_v7.sh + the HF teacher env. Drives examples/toml/sft_config/action_policy_libero_goal_edge_reptileinit_v22.toml
# (override with REPA_TOML_FILE). Same env as launch_sft_action_policy_libero_goal_edge_reptileinit_v7.sh plus HF_HOME (the frozen
# teacher facebook/dinov2-base is read from $HF_HOME/hub, downloaded once if absent). See docs/action_reptile_meta.md section 8.
#
# Required env vars:
#   LIBERO_ROOT            local LIBERO-Goal LeRobot dataset dir, e.g. <dir>/libero_goal
#   REPTILE_CKPT_PATH      <reptile job>/checkpoints/iter_XXXXXXXXX  (the directory that contains model/)
#   META_ACTION_INIT_PATH  <reptile job>/meta_action_init_iter_XXXXXX.pt (the matching action heads)
# Optional: WAN_VAE_PATH, IMAGINAIRE_OUTPUT_ROOT, NPROC_PER_NODE, MASTER_PORT, EXTRA_TAIL_OVERRIDES.
# Usage:
#   NPROC_PER_NODE=4 sr 4 48 examples/launch_sft_action_policy_libero_goal_edge_reptileinit_v22.sh

TOML_FILE="${REPA_TOML_FILE:-${TOML_FILE:-examples/toml/sft_config/action_policy_libero_goal_edge_reptileinit_v22.toml}}"
: "${MASTER_PORT:=50087}"   # libero_goal v17-v22 = 50082-50087; distinct from every other cosmos_hs* launcher (50004-50081)
export MASTER_PORT
: "${COSMOS_STORAGE:=$HOME/project/cosmos_storage}"
export COSMOS_STORAGE
export HF_HOME="${HF_HOME:-$COSMOS_STORAGE/hf_cache}" HF_HUB_CACHE="${HF_HUB_CACHE:-$COSMOS_STORAGE/hf_cache/hub}"
# The packing loaders ship many tensors per batch (+ one native uint8 clip per sample for the teacher); lift the
# 1024 soft fd limit of the compute nodes (train.py also switches to file_system tensor sharing).
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export LIBERO_ROOT="${LIBERO_ROOT:-}"
export REPTILE_CKPT_PATH="${REPTILE_CKPT_PATH:-}"
export META_ACTION_INIT_PATH="${META_ACTION_INIT_PATH:-}"
# _sft_launcher_common.sh validates BASE_CHECKPOINT_PATH / defaults WAN_VAE_PATH when this is set; the TOML itself
# reads REPTILE_CKPT_PATH.
: "${BASE_CHECKPOINT_PATH:=${REPTILE_CKPT_PATH}}"

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$LIBERO_ROOT'\'')." >&2; exit 1; }; [[ -d "$REPTILE_CKPT_PATH/model" ]] || { echo "ERROR: REPTILE_CKPT_PATH must be a Reptile DCP iteration dir containing model/ (got: '\''$REPTILE_CKPT_PATH'\''). Run examples/launch_reptile_meta_edge.sh first." >&2; exit 1; }; [[ -f "$META_ACTION_INIT_PATH" ]] || { echo "ERROR: META_ACTION_INIT_PATH must point at the Reptile meta_action_init*.pt (got: '\''$META_ACTION_INIT_PATH'\'')." >&2; exit 1; }; [[ -d "$HF_HUB_CACHE/models--facebook--dinov2-base" ]] || echo ">>> WARNING: facebook/dinov2-base not in $HF_HUB_CACHE; it will be downloaded (needs internet on this node)." >&2'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
