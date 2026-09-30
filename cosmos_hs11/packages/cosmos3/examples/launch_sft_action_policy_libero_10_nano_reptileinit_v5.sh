#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# cosmos_hs11: Cosmos3-NANO LIBERO-10 few-shot (3 demos/task) FULL fine-tune warm-started from a full-mode NANO Reptile
# checkpoint + DINOv2 ViT-L/14 REPA distillation on the last MoT block with LESS POOLING (target_subgrid_thw=[1,2,2], v5). Drives
# examples/toml/sft_config/action_policy_libero_10_nano_reptileinit_v5.toml (override with TOML_FILE / REPA_TOML_FILE).
# Extra env: HF_HOME (frozen teacher facebook/dinov2-large read from $HF_HOME/hub). See docs/action_reptile_meta.md section 9.
#
# Required env vars:
#   LIBERO_ROOT            local LIBERO-10 LeRobot dataset dir, e.g. <dir>/libero_10
#   REPTILE_CKPT_PATH      <reptile job>/checkpoints/iter_XXXXXXXXX  (the directory that contains model/)
#   META_ACTION_INIT_PATH  <reptile job>/meta_action_init_iter_XXXXXX.pt (the matching action heads)
# Optional: WAN_VAE_PATH, IMAGINAIRE_OUTPUT_ROOT, NPROC_PER_NODE, MASTER_PORT, EXTRA_TAIL_OVERRIDES.
# Usage:
#   NPROC_PER_NODE=8 sr 8 48 examples/launch_sft_action_policy_libero_10_nano_reptileinit_v5.sh

TOML_FILE="${REPA_TOML_FILE:-${TOML_FILE:-examples/toml/sft_config/action_policy_libero_10_nano_reptileinit_v5.toml}}"
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
export MASTER_PORT=${MASTER_PORT:-50053}   # distinct from the plain nano reptileinit (50027) and v2 (50050) launchers
# _sft_launcher_common.sh validates BASE_CHECKPOINT_PATH / defaults WAN_VAE_PATH when this is set; the TOML itself
# reads REPTILE_CKPT_PATH.
: "${BASE_CHECKPOINT_PATH:=${REPTILE_CKPT_PATH}}"

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$LIBERO_ROOT'\'')." >&2; exit 1; }; [[ -d "$REPTILE_CKPT_PATH/model" ]] || { echo "ERROR: REPTILE_CKPT_PATH must be a Reptile DCP iteration dir containing model/ (got: '\''$REPTILE_CKPT_PATH'\''). Run examples/launch_reptile_meta_nano.sh first." >&2; exit 1; }; [[ -f "$META_ACTION_INIT_PATH" ]] || { echo "ERROR: META_ACTION_INIT_PATH must point at the Reptile meta_action_init*.pt (got: '\''$META_ACTION_INIT_PATH'\'')." >&2; exit 1; }; [[ -d "$HF_HUB_CACHE/models--facebook--dinov2-large" ]] || echo ">>> WARNING: facebook/dinov2-large not in $HF_HUB_CACHE; it will be downloaded (needs internet on this node)." >&2'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
