#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# cosmos_hs09_2: LIBERO-10 few-shot post-training (3 demos/task) from the META-LEARNED init (cosmos_hs09
# launch_sft_action_policy_libero_10_edge_metainit.sh) + the cosmos_hs10 v7 DINOv2 ViT-B/14 REPA distillation loss.
# Drives examples/toml/sft_config/action_policy_libero_10_edge_metainit_repa_dinov2.toml (2 GPUs, shard 2) and
# requires META_ACTION_INIT_PATH (= <meta run dir>/meta_action_init[_iter_XXXXXX].pt from train_action_meta).
#
# Required env vars:
#   LIBERO_ROOT            local LIBERO-10 LeRobot dataset dir, e.g. <dir>/libero_10 (contains meta/info.json)
#   META_ACTION_INIT_PATH  meta_action_init.pt written by cosmos_framework.scripts.train_action_meta
# Optional env vars:
#   REPA_TOML_FILE         recipe TOML; default the meta-init + DINOv2 recipe above
#   BASE_CHECKPOINT_PATH   default: examples/checkpoints/Cosmos3-Edge   (mid-trained Cosmos3-Edge DCP)
#   WAN_VAE_PATH           default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   HF_HOME                Hugging Face cache; the frozen teacher is facebook/dinov2-base (loaded from
#                          $HF_HOME/hub first, downloaded once if absent -- do that on a node with internet)
#   IMAGINAIRE_OUTPUT_ROOT, NPROC_PER_NODE (2), MASTER_PORT (default 50018), EXTRA_TAIL_OVERRIDES
#                          (space-separated Hydra overrides, e.g. "model.config.repa.loss_weight=1.0 trainer.max_iter=5")
#
# Usage (single node, 2 GPUs):
#   export META_ACTION_INIT_PATH=$IMAGINAIRE_OUTPUT_ROOT/cosmos3_action_meta/fewshot_meta_lora/edge_meta_lora_r32_fomaml_adam_k5q5_w8_seed42/meta_action_init_iter_001000.pt
#   NPROC_PER_NODE=2 sr 2 48 examples/launch_sft_action_policy_libero_10_edge_metainit_repa_dinov2.sh

TOML_FILE="${REPA_TOML_FILE:-examples/toml/sft_config/action_policy_libero_10_edge_metainit_repa_dinov2.toml}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"
: "${MASTER_PORT:=50018}"   # distinct from the hs09 (50012/50015/50017) and hs10 (50016) launchers
export MASTER_PORT

export LIBERO_ROOT="${LIBERO_ROOT:-}"
export META_ACTION_INIT_PATH="${META_ACTION_INIT_PATH:-}"

# The packing loaders ship many tensors per batch (+ one native uint8 clip per sample for the DINOv2 teacher);
# lift the 1024 soft fd limit of the compute nodes (train.py also switches to file_system tensor sharing).
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
# NCCL_IB_DISABLE (single node) and PYTORCH_CUDA_ALLOC_CONF=expandable_segments are set by _sft_launcher_common.sh.

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$LIBERO_ROOT'\'')." >&2; exit 1; }; [[ -f "$META_ACTION_INIT_PATH" ]] || { echo "ERROR: META_ACTION_INIT_PATH must point at a meta_action_init.pt (got: '\''$META_ACTION_INIT_PATH'\''). Run examples/launch_meta_action_fewshot_lora_edge.sh first." >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
