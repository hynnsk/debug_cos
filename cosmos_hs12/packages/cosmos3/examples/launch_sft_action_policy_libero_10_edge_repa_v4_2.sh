#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for action_policy_libero_edge_repa, variant v4 -- Cosmos3-Edge LIBERO-10 few-shot
# action-policy post-training + V-JEPA 2.1 REPA loss whose student-side projector is a SINGLE LINEAR LAYER
# (model.repa.projector_type = "linear") instead of the REPA 3-layer SiLU MLP used by the base / v2 / v3 recipes.
# Drives cosmos_framework.scripts.train against examples/toml/sft_config/action_policy_libero_10_edge_repa_v4.toml
# (2 GPUs, shard 2) by default. See docs/action_policy_libero_repa_vjepa.md.
#
# Required env vars:
#   LIBERO_ROOT            local LIBERO-10 LeRobot dataset dir (contains meta/info.json)
# Optional env vars:
#   REPA_TOML_FILE         recipe TOML; default the v4 (ViT-B / block-8 / avgpool / linear projector) recipe.
#                            MLP-projector recipes: examples/toml/sft_config/action_policy_libero_10_edge_repa{,_l14,_vitl,_v2,_v3}.toml
#   BASE_CHECKPOINT_PATH   default: examples/checkpoints/Cosmos3-Edge   (mid-trained Cosmos3-Edge DCP)
#   WAN_VAE_PATH           default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   COSMOS_STORAGE         teacher checkpoints are read from $COSMOS_STORAGE/checkpoints/vjepa2_1/
#                          (vjepa2_1_vitb_dist_vitG_384.pt / vjepa2_1_vitl_dist_vitG_384.pt; downloaded if absent)
#   MASTER_PORT            default 50016 (distinct from the hs08/hs09 launchers)
#   EXTRA_TAIL_OVERRIDES   space-separated Hydra overrides, e.g. "model.config.repa.loss_weight=1.0 trainer.max_iter=5"
#
# Usage (single node, 2 GPUs):
#   NPROC_PER_NODE=2 bash examples/launch_sft_action_policy_libero_10_edge_repa_v4.sh
#   EXTRA_TAIL_OVERRIDES="model.config.repa.layer_index=28 job.name=hs10_repa_l28_linear" NPROC_PER_NODE=2 bash examples/launch_sft_action_policy_libero_10_edge_repa_v4.sh

TOML_FILE="${REPA_TOML_FILE:-examples/toml/sft_config/action_policy_libero_10_edge_repa_v4_2.toml}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"
: "${MASTER_PORT:=50016}"
export MASTER_PORT

# LIBEROLeRobotDataset reads ${oc.env:LIBERO_ROOT} directly (a LOCAL LeRobot dir).
export LIBERO_ROOT="${LIBERO_ROOT:-}"

# Single-node runs on this cluster: the IB/RoCE stack of some nodes (e.g. `bob`) segfaults NCCL at the first
# collective; IB is not needed within a node. Export NCCL_IB_DISABLE yourself to override.
if [[ -z "${NNODES:-}" || "${NNODES:-1}" == "1" ]]; then
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
fi
# The packing loaders ship many tensors per batch (+ one native clip per sample for the teacher); lift the
# 1024 soft fd limit of the compute nodes (train.py also switches to file_system tensor sharing).
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
# The 2-GPU recipe runs within ~1 GB of the 44 GiB cards; expandable segments avoid allocator fragmentation
# (the OOM message itself recommends it). Export PYTORCH_ALLOC_CONF yourself to override.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$LIBERO_ROOT'\''). See docs/action_policy_libero_posttrain.md" >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
