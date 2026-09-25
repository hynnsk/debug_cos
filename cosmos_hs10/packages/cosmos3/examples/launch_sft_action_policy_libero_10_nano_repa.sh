#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for action_policy_libero_nano_repa -- Cosmos3-Nano LIBERO-10 few-shot action-policy
# post-training + REPA loss (cosmos_hs10). Nano needs one 8 x 48 GB node (FSDP shard 8 x 32 windows/rank).
# Drives cosmos_framework.scripts.train against the TOML in REPA_TOML_FILE (default: the v1 recipe).
# See docs/action_policy_libero_repa_vjepa.md, section 8.
#
# Required env vars:
#   LIBERO_ROOT            local LIBERO-10 LeRobot dataset dir (contains meta/info.json)
# Optional env vars:
#   REPA_TOML_FILE         recipe TOML; default examples/toml/sft_config/action_policy_libero_10_nano_repa_v1.toml.
#                          Variants: *_nano_repa_v7.toml (DINOv2 ViT-L), *_nano_repa_v10.toml (spatial-normalized targets);
#                          the launch_sft_action_policy_libero_10_nano_repa_v{1,7,10}.sh wrappers pick them.
#   BASE_CHECKPOINT_PATH   default: examples/checkpoints/Cosmos3-Nano   (Cosmos3-Nano DCP, e.g. $COSMOS_STORAGE/checkpoints/Cosmos3-Nano)
#   WAN_VAE_PATH           default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   COSMOS_STORAGE / HF_HOME  V-JEPA 2.1 checkpoints under $COSMOS_STORAGE/checkpoints/vjepa2_1/, DINOv2 in the HF cache
#   NPROC_PER_NODE         default 8 (must equal the TOML's data_parallel_shard_degree)
#   MASTER_PORT            default 50030
#   EXTRA_TAIL_OVERRIDES   space-separated Hydra overrides, e.g. "model.config.repa.layer_index=14 trainer.max_iter=5"
#
# Usage (single node, 8 GPUs):
#   NPROC_PER_NODE=8 sr 8 48 bash examples/launch_sft_action_policy_libero_10_nano_repa.sh

TOML_FILE="${REPA_TOML_FILE:-examples/toml/sft_config/action_policy_libero_10_nano_repa_v1.toml}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"
: "${MASTER_PORT:=50030}"
: "${NPROC_PER_NODE:=4}"
export MASTER_PORT NPROC_PER_NODE

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
# Expandable segments avoid allocator fragmentation on the 44-48 GiB cards. Export PYTORCH_ALLOC_CONF to override.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '"'"'$LIBERO_ROOT'"'"'). See docs/action_policy_libero_posttrain.md" >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
