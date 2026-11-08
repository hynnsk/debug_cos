#!/usr/bin/env bash
# v151: REPA from the FROZEN mid-trained Cosmos3-Edge network on CLEAN predicted frames (teacher_input clean_context): student block 24 -> teacher block 24, feature cosine (w 5) + relation loss (w 5, l1, margin 0.1); otherwise the v111 recipe (ranking hinge: relative margin 0.3, full gradient through E_neg (negative_grad_scale 1.0), weight 1; fresh action heads, Reptile theta). Needs BASE_CHECKPOINT_PATH = the mid-trained Cosmos3-Edge DCP (the frozen teacher).
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# LIBERO-10 Reptile post-training = v104 (action-language ranking loss) + the cosmos_hs10 DINOv2 ViT-B/14 REPA
# distillation (experiment action_policy_libero_edge_repa, [model.repa] + [model.language_ranking] in the TOML).
# Drives action_policy_libero_10_edge_reptileinit_v151.toml. Copied from launch_sft_action_policy_libero_10_edge_reptileinit_v130.sh (same env):
# HF_HOME (the frozen teacher facebook/dinov2-base is read from $HF_HOME/hub, downloaded once if absent).
# Required: LIBERO_ROOT=<path>/libero_10, REPTILE_CKPT_PATH, META_ACTION_INIT_PATH.
# Optional: WAN_VAE_PATH, NPROC_PER_NODE, MASTER_PORT, IMAGINAIRE_OUTPUT_ROOT, EXTRA_TAIL_OVERRIDES.
# See docs/action_policy_libero_language_ranking.md.

TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_libero_10_edge_reptileinit_v151.toml}"
: "${COSMOS_STORAGE:=$HOME/project/cosmos_storage}"
export COSMOS_STORAGE
# Frozen DINOv2 teacher (facebook/dinov2-base) is loaded from the HF hub cache; GPU nodes have no internet.
export HF_HOME="${HF_HOME:-$COSMOS_STORAGE/hf_cache}" HF_HUB_CACHE="${HF_HUB_CACHE:-$COSMOS_STORAGE/hf_cache/hub}"
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export LIBERO_ROOT="${LIBERO_ROOT:-}"
export REPTILE_CKPT_PATH="${REPTILE_CKPT_PATH:-}"
export META_ACTION_INIT_PATH="${META_ACTION_INIT_PATH:-}"
: "${MASTER_PORT:=50892}"
export MASTER_PORT
# dcp_init REPA teacher: the TOML reads BASE_CHECKPOINT_PATH as the FROZEN mid-trained Cosmos3-Edge teacher network
# (repa.teacher_checkpoint_path), so it must be set explicitly -- no fallback to REPTILE_CKPT_PATH here, unlike the
# other reptileinit launchers. The student still loads REPTILE_CKPT_PATH ([checkpoint].load_path).
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-}"

EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$LIBERO_ROOT'\'')." >&2; exit 1; }; [[ -d "$REPTILE_CKPT_PATH/model" ]] || { echo "ERROR: REPTILE_CKPT_PATH must be a Reptile DCP iteration dir containing model/ (got: '\''$REPTILE_CKPT_PATH'\''). Run examples/launch_reptile_meta_edge.sh first." >&2; exit 1; }; [[ -f "$META_ACTION_INIT_PATH" ]] || { echo "ERROR: META_ACTION_INIT_PATH must point at the Reptile meta_action_init*.pt (got: '\''$META_ACTION_INIT_PATH'\'')." >&2; exit 1; }; [[ -d "$HF_HUB_CACHE/models--facebook--dinov2-base" ]] || echo ">>> WARNING: facebook/dinov2-base not in $HF_HUB_CACHE; it will be downloaded (needs internet on this node)." >&2; [[ -d "$BASE_CHECKPOINT_PATH/model" ]] || { echo "ERROR: BASE_CHECKPOINT_PATH must be the mid-trained Cosmos3-Edge DCP dir containing model/ (the dcp_init REPA teacher; got: '\''$BASE_CHECKPOINT_PATH'\'')." >&2; exit 1; }; [[ "$(cd "$BASE_CHECKPOINT_PATH" && pwd -P)" != "$(cd "$REPTILE_CKPT_PATH" && pwd -P)" ]] || { echo "ERROR: BASE_CHECKPOINT_PATH equals REPTILE_CKPT_PATH: the dcp_init teacher would be the Reptile init itself (that is self_init, v126-v129)." >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
