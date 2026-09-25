#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# cosmos_hs09_2: META INIT -> full FT + the cosmos_hs10 v10 REPA loss (V-JEPA 2.1 ViT-B/16 teacher, per-frame
# spatially normalized cosine). Two-GPU wrapper around launch_sft_action_policy_libero_10_edge_metainit_repa_dinov2.sh
# (the generic meta-init + REPA launcher; REPA_TOML_FILE selects the recipe) driving
# examples/toml/sft_config/action_policy_libero_10_edge_metainit_repa_v10.toml.
#
# Required env vars: LIBERO_ROOT, META_ACTION_INIT_PATH (see the wrapped launcher). The frozen teacher is read from
# $COSMOS_STORAGE/checkpoints/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt (downloaded on first use if absent -- compute
# nodes have no internet, so make sure it is there; it is on this cluster).
# Optional: BASE_CHECKPOINT_PATH, WAN_VAE_PATH, IMAGINAIRE_OUTPUT_ROOT, NPROC_PER_NODE (2), MASTER_PORT (50019),
#           EXTRA_TAIL_OVERRIDES (e.g. "model.config.repa.loss_weight=1.0 model.config.repa.layer_index=14").
#
# Usage (single node, 2 GPUs):
#   export META_ACTION_INIT_PATH=$IMAGINAIRE_OUTPUT_ROOT/cosmos3_action_meta/fewshot_meta_lora/edge_meta_lora_r32_fomaml_adam_k5q5_w8_seed42/meta_action_init_iter_001000.pt
#   NPROC_PER_NODE=2 sr 2 48 examples/launch_sft_action_policy_libero_10_edge_metainit_repa_v10.sh

export REPA_TOML_FILE="${REPA_TOML_FILE:-examples/toml/sft_config/action_policy_libero_10_edge_metainit_repa_v10.toml}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MASTER_PORT="${MASTER_PORT:-50019}"   # distinct from the dinov2 launcher (50018) and the hs09/hs10 launchers
exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_sft_action_policy_libero_10_edge_metainit_repa_dinov2.sh" "$@"
