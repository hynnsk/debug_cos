#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Two-GPU launcher for the v13 recipe: v12 + less pooling (target_subgrid_thw=[1,2,2], 2x2 teacher sub-cells per MoT token).
# Same env/overrides as launch_sft_action_policy_libero_10_edge_repa.sh (REPA_TOML_FILE, NPROC_PER_NODE, MASTER_PORT,
# EXTRA_TAIL_OVERRIDES, LIBERO_ROOT, BASE_CHECKPOINT_PATH, WAN_VAE_PATH, COSMOS_STORAGE).
export REPA_TOML_FILE="${REPA_TOML_FILE:-examples/toml/sft_config/action_policy_libero_10_edge_repa_v13.toml}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MASTER_PORT="${MASTER_PORT:-50047}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_sft_action_policy_libero_10_edge_repa.sh" "$@"
