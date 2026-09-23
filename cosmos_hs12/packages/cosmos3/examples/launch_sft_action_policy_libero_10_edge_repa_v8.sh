#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Two-GPU launcher for the v8 same-patch temporal-transition REPA recipe.
export REPA_TOML_FILE="${REPA_TOML_FILE:-examples/toml/sft_config/action_policy_libero_10_edge_repa_v8.toml}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MASTER_PORT="${MASTER_PORT:-50018}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_sft_action_policy_libero_10_edge_repa.sh" "$@"
