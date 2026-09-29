#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# 8-GPU launcher for the Nano REPA v10.6 recipe (v10.5 without pooling: 240 px teacher, target_subgrid_thw=[2,3,3]).
#   NPROC_PER_NODE=4 sr 4 48 bash examples/launch_sft_action_policy_libero_10_nano_repa_v10.6.sh   (shard 4 x 64 windows, as v10.2)
export REPA_TOML_FILE="${REPA_TOML_FILE:-examples/toml/sft_config/action_policy_libero_10_nano_repa_v10.6.toml}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-50047}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_sft_action_policy_libero_10_nano_repa.sh" "$@"
