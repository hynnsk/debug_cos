#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# 8-GPU launcher for the Nano REPA v10 recipe (V-JEPA 2.1 ViT-L/16 + spatial-normalized targets).
#   NPROC_PER_NODE=8 sr 8 48 bash examples/launch_sft_action_policy_libero_10_nano_repa_v10.sh
export REPA_TOML_FILE="${REPA_TOML_FILE:-examples/toml/sft_config/action_policy_libero_10_nano_repa_v10.3.toml}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-50037}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_sft_action_policy_libero_10_nano_repa.sh" "$@"
