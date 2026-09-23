#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Based on launch_sft_action_policy_libero_10_edge_repa.sh.
# JEPA_VARIANT=dense (default) | masked_only | linear
# Set LIBERO_ROOT, BASE_CHECKPOINT_PATH, WAN_VAE_PATH and COSMOS_STORAGE.
# Other overrides are the same as the original REPA launcher.
set -euo pipefail
case "${JEPA_VARIANT:-masked_only}" in
    dense) recipe=action_policy_libero_10_edge_masked_jepa ;;
    masked_only) recipe=action_policy_libero_10_edge_masked_jepa_masked_only ;;
    linear) recipe=action_policy_libero_10_edge_masked_jepa_linear ;;
    *) echo "ERROR: JEPA_VARIANT must be dense, masked_only, or linear" >&2; exit 2 ;;
esac
export REPA_TOML_FILE="${REPA_TOML_FILE:-examples/toml/sft_config/${recipe}.toml}"
export MASTER_PORT="${MASTER_PORT:-50019}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_sft_action_policy_libero_10_edge_repa.sh" "$@"
