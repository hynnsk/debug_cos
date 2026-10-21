#!/usr/bin/env bash
# Quantile normalization stats for the four LIBERO_LeRobot_v3 suites (state + actions; images are skipped). CPU, ~1 min/suite.
# Writes assets/pi05_libero_v3_<suite>_all_base/<suite>/norm_stats.json, which all six recipes of a suite share.
# Usage: bash scripts/hs11/compute_norm_stats_libero_v3.sh [suite ...]   (default: all four suites)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SUITES=("$@"); [[ ${#SUITES[@]} -gt 0 ]] || SUITES=(libero_10 libero_goal libero_object libero_spatial)
for suite in "${SUITES[@]}"; do
    echo ">>> $(date '+%H:%M:%S') norm stats for $suite"
    HS11_SKIP_IMAGES=1 JAX_PLATFORMS=cpu "$PY" scripts/compute_norm_stats.py --config-name "pi05_libero_v3_${suite}_all_base"
done
