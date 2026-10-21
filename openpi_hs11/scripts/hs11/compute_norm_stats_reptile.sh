#!/usr/bin/env bash
# Per-embodiment quantile stats for Reptile (fractal, bridge, robomind_ur, robomind_franka, molmoact2_yam). CPU, no video decode.
# Writes assets/pi05_reptile/<embodiment>/norm_stats.json. Extra args go to scripts/hs11/compute_norm_stats_reptile.py.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
JAX_PLATFORMS=cpu "$PY" scripts/hs11/compute_norm_stats_reptile.py --assets-dir ./assets/pi05_reptile "$@"
