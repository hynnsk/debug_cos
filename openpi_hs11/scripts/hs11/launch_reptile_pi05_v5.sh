#!/usr/bin/env bash
# Reptile meta pre-training of pi05_base (full theta), variant v5. Drives scripts/hs11/toml/reptile_pi05_v5.toml (section.key=value overrides pass through); see the TOML header for the rationale.
# Usage: sr 4 48 scripts/hs11/launch_reptile_pi05_v5.sh            smoke: ... launch_reptile_pi05_v5.sh outer.max_iter=2 checkpoint.save_interval=1 wandb.enabled=false
#   dry run (CPU): JAX_PLATFORMS=cpu .venv/bin/python scripts/train_reptile_toml.py --toml scripts/hs11/toml/reptile_pi05_v5.toml --dryrun
# Prerequisites: scripts/hs11/download_pi05_base.sh, scripts/hs11/compute_norm_stats_reptile.sh (both done once).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TOML_FILE="${TOML_FILE:-scripts/hs11/toml/reptile_pi05_v5.toml}"
for e in fractal bridge robomind_ur robomind_franka molmoact2_yam; do
    [[ -f "assets/pi05_reptile/$e/norm_stats.json" ]] || { echo "ERROR: assets/pi05_reptile/$e/norm_stats.json missing; run scripts/hs11/compute_norm_stats_reptile.sh" >&2; exit 1; }
done
echo ">>> $(date '+%H:%M:%S') train_reptile_toml: $TOML_FILE $*"
exec "$PY" scripts/train_reptile_toml.py --toml "$TOML_FILE" "$@"
