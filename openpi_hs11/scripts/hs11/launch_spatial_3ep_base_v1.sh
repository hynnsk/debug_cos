#!/usr/bin/env bash
# libero_spatial / 3 demos/task (the final cosmos_hs11 subset) / pi05_base init (baseline). Drives scripts/hs11/toml/spatial_3ep_base_v1.toml.
# Usage: sr 4 48 scripts/hs11/launch_spatial_3ep_base_v1.sh [section.key=value ...]     (TOML_FILE=<other toml> to swap the file)
#   e.g. ... launch_spatial_3ep_base_v1.sh train.num_train_steps=20 train.save_interval=10 wandb.enabled=false   # GPU smoke
#   dry run (CPU):  JAX_PLATFORMS=cpu .venv/bin/python scripts/train_toml.py --toml scripts/hs11/toml/spatial_3ep_base_v1.toml --dryrun
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TOML_FILE="${TOML_FILE:-scripts/hs11/toml/spatial_3ep_base_v1.toml}"
echo ">>> $(date '+%H:%M:%S') train_toml: $TOML_FILE $*"
exec "$PY" scripts/train_toml.py --toml "$TOML_FILE" "$@"
