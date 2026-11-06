#!/usr/bin/env bash
# libero_goal / 3 demos/task (the final cosmos_hs11 subset) / Reptile init. Drives scripts/hs11/toml/goal_3ep_reptile_v8.4.toml.
# Usage: sr 4 48 scripts/hs11/launch_goal_3ep_reptile_v8.4.sh [section.key=value ...]     (TOML_FILE=<other toml> to swap the file)
#   e.g. ... launch_goal_3ep_reptile_v8.4.sh train.num_train_steps=20 train.save_interval=10 wandb.enabled=false   # GPU smoke
#   dry run (CPU):  JAX_PLATFORMS=cpu .venv/bin/python scripts/train_toml.py --toml scripts/hs11/toml/goal_3ep_reptile_v8.4.toml --dryrun
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TOML_FILE="${TOML_FILE:-scripts/hs11/toml/goal_3ep_reptile_v8.4.toml}"
# REPTILE_PARAMS=<reptile ckpt>/<meta-iter>/params overrides [checkpoint].init_params of the TOML.
if [[ -n "${REPTILE_PARAMS:-}" ]]; then set -- "checkpoint.init_params=$REPTILE_PARAMS" "$@"; fi
echo ">>> $(date '+%H:%M:%S') train_toml: $TOML_FILE $*"
exec "$PY" scripts/train_toml.py --toml "$TOML_FILE" "$@"
