#!/usr/bin/env bash
# LIBERO_LeRobot_v3 post-training: suite x data x init.
#   suite: libero_10 | libero_goal | libero_object | libero_spatial
#   data : all (every episode) | 3ep (3 demos/task, the final cosmos_hs11 jsons) | 1ep (1 demo/task)
#   init : base (pi05_base) | reptile (REPTILE_PARAMS=<reptile ckpt>/<step>/params)
# Usage: EXP_NAME=<run> [REPTILE_PARAMS=...] NGPU=2 sr 2 48 scripts/hs11/launch_sft_libero_v3.sh <suite> <data> <init> [extra tyro flags]
#   e.g. EXP_NAME=goal_3ep_base_v1 sr 2 48 scripts/hs11/launch_sft_libero_v3.sh libero_goal 3ep base
#        EXP_NAME=goal_3ep_reptile_v1 REPTILE_PARAMS=checkpoints/pi05_reptile/reptile_v1/1000/params sr 2 48 \
#            scripts/hs11/launch_sft_libero_v3.sh libero_goal 3ep reptile
# Global batch 64 (override --batch-size), fsdp over all visible GPUs. Checkpoints: checkpoints/<config>/<EXP_NAME>/<step>/.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SUITE="${1:?suite}"; DATA="${2:?data: all|3ep|1ep}"; INIT="${3:?init: base|reptile}"; shift 3
: "${EXP_NAME:?set EXP_NAME}"
CONFIG="pi05_libero_v3_${SUITE}_${DATA}_${INIT}"
[[ -d "$LIBERO_V3_ROOT/$SUITE/meta" ]] || { echo "ERROR: $LIBERO_V3_ROOT/$SUITE is not a LeRobot dir" >&2; exit 1; }
STATS="assets/pi05_libero_v3_${SUITE}_all_base/${SUITE}/norm_stats.json"
[[ -f "$STATS" ]] || { echo "ERROR: $STATS missing; run scripts/hs11/compute_norm_stats_libero_v3.sh $SUITE" >&2; exit 1; }
if [[ "$INIT" == "reptile" ]]; then
    : "${REPTILE_PARAMS:?set REPTILE_PARAMS=<reptile checkpoint dir>/<step>/params}"
    [[ -d "$REPTILE_PARAMS" ]] || { echo "ERROR: REPTILE_PARAMS dir not found: $REPTILE_PARAMS" >&2; exit 1; }
    export HS11_REPTILE_PARAMS="$(cd "$REPTILE_PARAMS" && pwd)"
fi
# GPU count as JAX will see it inside the allocation (cgroup / CUDA_VISIBLE_DEVICES aware; nvidia-smi -L may list the whole node).
NGPU="${NGPU:-$(XLA_PYTHON_CLIENT_PREALLOCATE=false "$PY" -c "import jax; print(jax.device_count())" 2>/dev/null || echo 0)}"
[[ "$NGPU" -ge 1 ]] || { echo "ERROR: no GPU visible" >&2; exit 1; }
echo ">>> $(date '+%H:%M:%S') $CONFIG exp=$EXP_NAME gpus=$NGPU reptile=${HS11_REPTILE_PARAMS:-} extra: $*"
exec "$PY" scripts/train.py "$CONFIG" --exp-name "$EXP_NAME" --fsdp-devices "$NGPU" "$@"
