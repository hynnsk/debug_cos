#!/usr/bin/env bash
# Reptile meta-training of pi05_base over the five robot_fewshot embodiments (full-parameter theta, JAX FSDP).
# Defaults = cosmos_hs11 action_reptile_meta_edge_v4.toml: K=32 demos x 8 windows (256 support), 10 inner AdamW steps of the
# full support set (lr 5e-5, 2-step ramp, clip 1.0, Adam reset per episode), q=8 query demos, eps 0.5 -> 0.05 over 1000
# meta-iterations, save every 250.
# Usage (4 GPUs on one node):   EXP_NAME=reptile_v1 NGPU=4 sr 4 48 scripts/hs11/launch_reptile_pi05.sh
#   extra flags pass through, e.g. ... scripts/hs11/launch_reptile_pi05.sh --max-iter 500 --inner-batch-size 16
#   joint-SFT control (k=1):    ... scripts/hs11/launch_reptile_pi05.sh --inner-steps 1 --meta-lr 1.0 --meta-lr-min-ratio 1.0 --no-inner-reset-optimizer
# Prerequisites: scripts/hs11/download_pi05_base.sh, scripts/hs11/compute_norm_stats_reptile.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
: "${EXP_NAME:?set EXP_NAME (checkpoints -> checkpoints/pi05_reptile/<EXP_NAME>)}"
# GPU count as JAX will see it inside the allocation (cgroup / CUDA_VISIBLE_DEVICES aware; nvidia-smi -L may list the whole node).
NGPU="${NGPU:-$(XLA_PYTHON_CLIENT_PREALLOCATE=false "$PY" -c "import jax; print(jax.device_count())" 2>/dev/null || echo 0)}"
[[ "$NGPU" -ge 1 ]] || { echo "ERROR: no GPU visible" >&2; exit 1; }
for e in google_robot_rt1 bridge_v2 robomind/ur_1rgb robomind/franka_1rgb yam/repos; do
    [[ -d "$ROBOT_FEWSHOT_ROOT/$e" ]] || { echo "ERROR: $ROBOT_FEWSHOT_ROOT/$e missing" >&2; exit 1; }
done
for e in fractal bridge robomind_ur robomind_franka molmoact2_yam; do
    [[ -f "assets/pi05_reptile/$e/norm_stats.json" ]] || { echo "ERROR: assets/pi05_reptile/$e/norm_stats.json missing; run scripts/hs11/compute_norm_stats_reptile.sh" >&2; exit 1; }
done
echo ">>> $(date '+%H:%M:%S') Reptile pi05: exp=$EXP_NAME gpus=$NGPU (fsdp over all) extra: $*"
exec "$PY" scripts/train_reptile.py --exp-name "$EXP_NAME" --fsdp-devices "$NGPU" "$@"
