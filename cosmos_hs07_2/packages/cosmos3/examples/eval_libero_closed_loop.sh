#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Closed-loop LIBERO-10 evaluation of a post-trained checkpoint: starts the action policy server
# (this repo's cosmos3-pt env, 1 GPU) in the background, runs the LIBERO simulator client in the
# separate libero-eval env, writes summary.json, then stops the server.
#
# Required env vars:
#   RUN_ROOT      training run dir, e.g. $IMAGINAIRE_OUTPUT_ROOT/cosmos3_action_libero/edge_libero10_fewshot/<name>
#   ITER          checkpoint iteration, e.g. 1500 (-> $RUN_ROOT/checkpoints/iter_000001500)
#   OUT_DIR       where summary.json / gifs go
# Optional:
#   LIBERO_REPO   (default $HOME/project/LIBERO)     LIBERO_PY (default ~/anaconda3/envs/libero-eval/bin/python)
#   NUM_TRIALS    trials per task (default 50)        PORT (default 8000)      TASK_SUITE (default libero_10)
#   SERVER_PY     python of the training env (default: current `python`)
#
# Usage (inside a 1-GPU allocation):
#   RUN_ROOT=... ITER=1500 OUT_DIR=$COSMOS_STORAGE/outputs/eval/<name>_iter1500 sr 1 48 examples/eval_libero_closed_loop.sh

set -uo pipefail
: "${RUN_ROOT:?set RUN_ROOT}"; : "${ITER:?set ITER}"; : "${OUT_DIR:?set OUT_DIR}"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBERO_REPO="${LIBERO_REPO:-$HOME/project/LIBERO}"
LIBERO_PY="${LIBERO_PY:-$HOME/anaconda3/envs/libero-eval/bin/python}"
SERVER_PY="${SERVER_PY:-python}"
NUM_TRIALS="${NUM_TRIALS:-50}"
PORT="${PORT:-8000}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
CKPT="$RUN_ROOT/checkpoints/iter_$(printf '%09d' "$ITER")"
TRAIN_CONFIG="$RUN_ROOT/config.yaml"
ACTION_STATS="$WORKDIR/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json"
[[ -d "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }
[[ -f "$TRAIN_CONFIG" ]] || { echo "ERROR: config not found: $TRAIN_CONFIG" >&2; exit 1; }
mkdir -p "$OUT_DIR"

cd "$WORKDIR"
echo ">>> $(date '+%H:%M:%S') starting policy server on port $PORT for $CKPT"
PYTHONPATH=. "$SERVER_PY" -m cosmos_framework.scripts.action_policy_server_libero \
    --checkpoint-path "$CKPT" \
    --config-file "$TRAIN_CONFIG" \
    --action-stats-path "$ACTION_STATS" \
    --action-normalization quantile_rot \
    --raw-action-dim 10 \
    --fps 20 \
    --port "$PORT" > "$OUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

echo ">>> $(date '+%H:%M:%S') waiting for the server..."
for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/info" > /dev/null 2>&1; then break; fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "ERROR: server exited early, see $OUT_DIR/server.log" >&2; exit 1; fi
    sleep 5
done

echo ">>> $(date '+%H:%M:%S') running closed-loop eval ($TASK_SUITE, $NUM_TRIALS trials/task)"
env PYTHONPATH="$LIBERO_REPO:$WORKDIR" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    "$LIBERO_PY" "$WORKDIR/cosmos_framework/simulation/libero/closed_loop_eval.py" \
    --server_url "http://127.0.0.1:$PORT" \
    --task_suite "$TASK_SUITE" \
    --num_trials_per_task "$NUM_TRIALS" \
    --action_horizon 16 \
    --camera agentview,wrist \
    --action_space frame_wise_relative \
    --rotation_space 6d \
    --action_dim 10 \
    --gripper_mode zero_one \
    --mujoco_gl egl \
    --timeout 300 \
    --save_gifs --gif_fps 20 \
    --output_dir "$OUT_DIR" 2>&1 | tee "$OUT_DIR/eval.log"
EXIT_CODE=${PIPESTATUS[0]}
echo ">>> $(date '+%H:%M:%S') Done (exit $EXIT_CODE); summary: $OUT_DIR/summary.json"
exit $EXIT_CODE
