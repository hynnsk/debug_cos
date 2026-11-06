#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# cosmos_hs11: OPEN-LOOP action accuracy of a RoboCasa365 checkpoint on held-out demo windows (no simulator).
# Starts one action_policy_server_robocasa per GPU (same server as the closed-loop eval), then runs
# cosmos_framework/simulation/robocasa/openloop_eval.py, which feeds the first frame + language + EEF state token of
# val-split windows and compares the sampled 16-step chunk with the demonstrator's actions (env 12D space).
# See the docstring of openloop_eval.py for how to read the numbers.
#
# Env vars:
#   RUN_ROOT   (required) job dir holding config.yaml + checkpoints/iter_XXXXXXXXX
#   ITER       (default 3000)
#   OUT_DIR    (default $COSMOS_STORAGE/outputs/eval_robocasa_openloop/<run>_iter<ITER>)
#   ROBOCASA_ROOT (default $COSMOS_STORAGE/data/robocasa365/v30/target/atomic)
#   NUM_SERVERS (default SLURM_GPUS_ON_NODE, else 1): one server per GPU, CUDA_VISIBLE_DEVICES=i, port BASE_PORT+i
#   BASE_PORT (default 8901), WINDOWS_PER_TASK (default 12), TASKS (default: all 18), SERVER_PY (cosmos3-pt python)
#   SPLIT      val (default, held-out demos) | train (the run's training demos; set EPISODES_PER_TASK=50/150 to match the
#              run's fraction). val vs train tells underfitting (both bad) from overfitting (train good, val bad).
# Usage:
#   RUN_ROOT=$COSMOS_STORAGE/outputs/cosmos3_action_robocasa/edge_robocasa365_atomic/edge_robocasa365_atomic_10pct_fullft_baseinit \
#     ITER=3000 sr 4 48 --qos=<qos> examples/eval_robocasa_openloop.sh

set -uo pipefail
: "${COSMOS_STORAGE:=$HOME/project/cosmos_storage}"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT=<job dir with config.yaml + checkpoints/>}"
ITER="${ITER:-3000}"
RUN_NAME="$(basename "$RUN_ROOT")"
OUT_DIR="${OUT_DIR:-$COSMOS_STORAGE/outputs/eval_robocasa_openloop/${RUN_NAME}_iter${ITER}_${SPLIT}}"
ROBOCASA_ROOT="${ROBOCASA_ROOT:-$COSMOS_STORAGE/data/robocasa365/v30/target/atomic}"
NUM_SERVERS="${NUM_SERVERS:-${SLURM_GPUS_ON_NODE:-1}}"
BASE_PORT="${BASE_PORT:-8901}"
WINDOWS_PER_TASK="${WINDOWS_PER_TASK:-12}"
SPLIT="${SPLIT:-val}"                      # val (held-out demos) | train (the run's own training demos)
EPISODES_PER_TASK="${EPISODES_PER_TASK:-}"  # with SPLIT=train: 50 / 150 / empty = all, i.e. the run's data fraction
TASKS="${TASKS:-}"
SERVER_PY="${SERVER_PY:-$HOME/anaconda3/envs/cosmos3-pt/bin/python}"
CKPT="$RUN_ROOT/checkpoints/$(printf 'iter_%09d' "$ITER")"
TRAIN_CONFIG="$RUN_ROOT/config.yaml"
[[ -d "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }
[[ -f "$TRAIN_CONFIG" ]] || { echo "ERROR: config not found: $TRAIN_CONFIG" >&2; exit 1; }
[[ -x "$SERVER_PY" ]] || { echo "ERROR: SERVER_PY not found: $SERVER_PY" >&2; exit 1; }
# bob/greco: broken IB stack segfaults the first NCCL collective (same guard as the closed-loop eval script).
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
mkdir -p "$OUT_DIR"
cd "$WORKDIR"
echo ">>> $(date '+%H:%M:%S') $RUN_NAME iter $ITER -> $OUT_DIR ($NUM_SERVERS server(s) on $(hostname -s))"

PIDS=(); URLS=()
for ((i = 0; i < NUM_SERVERS; i++)); do
    PORT=$((BASE_PORT + i))
    CUDA_VISIBLE_DEVICES="$i" MASTER_PORT=$((29500 + i)) PYTHONPATH=. "$SERVER_PY" -m cosmos_framework.scripts.action_policy_server_robocasa \
        --checkpoint-path "$CKPT" --config-file "$TRAIN_CONFIG" --fps 20 \
        --output-dir "$OUT_DIR/server$i" --port "$PORT" > "$OUT_DIR/server$i.log" 2>&1 &
    PIDS+=($!); URLS+=("http://127.0.0.1:$PORT")
done
trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT

echo ">>> $(date '+%H:%M:%S') waiting for ${#URLS[@]} server(s)..."
for u in "${URLS[@]}"; do
    for _ in $(seq 1 360); do
        if curl -sf "$u/info" > /dev/null 2>&1; then break; fi
        alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=$((alive + 1)); done
        [[ "$alive" -eq "${#PIDS[@]}" ]] || { echo "ERROR: a server exited early, see $OUT_DIR/server*.log" >&2; exit 1; }
        sleep 5
    done
done
curl -s "${URLS[0]}/info" | tee "$OUT_DIR/server_info.json"; echo

TASK_FLAG=()
# shellcheck disable=SC2206
[[ -n "$TASKS" ]] && TASK_FLAG=(--tasks $TASKS)
PYTHONPATH=. "$SERVER_PY" cosmos_framework/simulation/robocasa/openloop_eval.py \
    --server-urls "${URLS[@]}" --robocasa-root "$ROBOCASA_ROOT" --out-dir "$OUT_DIR" \
    --windows-per-task "$WINDOWS_PER_TASK" --split "$SPLIT" ${EPISODES_PER_TASK:+--episodes-per-task "$EPISODES_PER_TASK"} \
    "${TASK_FLAG[@]}" 2>&1 | tee "$OUT_DIR/openloop.log"
RC=${PIPESTATUS[0]}
echo ">>> $(date '+%H:%M:%S') done (exit $RC): $OUT_DIR/openloop_summary.json"
exit "$RC"
