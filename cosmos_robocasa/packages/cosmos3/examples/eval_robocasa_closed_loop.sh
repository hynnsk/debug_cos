#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Closed-loop evaluation of a post-trained checkpoint on the ORIGINAL RoboCasa benchmark (24 atomic tasks, v0.2 simulator)
# with Cosmos Policy's protocol (cosmos_robocasa): starts the RoboCasa action policy server (this repo's cosmos3-pt env,
# 1 GPU) in the background with the recipe's action denormalization, runs the simulator client
# (cosmos_framework/simulation/robocasa_v02/closed_loop_eval.py in the robocasa-v02-eval env, EGL on the same GPU) task by
# task (object split B, the 5 official test scenes x 10 episodes, one seeded env per episode, registry horizon, first
# success), writes <OUT_DIR>/<task>/results.json (+ rollout mp4s / init pngs) and <OUT_DIR>/summary.json, then stops the
# server. Tasks that already have a results.json are skipped, so an interrupted run resumes.
#
# The contract is fixed below to action_policy_robocasa_edge: chunk 16 executed fully (ACTION_HORIZON=16; Cosmos Policy
# predicts 32 and executes 16), camera_set left_wrist at 224 px, state token min/max-normalized with the bundled stats,
# actions quantile_rot-denormalized by the server with the same stats, guidance 1.0 (no prompt dropout in training).
#
# Required env vars:
#   RUN_ROOT      training run dir, e.g. $IMAGINAIRE_OUTPUT_ROOT/cosmos3_action_robocasa/edge_robocasa_h50/<name>
#   ITER          checkpoint iteration, e.g. 6000 (-> $RUN_ROOT/checkpoints/iter_000006000)
#   OUT_DIR       where results go, e.g. $COSMOS_STORAGE/outputs/eval_robocasa/<name>_iter6000_seed195
# Optional:
#   ROBOCASA_PY   python of the simulator env (default $HOME/anaconda3/envs/robocasa-v02-eval/bin/python)
#   SERVER_PY     python of the training env (default: current `python`, i.e. the cosmos3-pt env)
#   NUM_TRIALS    rollouts per task (default 50 = Cosmos Policy)
#   SEED          base env seed (default 195; Cosmos Policy averages seeds 195 / 196 / 197)
#   DETERMINISTIC 1 (default) seeds each episode env Cosmos-Policy style; 0 = unseeded
#   TASKS         space-separated subset of the 24 tasks (default: all)
#   STATS         stats JSON (default: the bundled robocasa_cosmos_policy_frame_wise_relative_rot6d.json)
#   PORT (8900), ACTION_HORIZON (16), CAMERA_SET (left_wrist), NUM_STEPS (server denoising steps, default 30),
#   SAVE_GEN_VIDEO (0|1), NO_VIDEO (0|1, skip rollout mp4s), TIMEOUT (default 900 s per request / server wait)
# Usage (inside a 1-GPU allocation):
#   RUN_ROOT=... ITER=6000 OUT_DIR=... sr 1 48 --exclude haring examples/eval_robocasa_closed_loop.sh

set -uo pipefail
: "${RUN_ROOT:?set RUN_ROOT}"; : "${ITER:?set ITER}"; : "${OUT_DIR:?set OUT_DIR}"
: "${COSMOS_STORAGE:=$HOME/project/cosmos_storage}"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOCASA_PY="${ROBOCASA_PY:-$HOME/anaconda3/envs/robocasa-v02-eval/bin/python}"
SERVER_PY="${SERVER_PY:-python}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SEED="${SEED:-195}"
DETERMINISTIC="${DETERMINISTIC:-1}"
PORT="${PORT:-8900}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"
CAMERA_SET="${CAMERA_SET:-left_wrist}"
NUM_STEPS="${NUM_STEPS:-30}"
SAVE_GEN_VIDEO="${SAVE_GEN_VIDEO:-0}"
NO_VIDEO="${NO_VIDEO:-0}"
TIMEOUT="${TIMEOUT:-900}"
STATS="${STATS:-$WORKDIR/cosmos_framework/data/generator/action/normalizer_stats/robocasa_cosmos_policy_frame_wise_relative_rot6d.json}"
# bob / greco have a broken IB/RoCE stack: the server's first collective (1-rank broadcast in OmniInference.create) SIGSEGVs
# unless IB is off. Everything here is single-node.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
TASKS="${TASKS:-PnPCounterToCab PnPCabToCounter PnPCounterToSink PnPSinkToCounter PnPCounterToMicrowave PnPMicrowaveToCounter PnPCounterToStove PnPStoveToCounter OpenSingleDoor CloseSingleDoor OpenDoubleDoor CloseDoubleDoor OpenDrawer CloseDrawer TurnOnSinkFaucet TurnOffSinkFaucet TurnSinkSpout TurnOnStove TurnOffStove CoffeeSetupMug CoffeeServeMug CoffeePressButton TurnOnMicrowave TurnOffMicrowave}"
CKPT="$RUN_ROOT/checkpoints/iter_$(printf '%09d' "$ITER")"
TRAIN_CONFIG="$RUN_ROOT/config.yaml"
[[ -d "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }
[[ -f "$TRAIN_CONFIG" ]] || { echo "ERROR: config not found: $TRAIN_CONFIG" >&2; exit 1; }
[[ -f "$STATS" ]] || { echo "ERROR: stats JSON not found: $STATS" >&2; exit 1; }
[[ -x "$ROBOCASA_PY" ]] || { echo "ERROR: ROBOCASA_PY not found: $ROBOCASA_PY (run examples/setup_robocasa_v02_eval.sh)" >&2; exit 1; }
"$ROBOCASA_PY" -c "import robosuite, robocasa" 2>/dev/null || { echo "ERROR: $ROBOCASA_PY cannot import robosuite/robocasa (run examples/setup_robocasa_v02_eval.sh)" >&2; exit 1; }
mkdir -p "$OUT_DIR"
cd "$WORKDIR"

echo ">>> $(date '+%H:%M:%S') starting RoboCasa policy server on port $PORT for $CKPT"
PYTHONPATH=. "$SERVER_PY" -m cosmos_framework.scripts.action_policy_server_robocasa \
    --checkpoint-path "$CKPT" \
    --config-file "$TRAIN_CONFIG" \
    --fps 20 \
    --num-steps "$NUM_STEPS" \
    --action-stats-path "$STATS" \
    --action-normalization quantile_rot \
    --output-dir "$OUT_DIR/server" \
    --port "$PORT" > "$OUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

echo ">>> $(date '+%H:%M:%S') waiting for the server..."
for _ in $(seq 1 360); do
    if curl -sf "http://127.0.0.1:$PORT/info" > /dev/null 2>&1; then break; fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "ERROR: server exited early, see $OUT_DIR/server.log" >&2; exit 1; fi
    sleep 5
done
curl -s "http://127.0.0.1:$PORT/info" | tee "$OUT_DIR/server_info.json"; echo

EXTRA=()
[[ "$SAVE_GEN_VIDEO" == "1" ]] && EXTRA+=(--save-gen-video)
[[ "$NO_VIDEO" == "1" ]] && EXTRA+=(--no-video)
[[ "$DETERMINISTIC" == "0" ]] && EXTRA+=(--no-deterministic)

FAILED=0
for t in $TASKS; do
    TASK_OUT="$OUT_DIR/$t"
    if [[ -f "$TASK_OUT/results.json" ]] && [[ "$("$ROBOCASA_PY" -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$TASK_OUT/results.json")" -ge "$NUM_TRIALS" ]]; then
        echo ">>> $(date '+%H:%M:%S') $t: results.json complete, skipping"; continue
    fi
    mkdir -p "$TASK_OUT"
    echo ">>> $(date '+%H:%M:%S') $t: $NUM_TRIALS rollouts"
    env PYTHONPATH="$WORKDIR" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
        "$ROBOCASA_PY" "$WORKDIR/cosmos_framework/simulation/robocasa_v02/closed_loop_eval.py" \
        --server-url "http://127.0.0.1:$PORT" \
        --task "$t" \
        --output-dir "$TASK_OUT" \
        --num-trials "$NUM_TRIALS" \
        --seed "$SEED" \
        --action-horizon "$ACTION_HORIZON" \
        --camera-set "$CAMERA_SET" \
        --image-size 224 \
        --state-stats-path "$STATS" \
        --state-normalization minmax \
        --timeout "$TIMEOUT" \
        "${EXTRA[@]}" 2>&1 | tee "$TASK_OUT/eval.log"
    RC=${PIPESTATUS[0]}
    if [[ "$RC" != "0" ]]; then
        echo ">>> $(date '+%H:%M:%S') $t: client FAILED (exit $RC), see $TASK_OUT/eval.log" >&2; FAILED=$((FAILED + 1))
        if ! kill -0 $SERVER_PID 2>/dev/null; then echo "ERROR: the server died, see $OUT_DIR/server.log" >&2; exit 1; fi
    fi
done

echo ">>> $(date '+%H:%M:%S') summarizing"
# shellcheck disable=SC2086
"$ROBOCASA_PY" "$WORKDIR/cosmos_framework/simulation/robocasa/summarize_results.py" "$OUT_DIR" --tasks $TASKS
echo ">>> $(date '+%H:%M:%S') Done ($FAILED failed task(s)); summary: $OUT_DIR/summary.json"
exit $FAILED
