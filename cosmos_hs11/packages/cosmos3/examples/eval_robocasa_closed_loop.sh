#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Closed-loop RoboCasa365 target/atomic evaluation of a post-trained checkpoint (cosmos_hs11): starts the RoboCasa action
# policy server (this repo's cosmos3-pt env, 1 GPU) in the background, runs the simulator client
# (cosmos_framework/simulation/robocasa/closed_loop_eval.py in the separate robocasa-eval env, EGL on the same GPU) task
# by task with the official Atomic-Seen protocol (target object split, the 10 target kitchens, official per-task horizon,
# first-success latching), writes <OUT_DIR>/<task>/results.json (+ rollout mp4s / init pngs) and <OUT_DIR>/summary.json,
# then stops the server. Tasks that already have a results.json are skipped, so an interrupted run resumes.
#
# The client flags MUST match the training contract of action_policy_robocasa_edge (they are fixed below):
#   --action-horizon 16 (chunk_length 16) --camera-set left_wrist --use-state --use-base-action --base-encoding raw_armfirst
# (raw_armfirst = 15D [eef_pos(3), eef_rot6d(6), gripper(1), base_motion(4), control_mode(1)]; BASE_ENCODING=raw for a
# checkpoint of the Nano recipe's base-first contract)
#
# Required env vars:
#   RUN_ROOT      training run dir, e.g. $IMAGINAIRE_OUTPUT_ROOT/cosmos3_action_robocasa/edge_robocasa365_atomic/<name>
#   ITER          checkpoint iteration, e.g. 3000 (-> $RUN_ROOT/checkpoints/iter_000003000)
#   OUT_DIR       where results go, e.g. $COSMOS_STORAGE/outputs/eval_robocasa/<name>_iter3000
# Optional:
#   ROBOCASA_V21_ROOT  v2.1 datasets WITH extras/dataset_meta.json (default $COSMOS_STORAGE/data/robocasa365/v1.0/target/atomic)
#   ROBOCASA_PY        python of the simulator env (default $HOME/anaconda3/envs/robocasa-eval/bin/python)
#   SERVER_PY          python of the training env (default: current `python`, i.e. the cosmos3-pt env)
#   NUM_TRIALS         rollouts per task (default 30 = the RoboCasa365 paper's count; the leaderboard uses 50)
#   SEED               env seed -> the same test scenes for every checkpoint (default 0; unset with SEED="" for the
#                      official nondeterministic sampling)
#   TASKS              space-separated subset of the 18 tasks (default: all)
#   PORT (8900), ACTION_HORIZON (16), BASE_ENCODING (raw_armfirst), SAVE_GEN_VIDEO (0|1, default 0),
#   TIMEOUT (default 900 s per request / server wait)
# Usage (inside a 1-GPU allocation):
#   RUN_ROOT=... ITER=3000 OUT_DIR=$COSMOS_STORAGE/outputs/eval_robocasa/<name>_iter3000 sr 1 48 examples/eval_robocasa_closed_loop.sh

set -uo pipefail
: "${RUN_ROOT:?set RUN_ROOT}"; : "${ITER:?set ITER}"; : "${OUT_DIR:?set OUT_DIR}"
: "${COSMOS_STORAGE:=$HOME/project/cosmos_storage}"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOCASA_V21_ROOT="${ROBOCASA_V21_ROOT:-$COSMOS_STORAGE/data/robocasa365/v1.0/target/atomic}"
ROBOCASA_PY="${ROBOCASA_PY:-$HOME/anaconda3/envs/robocasa-eval/bin/python}"
SERVER_PY="${SERVER_PY:-python}"
NUM_TRIALS="${NUM_TRIALS:-30}"
SEED="${SEED-0}"
PORT="${PORT:-8900}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"
BASE_ENCODING="${BASE_ENCODING:-raw_armfirst}"
SAVE_GEN_VIDEO="${SAVE_GEN_VIDEO:-0}"
TIMEOUT="${TIMEOUT:-900}"
# Same guard as examples/_sft_launcher_common.sh: nodes bob / greco have a broken IB/RoCE stack and SIGSEGV inside
# ncclCommInitRank at the FIRST collective, which for this server is the 1-rank broadcast_object_list in
# OmniInference.create() (seen on bob 2026-11-01, "Fatal Python error: Segmentation fault" in distributed_c10d.broadcast).
# Everything here is single-node, so IB is never needed. Export NCCL_IB_DISABLE yourself to override.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
TASKS="${TASKS:-CloseBlenderLid CloseFridge CloseToasterOvenDoor CoffeeSetupMug NavigateKitchen OpenCabinet OpenDrawer OpenStandMixerHead PickPlaceCounterToCabinet PickPlaceCounterToStove PickPlaceDrawerToCounter PickPlaceSinkToCounter PickPlaceToasterToCounter SlideDishwasherRack TurnOffStove TurnOnElectricKettle TurnOnMicrowave TurnOnSinkFaucet}"
CKPT="$RUN_ROOT/checkpoints/iter_$(printf '%09d' "$ITER")"
TRAIN_CONFIG="$RUN_ROOT/config.yaml"
[[ -d "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }
[[ -f "$TRAIN_CONFIG" ]] || { echo "ERROR: config not found: $TRAIN_CONFIG" >&2; exit 1; }
[[ -x "$ROBOCASA_PY" ]] || { echo "ERROR: ROBOCASA_PY not found: $ROBOCASA_PY (run examples/setup_robocasa365.sh, step env)" >&2; exit 1; }
"$ROBOCASA_PY" -c "import robosuite, robocasa" 2>/dev/null || { echo "ERROR: $ROBOCASA_PY cannot import robosuite/robocasa (run examples/setup_robocasa365.sh, step env)" >&2; exit 1; }
for t in $TASKS; do
    ls -d "$ROBOCASA_V21_ROOT/$t"/*/lerobot/extras/dataset_meta.json >/dev/null 2>&1 \
        || { echo "ERROR: no $ROBOCASA_V21_ROOT/$t/<date>/lerobot/extras/dataset_meta.json (run examples/setup_robocasa365.sh, step data)" >&2; exit 1; }
done
mkdir -p "$OUT_DIR"
cd "$WORKDIR"

echo ">>> $(date '+%H:%M:%S') starting RoboCasa policy server on port $PORT for $CKPT"
PYTHONPATH=. "$SERVER_PY" -m cosmos_framework.scripts.action_policy_server_robocasa \
    --checkpoint-path "$CKPT" \
    --config-file "$TRAIN_CONFIG" \
    --fps 20 \
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

GEN_FLAG=()
[[ "$SAVE_GEN_VIDEO" == "1" ]] && GEN_FLAG=(--save-gen-video)
SEED_FLAG=()
[[ -n "$SEED" ]] && SEED_FLAG=(--seed "$SEED")

FAILED=0
for t in $TASKS; do
    DS_DIR="$(ls -d "$ROBOCASA_V21_ROOT/$t"/*/lerobot | head -1)"
    TASK_OUT="$OUT_DIR/$t"
    if [[ -f "$TASK_OUT/results.json" ]]; then
        echo ">>> $(date '+%H:%M:%S') $t: results.json exists, skipping"; continue
    fi
    mkdir -p "$TASK_OUT"
    echo ">>> $(date '+%H:%M:%S') $t: $NUM_TRIALS rollouts (dataset $DS_DIR)"
    env PYTHONPATH="$WORKDIR" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
        "$ROBOCASA_PY" "$WORKDIR/cosmos_framework/simulation/robocasa/closed_loop_eval.py" \
        --server-url "http://127.0.0.1:$PORT" \
        --dataset-dir "$DS_DIR" \
        --output-dir "$TASK_OUT" \
        --num-test-episodes "$NUM_TRIALS" \
        --action-horizon "$ACTION_HORIZON" \
        --camera-set left_wrist \
        --use-state \
        --use-base-action --base-encoding "$BASE_ENCODING" \
        --success-latch 1 --image-size 256 --cam-size 256 \
        --timeout "$TIMEOUT" \
        "${SEED_FLAG[@]}" "${GEN_FLAG[@]}" 2>&1 | tee "$TASK_OUT/eval.log"
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
