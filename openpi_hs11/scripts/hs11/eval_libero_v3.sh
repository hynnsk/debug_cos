#!/usr/bin/env bash
# Closed-loop LIBERO evaluation of a trained checkpoint: policy server (openpi env, 1 GPU) + client (libero-eval env), one allocation.
# Usage (TOML-driven, recommended):
#   TOML_FILE=scripts/hs11/toml/libero10_3ep_base_v1.toml CKPT=checkpoints/pi05_libero_v3_libero_10_3ep_base/libero10_3ep_base_v1/4999 \
#   OUT_DIR=$COSMOS_STORAGE/outputs/eval_openpi/libero10_3ep_base_v1_4999 sr 1 48 scripts/hs11/eval_libero_v3.sh [client args]
# or with a registered config name: CONFIG=pi05_libero_v3_libero_10_3ep_base CKPT=... OUT_DIR=... sr 1 48 scripts/hs11/eval_libero_v3.sh
# sanity check of the harness with openpi's official LIBERO checkpoint (env-convention gripper -> --gripper passthrough):
#   CONFIG=pi05_libero CKPT=gs://openpi-assets/checkpoints/pi05_libero TASK_SUITE=libero_goal OUT_DIR=... sr 1 48 scripts/hs11/eval_libero_v3.sh --gripper passthrough
# Env: TASK_SUITE (default = the TOML's / config's suite), NUM_TRIALS (50), PORT (8000), LIBERO_REPO (~/project/LIBERO),
#      LIBERO_PY (~/anaconda3/envs/libero-eval/bin/python), REPLAN_STEPS (5), SAVE_GIFS=1, SERVER_MEM_FRACTION (0.6),
#      GUIDANCE (1.0 = off; e.g. 3.0 = classifier-free guidance on the prompt, TOML path only), UNCOND_PROMPT (""),
#      ALLOW_CPU_RENDER=1 (accept Mesa's llvmpipe when no NVIDIA EGL device opens in the job; ~5x slower, off by default).
# Rendering: the client probes the EGL device list for the GPU of this allocation (table in client.log, "GL renderer:" line
#   is the ground truth); inspect a node with  sr 1 48 ~/anaconda3/envs/libero-eval/bin/python examples/libero_v3/egl_probe.py
# Faster: split the tasks over several allocations (each its own GPU/PORT/OUT_DIR) and merge the summaries, e.g.
#   ... OUT_DIR=$BASE/t0-4 PORT=8100 sr 1 48 scripts/hs11/eval_libero_v3.sh --task_ids 0,1,2,3,4
#   ... OUT_DIR=$BASE/t5-9 PORT=8101 sr 1 48 scripts/hs11/eval_libero_v3.sh --task_ids 5,6,7,8,9
#   python scripts/hs11/merge_eval_summaries.py --out $BASE/summary.json $BASE/t0-4/summary.json $BASE/t5-9/summary.json
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
NUM_TRIALS="${NUM_TRIALS:-50}"; PORT="${PORT:-8000}"
: "${CKPT:?set CKPT (…/<step> dir holding params/ and assets/)}"; : "${OUT_DIR:?set OUT_DIR}"
if [[ -n "${TOML_FILE:-}" ]]; then
    CONFIG="$("$PY" -c "import tomllib,sys; print(tomllib.load(open(sys.argv[1],'rb'))['job']['config'])" "$TOML_FILE")"
    SERVER_CMD=("$PY" scripts/hs11/serve_policy_toml.py --toml "$TOML_FILE" --dir "$CKPT" --port "$PORT" --guidance "${GUIDANCE:-1.0}" --uncond-prompt "${UNCOND_PROMPT:-}")
else
    : "${CONFIG:?set TOML_FILE or CONFIG}"
    SERVER_CMD=("$PY" scripts/serve_policy.py --port "$PORT" policy:checkpoint --policy.config "$CONFIG" --policy.dir "$CKPT")
    [[ "${GUIDANCE:-1.0}" == "1.0" ]] || { echo "ERROR: GUIDANCE needs the TOML path (TOML_FILE=...), the stock serve_policy.py has no CFG" >&2; exit 1; }
fi
TASK_SUITE="${TASK_SUITE:-$(echo "$CONFIG" | sed -E 's/^pi05_libero_v3_(libero_[a-z0-9]+)_.*/\1/')}"
LIBERO_REPO="${LIBERO_REPO:-$HOME/project/LIBERO}"
LIBERO_PY="${LIBERO_PY:-$HOME/anaconda3/envs/libero-eval/bin/python}"
[[ "$CKPT" == gs://* || -d "$CKPT/params" ]] || { echo "ERROR: $CKPT/params not found" >&2; exit 1; }   # gs:// checkpoints are downloaded by the server
mkdir -p "$OUT_DIR"
echo ">>> $(date '+%H:%M:%S') serving $CONFIG from $CKPT on port $PORT"
# The server and the MuJoCo/EGL client share the allocated GPU: leave JAX room for the renderer (inference needs ~8 GB).
XLA_PYTHON_CLIENT_MEM_FRACTION="${SERVER_MEM_FRACTION:-0.6}" "${SERVER_CMD[@]}" > "$OUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
# Wait for the server (loading 3.3 B params + warm-up JIT compile takes a few minutes); /healthz is served by openpi's server.
for _ in $(seq 1 360); do
    if curl -sf "http://127.0.0.1:$PORT/healthz" > /dev/null 2>&1; then break; fi
    kill -0 $SERVER_PID 2>/dev/null || { echo "ERROR: server exited, see $OUT_DIR/server.log" >&2; exit 1; }
    sleep 5
done
curl -sf "http://127.0.0.1:$PORT/healthz" > /dev/null 2>&1 || { echo "ERROR: server not healthy after 30 min, see $OUT_DIR/server.log" >&2; exit 1; }
echo ">>> $(date '+%H:%M:%S') client: $TASK_SUITE, $NUM_TRIALS trials/task"
GIF_FLAG=""; [[ "${SAVE_GIFS:-0}" == "1" ]] && GIF_FLAG="--save_gifs"
[[ "${ALLOW_CPU_RENDER:-0}" == "1" ]] && GIF_FLAG="$GIF_FLAG --allow_cpu_render"
# TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1: LIBERO loads its init-state pickles with torch.load, which torch >= 2.6 rejects by default.
PYTHONPATH="$LIBERO_REPO:$HS11_ROOT/packages/openpi-client/src" MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    "$LIBERO_PY" examples/libero_v3/main_v3.py --port "$PORT" --task_suite "$TASK_SUITE" --num_trials_per_task "$NUM_TRIALS" \
    --replan_steps "${REPLAN_STEPS:-5}" --output_dir "$OUT_DIR" $GIF_FLAG "$@" 2>&1 | tee "$OUT_DIR/client.log"
CLIENT_STATUS=${PIPESTATUS[0]}
if [[ "$CLIENT_STATUS" -ne 0 || ! -f "$OUT_DIR/summary.json" ]]; then
    echo "ERROR: client exited with status $CLIENT_STATUS (see $OUT_DIR/client.log and $OUT_DIR/server.log)" >&2
    exit 1
fi
echo ">>> done: $OUT_DIR/summary.json ($("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"SR {d['overall_success_rate']:.3f} over {d['total_episodes']} episodes\")" "$OUT_DIR/summary.json"))"
