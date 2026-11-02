#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# cosmos_hs11: ONE-TIME RoboCasa365 setup for the Cosmos3-Edge post-training + closed-loop evaluation recipes
# (examples/toml/sft_config/action_policy_robocasa_edge_*.toml, examples/eval_robocasa_closed_loop.sh).
# Run it on a node WITH internet (the GPU nodes have none): the login node is fine, everything is CPU-only.
#
# Steps (STEPS="env assets data convert" by default; pass any subset, they are idempotent / resumable and run in this order):
#   env      conda env $ROBOCASA_ENV (python 3.11) with robosuite (ARISE master, 1.5.2) + robocasa (RoboCasa365 1.0.1) as
#            editable installs from $ROBOSUITE_REPO / $ROBOCASA_REPO (cloned if missing), plus requests + imageio-ffmpeg
#            for the eval client. Writes robocasa/macros_private.py with DATASET_BASE_PATH=$ROBOCASA_DATA so robocasa's
#            own registry code (horizons, dataset paths) resolves under $ROBOCASA_DATA/v1.0/... .
#            NOTE: robocasa pins numpy 2.2.5 / mujoco 3.3.1 / lerobot 0.3.3; cosmos-framework pins lerobot 0.4.x, which is
#            why the simulator lives in its own env (the eval runs two processes: policy server in the cosmos3-pt env,
#            simulator client in this env). Only this step needs the env; assets/data/convert do not.
#   assets   kitchen assets (zips from the official Box links, via cosmos_framework/scripts/download_robocasa365.py) into
#            $ROBOCASA_REPO/robocasa/models/assets. ASSET_TYPES="all" (default: tex tex_generative fixtures_lw
#            objs_objaverse objs_aigen objs_lw). The target demos use the default ("objaverse", "lightwheel") object
#            registries, so objs_objaverse (the big one) must stay in.
#   data     the 18 target/atomic human datasets (LeRobot v2.1 export + extras/dataset_meta.json, 500 demos/task) from the
#            official Box links -> $ROBOCASA_DATA/v1.0/target/atomic/<Task>/<date>/lerobot
#   convert  v2.1 -> v3.0 COPIES for the cosmos-framework reader (lerobot 0.4.x rejects v2.1) ->
#            $ROBOCASA_DATA/v30/target/atomic/<Task>/<date>/lerobot  (= ROBOCASA_ROOT for training).
#            Uses cosmos_framework/scripts/convert_robocasa_to_lerobot_v30.sh with the TRAIN env's python
#            (default $HOME/anaconda3/envs/cosmos3-pt/bin/python). Eval keeps reading the v2.1 dirs.
#
# Env vars (all optional):
#   COSMOS_STORAGE   default $HOME/project/cosmos_storage
#   ROBOCASA_DATA    default $COSMOS_STORAGE/data/robocasa365
#   ROBOCASA_REPO    default $HOME/project/robocasa      (git clone https://github.com/robocasa/robocasa)
#   ROBOSUITE_REPO   default $HOME/project/robosuite     (git clone https://github.com/ARISE-Initiative/robosuite)
#   ROBOCASA_ENV     default robocasa-eval               (conda env name)
#   CONDA_BASE       default $HOME/anaconda3
#   TRAIN_PY         default $HOME/anaconda3/envs/cosmos3-pt/bin/python  (steps assets/data/convert; any python >= 3.12)
#   ASSET_TYPES      default all
#   TASKS            default: the 18 target/atomic tasks (DEFAULT_ALL_ATOMIC_TASKS of robocasa_lerobot_dataset.py)
#   DOWNLOAD_SOURCE  auto (default) | box | hf -- where steps assets/data fetch from. Some clusters reset every TCP
#                    connection to utexas.box.com ("[Errno 104] Connection reset by peer" on every archive); "auto"
#                    then falls back to the verified Hugging Face mirrors (twilighted/Robocasa365-Assets,
#                    huiwon/robocasa365, see download_robocasa365.py), "hf" skips Box altogether.
#
# Usage:
#   bash examples/setup_robocasa365.sh                 # everything
#   STEPS="data convert" bash examples/setup_robocasa365.sh

set -euo pipefail

: "${COSMOS_STORAGE:=$HOME/project/cosmos_storage}"
ROBOCASA_DATA="${ROBOCASA_DATA:-$COSMOS_STORAGE/data/robocasa365}"
ROBOCASA_REPO="${ROBOCASA_REPO:-$HOME/project/robocasa}"
ROBOSUITE_REPO="${ROBOSUITE_REPO:-$HOME/project/robosuite}"
ROBOCASA_ENV="${ROBOCASA_ENV:-robocasa-eval}"
CONDA_BASE="${CONDA_BASE:-$HOME/anaconda3}"
TRAIN_PY="${TRAIN_PY:-$HOME/anaconda3/envs/cosmos3-pt/bin/python}"
ASSET_TYPES="${ASSET_TYPES:-all}"
STEPS="${STEPS:-env assets data convert}"
DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE:-auto}"
TASKS="${TASKS:-CloseBlenderLid CloseFridge CloseToasterOvenDoor CoffeeSetupMug NavigateKitchen OpenCabinet OpenDrawer OpenStandMixerHead PickPlaceCounterToCabinet PickPlaceCounterToStove PickPlaceDrawerToCounter PickPlaceSinkToCounter PickPlaceToasterToCounter SlideDishwasherRack TurnOffStove TurnOnElectricKettle TurnOnMicrowave TurnOnSinkFaucet}"

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOCASA_PY="$CONDA_BASE/envs/$ROBOCASA_ENV/bin/python"
DOWNLOADER="$WORKDIR/cosmos_framework/scripts/download_robocasa365.py"

log() { echo ">>> $(date '+%H:%M:%S') $*"; }
has_step() { [[ " $STEPS " == *" $1 "* ]]; }

[[ -x "$TRAIN_PY" ]] || { echo "ERROR: TRAIN_PY not found: $TRAIN_PY (the cosmos3-pt env python)" >&2; exit 1; }
[[ -d "$ROBOCASA_REPO/.git" ]] || git clone https://github.com/robocasa/robocasa "$ROBOCASA_REPO"
BOX_LINKS_DIR="$ROBOCASA_REPO/robocasa/models/assets/box_links"

# ------------------------------------------------------------------------------------------------ env
if has_step env; then
    log "[env] conda env $ROBOCASA_ENV + robosuite/robocasa editable installs"
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    [[ -d "$ROBOSUITE_REPO/.git" ]] || git clone https://github.com/ARISE-Initiative/robosuite "$ROBOSUITE_REPO"
    if [[ ! -x "$ROBOCASA_PY" ]]; then
        conda create -y -c conda-forge -n "$ROBOCASA_ENV" python=3.11
    fi
    conda activate "$ROBOCASA_ENV"
    python -V
    python -m pip install -e "$ROBOSUITE_REPO"
    python -m pip install -e "$ROBOCASA_REPO"
    python -m pip install requests imageio-ffmpeg
    # Private macros: DATASET_BASE_PATH tells robocasa's registry code where the datasets live. The shipped macros.py
    # ends with a star-import of macros_private itself, so the copy must drop that block.
    MACROS_PRIVATE="$ROBOCASA_REPO/robocasa/macros_private.py"
    python - "$ROBOCASA_REPO/robocasa/macros.py" "$MACROS_PRIVATE" "$ROBOCASA_DATA" <<'PY'
import re, sys
src, dst, data = sys.argv[1:4]
s = open(dst).read() if __import__("os").path.exists(dst) else open(src).read()
s = re.sub(r"\ntry:\n    from robocasa\.macros_private import \*.*", "\n", s, flags=re.S)
s = re.sub(r"^DATASET_BASE_PATH = .*$", f'DATASET_BASE_PATH = "{data}"', s, flags=re.M)
open(dst, "w").write(s)
print("wrote", dst, "with DATASET_BASE_PATH =", data)
PY
    python -c "import robosuite, robocasa, mujoco, requests, imageio_ffmpeg; from robocasa.macros import DATASET_BASE_PATH; print('robosuite', robosuite.__version__, 'mujoco', mujoco.__version__, 'DATASET_BASE_PATH', DATASET_BASE_PATH)"
    conda deactivate
fi

# ------------------------------------------------------------------------------------------------ assets
if has_step assets; then
    log "[assets] kitchen assets ($ASSET_TYPES, source $DOWNLOAD_SOURCE) -> $ROBOCASA_REPO/robocasa/models/assets"
    # shellcheck disable=SC2086
    "$TRAIN_PY" "$DOWNLOADER" assets --box-links "$BOX_LINKS_DIR/box_links_assets.json" \
        --assets-dir "$ROBOCASA_REPO/robocasa/models/assets" --types $ASSET_TYPES --source "$DOWNLOAD_SOURCE"
    du -sh "$ROBOCASA_REPO/robocasa/models/assets" || true
fi

# ------------------------------------------------------------------------------------------------ data
if has_step data; then
    log "[data] target/atomic human demos (18 tasks x 500, source $DOWNLOAD_SOURCE) -> $ROBOCASA_DATA/v1.0/target/atomic"
    mkdir -p "$ROBOCASA_DATA/v1.0"
    # shellcheck disable=SC2086
    "$TRAIN_PY" "$DOWNLOADER" datasets --box-links "$BOX_LINKS_DIR/box_links_ds.json" \
        --dest "$ROBOCASA_DATA/v1.0" --split target --kind atomic --tasks $TASKS --source "$DOWNLOAD_SOURCE"
    log "[data] verify"
    missing=0
    for t in $TASKS; do
        n="$(ls -d "$ROBOCASA_DATA"/v1.0/target/atomic/"$t"/*/lerobot 2>/dev/null | head -1 || true)"
        if [[ -n "$n" && -f "$n/meta/info.json" && -f "$n/extras/dataset_meta.json" ]]; then
            printf '  OK   %-28s %s\n' "$t" "$("$TRAIN_PY" -c "import json,sys; d=json.load(open(sys.argv[1]+'/meta/info.json')); print(d['total_episodes'],'episodes',d['total_frames'],'frames',d['fps'],'fps', d.get('codebase_version'))" "$n")"
        else
            echo "  MISS $t (needs <date>/lerobot/meta/info.json + extras/dataset_meta.json)"; missing=$((missing + 1))
        fi
    done
    (( missing == 0 )) || { echo "ERROR: $missing task(s) missing; re-run STEPS=data" >&2; exit 1; }
    du -sh "$ROBOCASA_DATA/v1.0" || true
fi

# ------------------------------------------------------------------------------------------------ convert
if has_step convert; then
    log "[convert] v2.1 -> v3.0 copies -> $ROBOCASA_DATA/v30/target/atomic (python: $TRAIN_PY)"
    cd "$WORKDIR"
    PATH="$(dirname "$TRAIN_PY"):$PATH" \
    SRC_ROOT="$ROBOCASA_DATA/v1.0/target/atomic" V30_ROOT="$ROBOCASA_DATA/v30/target/atomic" TASKS_OVERRIDE="$TASKS" \
        bash cosmos_framework/scripts/convert_robocasa_to_lerobot_v30.sh
    du -sh "$ROBOCASA_DATA/v30" || true
    log "[convert] done. Training: export ROBOCASA_ROOT=$ROBOCASA_DATA/v30/target/atomic ; eval: ROBOCASA_V21_ROOT=$ROBOCASA_DATA/v1.0/target/atomic"
fi

log "all requested steps done: $STEPS"
