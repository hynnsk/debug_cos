#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# cosmos_robocasa: ONE-TIME data setup for the ORIGINAL RoboCasa post-training recipes
# (examples/toml/sft_config/action_policy_robocasa_edge_{baseinit,reptileinit}.toml). CPU only; run it on a node WITH
# internet (the login node) -- the GPU nodes have none.
#
# Steps (STEPS="download convert stats" by default; any subset, idempotent / resumable, run in this order):
#   download  the Cosmos Policy RoboCasa export nvidia/RoboCasa-Cosmos-Policy (Hugging Face, CC BY 4.0):
#             success_only/kitchen_*/<Task>/<date>/*.hdf5 = 24 atomic tasks x ~50 human demos, replayed at 224x224 with
#             no-op filtering + success trimming (~144 GB raw uint8) + its dataset_statistics*.json + README.md
#             -> $ROBOCASA_HF_DIR. The 25 GB reason1 / 0.5 GB T5 embedding pickles and all_episodes/ (world-model /
#             value rollouts incl. failures) are NOT needed by Cosmos3 and are skipped.
#   convert   HDF5 -> LeRobot v3 shards $ROBOCASA_ROOT/<Task>/<date>/lerobot (AV1 mp4 per camera, parquet, the RoboCasa365
#             12D action / 16D state column contract; JOBS tasks in parallel) via
#             cosmos_framework/scripts/convert_robocasa_cosmos_policy_to_lerobot_v30.py (cosmos3-pt env: lerobot 0.4.x).
#   stats     q01/q99/mean/std/min/max of the reader's 10D actions and 10D state token over every converted frame ->
#             cosmos_framework/data/generator/action/normalizer_stats/robocasa_cosmos_policy_frame_wise_relative_rot6d.json
#             (read by action_normalization="quantile_rot" / state_normalization="minmax"; commit it with the recipes).
#
# Env vars (all optional):
#   COSMOS_STORAGE   default $HOME/project/cosmos_storage
#   ROBOCASA_HF_DIR  default $COSMOS_STORAGE/data/robocasa_cosmos_policy           (raw HF layout)
#   ROBOCASA_ROOT    default $COSMOS_STORAGE/data/robocasa_cosmos_policy_lerobot_v3 (= ROBOCASA_ROOT for training)
#   TRAIN_PY         default $HOME/anaconda3/envs/cosmos3-pt/bin/python            (hf CLI + lerobot live next to it)
#   JOBS             default 6   parallel task conversions (each ~1 CPU core for AV1 encoding + a few GB RAM)
#   TASKS            default: the 24 tasks (DEFAULT_ROBOCASA_COSMOS_POLICY_TASKS); space-separated subset to restrict
#   HF_HOME          default $COSMOS_STORAGE/hf_cache
#
# Usage:
#   bash examples/setup_robocasa.sh                       # everything
#   STEPS="convert stats" bash examples/setup_robocasa.sh # after a manual download

set -euo pipefail

: "${COSMOS_STORAGE:=$HOME/project/cosmos_storage}"
ROBOCASA_HF_DIR="${ROBOCASA_HF_DIR:-$COSMOS_STORAGE/data/robocasa_cosmos_policy}"
ROBOCASA_ROOT="${ROBOCASA_ROOT:-$COSMOS_STORAGE/data/robocasa_cosmos_policy_lerobot_v3}"
TRAIN_PY="${TRAIN_PY:-$HOME/anaconda3/envs/cosmos3-pt/bin/python}"
JOBS="${JOBS:-6}"
STEPS="${STEPS:-download convert stats}"
TASKS="${TASKS:-}"
export HF_HOME="${HF_HOME:-$COSMOS_STORAGE/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
HF_REPO="nvidia/RoboCasa-Cosmos-Policy"

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATS_OUT="$WORKDIR/cosmos_framework/data/generator/action/normalizer_stats/robocasa_cosmos_policy_frame_wise_relative_rot6d.json"
HF_CLI="$(dirname "$TRAIN_PY")/hf"

log() { echo ">>> $(date '+%H:%M:%S') $*"; }
has_step() { [[ " $STEPS " == *" $1 "* ]]; }

[[ -x "$TRAIN_PY" ]] || { echo "ERROR: TRAIN_PY not found: $TRAIN_PY (the cosmos3-pt env python)" >&2; exit 1; }

# ------------------------------------------------------------------------------------------------ download
if has_step download; then
    [[ -x "$HF_CLI" ]] || { echo "ERROR: hf CLI not found next to TRAIN_PY: $HF_CLI (pip install -U huggingface_hub)" >&2; exit 1; }
    log "[download] $HF_REPO success_only/kitchen_* (~144 GB) -> $ROBOCASA_HF_DIR"
    mkdir -p "$ROBOCASA_HF_DIR"
    "$HF_CLI" download "$HF_REPO" --repo-type dataset \
        --include 'success_only/kitchen_*/*' 'success_only/*.json' 'README.md' \
        --local-dir "$ROBOCASA_HF_DIR" --max-workers 8
    _n="$(find "$ROBOCASA_HF_DIR/success_only" -name '*.hdf5' -not -path '*/.cache/*' | wc -l)"
    log "[download] $_n HDF5 files under $ROBOCASA_HF_DIR/success_only ($(du -sh "$ROBOCASA_HF_DIR" | cut -f1))"
    (( _n >= 24 )) || { echo "ERROR: expected 24 task HDF5 files, found $_n; re-run STEPS=download" >&2; exit 1; }
fi

# ------------------------------------------------------------------------------------------------ convert
if has_step convert; then
    log "[convert] HDF5 -> LeRobot v3 shards -> $ROBOCASA_ROOT (jobs=$JOBS, python: $TRAIN_PY)"
    cd "$WORKDIR"
    # shellcheck disable=SC2086
    PYTHONPATH=. "$TRAIN_PY" -m cosmos_framework.scripts.convert_robocasa_cosmos_policy_to_lerobot_v30 \
        --src "$ROBOCASA_HF_DIR/success_only" --dst "$ROBOCASA_ROOT" --jobs "$JOBS" ${TASKS:+--tasks $TASKS}
    du -sh "$ROBOCASA_ROOT" || true
fi

# ------------------------------------------------------------------------------------------------ stats
if has_step stats; then
    log "[stats] aggregate per-shard arrays -> $STATS_OUT"
    cd "$WORKDIR"
    # shellcheck disable=SC2086
    PYTHONPATH=. "$TRAIN_PY" -m cosmos_framework.scripts.convert_robocasa_cosmos_policy_to_lerobot_v30 \
        --src "$ROBOCASA_HF_DIR/success_only" --dst "$ROBOCASA_ROOT" --stats-only --stats-out "$STATS_OUT" ${TASKS:+--tasks $TASKS}
    log "[stats] done. Training: export ROBOCASA_ROOT=$ROBOCASA_ROOT"
fi

log "all requested steps done: $STEPS"
