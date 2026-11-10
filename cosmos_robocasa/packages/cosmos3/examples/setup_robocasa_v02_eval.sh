#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# cosmos_robocasa: ONE-TIME simulator setup for the ORIGINAL RoboCasa (v0.2) closed-loop evaluation
# (examples/eval_robocasa_closed_loop.sh). Login node (internet), CPU only.
#
# Steps (STEPS="env assets" by default; idempotent):
#   env     conda env $ROBOCASA_V02_ENV (python 3.10) with robosuite==1.5.1 + mujoco==3.2.6 (Cosmos Policy's pins) and the
#           Cosmos Policy fork of robocasa v0.2 (github.com/moojink/robocasa-cosmos-policy = v0.2 + a removed numpy assert)
#           as an editable install from $ROBOCASA_V02_REPO (cloned if missing; its setup.py pins numpy 1.23.3 / numba
#           0.56.4 for python 3.10), plus requests / imageio-ffmpeg / scipy for the client. The RoboCasa365 env
#           (robocasa-eval) is NOT reusable: v0.2 and 1.0 are different packages with different assets.
#   assets  the v0.2 kitchen assets (4 zips, 3.7 GB: textures, fixtures, objaverse objects, generative textures; the
#           official downloader skips aigen_objs and the eval never needs it) into
#           $ROBOCASA_V02_REPO/robocasa/models/assets, then setup_macros.py (writes robocasa/macros_private.py).
#           ASSET_SOURCE=box uses the fork's download_kitchen_assets.py (utexas Box links); ASSET_SOURCE=hf fetches the
#           byte-identical mirror Tej3/robocasa-v02-kitchen-assets (Hugging Face; sizes + SHA256SUMS match the Box
#           originals, verified 2026-11-10) and unpacks it the same way -- for clusters that cannot reach utexas.box.com.
#           auto (default) probes Box first and falls back to hf.
#
# Env vars (all optional):
#   ROBOCASA_V02_REPO  default $HOME/project/robocasa_cosmos_policy
#   ROBOCASA_V02_ENV   default robocasa-v02-eval
#   CONDA_BASE         default $HOME/anaconda3
#   ASSET_SOURCE       auto (default) | box | hf
#   HF_HOME            default $COSMOS_STORAGE/hf_cache (hf source only)
#
# Usage: bash examples/setup_robocasa_v02_eval.sh            # STEPS="env assets"
#        STEPS=assets bash examples/setup_robocasa_v02_eval.sh

set -euo pipefail
ROBOCASA_V02_REPO="${ROBOCASA_V02_REPO:-$HOME/project/robocasa_cosmos_policy}"
ROBOCASA_V02_ENV="${ROBOCASA_V02_ENV:-robocasa-v02-eval}"
CONDA_BASE="${CONDA_BASE:-$HOME/anaconda3}"
STEPS="${STEPS:-env assets}"
ASSET_SOURCE="${ASSET_SOURCE:-auto}"
: "${COSMOS_STORAGE:=$HOME/project/cosmos_storage}"
export HF_HOME="${HF_HOME:-$COSMOS_STORAGE/hf_cache}"
PY="$CONDA_BASE/envs/$ROBOCASA_V02_ENV/bin/python"
ASSETS_DIR="$ROBOCASA_V02_REPO/robocasa/models/assets"
HF_ASSETS_REPO="Tej3/robocasa-v02-kitchen-assets"

log() { echo ">>> $(date '+%H:%M:%S') $*"; }
has_step() { [[ " $STEPS " == *" $1 "* ]]; }

[[ -d "$ROBOCASA_V02_REPO/.git" ]] || git clone https://github.com/moojink/robocasa-cosmos-policy "$ROBOCASA_V02_REPO"

if has_step env; then
    log "[env] conda env $ROBOCASA_V02_ENV (python 3.10) + robosuite 1.5.1 + mujoco 3.2.6 + robocasa v0.2 fork"
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    [[ -x "$PY" ]] || conda create -y -c conda-forge -n "$ROBOCASA_V02_ENV" python=3.10
    conda activate "$ROBOCASA_V02_ENV"
    python -m pip install "robosuite==1.5.1" "mujoco==3.2.6"
    python -m pip install -e "$ROBOCASA_V02_REPO"
    python -m pip install requests imageio imageio-ffmpeg scipy
    python - <<'PYCHK'
import numpy, mujoco, robosuite, robocasa, scipy, requests, imageio_ffmpeg  # noqa: F401
from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS
print("numpy", numpy.__version__, "mujoco", mujoco.__version__, "robocasa tasks", len(SINGLE_STAGE_TASK_DATASETS))
PYCHK
    conda deactivate
fi

if has_step assets; then
    [[ -x "$PY" ]] || { echo "ERROR: env $ROBOCASA_V02_ENV missing (run STEPS=env first)" >&2; exit 1; }
    if [[ "$ASSET_SOURCE" == "auto" ]]; then
        # Box answers 404 to HEAD but serves ranged GETs; a reset connection means the cluster firewalls utexas.box.com.
        if "$PY" - <<'PYPROBE'
import sys, urllib.request
req = urllib.request.Request("https://utexas.box.com/shared/static/otdsyfjontk17jdp24bkhy2hgalofbh4.zip", headers={"Range": "bytes=0-0"})
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        sys.exit(0 if r.status in (200, 206) else 1)
except Exception as e:
    print("box probe failed:", type(e).__name__, e); sys.exit(1)
PYPROBE
        then ASSET_SOURCE=box; else ASSET_SOURCE=hf; fi
        log "[assets] ASSET_SOURCE=auto -> $ASSET_SOURCE"
    fi
    if [[ "$ASSET_SOURCE" == "box" ]]; then
        log "[assets] v0.2 kitchen assets (utexas Box, 4 zips / 3.7 GB) -> $ASSETS_DIR"
        # The downloader asks "Proceed? (y/n)" and, per already-present folder, "overwrite? (y/n)": answer y, then n.
        ( echo y; yes n ) | "$PY" "$ROBOCASA_V02_REPO/robocasa/scripts/download_kitchen_assets.py" || true
    elif [[ "$ASSET_SOURCE" == "hf" ]]; then
        log "[assets] v0.2 kitchen assets (Hugging Face mirror $HF_ASSETS_REPO, 4 zips / 3.7 GB) -> $ASSETS_DIR"
        HF_CLI="$(dirname "$(command -v python || echo "$CONDA_BASE/bin/python")")/hf"; command -v hf >/dev/null && HF_CLI="$(command -v hf)"
        [[ -x "$HF_CLI" ]] || HF_CLI="$CONDA_BASE/envs/cosmos3-pt/bin/hf"
        [[ -x "$HF_CLI" ]] || { echo "ERROR: hf CLI not found (pip install -U huggingface_hub in any env and put hf on PATH)" >&2; exit 1; }
        ZIPS="$ASSETS_DIR/_hf_zips"; mkdir -p "$ZIPS"
        "$HF_CLI" download "$HF_ASSETS_REPO" --repo-type dataset --include 'archives/*' SHA256SUMS assets_manifest.json --local-dir "$ZIPS"
        ( cd "$ZIPS" && sha256sum -c SHA256SUMS ) || { echo "ERROR: SHA256 mismatch in $ZIPS (re-run STEPS=assets ASSET_SOURCE=hf)" >&2; exit 1; }
        # Same layout as the official downloader: each zip holds its top-level folder and is extracted into the PARENT of
        # its target (textures -> assets/, fixtures -> assets/, objaverse -> assets/objects/, generative_textures -> assets/).
        for pair in "textures:." "fixtures:." "objaverse:objects" "generative_textures:."; do
            name="${pair%%:*}"; parent="${pair##*:}"
            target="$ASSETS_DIR/$parent/$name"; target="${target//\/.\//\/}"
            if [[ -d "$target" ]]; then log "[assets] $name already extracted, skipping"; continue; fi
            mkdir -p "$ASSETS_DIR/$parent"
            log "[assets] unzip $name -> $ASSETS_DIR/$parent"
            "$PY" -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$ZIPS/archives/$name.zip" "$ASSETS_DIR/$parent"
        done
        rm -rf "$ZIPS"
    else
        echo "ERROR: ASSET_SOURCE must be auto, box or hf (got '$ASSET_SOURCE')" >&2; exit 1
    fi
    for d in textures fixtures objects/objaverse generative_textures; do
        [[ -d "$ASSETS_DIR/$d" ]] || { echo "ERROR: asset folder missing: $ASSETS_DIR/$d (re-run STEPS=assets, maybe ASSET_SOURCE=hf)" >&2; exit 1; }
    done
    # macros_private.py (the script prompts only when the file exists: keep it).
    echo n | "$PY" "$ROBOCASA_V02_REPO/robocasa/scripts/setup_macros.py" || true
    du -sh "$ASSETS_DIR" || true
    log "[assets] smoke: build one env headless (EGL/OSMesa may be unavailable on the login node; rendering is not needed here)"
    MUJOCO_GL="${MUJOCO_GL:-egl}" "$PY" - <<'PYCHK' || log "[assets] env smoke skipped/failed (fine on a node without EGL; the GPU nodes have it)"
import robosuite, robocasa  # noqa: F401
from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS
print("registry horizons:", {k: v["horizon"] for k, v in list(SINGLE_STAGE_TASK_DATASETS.items())[:3]})
PYCHK
fi
log "done: $STEPS"
