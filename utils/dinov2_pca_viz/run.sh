#!/usr/bin/env bash
# LIBERO-10 x DINOv2 ViT-B/14 PCA visualization (raw frames / patch-feature PCA / pooled REPA target), one PNG per task.
# Output: utils/dinov2_pca_viz/results/<run-name>/task{XX}_<task>.png (+ run_info.json).
#
# Usage (from anywhere; extra args are passed to visualize_dinov2_pca.py, see --help):
#   bash utils/dinov2_pca_viz/run.sh                      # CPU is enough (~1-2 min for all 10 tasks on the login node)
#   sr 1 48 utils/dinov2_pca_viz/run.sh                   # or on a GPU node (bf16 autocast, a few seconds)
#   bash utils/dinov2_pca_viz/run.sh --tasks 0,3 --window-pos random --seed 1
#   bash utils/dinov2_pca_viz/run.sh --target-grid 8 5 5  # 2 frames -> 1 pooled map
#   bash utils/dinov2_pca_viz/run.sh --pca-per-view       # separate PCA per camera view
#   bash utils/dinov2_pca_viz/run.sh --pooled-pca refit   # own PCA for the pooled row instead of the shared basis
#   bash utils/dinov2_pca_viz/run.sh --pca-scope global   # one PCA basis for all 10 tasks (colors comparable across PNGs)
#
# Env (same defaults as the cosmos_hs* launchers): COSMOS_STORAGE, HF_HOME/HF_HUB_CACHE (facebook/dinov2-base snapshot),
# LIBERO_ROOT (LeRobot v3 dir of LIBERO-10), CONDA_ENV (default cosmos3-pt), OMP_NUM_THREADS (default 8).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-cosmos3-pt}"

# --- conda env ------------------------------------------------------------------------------------------------
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        for d in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3"; do
            [[ -f "$d/etc/profile.d/conda.sh" ]] && { source "$d/etc/profile.d/conda.sh"; break; }
        done
    fi
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV"
    else
        echo "ERROR: conda not found; activate the '$CONDA_ENV' env yourself and re-run." >&2
        exit 1
    fi
fi

# --- paths ----------------------------------------------------------------------------------------------------
export COSMOS_STORAGE="${COSMOS_STORAGE:-$HOME/project/cosmos_storage}"
export HF_HOME="${HF_HOME:-$COSMOS_STORAGE/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export LIBERO_ROOT="${LIBERO_ROOT:-$COSMOS_STORAGE/data/LIBERO_LeRobot_v3/libero_10}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# GPU nodes have no internet: stay offline when the DINOv2 snapshot is already cached.
if [[ -d "$HF_HUB_CACHE/models--facebook--dinov2-base" ]]; then
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
else
    echo ">>> WARNING: facebook/dinov2-base not in $HF_HUB_CACHE; it will be downloaded (needs internet on this node)." >&2
fi
[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must contain meta/info.json (got: '$LIBERO_ROOT')." >&2; exit 1; }

echo ">>> env=$CONDA_ENV  python=$(command -v python)  LIBERO_ROOT=$LIBERO_ROOT"
echo ">>> results -> $HERE/results/"
exec python "$HERE/visualize_dinov2_pca.py" --out-dir "$HERE/results" "$@"
