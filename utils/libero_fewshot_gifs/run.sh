#!/usr/bin/env bash
# Export the 30 few-shot LIBERO demos of a suite (<suite>_3ep_per_task_seed42.json) as GIFs:
#   utils/<suite>_3ep_seed42_gifs/<text instruction>/ep{XXX}.gif   (3 episodes per task folder) + manifest.json
#
# Usage (from anywhere; extra args go to make_gifs.py, see --help):
#   bash utils/libero_fewshot_gifs/run.sh --suite libero_10        # default suite; both views side by side, 20 fps, native res
#   bash utils/libero_fewshot_gifs/run.sh --suite libero_goal
#   for s in libero_10 libero_goal libero_object libero_spatial; do bash utils/libero_fewshot_gifs/run.sh --suite $s; done
#   bash utils/libero_fewshot_gifs/run.sh --suite libero_object --views image          # 3rd-person camera only
#   bash utils/libero_fewshot_gifs/run.sh --suite libero_10 --stride 2 --scale 0.5 --out-dir <dir>   # 10 fps, 128 px per view
#   bash utils/libero_fewshot_gifs/run.sh --suite libero_10 --episode-subset libero_10_10ep_per_task_seed42.json --out-dir <dir>
#
# CPU only (~15 s per suite with 8 workers). Env: COSMOS_STORAGE (LeRobot dirs under data/LIBERO_LeRobot_v3/), CONDA_ENV (cosmos3-pt).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-cosmos3-pt}"

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

export COSMOS_STORAGE="${COSMOS_STORAGE:-$HOME/project/cosmos_storage}"
echo ">>> env=$CONDA_ENV  python=$(command -v python)  COSMOS_STORAGE=$COSMOS_STORAGE"
exec python "$HERE/make_gifs.py" "$@"
