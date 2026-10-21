#!/usr/bin/env bash
# Shared environment for the cosmos_hs11 openpi launchers. Source this file; it is safe under `set -u`.
HS11_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HS11_ROOT
export PATH="$HOME/.local/bin:$PATH"                     # uv
export COSMOS_STORAGE="${COSMOS_STORAGE:-$HOME/project/cosmos_storage}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$COSMOS_STORAGE/openpi_cache}"   # gs://openpi-assets cache (pi05_base)
export LIBERO_V3_ROOT="${LIBERO_V3_ROOT:-$COSMOS_STORAGE/data/LIBERO_LeRobot_v3}"
export ROBOT_FEWSHOT_ROOT="${ROBOT_FEWSHOT_ROOT:-$COSMOS_STORAGE/data/robot_fewshot}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$COSMOS_STORAGE/uv_cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$COSMOS_STORAGE/uv_python}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
# openpi's data loader spawns workers that import jax; keep them off the GPU.
export JAX_TRACEBACK_FILTERING=off
PY="$HS11_ROOT/.venv/bin/python"
[[ -x "$PY" ]] || { echo "ERROR: $PY not found; run: cd $HS11_ROOT && GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11" >&2; exit 1; }
cd "$HS11_ROOT"
