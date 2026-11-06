#!/usr/bin/env bash

# User-defined variables
PORT=${PORT:-8701}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
EXP_NAME=${EXP_NAME:?EXP_NAME is required}
CONFIG=${CONFIG:?CONFIG is required}

# Fixed variables
MUJOCO_GL=osmesa
PYOPENGL_PLATFORM=osmesa
ALLOW_CPU_RENDER=1

LIBERO_REPO=/mnt/ufs/users/jiyoung-kim/project/backend/debug_cos/LIBERO
LIBERO_PY=/home/ad-group/miniconda3/envs/osmesa_test/bin/python

TOML_FILE="scripts/hs11/toml/${EXP_NAME}.toml"
CKPT="checkpoints/${CONFIG}/${EXP_NAME}/4999"
OUT_DIR="${COSMOS_STORAGE}/outputs/eval_openpi/${EXP_NAME}_4999"

# Run evaluation
export PORT MUJOCO_GL PYOPENGL_PLATFORM ALLOW_CPU_RENDER
export LIBERO_REPO LIBERO_PY TOML_FILE CKPT OUT_DIR
export CUDA_VISIBLE_DEVICES

NPROC_PER_NODE=1 \
NUM_TRIALS=50 \
SAVE_GIFS=1 \
scripts/hs11/eval_libero_v3.sh
