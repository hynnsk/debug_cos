#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# cosmos_hs11: REPTILE meta-training of Cosmos3-Edge over robot embodiments (full-parameter by default;
# TOML_FILE=examples/toml/sft_config/action_reptile_meta_lora_edge.toml for the LoRA mode). See docs/action_reptile_meta.md.
# Drives cosmos_framework.scripts.train_action_reptile with
# examples/toml/sft_config/action_reptile_meta_edge.toml. See docs/action_fewshot_meta.md.
#
# Required env vars:
#   ROBOT_FEWSHOT_ROOT    parent dir of google_robot_rt1/ bridge_v2/ robomind/{ur,franka}_1rgb/ yam/repos/
#                         (default: $COSMOS_STORAGE/data/robot_fewshot when COSMOS_STORAGE is set)
# Optional env vars:
#   BASE_CHECKPOINT_PATH  Cosmos3-Edge DCP dir (default: examples/checkpoints/Cosmos3-Edge)
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   IMAGINAIRE_OUTPUT_ROOT, NPROC_PER_NODE, MASTER_PORT, EXTRA_TAIL_OVERRIDES (see _sft_launcher_common.sh)
#
# Usage (4 GPUs on one node via the site SLURM wrapper):
#   NPROC_PER_NODE=4 sr 4 48 examples/launch_reptile_meta_edge.sh
# Smoke (2 GPUs, 5 iterations, tiny episodes):
#   EXTRA_TAIL_OVERRIDES="trainer.max_iter=5 job.wandb_mode=disabled" NPROC_PER_NODE=2 sr 2 48 \
#     examples/launch_reptile_meta_edge.sh

set -uo pipefail

TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_reptile_meta_edge2.toml}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"
if [[ -z "${ROBOT_FEWSHOT_ROOT:-}" && -n "${COSMOS_STORAGE:-}" ]]; then
    ROBOT_FEWSHOT_ROOT="$COSMOS_STORAGE/data/robot_fewshot"
fi
export ROBOT_FEWSHOT_ROOT="${ROBOT_FEWSHOT_ROOT:-}"

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$TOML_FILE" = /* ]] || TOML_FILE="$WORKDIR/$TOML_FILE"
[[ "$BASE_CHECKPOINT_PATH" = /* ]] || BASE_CHECKPOINT_PATH="$WORKDIR/$BASE_CHECKPOINT_PATH"
WAN_VAE_PATH="${WAN_VAE_PATH:-examples/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
[[ "$WAN_VAE_PATH" = /* ]] || WAN_VAE_PATH="$WORKDIR/$WAN_VAE_PATH"
export BASE_CHECKPOINT_PATH WAN_VAE_PATH

echo ">>> $(date '+%H:%M:%S') Checking inputs..."
[[ -f "$TOML_FILE" ]] || { echo "ERROR: TOML not found: $TOML_FILE" >&2; exit 1; }
[[ -d "$BASE_CHECKPOINT_PATH" ]] || { echo "ERROR: BASE_CHECKPOINT_PATH not found: $BASE_CHECKPOINT_PATH" >&2; exit 1; }
[[ -f "$WAN_VAE_PATH" ]] || { echo "ERROR: WAN_VAE_PATH not found: $WAN_VAE_PATH" >&2; exit 1; }
[[ -n "$ROBOT_FEWSHOT_ROOT" && -d "$ROBOT_FEWSHOT_ROOT" ]] || { echo "ERROR: ROBOT_FEWSHOT_ROOT must point at the robot_fewshot data dir (got '$ROBOT_FEWSHOT_ROOT')" >&2; exit 1; }
for sub in google_robot_rt1 bridge_v2 robomind/ur_1rgb robomind/franka_1rgb yam/repos; do
    [[ -e "$ROBOT_FEWSHOT_ROOT/$sub" ]] || { echo "ERROR: missing $ROBOT_FEWSHOT_ROOT/$sub" >&2; exit 1; }
done
STATS="$WORKDIR/cosmos_framework/data/generator/action/normalizer_stats/molmoact2_yam_stats.json"
[[ -f "$STATS" ]] || { echo "ERROR: YAM normalization stats missing ($STATS). Run tools/compute_action_stats_from_dataset.py --embodiment molmoact2_yam" >&2; exit 1; }

OUTPUT_ROOT="${OUTPUT_ROOT:-$WORKDIR/outputs/train}"
LOG_DIR="$OUTPUT_ROOT/logs"
TOML_STEM="$(basename "$TOML_FILE" .toml)"
LOG_FILE="$LOG_DIR/${LOG_FILENAME:-${TOML_STEM}_reptile.log}"
IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}"
mkdir -p "$LOG_DIR"

TAIL_OVERRIDES=( ${EXTRA_TAIL_OVERRIDES:-} )
TRAILING_ARGS=()
if (( ${#TAIL_OVERRIDES[@]} > 0 )); then
    TRAILING_ARGS=(-- "${TAIL_OVERRIDES[@]}")
fi

# Some vram48 nodes (observed on `bob`) have a broken irdma0/RoCE stack: every rank segfaults
# inside ncclCommInitRank immediately after "NCCL INFO Initialized NET plugin IB", so the job dies
# at the first collective with SIGSEGV and no Python traceback. A single-node run never needs the
# network transport (NVLink/PCIe/SHM cover intra-node NCCL), so turn IB off unless this is a
# multi-node launch. Export NCCL_IB_DISABLE yourself to override.
if [[ -z "${NNODES:-}" || "${NNODES}" == "1" ]]; then
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
fi

# Same as _sft_launcher_common.sh: the full-theta run sits at ~38 GiB allocated / ~44 GiB reserved on a
# 45 GiB A40, and the 5+ GiB "reserved but unallocated" fragmentation is what leaves NCCL/DCP with no
# room at save time. expandable_segments lets the allocator grow segments instead of fragmenting.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Drop GPUs that fail a torch init (haring has one) and dodge a MASTER_PORT another torchrun of ours holds on
# this node; see _node_guards.sh. Ask for one GPU more than NPROC_PER_NODE on a node with a broken GPU.
source "$(dirname "${BASH_SOURCE[0]}")/_node_guards.sh"
select_healthy_gpus "${NPROC_PER_NODE:-}" || exit 1
NPROC_PER_NODE="${NPROC_PER_NODE:-${GPU_HEALTHY:-${SLURM_GPUS_ON_NODE:-4}}}"
_USABLE="${GPU_HEALTHY:-${SLURM_GPUS_ON_NODE:-}}"
if [[ -n "$_USABLE" && "$NPROC_PER_NODE" != "$_USABLE" ]]; then
    echo ">>> WARNING: NPROC_PER_NODE=$NPROC_PER_NODE but $_USABLE usable GPU(s) are allocated; the rest stay idle." >&2
fi
: "${MASTER_PORT:=50016}"
pick_free_master_port || exit 1
TORCHRUN_ARGS=(--nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT")
[[ -n "${NNODES:-}" ]]      && TORCHRUN_ARGS+=(--nnodes="$NNODES")
[[ -n "${NODE_RANK:-}" ]]   && TORCHRUN_ARGS+=(--node_rank="$NODE_RANK")
[[ -n "${MASTER_ADDR:-}" ]] && TORCHRUN_ARGS+=(--master_addr="$MASTER_ADDR")

# DataLoader workers ship ~1k tensors per meta-episode; the compute nodes' default soft nofile
# limit (1024) is too small (the trainer also switches to the file_system sharing strategy).
ulimit -n "$(ulimit -Hn)" 2>/dev/null || ulimit -n 65536 2>/dev/null || true
echo ">>> $(date '+%H:%M:%S') nofile limit: $(ulimit -n)"

cd "$WORKDIR"
echo ">>> $(date '+%H:%M:%S') WORKDIR:    $WORKDIR"
echo ">>> $(date '+%H:%M:%S') TOML:       $TOML_FILE"
echo ">>> $(date '+%H:%M:%S') data root:  $ROBOT_FEWSHOT_ROOT"
echo ">>> $(date '+%H:%M:%S') checkpoint: $BASE_CHECKPOINT_PATH"
echo ">>> $(date '+%H:%M:%S') ranks/node: $NPROC_PER_NODE"
echo ">>> $(date '+%H:%M:%S') log:        $LOG_FILE"

IMAGINAIRE_OUTPUT_ROOT="$IMAGINAIRE_OUTPUT_ROOT" PYTHONPATH=. \
    torchrun "${TORCHRUN_ARGS[@]}" -m cosmos_framework.scripts.train_action_reptile \
    --sft-toml="$TOML_FILE" \
    "${TRAILING_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo ">>> $(date '+%H:%M:%S') Done (exit $EXIT_CODE)"
exit $EXIT_CODE
