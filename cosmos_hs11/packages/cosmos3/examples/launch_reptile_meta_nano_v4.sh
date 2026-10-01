#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# cosmos_hs11: REPTILE meta-training of Cosmos3-NANO (Qwen3-VL-8B MoT) over robot embodiments -- the Nano twin of
# launch_reptile_meta_edge.sh. Drives cosmos_framework.scripts.train_action_reptile with
# examples/toml/sft_config/action_reptile_meta_nano.toml (TOML_FILE=... for the bs128 / LoRA / smoke variants).
# See docs/action_reptile_meta.md section 9.
#
# Required env vars:
#   ROBOT_FEWSHOT_ROOT    parent dir of google_robot_rt1/ bridge_v2/ robomind/{ur,franka}_1rgb/ yam/repos/
#                         (default: $COSMOS_STORAGE/data/robot_fewshot when COSMOS_STORAGE is set)
# Optional env vars:
#   BASE_CHECKPOINT_PATH  Cosmos3-Nano DCP dir (default: examples/checkpoints/Cosmos3-Nano; on this cluster
#                         $COSMOS_STORAGE/checkpoints/Cosmos3-Nano)
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   IMAGINAIRE_OUTPUT_ROOT, NPROC_PER_NODE, MASTER_PORT (default 50026), EXTRA_TAIL_OVERRIDES, SKIP_GPU_PROBE
#
# Usage (one 8 x 48 GB node; the TOML is FSDP shard 8):
#   NPROC_PER_NODE=8 sr 8 48 examples/launch_reptile_meta_nano.sh
# Smoke (2 GPUs, tiny episodes, no DCP):
#   TOML_FILE=examples/toml/sft_config/action_reptile_meta_nano_smoke.toml NPROC_PER_NODE=2 sr 2 48 \
#     examples/launch_reptile_meta_nano.sh

set -uo pipefail

TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_reptile_meta_nano_v4.toml}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"
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

NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}"
if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "$NPROC_PER_NODE" != "$SLURM_GPUS_ON_NODE" ]]; then
    echo ">>> WARNING: NPROC_PER_NODE=$NPROC_PER_NODE but SLURM allocated $SLURM_GPUS_ON_NODE GPU(s)." >&2
fi
TORCHRUN_ARGS=(--nproc_per_node="$NPROC_PER_NODE" --master_port="${MASTER_PORT:-50026}")
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
