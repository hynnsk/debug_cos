# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Shared launch plumbing for examples/launch_sft_<recipe>.sh — the
# structured-TOML / pydantic-schema flow that drives cosmos_framework.scripts.train.
#
# Caller MUST set before sourcing:
#   TOML_FILE            recipe TOML, e.g. "examples/toml/sft_config/<recipe>.toml".
#                        Absolute or repo-root-relative.
#
# Caller MAY set before sourcing (presence drives which existence checks fire):
#   DATASET_PATH         recipe-local dataset dir, e.g. "examples/data/<name>".
#                        If unset, no dataset existence check fires
#                        (reasoner / HF-streaming case).
#   BASE_CHECKPOINT_PATH recipe-local base DCP dir, e.g. "examples/checkpoints/<name>".
#                        Setting it also enables WAN_VAE_PATH plumbing + check.
#   WAN_VAE_PATH         override the default examples/checkpoints/wan22_vae/Wan2.2_VAE.pth.
#   EXTRA_DATASET_CHECK  bash snippet (string) eval'd after the default checks.
#   TAIL_OVERRIDES       bash array of Hydra CLI overrides appended after `--`
#                        (e.g. data_setting.max_tokens=16000 for VLM smokes).
#   MASTER_PORT          torchrun --master_port; default 50012.
#   NPROC_PER_NODE       torchrun --nproc_per_node; default SLURM_GPUS_ON_NODE, else 4.
#                        Warns when it disagrees with the SLURM allocation.
#   NNODES               torchrun --nnodes; multi-node only (unset = single-node).
#   NODE_RANK            torchrun --node_rank; this worker's 0-based index.
#   MASTER_ADDR          torchrun --master_addr; rank-0 host (multi-node only — it
#                        has no torchrun env fallback, so it must be passed here).
#   LOG_FILENAME         override $LOG_DIR/${LOG_FILENAME}
#                        (default <toml-stem>_sft.log).
#
# Absolute paths are passed through; relative paths are anchored to the repo
# root (the parent of this examples/ directory). Paths set in the caller's
# shell via `export DATASET_PATH=...` etc. win over the launcher's defaults
# (use the `: "${VAR:=default}"` idiom in the launcher to preserve this).

set -uo pipefail

: "${TOML_FILE:?TOML_FILE must be set before sourcing _sft_launcher_common.sh}"

# Repo root = parent of the wrapper's directory (examples/).
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd)"

# Anchor relative paths to $WORKDIR.
[[ "$TOML_FILE" = /* ]] || TOML_FILE="$WORKDIR/$TOML_FILE"

if [[ -n "${DATASET_PATH:-}" ]]; then
    [[ "$DATASET_PATH" = /* ]] || DATASET_PATH="$WORKDIR/$DATASET_PATH"
    export DATASET_PATH
fi

if [[ -n "${BASE_CHECKPOINT_PATH:-}" ]]; then
    [[ "$BASE_CHECKPOINT_PATH" = /* ]] || BASE_CHECKPOINT_PATH="$WORKDIR/$BASE_CHECKPOINT_PATH"
    WAN_VAE_PATH="${WAN_VAE_PATH:-examples/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
    [[ "$WAN_VAE_PATH" = /* ]] || WAN_VAE_PATH="$WORKDIR/$WAN_VAE_PATH"
    export BASE_CHECKPOINT_PATH WAN_VAE_PATH
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$WORKDIR/outputs/train}"
LOG_DIR="$OUTPUT_ROOT/logs"
TOML_STEM="$(basename "$TOML_FILE" .toml)"
LOG_FILE="$LOG_DIR/${LOG_FILENAME:-${TOML_STEM}_sft.log}"
IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}"
mkdir -p "$LOG_DIR"

echo ">>> $(date '+%H:%M:%S') Checking inputs..."
[[ -f "$TOML_FILE" ]] || { echo "ERROR: TOML not found: $TOML_FILE" >&2; exit 1; }
if [[ -n "${DATASET_PATH:-}" ]]; then
    [[ -d "$DATASET_PATH" ]] || { echo "ERROR: DATASET_PATH not found: $DATASET_PATH (run Step 1 of docs/training.md, or export DATASET_PATH=<path>)" >&2; exit 1; }
fi
if [[ -n "${BASE_CHECKPOINT_PATH:-}" ]]; then
    [[ -d "$BASE_CHECKPOINT_PATH" ]] || { echo "ERROR: BASE_CHECKPOINT_PATH not found: $BASE_CHECKPOINT_PATH (run Step 2 of docs/training.md, or export BASE_CHECKPOINT_PATH=<path>)" >&2; exit 1; }
    [[ -f "$WAN_VAE_PATH" ]]         || { echo "ERROR: WAN_VAE_PATH not found: $WAN_VAE_PATH (run Step 1 of docs/training.md, or export WAN_VAE_PATH=<path>)" >&2; exit 1; }
fi
if [[ -n "${EXTRA_DATASET_CHECK:-}" ]]; then eval "$EXTRA_DATASET_CHECK"; fi

cd "$WORKDIR"
echo ">>> $(date '+%H:%M:%S') WORKDIR:    $WORKDIR"
echo ">>> $(date '+%H:%M:%S') TOML:       $TOML_FILE"
[[ -n "${DATASET_PATH:-}" ]]         && echo ">>> $(date '+%H:%M:%S') dataset:    $DATASET_PATH"
[[ -n "${BASE_CHECKPOINT_PATH:-}" ]] && echo ">>> $(date '+%H:%M:%S') checkpoint: $BASE_CHECKPOINT_PATH"
echo ">>> $(date '+%H:%M:%S') log:        $LOG_FILE"

# Default empty if caller didn't set; safe under set -u.
[[ ${TAIL_OVERRIDES+x} ]] || TAIL_OVERRIDES=()

TRAILING_ARGS=()
if (( ${#TAIL_OVERRIDES[@]} > 0 )); then
    TRAILING_ARGS=(-- "${TAIL_OVERRIDES[@]}")
fi

# Ranks per node. An explicit NPROC_PER_NODE still wins, but the fallback follows the
# SLURM allocation so `srun --gres=gpu:N <wrapper>` uses all N GPUs without the caller
# having to restate N. A stale exported value silently leaves GPUs idle (a 2 on a
# 4-GPU allocation halves the global batch and never fails), so say so loudly instead.
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "$NPROC_PER_NODE" != "$SLURM_GPUS_ON_NODE" ]]; then
    echo ">>> WARNING: NPROC_PER_NODE=$NPROC_PER_NODE but SLURM allocated $SLURM_GPUS_ON_NODE GPU(s) on $(hostname -s). Unset NPROC_PER_NODE to use the whole allocation." >&2
fi
echo ">>> $(date '+%H:%M:%S') ranks/node: $NPROC_PER_NODE${SLURM_GPUS_ON_NODE:+ (SLURM gres: $SLURM_GPUS_ON_NODE)}"

# torchrun topology. Single-node by default; a SLURM/Lepton wrapper sets NNODES /
# NODE_RANK / MASTER_ADDR for multi-node. Each is appended only when set, so with all
# three unset the invocation is identical to the single-node case.
# Some vram48 nodes (observed on `bob`) have a broken irdma0/RoCE stack: every rank segfaults
# inside ncclCommInitRank immediately after "NCCL INFO Initialized NET plugin IB", so the job dies
# at the first collective with SIGSEGV and no Python traceback. A single-node run never needs the
# network transport (NVLink/PCIe/SHM cover intra-node NCCL), so turn IB off unless this is a
# multi-node launch. Export NCCL_IB_DISABLE yourself to override.
if [[ -z "${NNODES:-}" || "${NNODES}" == "1" ]]; then
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
fi

# Caching-allocator fragmentation is what tips near-capacity runs over (OOM at 39.6 GiB allocated + 3 GiB
# "reserved but unallocated" right after a validation pass). expandable_segments lets the allocator grow
# segments instead of fragmenting; export your own value to override.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Guard: the TOML's data_parallel_shard_degree vs the number of ranks. The framework treats
# data_parallel_replicate_degree=1 as "auto" and fills replicate = NPROC / shard, so launching a
# shard-2 TOML on 4 GPUs silently becomes HSDP 2x2 with DOUBLE the global batch (bs/rank x NPROC).
# Strip the inline TOML comment BEFORE extracting the value: the greedy `.*=` otherwise latches onto the last
# `=<digits>` of the line, e.g. the `NPROC_PER_NODE=4` inside the comment of the metainit TOML, and reports a
# spurious "shard 4 x replicate 0" warning while the framework (which parses the TOML properly) runs shard 2.
_SHARD="$(grep -E '^\s*data_parallel_shard_degree\s*=' "$TOML_FILE" | head -1 | sed -E 's/#.*//; s/.*=\s*([0-9]+).*/\1/')"
_BS="$(grep -E '^\s*max_samples_per_batch\s*=' "$TOML_FILE" | head -1 | sed -E 's/#.*//; s/.*=\s*([0-9]+).*/\1/')"
if [[ -n "$_SHARD" && "$_SHARD" != "$NPROC_PER_NODE" ]]; then
    echo ">>> WARNING: TOML data_parallel_shard_degree=$_SHARD but NPROC_PER_NODE=$NPROC_PER_NODE -> the framework will run" >&2
    echo ">>>          HSDP shard $_SHARD x replicate $((NPROC_PER_NODE / _SHARD)); global batch = ${_BS:-?} x $NPROC_PER_NODE ranks" >&2
    echo ">>>          (not ${_BS:-?} x $_SHARD). Set NPROC_PER_NODE=$_SHARD, or change the TOML, if that is not intended." >&2
fi

TORCHRUN_ARGS=(--nproc_per_node="$NPROC_PER_NODE" --master_port="${MASTER_PORT:-50012}")
[[ -n "${NNODES:-}" ]]      && TORCHRUN_ARGS+=(--nnodes="$NNODES")
[[ -n "${NODE_RANK:-}" ]]   && TORCHRUN_ARGS+=(--node_rank="$NODE_RANK")
[[ -n "${MASTER_ADDR:-}" ]] && TORCHRUN_ARGS+=(--master_addr="$MASTER_ADDR")

IMAGINAIRE_OUTPUT_ROOT="$IMAGINAIRE_OUTPUT_ROOT" PYTHONPATH=. \
    torchrun "${TORCHRUN_ARGS[@]}" -m cosmos_framework.scripts.train \
    --sft-toml="$TOML_FILE" \
    "${TRAILING_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo ">>> $(date '+%H:%M:%S') Done (exit $EXIT_CODE)"
exit $EXIT_CODE
