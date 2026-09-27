# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Sourced by the launchers right before torchrun. Two guards against per-node problems that SLURM does not
# know about (both are no-ops on a healthy node):
#
#   select_healthy_gpus [N]   Probe every GPU SLURM gave this job (cgroup-constrained, so "every GPU" is exactly
#                             what nvidia-smi lists here) with a 1-GPU torch init in a subprocess, then export
#                             CUDA_VISIBLE_DEVICES = the first N healthy ones (all healthy ones when N is empty).
#                             Sets GPU_HEALTHY (count) and GPU_BROKEN ("idx@pci ..." of the dropped ones).
#                             Returns 1 when fewer than N GPUs are healthy.
#                             Why: node `haring` has an A40 (PCI 57:00.0) stuck in "GPU requires reset" (since
#                             2026-09-26). SLURM still allocates it, and torch's lazy CUDA init then fails or
#                             hangs on EVERY rank as soon as that device is merely visible. This SLURM setup lets
#                             nobody pick GPU indices, so on such a node ask for one GPU more than you need
#                             (`--gres=gpu:3` with NPROC_PER_NODE=2) and let this drop the bad one.
#   pick_free_master_port     If $MASTER_PORT is already bound on this host (another torchrun of ours on the
#                             same node), move to the next free port and say so. Single-node only: a multi-node
#                             launch must keep the port the other workers were given.
#
# SKIP_GPU_PROBE=1 disables the probe. The probe uses `python` from PATH (the same env torchrun comes from) and
# costs ~5 s (all GPUs are probed in parallel).

# Pin CUDA's device order so the probe and the training run agree on what "index i" is. Note that a GPU that needs
# a reset is put LAST by the CUDA driver whatever the order (verified on haring: nvidia-smi lists 56,57,CE but CUDA
# enumerates 56,CE,<broken 57>), so a CUDA index is not an nvidia-smi index there; PCI ids below therefore come
# from torch for the healthy GPUs and by elimination against nvidia-smi's list for the broken ones.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

select_healthy_gpus() {
    local want="${1:-}"
    GPU_HEALTHY="" GPU_BROKEN=""
    if [[ "${SKIP_GPU_PROBE:-0}" == "1" ]] || ! command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi
    local -a pci=()
    mapfile -t pci < <(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | tr -d ' ')
    (( ${#pci[@]} > 0 )) || return 0
    # Candidates = what SLURM handed this job. Under cgroup device constraints (this cluster) that is every GPU
    # nvidia-smi lists, renumbered 0..N-1; elsewhere SLURM only sets CUDA_VISIBLE_DEVICES to the allocated
    # physical indices, so honour that list instead of probing (and possibly grabbing) other jobs' GPUs.
    local -a cand=()
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=, read -r -a cand <<< "$CUDA_VISIBLE_DEVICES"
    else
        local j
        for (( j = 0; j < ${#pci[@]}; j++ )); do cand+=("$j"); done
    fi
    local n=${#cand[@]} i
    for i in "${cand[@]}"; do
        [[ "$i" =~ ^[0-9]+$ ]] || return 0   # UUID / MIG style lists: leave the allocation alone
    done
    local py
    py="$(command -v python)" || return 0
    local tmp
    tmp="$(mktemp -d)" || return 0
    local probe='import torch; torch.empty(1, device="cuda"); p = torch.cuda.get_device_properties(0); print(f"{p.pci_domain_id:08X}:{p.pci_bus_id:02X}:{p.pci_device_id:02X}.0")'
    for i in "${cand[@]}"; do
        (
            CUDA_VISIBLE_DEVICES=$i timeout 120 "$py" -c "$probe" >"$tmp/$i.log" 2>&1 && touch "$tmp/$i.ok"
        ) &
    done
    wait
    local -a healthy=() broken=() healthy_pci=()
    for i in "${cand[@]}"; do
        if [[ -f "$tmp/$i.ok" ]]; then
            healthy+=("$i")
            healthy_pci+=("$(tail -n 1 "$tmp/$i.log")")
        else
            broken+=("$i")
            echo ">>> CUDA device $i failed the torch probe: $(tail -n 1 "$tmp/$i.log" 2>/dev/null)" >&2
        fi
    done
    rm -rf "$tmp"
    GPU_HEALTHY=${#healthy[@]}
    GPU_BROKEN=""
    if (( ${#broken[@]} > 0 )); then
        # nvidia-smi entries no healthy probe claimed = the unusable devices (only meaningful when the job sees
        # exactly its own GPUs, i.e. the cgroup case; otherwise the leftover list would include other jobs' GPUs).
        local -a leftover=() b h
        if (( n == ${#pci[@]} )); then
            for b in "${pci[@]}"; do
                for h in "${healthy_pci[@]:-}"; do [[ "$h" == "$b" ]] && continue 2; done
                leftover+=("$b")
            done
        fi
        local IFS=,
        GPU_BROKEN="idx:${broken[*]} pci:${leftover[*]:-?}"
        echo ">>> WARNING: $(hostname -s): ${#broken[@]} of $n allocated GPU(s) are unusable and dropped from CUDA_VISIBLE_DEVICES ($GPU_BROKEN; healthy: ${healthy_pci[*]:-none})" >&2
    fi
    if (( GPU_HEALTHY == 0 )) || { [[ -n "$want" ]] && (( want > GPU_HEALTHY )); }; then
        echo "ERROR: NPROC_PER_NODE=${want:-<all>} but only $GPU_HEALTHY of the $n allocated GPU(s) on $(hostname -s) work (broken: ${GPU_BROKEN:-none})." >&2
        echo "       Re-submit with one GPU more than NPROC_PER_NODE (e.g. --gres=gpu:$((${want:-n} + 1))), or exclude the node (srun -x $(hostname -s))." >&2
        return 1
    fi
    local -a keep=("${healthy[@]}")
    [[ -n "$want" ]] && keep=("${healthy[@]:0:$want}")
    local IFS=,
    export CUDA_VISIBLE_DEVICES="${keep[*]}"
    echo ">>> $(date '+%H:%M:%S') GPUs:       CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (healthy $GPU_HEALTHY/$n on $(hostname -s))"
}

pick_free_master_port() {
    if [[ -n "${NNODES:-}" && "${NNODES}" != "1" ]]; then
        return 0
    fi
    command -v ss >/dev/null 2>&1 || return 0
    local orig="${MASTER_PORT:-50012}" port tries=0
    port=$orig
    while ss -Hltn "sport = :$port" 2>/dev/null | grep -q .; do
        port=$((port + 1))
        tries=$((tries + 1))
        if (( tries > 200 )); then
            echo "ERROR: no free rendezvous port in $orig..$port on $(hostname -s)." >&2
            return 1
        fi
    done
    if [[ "$port" != "$orig" ]]; then
        echo ">>> WARNING: MASTER_PORT=$orig is already in use on $(hostname -s) (another torchrun of ours on this node?); using $port instead." >&2
    fi
    export MASTER_PORT=$port
}
