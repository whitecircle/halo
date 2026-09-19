#!/bin/bash
# MFU sweep at 4k/8k/16k sequence lengths across the SFT and SMPO benchmarks, extracting the headline
# summary lines plus the machine-readable __HALO_BENCH__ JSON sentinel from each run.
#   ./tests/gpu/profiling/run_mfu_benchmarks.sh [--gpus=N] [--ep=N] [--steps=N]
#   GPUS=8 EP=8 ./tests/gpu/profiling/run_mfu_benchmarks.sh
set -e
cd "$(dirname "$0")/../../.."
# Benchmarks import `tests.common` at module load; torchrun puts the script dir on
# sys.path[0], not the project root, so the root must be on PYTHONPATH.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# One home for master ports (tests/common/ports.py): a fixed literal races the previous launch, whose
# rendezvous socket is still in TIME_WAIT after `cleanup`. Assigned on its own line at every call
# site — a command substitution inside an argument list does not trip `set -e`.
alloc_port() { python -c 'from tests.common.ports import free_port; print(free_port())'; }

GPUS=${GPUS:-2}
EP=${EP:-2}
STEPS=${STEPS:-10}
WARMUP=${WARMUP:-2}
for arg in "$@"; do
    case $arg in
        --gpus=*) GPUS="${arg#*=}" ;;
        --ep=*) EP="${arg#*=}" ;;
        --steps=*) STEPS="${arg#*=}" ;;
        --warmup=*) WARMUP="${arg#*=}" ;;
    esac
done

cleanup() {
    # Only this repo's GPU-benchmark launches: a bare `torchrun|python` pattern reaps a co-tenant's
    # job. $$/$PPID are excluded — the runner and its launcher carry `tests/gpu` in their cmdline too.
    pgrep -f '(torchrun|python).*tests/gpu/' 2>/dev/null \
        | grep -vx -e "$$" -e "$PPID" \
        | xargs -r kill -9 2>/dev/null || true
    sleep 3
}

echo "=========================================="
echo "MFU Benchmark Sweep"
echo "  GPUs: $GPUS, EP: $EP, Steps: $STEPS"
echo "  Sequence lengths: 4096, 8192, 16384"
echo "=========================================="

for SEQ in 4096 8192 16384; do
    echo ""
    echo "########## seq=$SEQ ##########"

    echo "=== SFT EP=$EP seq=$SEQ ==="
    cleanup
    port=$(alloc_port)
    timeout 300 torchrun --nproc_per_node=$GPUS --master_port="$port" \
        tests/gpu/profiling/benchmark_sft_ep.py \
        --ep $EP --seq $SEQ --steps $STEPS --warmup $WARMUP 2>&1 | \
        grep -E "__HALO_BENCH__|tokens/s/GPU|peak memory \(GB\)|avg step time|MFU %" || echo "FAILED"
    cleanup

    echo "=== SFT EP=$EP+CP=$EP seq=$SEQ ==="
    cleanup
    port=$(alloc_port)
    timeout 300 torchrun --nproc_per_node=$GPUS --master_port="$port" \
        tests/gpu/profiling/benchmark_sft_ep_cp.py \
        --ep $EP --cp $EP --seq $SEQ --steps $STEPS --warmup $WARMUP 2>&1 | \
        grep -E "__HALO_BENCH__|tokens/s/GPU|peak memory \(GB\)|avg step time|MFU %" || echo "FAILED"
    cleanup

    echo "=== SMPO EP=$EP seq=$SEQ ==="
    cleanup
    port=$(alloc_port)
    timeout 300 torchrun --nproc_per_node=$GPUS --master_port="$port" \
        tests/gpu/profiling/benchmark_smpo_ep.py \
        --ep $EP --seq $SEQ --steps $STEPS --warmup $WARMUP 2>&1 | \
        grep -E "__HALO_BENCH__|tokens/s/GPU|peak memory \(GB\)|avg step time|MFU %" || echo "FAILED"
    cleanup

    echo "=== SMPO EP=$EP+CP=$EP seq=$SEQ ==="
    cleanup
    port=$(alloc_port)
    timeout 300 torchrun --nproc_per_node=$GPUS --master_port="$port" \
        tests/gpu/profiling/benchmark_smpo_ep_cp.py \
        --ep $EP --cp $EP --seq $SEQ --steps $STEPS --warmup $WARMUP 2>&1 | \
        grep -E "__HALO_BENCH__|tokens/s/GPU|peak memory \(GB\)|avg step time|MFU %" || echo "FAILED"
    cleanup

    echo "=== SFT Dense (FSDP) seq=$SEQ ==="
    cleanup
    port=$(alloc_port)
    timeout 300 torchrun --nproc_per_node=$GPUS --master_port="$port" \
        tests/gpu/profiling/benchmark_sft_dense.py \
        --seq $SEQ --steps $STEPS --warmup $WARMUP 2>&1 | \
        grep -E "__HALO_BENCH__|tokens/s/GPU|peak memory \(GB\)|avg step time|MFU %" || echo "FAILED"
    cleanup
done

echo ""
echo "=========================================="
echo "MFU sweep complete."
echo "=========================================="
