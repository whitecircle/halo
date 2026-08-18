#!/bin/bash
# EP+TP benchmark runner.
#
# Runs SFT benchmarks with combined Expert Parallelism and Tensor Parallelism
# at different EP/TP size combinations. Requires MoE model weights and DeepEP.
#
# EP+TP mode: attention layers are sharded via TP (DTensor), experts are
# distributed via EP (DeepEP all-to-all). TP must be node-local. DP size
# is reduced by TP (world_size / tp_size).
#
# Usage:
#   ./tests/gpu/profiling/run_ep_tp_benchmarks.sh [--gpus=N] [--seq=N] [--steps=N] [--quick]
#   GPUS=8 ./tests/gpu/profiling/run_ep_tp_benchmarks.sh
set -e
cd "$(dirname "$0")/../../.."
# Benchmarks import `tests.common` at module load; torchrun puts the script dir on
# sys.path[0], not the project root, so the root must be on PYTHONPATH.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# One home for master ports (tests/common/ports.py): a fixed literal races the previous launch,
# whose rendezvous socket is still in TIME_WAIT after `cleanup`.
# Assigned on its own line at the call site: a command substitution inside an argument list does not
# trip `set -e`, so a failing allocation would hand torchrun a bare `--master_port=`.
alloc_port() { python -c 'from tests.common.ports import free_port; print(free_port())'; }

GPUS=${GPUS:-8}
SEQ=${SEQ:-4096}
STEPS=${STEPS:-10}
WARMUP=${WARMUP:-2}
QUICK=false
for arg in "$@"; do
    case $arg in
        --gpus=*) GPUS="${arg#*=}" ;;
        --seq=*) SEQ="${arg#*=}" ;;
        --steps=*) STEPS="${arg#*=}" ;;
        --warmup=*) WARMUP="${arg#*=}" ;;
        --quick) QUICK=true ;;
    esac
done

if [ "$QUICK" = true ]; then
    STEPS=5
    WARMUP=1
fi

PASSED=0
FAILED=0
RESULTS=""

cleanup() {
    # Only this repo's GPU-benchmark launches: a bare `torchrun|python` pattern reaps a co-tenant's
    # job. $$/$PPID are excluded — the runner and its launcher carry `tests/gpu` in their cmdline too.
    pgrep -f '(torchrun|python).*tests/gpu/' 2>/dev/null \
        | grep -vx -e "$$" -e "$PPID" \
        | xargs -r kill -9 2>/dev/null || true
    sleep 3
}

run_bench() {
    local name=$1
    local ep=$2
    local tp=$3
    local seq=$4

    echo ""
    echo "========================================"
    echo "Benchmark: $name"
    echo "  EP=$ep, TP=$tp, GPUs=$GPUS, seq=$seq, steps=$STEPS"
    echo "========================================"

    cleanup
    local output
    local bench_script="tests/gpu/profiling/benchmark_sft_ep_tp.py"
    local bench_args="--ep $ep --tp $tp --seq $seq --steps $STEPS --warmup $WARMUP"

    local port
    port=$(alloc_port)
    if output=$(timeout 600 torchrun --nproc_per_node=$GPUS --master_port="$port" \
        $bench_script $bench_args 2>&1); then
        echo "$output"
        local summary
        summary=$(echo "$output" | grep -E "__HALO_BENCH__|tokens/s/GPU|peak memory \(GB\)|avg step time|MFU %" || true)
        if [ -n "$summary" ]; then
            RESULTS="${RESULTS}\n--- $name ---\n${summary}\n"
        fi
        echo "PASS: $name"
        ((PASSED++)) || true
    else
        echo "$output"
        echo "FAIL: $name"
        ((FAILED++)) || true
    fi
    cleanup
}

echo "=========================================="
echo "EP+TP Benchmark Suite (GPUs: $GPUS, seq: $SEQ)"
echo "=========================================="

# EP-only baseline (all GPUs as EP, no TP)
run_bench "SFT EP=$GPUS (baseline)" "$GPUS" 1 "$SEQ"

# EP+TP=2 (TP shards attention across 2 GPUs, EP distributes experts)
if [ "$GPUS" -ge 2 ]; then
    EP_SIZE=$GPUS
    run_bench "SFT EP=$EP_SIZE TP=2" "$EP_SIZE" 2 "$SEQ"
fi

# EP+TP=4 (only if 4+ GPUs)
if [ "$GPUS" -ge 4 ]; then
    EP_SIZE=$GPUS
    run_bench "SFT EP=$EP_SIZE TP=4" "$EP_SIZE" 4 "$SEQ"
fi

# No half-EP cell: ep_size < world with ep_size > 2 forms multiple >2-rank dispatch groups whose
# combine barriers race FSDP2's DP-wide NCCL, and ParallelismConfig rejects it at config time (TP
# does not shrink ep_group_size, so ep4+tp2 on 8 lands on the same rejection). Sharding experts
# finer than the domain is ETP's job, not EP+TP's.

# Long sequence comparison (if not in quick mode)
if [ "$QUICK" = false ] && [ "$GPUS" -ge 4 ]; then
    for LONG_SEQ in 8192 16384; do
        EP_SIZE=$GPUS
        run_bench "SFT EP=$EP_SIZE TP=2 seq=$LONG_SEQ" "$EP_SIZE" 2 "$LONG_SEQ"
        cleanup
    done
fi

echo ""
echo "=========================================="
echo "EP+TP Benchmark Summary: $PASSED completed, $FAILED failed"
echo "=========================================="
if [ -n "$RESULTS" ]; then
    echo ""
    echo "=========================================="
    echo "Performance Summary"
    echo "=========================================="
    echo -e "$RESULTS"
fi
exit $FAILED
