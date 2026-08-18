#!/bin/bash
# Master runner for all MFU/TFLOPS benchmarks.
#
# Runs each benchmark at seq=4096 with default steps (10) and warmup (2).
# Collects the headline summary lines (tokens/s/GPU, peak memory, step time) plus the
# machine-readable __HALO_BENCH__ JSON sentinel from each run.
#
# Usage:
#   ./tests/gpu/profiling/run_all_benchmarks.sh [--gpus=N] [--ep=N] [--seq=N] [--steps=N]
#   GPUS=8 EP=8 ./tests/gpu/profiling/run_all_benchmarks.sh
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

GPUS=${GPUS:-2}
EP=${EP:-2}
SEQ=${SEQ:-4096}
STEPS=${STEPS:-10}
WARMUP=${WARMUP:-2}
for arg in "$@"; do
    case $arg in
        --gpus=*) GPUS="${arg#*=}" ;;
        --ep=*) EP="${arg#*=}" ;;
        --seq=*) SEQ="${arg#*=}" ;;
        --steps=*) STEPS="${arg#*=}" ;;
        --warmup=*) WARMUP="${arg#*=}" ;;
    esac
done

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
    local script=$2
    local extra_args=${3:-}

    echo ""
    echo "========================================"
    echo "Benchmark: $name"
    echo "  GPUs=$GPUS, seq=$SEQ, steps=$STEPS, warmup=$WARMUP"
    echo "========================================"

    cleanup
    local output
    local port
    port=$(alloc_port)
    if output=$(timeout 600 torchrun --nproc_per_node=$GPUS --master_port="$port" \
        $script --seq $SEQ --steps $STEPS --warmup $WARMUP $extra_args 2>&1); then
        echo "$output"
        # Extract summary lines
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
echo "All Benchmarks (GPUs: $GPUS, EP: $EP, seq: $SEQ)"
echo "=========================================="

# Dense SFT (FSDP, no EP)
run_bench "SFT Dense (FSDP)" \
    "tests/gpu/profiling/benchmark_sft_dense.py"

# Dense SFT (TP=2)
if [ "$GPUS" -ge 2 ]; then
    run_bench "SFT Dense (TP=2)" \
        "tests/gpu/profiling/benchmark_sft_dense.py" "--tp 2"
fi

# MoE SFT with EP
run_bench "SFT MoE (EP=$EP)" \
    "tests/gpu/profiling/benchmark_sft_ep.py" "--ep $EP"

# MoE SFT with EP+CP
run_bench "SFT MoE (EP=$EP + CP=$EP)" \
    "tests/gpu/profiling/benchmark_sft_ep_cp.py" "--ep $EP --cp $EP"

# SMPO with EP
run_bench "SMPO MoE (EP=$EP)" \
    "tests/gpu/profiling/benchmark_smpo_ep.py" "--ep $EP"

# SMPO with EP+CP
run_bench "SMPO MoE (EP=$EP + CP=$EP)" \
    "tests/gpu/profiling/benchmark_smpo_ep_cp.py" "--ep $EP --cp $EP"

echo ""
echo "=========================================="
echo "Benchmark Summary: $PASSED completed, $FAILED failed"
echo "=========================================="
if [ -n "$RESULTS" ]; then
    echo ""
    echo "=========================================="
    echo "Performance Summary"
    echo "=========================================="
    echo -e "$RESULTS"
fi
exit $FAILED
