#!/bin/bash
# The rows `make bench` measures on one 8-GPU node. Each row is one torchrun launch whose result lines carry
# the key of its golden tests/baselines/bench/<key>.json, which tests/common/equivalence/bench_band.py holds
# the run to (make bench BAND=1). A baseline records the command that measured it, so editing a row means
# re-recording its baseline (make bench RECORD=1).
#   ./tests/gpu/profiling/run_baseline_benchmarks.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
# Benchmarks import `tests.common` at module load; torchrun puts the script dir on
# sys.path[0], not the project root, so the root must be on PYTHONPATH.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

GPUS=8
# Every row: seq 4096, batch 1, gradient checkpointing, the production attention and grouped-GEMM defaults.
SHARED_ARGS=(--seq 4096 --steps 20 --warmup 3)
# A hang guard, not a time budget: a cold kernel cache compiles for minutes.
ROW_TIMEOUT_SECONDS=1800
DENSE=tests/gpu/profiling/benchmark_sft_dense.py
MOE=tests/gpu/profiling/benchmark_sft_ep.py

# One home for master ports (tests/common/ports.py): a fixed literal races the previous launch, whose
# rendezvous socket is still in TIME_WAIT. Assigned on its own line at every call site — a command
# substitution inside an argument list does not trip `set -e`.
alloc_port() { python -c 'from tests.common.ports import free_port; print(free_port())'; }

visible=$(python -c 'import torch; print(torch.cuda.device_count())')
if [ "$visible" -lt "$GPUS" ]; then
    echo "make bench measures on $GPUS GPUs; $visible are visible"
    exit 1
fi

FAILED=0
run_row() {
    local name=$1
    shift
    echo ""
    echo "=== $name ==="
    local port
    port=$(alloc_port)
    if timeout "$ROW_TIMEOUT_SECONDS" torchrun --nproc_per_node="$GPUS" --master_port="$port" \
        "$@" "${SHARED_ARGS[@]}"; then
        echo "PASS: $name"
    else
        echo "FAIL: $name"
        FAILED=$((FAILED + 1))
    fi
}

run_row "Qwen3-8B FSDP2" "$DENSE" --model qwen3-8b
run_row "Qwen3-8B CP2" "$DENSE" --model qwen3-8b --cp 2
run_row "Qwen3-8B TP2" "$DENSE" --model qwen3-8b --tp 2
run_row "gpt-oss-20b EP1" "$MOE" --model gpt-oss-20b --ep 1
run_row "gpt-oss-20b EP2" "$MOE" --model gpt-oss-20b --ep 2
run_row "gpt-oss-20b EP8 + gathered save" "$MOE" --model gpt-oss-20b --ep 8 --measure_save
run_row "gpt-oss-20b EP8 x TP2" "$MOE" --model gpt-oss-20b --ep 8 --tp 2
run_row "gpt-oss-20b EP4 x ETP2" "$MOE" --model gpt-oss-20b --ep 4 --etp 2

echo ""
echo "$FAILED row(s) failed"
exit "$FAILED"
