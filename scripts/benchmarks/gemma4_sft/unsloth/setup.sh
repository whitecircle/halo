#!/bin/bash
# Re-create the Unsloth 2026.9.11 benchmark environment from scratch (host, no docker).
# unsloth 2026.9.11 pins torch<2.13.0, transformers<=5.5.0, trl<=0.24.0, datasets<4.4.0
# -> newest accepted: torch 2.12.1+cu130, transformers 5.5.0, trl 0.24.0.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../paths.env"
set -euo pipefail
export UV_CACHE_DIR=${UV_CACHE_DIR} TMPDIR=${BENCH_TMP}
UV=uv
B=${BENCH_ROOT}/unsloth
mkdir -p $B ${BENCH_TMP} ${UV_CACHE_DIR}
cd $B
$UV venv -p 3.12 venv
$UV pip install -p venv/bin/python "unsloth==2026.9.11" "unsloth_zoo==2026.9.7" "torch==2.12.1" \
    --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
# exact pins: $UV pip install -p venv/bin/python -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
# Model + data: HF_HOME=${HF_CACHE}, `hf download google/gemma-4-26B-A4B-it` (token via HF_TOKEN env);
# dataset built by build_dataset.py (copied here).
