#!/bin/bash
# Stage the inputs docker_launch.sh reads, from scripts/benchmarks alone:
#   1. $BENCH_ROOT/axolotl/data_axolotl[_<SEQ>].jsonl  from the canonical tokens at BENCH_SEQ (sha256-checked)
#   2. $BENCH_ROOT/axolotl/prepared[_<SEQ>]/           Axolotl's tokenized cache, built once in the official image
#   3. the two derived images (Dockerfiles in docker/)
# Needs $BENCH_CANON (prepare_data.sh). Paths: see paths.env.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../paths.env"
set -euo pipefail
HERE=$BUNDLE/gemma4_sft/axolotl
OFFICIAL=axolotlai/axolotl:0.19.0-py3.12-cu130-2.13.0@sha256:b38049f2454ceafc58c7a8269c7b400a75416e15b0f308a33850d48ae11e384d
mkdir -p "$BENCH_ROOT/axolotl"
python3 "$HERE/build_data.py"
CFG=$(render_config "$HERE/configs/L_fsdp2_groupedmm_nogc_hybridfa2_noreshard.yaml")
docker run --rm "${BENCH_DOCKER_MOUNTS[@]}" -e PYTHONPATH="$HERE" -w "$BENCH_ROOT/axolotl" --entrypoint bash "$OFFICIAL" \
  -lc "python -m axolotl.cli.preprocess $CFG"
docker build -f "$HERE/docker/Dockerfile.deepep" -t axolotl-bench:0.19.0-deepep "$HERE/docker"
docker build -f "$HERE/docker/Dockerfile.sonic_triton36" -t axolotl-bench:0.19.0-sonic-triton360 "$HERE/docker"
