#!/usr/bin/env bash
# Reproduce the Gemma 4 kernel microbenchmarks inside a Halo image on one GPU.
# Usage: [IMAGE=...] [OUT=<results dir>] [GPU_FLAG=...] bash tests/gpu/profiling/gemma4/run_all.sh
#   IMAGE     Halo image (public.ecr.aws/whitecircle/halo:blackwell)
#   OUT       results directory (./gemma4-kernel-results)
#   GPU_FLAG  how docker attaches the GPU, via CDI (--device nvidia.com/gpu=all)
set -euo pipefail
IMAGE=${IMAGE:-public.ecr.aws/whitecircle/halo:blackwell}
OUT=${OUT:-$PWD/gemma4-kernel-results}
GPU_FLAG=${GPU_FLAG:---device nvidia.com/gpu=all}
mkdir -p "$OUT"
run() {
  # shellcheck disable=SC2086  # GPU_FLAG is intentionally word-split
  docker run --rm $GPU_FLAG --ipc=host -e CUDA_VISIBLE_DEVICES=0 -e PYTHONPATH=/workspace \
    -v "$PWD":/workspace -v "$OUT":/out -w /workspace "$IMAGE" "$@"
}
docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE" > "$OUT/image.txt" 2>/dev/null || echo "$IMAGE" > "$OUT/image.txt"
git rev-parse HEAD > "$OUT/commit.txt"
run python tests/gpu/profiling/gemma4/bench_moe_block.py --out /out/moe_block.json
