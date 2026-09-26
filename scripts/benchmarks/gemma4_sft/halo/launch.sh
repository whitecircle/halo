#!/usr/bin/env bash
# Halo rows of the Gemma 4 26B-A4B SFT comparison, in the published image: two runs of the headline configuration.
#   HALO_TREE   Halo checkout to measure (paths.env; default: the repository holding these scripts).
#   OUT         results directory (default: $BENCH_ROOT/results/halo-<commit>).
# Prereqs: prepare_data.sh; docker pull public.ecr.aws/whitecircle/halo:blackwell
#          (sha256:6ece05e7b748453ce8e1dd878177b036a60971d9c197391670f986045f09ae77).
# GPUs attach through CDI: $GPU_FLAGS (paths.env), --device nvidia.com/gpu=all by default.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../paths.env"
set -euo pipefail
OUT=${OUT:-$BENCH_ROOT/results/halo-$(git -C "$HALO_TREE" rev-parse --short HEAD)}
mkdir -p "$OUT"
for i in 1 2; do "$BUNDLE/gemma4_sft/halo/run.sh" "$BUNDLE/gemma4_sft/halo/gemma4-ep2.yaml" "$OUT/gemma4-ep2-run$i.json"; done
