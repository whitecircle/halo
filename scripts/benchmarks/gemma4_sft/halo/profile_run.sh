#!/usr/bin/env bash
# Profile the headline configuration in the Halo image: 14 steps, torch.profiler via Halo's own
# TorchProfilerCallback (wait 9, warmup 1, active 3 -> steps 11-13 recorded, rank 0 only).
# Usage: profile_run.sh [config.yaml]   (default: gemma4-ep2.yaml); trace under $BENCH_ROOT/halo/prof/<config>/.
# Stage breakdown of the trace: scripts/analysis/gemma4_stage_breakdown.py <trace> <step_ms> 3
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../paths.env"
set -euo pipefail
CFG=${1:-$BUNDLE/gemma4_sft/halo/gemma4-ep2.yaml}
OUTD=$BENCH_ROOT/halo/prof/$(basename "$CFG" .yaml); rm -rf "$OUTD"; mkdir -p "$OUTD"
"$BUNDLE/gemma4_sft/halo/run.sh" "$CFG" "$OUTD/bench.json" --max_steps=14 --enable_torch_profiler=true \
  --profiler_wait=9 --profiler_warmup=1 --profiler_active=3 --profiler_ranks=0 --profiler_output_dir="$OUTD/trace"
find "$OUTD" -maxdepth 3 | head -30
