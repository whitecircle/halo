#!/usr/bin/env bash
# Megatron Bridge - the protocol Docker launch (to be run by the coordinator; the benchmarking agent's GPU
# docker launches are refused by its permission policy). Gemma 4 26B-A4B full SFT, 2x B300, 25 steps.
#
#   bash $BUNDLE/gemma4_sft/megatron_bridge/docker_launch.sh            # everything
#   bash .../docker_launch.sh build|convert|runs                                    # single phase
#
# Each training config = one `docker run` inside `flock ${GPU_LOCK}` (~4-5 min incl. model init +
# checkpoint load; the lock is re-acquired per config so no hold exceeds ~25 min). Results (one JSON + log per
# run, with per-step losses, token-weighted losses, grad norms, step times, row order + data verification) go to
# ${BENCH_ROOT}/results/megatron_bridge/runs/.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../paths.env"
set -uo pipefail

OFFICIAL=nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf  # nemo:26.08.01: Bridge 0.6.1 / MCore 0.19.1 / TE 2.16 / torch 2.13 nv26.6
DERIVED=gemma-bench/megatron-bridge:0.6.2-nemo26.08.01   # docker/Dockerfile: FROM $OFFICIAL + megatron-bridge 0.6.2 + megatron-core 0.19.2
R=${BENCH_ROOT}/results/megatron_bridge
W=$BUNDLE/gemma4_sft/megatron_bridge       # train_gemma4.py, train.py, convert.py, read from scripts/benchmarks (mounted)
OUT=$R/runs
CKPT=${BENCH_ROOT}/megatron/ckpt/gemma4-26b-a4b-it   # AutoBridge torch_dist conversion (Gemma4VLModel, TP1/EP1)
mkdir -p "$OUT"

DOCKER_COMMON=(--rm $GPU_FLAGS --ipc=host --ulimit memlock=-1 --shm-size=128g "${BENCH_DOCKER_MOUNTS[@]}"
  -e HF_HOME=${HF_CACHE} -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e TMPDIR=${BENCH_TMP}
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e PYTHONPATH=$W -w $W --entrypoint torchrun)

build() {
  docker build -t "$DERIVED" "$BUNDLE/gemma4_sft/megatron_bridge/docker"
  docker image inspect --format '{{.Id}} {{index .RepoDigests 0}}' "$OFFICIAL" > "$OUT/image_official.txt" 2>&1 || true
  docker image inspect --format '{{.Id}}' "$DERIVED" > "$OUT/image_derived.txt"
}

convert() {  # HF google/gemma-4-26B-A4B-it -> Megatron torch_dist (skipped if it already exists)
  if [ -f "$CKPT/latest_checkpointed_iteration.txt" ]; then echo "checkpoint exists: $CKPT"; return; fi
  flock ${GPU_LOCK} timeout 1800 docker run --name mb_convert "${DOCKER_COMMON[@]}" "$DERIVED" \
    --nproc_per_node=1 --master_port=29611 convert.py > "$OUT/convert.log" 2>&1 && echo "convert OK" || echo "convert FAILED"
}

run_cfg() {  # $1 image-key (official|derived)  $2 run name  $3.. MB_* knobs
  local key=$1 name=$2; shift 2
  local image=$OFFICIAL; [ "$key" = derived ] && image=$DERIVED
  local tag=${key}_${name}
  local envs=(-e MB_TP=1 -e MB_ETP=1 -e MB_EP=2 -e MB_RECOMPUTE=none -e MB_CKPT=$CKPT -e MB_OUT_JSON=$OUT/$tag.json)
  for kv in "$@"; do envs+=(-e "$kv"); done
  # Host CPU load is recorded before/after each run. On 2026-09-24 a load average of ~235 (192 threads) from
  # unrelated jobs cut the same config from 5,439 to ~1,800 tok/s - do not use throughput from overloaded runs.
  exec 9>${GPU_LOCK}; flock 9
  uptime > "$OUT/$tag.load_before"
  timeout 1500 docker run --name mb_$tag "${DOCKER_COMMON[@]}" "${envs[@]}" "$image" \
    --nproc_per_node=2 --master_port=29612 train.py > "$OUT/$tag.log" 2>&1
  local rc=$?
  uptime > "$OUT/$tag.load_after"
  flock -u 9; exec 9>&-
  [ $rc -eq 0 ] && echo "$tag OK" || echo "$tag FAILED rc=$rc (see $OUT/$tag.log)"
}

runs() {
  # 1) headline: EP2, allgather dispatcher, TE GroupedLinear experts, no recompute - 2 repeats per image
  run_cfg official ep2_allgather_r1   MB_DISPATCHER=allgather
  run_cfg official ep2_allgather_r2   MB_DISPATCHER=allgather
  run_cfg derived  ep2_allgather_r1   MB_DISPATCHER=allgather
  run_cfg derived  ep2_allgather_r2   MB_DISPATCHER=allgather
  # 2) dispatchers (derived image = Bridge 0.6.2)
  run_cfg derived  ep2_alltoall       MB_DISPATCHER=alltoall
  run_cfg derived  ep2_flex_deepep    MB_DISPATCHER=flex MB_FLEX_BACKEND=deepep
}

case "${1:-all}" in
  build) build ;; convert) convert ;; runs) runs ;;
  all) build; convert; runs ;;
  *) echo "usage: $0 [all|build|convert|runs]"; exit 2 ;;
esac
