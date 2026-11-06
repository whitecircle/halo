#!/bin/bash
# the protocol, Unsloth. The newest official image (unsloth/unsloth:core = nightly 2026-09-22,
# sha256:3904d453cedc872efc692269baff177fdeecc16e495a042b925a4f57f7478512) ships unsloth 2026.9.9 + torch 2.11.0+cu128 and
# fails on B300 (sm_103) at the first training step (CUTLASS "Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED",
# then CUDA "unspecified launch failure"); no newer image exists as of 2026-09-24. The measured runs therefore use the
# host venv (setup.sh): MODE=host (default) below. MODE=docker is the image variant, kept for when a working image appears.
# Usage: [MODE=host|docker] [BENCH_OPTIM=adamw_torch_fused|adamw_8bit] [BENCH_GC=false|unsloth|true] launch.sh NAME...
# AdamW per PROTOCOL.txt is set in train_unsloth.py (betas 0.9/0.999, eps 1e-8, wd 0, lr 1e-5 constant, clip 1.0).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../paths.env"
set -u
H=${BENCH_ROOT}/results/unsloth
MODE=${MODE:-host}
IMG=unsloth/unsloth:core@sha256:3904d453cedc872efc692269baff177fdeecc16e495a042b925a4f57f7478512
MODEL=$(ls -d ${HF_CACHE}/hub/models--google--gemma-4-26B-A4B-it/snapshots/*/ | head -1)
mkdir -p $H/runs ${BENCH_TMP}
NAMES=${@:-adamw_nogc_rep1 adamw_nogc_rep2}
for NAME in $NAMES; do
  ENVS="HF_HOME=${HF_CACHE} TMPDIR=${BENCH_TMP} HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
        UNSLOTH_COMPILE_LOCATION=${BENCH_TMP}/unsloth_compiled_cache_$NAME BENCH_MODEL=$MODEL BENCH_OUT=$H/runs/$NAME.json
        BENCH_OPTIM=${BENCH_OPTIM:-adamw_torch_fused} BENCH_GC=${BENCH_GC:-false} BENCH_TW_HOOK=${BENCH_TW_HOOK:-1}
        UNSLOTH_DISABLE_AUTO_PADDING_FREE=${UNSLOTH_DISABLE_AUTO_PADDING_FREE:-0} UNSLOTH_ENABLE_LOGGING=${UNSLOTH_ENABLE_LOGGING:-0}"
  if [ "$MODE" = docker ]; then
    DENV=$(for e in $ENVS; do printf -- "-e %s " "$e"; done)
    flock ${GPU_LOCK} timeout 1800 docker run --rm --name unsloth_$NAME $GPU_FLAGS \
      --ipc=host --ulimit memlock=-1 --shm-size=128g --network host -w ${BENCH_TMP} "${BENCH_DOCKER_MOUNTS[@]}" $DENV \
      --entrypoint "" $IMG /opt/unsloth-venv/bin/torchrun --nproc_per_node=2 --master_port=29632 $BUNDLE/gemma4_sft/unsloth/train_unsloth.py \
      > $H/runs/$NAME.log 2>&1
  else
    (cd ${BENCH_TMP} && env $ENVS flock ${GPU_LOCK} timeout 1800 ${BENCH_ROOT}/unsloth/venv/bin/torchrun \
      --nproc_per_node=2 --master_port=29631 $BUNDLE/gemma4_sft/unsloth/train_unsloth.py > $H/runs/$NAME.log 2>&1)
  fi
  echo "exit $?" >> $H/runs/$NAME.log
done
