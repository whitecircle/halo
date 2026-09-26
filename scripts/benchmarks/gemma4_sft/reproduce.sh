#!/usr/bin/env bash
# The Gemma 4 26B-A4B SFT comparison end to end: data, every framework, summary table.
#
# Usage: BENCH_ROOT=<workdir> HF_TOKEN=<token> [BENCH_SEQ=16384] bash reproduce.sh [framework...]
#   frameworks: halo axolotl automodel megatron_bridge ms_swift unsloth (default: all, in this order)
#   BENCH_SEQ   tokens per row (paths.env; default 2048). Rows longer than 2048 are packed protocol rows.
#   RUNNER      docker (default): each run in the framework's official image, GPUs through $GPU_FLAGS (CDI).
#               direct: run in the current environment, which must already be the framework's image
#               (e.g. a scheduler that starts that image); then name exactly one framework.
#   SKIP_DATA=1 skip prepare_data.sh (datasets already built under BENCH_ROOT). With RUNNER=direct the datasets are
#               built beforehand by `RUNNER=direct prepare_data.sh` in the Halo image.
#
# Each framework starts from its 2,048-token headline configuration (PROTOCOL.txt). When that runs out of GPU
# memory, the framework's own activation checkpointing is turned on; the first configuration that fits runs twice.
# Needs 2 GPUs. Results: $BENCH_ROOT/results/s$BENCH_SEQ/<framework>/<config>_r<i>.{json,log} and
# $BENCH_ROOT/results/s$BENCH_SEQ/summary.tsv (tokens/s, peak memory, host load per run).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../paths.env"
set -uo pipefail
RUNNER=${RUNNER:-docker}
FRAMEWORKS=("$@"); [ ${#FRAMEWORKS[@]} -gt 0 ] || FRAMEWORKS=(halo axolotl automodel megatron_bridge ms_swift unsloth)
HERE=$BUNDLE/gemma4_sft
RES=$BENCH_ROOT/results/s$BENCH_SEQ
HALO_IMAGE=${HALO_IMAGE:-public.ecr.aws/whitecircle/halo:blackwell}
AXOLOTL_IMAGE=axolotlai/axolotl:0.19.0-py3.12-cu130-2.13.0@sha256:b38049f2454ceafc58c7a8269c7b400a75416e15b0f308a33850d48ae11e384d
AUTOMODEL_IMAGE=nvcr.io/nvidia/nemo-automodel:26.08.00
MEGATRON_IMAGE=nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf
SWIFT_IMAGE=modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda13.0.3-py312-torch2.13.0-vllm0.28.0-modelscope1.40.0-swift4.5.3
MODEL_ID=google/gemma-4-26B-A4B-it

# in_image <image> <workdir> [VAR=value ...] -- <command...>: run in the image (docker) or here (direct);
# RUN_TIMEOUT (seconds) bounds the run when set.
in_image() {
  local image=$1 wd=$2; shift 2; local envs=()
  while [ "$1" != -- ]; do envs+=("$1"); shift; done; shift
  envs+=(CUDA_DEVICE_MAX_CONNECTIONS=1 WANDB_DISABLED=true WANDB_MODE=disabled)
  if [ "$RUNNER" = docker ]; then
    local flags=(); for kv in "${envs[@]}"; do flags+=(-e "$kv"); done
    # shellcheck disable=SC2086  # GPU_FLAGS is intentionally word-split
    ${RUN_TIMEOUT:+timeout "$RUN_TIMEOUT"} docker run --rm $GPU_FLAGS --ipc=host --ulimit memlock=-1 --shm-size=128g "${BENCH_DOCKER_MOUNTS[@]}" "${flags[@]}" \
      -w "$wd" --entrypoint "" "$image" "$@"
  else
    (cd "$wd" && ${RUN_TIMEOUT:+timeout "$RUN_TIMEOUT"} env "${envs[@]}" "$@")
  fi
}

state() { echo "[reproduce] $1 $(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg)"; }

# attempt <name> <command...>: 0 = finished, 2 = out of GPU memory, 1 = any other failure.
attempt() {
  local name=$1; shift; local log=$OUT/$name.log
  echo "load_before $(cut -d' ' -f1-3 /proc/loadavg)" > "$OUT/$name.load"
  state "start $FW/$name"
  RUN_TIMEOUT=3000 "$@" > "$log" 2>&1; local rc=$?
  echo "load_after $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/$name.load"
  echo "exit $rc" >> "$log"
  if grep -qiE "OutOfMemoryError|CUDA out of memory|CUDA error: out of memory" "$log"; then state "OOM $FW/$name"; return 2; fi
  if [ $rc -eq 0 ] && [ -s "$OUT/$name.json" ]; then state "OK $FW/$name"; return 0; fi
  state "FAILED $FW/$name (exit $rc, see $log)"; tail -20 "$log" >&2; return 1
}

# fit_and_repeat <config...>: configurations in order of preference; the first that fits runs twice.
fit_and_repeat() {
  local c r
  for c in "$@"; do
    "$c" "${c}_r1"; r=$?
    if [ $r -eq 0 ]; then "$c" "${c}_r2"; return; fi
    [ $r -eq 2 ] || return 1
  done
  state "no configuration of $FW fits at $BENCH_SEQ tokens"
}

model_dir() { ls -d "$HF_CACHE"/hub/models--google--gemma-4-26B-A4B-it/snapshots/*/ | head -1; }

# ---- Halo (the published image, this checkout mounted) ----
halo_run() {
  local n=$1; shift; local cfg; cfg=$(render_config "$HERE/halo/gemma4-ep2.yaml")
  attempt "$n" in_image "$HALO_IMAGE" "$HALO_TREE" PYTHONPATH="$HALO_TREE" BENCH_OUT="$OUT/$n.json" \
    HALO_DATA_ROOT="$BENCH_ROOT/halo/data_root" TRITON_CACHE_DIR="$BENCH_ROOT/halo/cache/triton" \
    EP_JIT_CACHE_DIR="$BENCH_ROOT/halo/cache/ep_jit" -- \
    python -m torch.distributed.run --nproc_per_node=2 --master_port=29641 "$HERE/halo/bench_sft.py" "$cfg" "$@"
}
halo_nogc() { halo_run "$1"; }
halo_gc() { halo_run "$1" --gradient_checkpointing=true; }
run_halo() { fit_and_repeat halo_nogc halo_gc; }

# ---- Axolotl 0.19.0 (official image; config L: FSDP2 no-reshard, grouped_mm, hybrid FA2) ----
AX_CFG=""
axolotl_run() {
  local n=$1; shift
  attempt "$n" in_image "$AXOLOTL_IMAGE" "$BENCH_ROOT/axolotl" PYTHONPATH="$HERE/axolotl" AXOLOTL_DO_NOT_TRACK=1 \
    TRITON_CACHE_DIR="$BENCH_ROOT/axolotl/triton" BENCH_EXIT_AFTER=1 BENCH_OUT="$OUT/$n.json" -- \
    torchrun --nproc_per_node 2 --master_port 29611 -m axolotl.cli.train "$AX_CFG" "$@"
}
axolotl_nogc() { axolotl_run "$1"; }
axolotl_gc() { axolotl_run "$1" --gradient_checkpointing=True; }
run_axolotl() {
  mkdir -p "$BENCH_ROOT/axolotl"
  in_image "$AXOLOTL_IMAGE" "$BENCH_ROOT/axolotl" -- python3 "$HERE/axolotl/build_data.py" || return 1
  AX_CFG=$(render_config "$HERE/axolotl/configs/L_fsdp2_groupedmm_nogc_hybridfa2_noreshard.yaml")
  in_image "$AXOLOTL_IMAGE" "$BENCH_ROOT/axolotl" PYTHONPATH="$HERE/axolotl" -- \
    python -m axolotl.cli.preprocess "$AX_CFG" > "$OUT/preprocess.log" 2>&1 || { state "axolotl preprocess failed"; return 1; }
  fit_and_repeat axolotl_nogc axolotl_gc
}

# ---- NeMo AutoModel 26.08 (official image; EP2 + FSDP2, DeepEP) ----
automodel_run() {
  local n=$1; shift; local cfg; cfg=$(render_config "$HERE/automodel/gemma4_26b_a4b_ep2_bench.yaml")
  attempt "$n" in_image "$AUTOMODEL_IMAGE" "$HERE/automodel" PYTHONPATH="$HERE/automodel" HF_HUB_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHINDUCTOR_CACHE_DIR="$BENCH_TMP/inductor_automodel" \
    BENCH_OUT="$OUT/$n.json" BENCH_ORDER_LOG="$OUT/$n.order" BENCH_TW_CHECK=0 -- \
    torchrun --nproc_per_node=2 --master_port=29579 run_bench.py --config "$cfg" "$@"
}
automodel_noac() { automodel_run "$1"; }
automodel_ac() { automodel_run "$1" --distributed.activation_checkpointing true; }
run_automodel() { fit_and_repeat automodel_noac automodel_ac; }

# ---- Megatron Bridge (official NeMo 26.08.01 image; EP2 allgather, TE GroupedLinear) ----
MB_CKPT=$BENCH_ROOT/megatron/ckpt/gemma4-26b-a4b-it
megatron_run() {
  local n=$1; shift; local envs=(); envs=("$@")
  attempt "$n" in_image "$MEGATRON_IMAGE" "$HERE/megatron_bridge" PYTHONPATH="$HERE/megatron_bridge" HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MB_TP=1 MB_ETP=1 MB_EP=2 \
    MB_DISPATCHER=allgather MB_CKPT="$MB_CKPT" MB_OUT_JSON="$OUT/$n.json" "${envs[@]}" -- \
    torchrun --nproc_per_node=2 --master_port=29612 train.py
}
megatron_norecompute() { megatron_run "$1" MB_RECOMPUTE=none; }
megatron_fullrecompute() { megatron_run "$1" MB_RECOMPUTE=full; }
# Megatron keeps fp32 master weights and fp32 Adam moments; when recompute alone does not fit, TE's precision-aware
# optimizer (bf16 moments) is its remaining memory lever.
megatron_fullrecompute_bf16moments() { megatron_run "$1" MB_RECOMPUTE=full MB_PRECISION_AWARE_OPT=1; }
run_megatron_bridge() {
  if [ ! -f "$MB_CKPT/latest_checkpointed_iteration.txt" ]; then  # HF -> Megatron torch_dist, once
    in_image "$MEGATRON_IMAGE" "$HERE/megatron_bridge" PYTHONPATH="$HERE/megatron_bridge" -- \
      torchrun --nproc_per_node=1 --master_port=29611 convert.py > "$OUT/convert.log" 2>&1 || { state "megatron convert failed"; return 1; }
  fi
  fit_and_repeat megatron_norecompute megatron_fullrecompute megatron_fullrecompute_bf16moments
}

# ---- MS-SWIFT 4.5.3 (official image; FSDP2 full_shard, canonical tokens injected) ----
ms_swift_run() {
  local n=$1 ac=$2
  attempt "$n" in_image "$SWIFT_IMAGE" "$BENCH_TMP" HF_HUB_OFFLINE=1 MODELSCOPE_CACHE="$BENCH_TMP/modelscope" \
    PYTORCH_ALLOC_CONF=expandable_segments:True BENCH_FSDP_NO_RAM_EFFICIENT=1 BENCH_CANONICAL_INJECT=1 \
    BENCH_TW_HOOK=1 BENCH_OUT="$OUT/$n.json" -- \
    torchrun --nproc_per_node=2 --master_port=29615 "$HERE/ms_swift/run_swift.py" \
      --model "$(model_dir)" --use_hf true --tuner_type full --torch_dtype bfloat16 --freeze_vit false --freeze_aligner false \
      --dataset "$BENCH_MESSAGES" --split_dataset_ratio 0 --dataset_shuffle false --train_dataloader_shuffle false \
      --max_length "$BENCH_SEQ" --truncation_strategy right --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
      --learning_rate 1e-5 --lr_scheduler_type constant --warmup_ratio 0 --warmup_steps 0 --weight_decay 0 --max_grad_norm 1.0 \
      --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 --max_steps "$BENCH_STEPS" --save_strategy no --eval_strategy no \
      --logging_steps 1 --report_to none --attn_impl sdpa --dataloader_num_workers 2 --dataset_num_proc 8 \
      --fsdp "$HERE/ms_swift/fsdp2_gemma4_ac_$ac.json" --gradient_checkpointing false --use_liger_kernel false \
      --output_dir "$BENCH_TMP/swift_out/$n"
}
ms_swift_noac() { ms_swift_run "$1" false; }
ms_swift_ac() { ms_swift_run "$1" true; }
run_ms_swift() { fit_and_repeat ms_swift_noac ms_swift_ac; }

# ---- Unsloth 2026.9.11 (its own venv, built by setup.sh inside the Halo image: the official image fails on sm_103) ----
unsloth_run() {
  local n=$1 gc=$2
  attempt "$n" in_image "$HALO_IMAGE" "$BENCH_TMP" HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
    UNSLOTH_COMPILE_LOCATION="$BENCH_TMP/unsloth_compiled_$n" BENCH_MODEL="$(model_dir)" BENCH_OUT="$OUT/$n.json" \
    BENCH_OPTIM=adamw_torch_fused BENCH_GC="$gc" BENCH_TW_HOOK=1 -- \
    "$BENCH_ROOT/unsloth/venv/bin/torchrun" --nproc_per_node=2 --master_port=29631 "$HERE/unsloth/train_unsloth.py"
}
unsloth_nogc() { unsloth_run "$1" false; }
unsloth_gc() { unsloth_run "$1" unsloth; }
run_unsloth() {
  [ -x "$BENCH_ROOT/unsloth/venv/bin/torchrun" ] ||
    in_image "$HALO_IMAGE" "$BENCH_ROOT" -- bash "$HERE/unsloth/setup.sh" > "$OUT/setup.log" 2>&1 || { state "unsloth setup failed"; return 1; }
  fit_and_repeat unsloth_nogc unsloth_gc
}

# ---- data, runs, summary ----
if [ "${SKIP_DATA:-0}" != 1 ]; then
  RUNNER=$RUNNER bash "$HERE/prepare_data.sh" || exit 1  # direct: needs the Halo image (a separate step)
fi
for FW in "${FRAMEWORKS[@]}"; do
  OUT=$RES/$FW; mkdir -p "$OUT"
  "run_$FW" || state "$FW stopped"
done
python3 "$HERE/summarize.py" "$RES"
