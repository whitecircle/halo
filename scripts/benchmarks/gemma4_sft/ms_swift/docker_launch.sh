#!/bin/bash
# the protocol: ms-swift 4.5.3 official image, headline config (FSDP2 full_shard, no activation checkpointing,
# no liger, sdpa, all params trainable), canonical tokens/labels, sequential order, 2 repeats.
# Usage: docker_launch.sh [run_name ...]   (default: fsdp2_noac_rep1 fsdp2_noac_rep2)
# BENCH_TW_HOOK=1 (default) computes the independent token-weighted CE each step (one extra no-grad lm_head matmul +
# CE over the 2x2048 tokens, inside the timed step); BENCH_TW_HOOK=0 skips it - ms-swift's logged loss already is the
# token-weighted global mean (|logged - independent| <= 0.01 in the measured runs).
# AdamW per PROTOCOL.txt: ms-swift's own default adam_beta2 is 0.95, so --adam_beta2 0.999 is passed.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../paths.env"
set -u
H=${BENCH_ROOT}/results/ms_swift
IMG=modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda13.0.3-py312-torch2.13.0-vllm0.28.0-modelscope1.40.0-swift4.5.3
# digest: modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope@sha256:cb8aadb783de32cd9df894b0d44b96161b31bed78d1eeafd59e9563a518743c3
MODEL=$(ls -d ${HF_CACHE}/hub/models--google--gemma-4-26B-A4B-it/snapshots/*/ | head -1)
mkdir -p $H/runs ${BENCH_TMP}
NAMES=${@:-fsdp2_noac_rep1 fsdp2_noac_rep2}
for NAME in $NAMES; do
  flock ${GPU_LOCK} timeout 1800 docker run --rm --name swift_$NAME $GPU_FLAGS \
    --ipc=host --ulimit memlock=-1 --shm-size=128g --network host \
    "${BENCH_DOCKER_MOUNTS[@]}" -e HF_HOME=${HF_CACHE} -e HF_HUB_OFFLINE=1 -e TMPDIR=${BENCH_TMP} -e MODELSCOPE_CACHE=${BENCH_TMP}/modelscope \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e BENCH_FSDP_NO_RAM_EFFICIENT=1 -e BENCH_CANONICAL_INJECT=1 -e BENCH_TW_HOOK=${BENCH_TW_HOOK:-1} -e BENCH_OUT=$H/runs/$NAME.json \
    --entrypoint "" $IMG \
    torchrun --nproc_per_node=2 --master_port=29615 $BUNDLE/gemma4_sft/ms_swift/run_swift.py \
      --model $MODEL --use_hf true --tuner_type full --torch_dtype bfloat16 \
      --freeze_vit false --freeze_aligner false \
      --dataset ${BENCH_MESSAGES} --split_dataset_ratio 0 \
      --dataset_shuffle false --train_dataloader_shuffle false \
      --max_length ${BENCH_SEQ} --truncation_strategy right \
      --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
      --learning_rate 1e-5 --lr_scheduler_type constant --warmup_ratio 0 --warmup_steps 0 --weight_decay 0 --max_grad_norm 1.0 \
      --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 \
      --max_steps ${BENCH_STEPS} --save_strategy no --eval_strategy no --logging_steps 1 --report_to none \
      --attn_impl sdpa --dataloader_num_workers 2 --dataset_num_proc 8 \
      --fsdp $BUNDLE/gemma4_sft/ms_swift/fsdp2_gemma4_ac_false.json --gradient_checkpointing false --use_liger_kernel false \
      --output_dir ${BENCH_TMP}/swift_bench_out/$NAME \
      > $H/runs/$NAME.log 2>&1
  echo "exit $?" >> $H/runs/$NAME.log
done
