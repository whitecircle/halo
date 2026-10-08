#!/usr/bin/env python
"""
VLM (Vision Language Model) SFT Training Test.

Smoke test: DistributedSFTTrainer trains a VLM model (AutoModelForImageTextToText) on text-only
synthetic conversations, exercising the VLM code path without image data. It checks only that the
logged losses are finite; no loss is compared against a reference.

Test Setup:
- Model: Qwen/Qwen3-VL-2B-Instruct (VLM with text+vision capabilities)
- Parallelism: the mixin's FSDP2 data parallelism (no EP/CP/TP)
- Dataset: the shared synthetic math SFT set, text-only (no images), 30% multi-turn

Test Phases:
1. Load VLM model via AutoModelForImageTextToText + AutoProcessor
2. Create synthetic text-only dataset with chat-templated conversations
3. Train for 5 steps with gradient checkpointing (Liger off)
4. Validate: training completes, loss is finite

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_vlm.py
"""

import torch
from trl import ModelConfig, SFTConfig

from src.distributed.loading.vlm_setup import load_model_for_training
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_VL_2B
from tests.common.utils import log, training_run_checks

MODEL_NAME = QWEN3_VL_2B
MAX_STEPS = 5
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 2e-5
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
SEED = 42


def run(ctx):
    log(f"\n{'=' * 70}")
    log("  VLM SFT Training Test (Text-only, no EP/CP/TP)")
    log(f"  World size: {ctx.world_size}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  Max steps: {MAX_STEPS}, Batch size: {BATCH_SIZE}")
    log(f"  Seq length: {MAX_SEQ_LENGTH}")
    log(f"{'=' * 70}")

    # the loader reads dtype/liger settings off these
    config = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        fsdp="",  # Mixin handles FSDP wrapping
    )
    parallelism_config = ParallelismConfig()

    # production entry point, so VLM detection and processor load are under test too
    log("\n[1/5] Loading VLM via load_model_for_training (production path)...")
    model_config = ModelConfig(
        model_name_or_path=MODEL_NAME, attn_implementation="flash_attention_2", trust_remote_code=True
    )
    model, processing_class, tokenizer, is_vlm = load_model_for_training(
        model_config, config, parallelism_config, vlm_run=False
    )
    assert is_vlm, "Qwen3-VL must be detected as a VLM by load_model_for_training"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"  Model: {type(model).__name__}, processor: {type(processing_class).__name__}")

    log("\n[2/5] Creating synthetic text-only datasets...")
    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1)
    log(f"  Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")
    if ctx.rank == 0:
        log(f"  Sample (truncated): {train_dataset[0]['text'][:200]}...")

    log("\n[3/5] Configuring trainer...")
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        parallelism_config=parallelism_config,
    )
    log(f"  Trainer created: {type(trainer).__name__}")
    log(f"  Parallelism: {parallelism_config.mode_string or 'dp'}")

    log("\n[5/5] Training...")
    train_result = trainer.train()
    return {"checks": training_run_checks(train_result, trainer, MAX_STEPS)}


main = gpu_test_main(min_world_size=2, prefix="test_sft_vlm")(run)

if __name__ == "__main__":
    main()
