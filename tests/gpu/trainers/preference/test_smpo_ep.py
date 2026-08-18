#!/usr/bin/env python
"""
Test script for SMPO trainer with Expert Parallelism (EP=2) on GptOss-20B MoE.

Validates that SmoothMarginPOTrainer works correctly with expert parallelism
enabled, using synthetic preference data (math Q&A with 30% multi-turn prompts).

Test validates:
1. EP model loading via load_distributed_model
2. SMPO training completes without errors
3. Loss values are finite (no NaN/Inf)
4. Training produces reasonable metrics

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/preference/test_smpo_ep.py

Requirements:
    - 2x GPUs with >=80GB memory each
    - DeepEP installed
    - Model: unsloth/gpt-oss-20b-BF16 (auto-downloaded)
"""

import sys
import traceback

import torch
from accelerate import PartialState

from src.configs.smpo_config import SmoothMarginPOConfig
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.preference.smpo import SmoothMarginPOTrainer
from tests.common.datasets import create_preference_dataset
from tests.common.distributed import (
    cleanup_dirs,
    ensure_model_downloaded,
    init_distributed,
    setup_cache_dirs,
    teardown_distributed,
)
from tests.common.models import GPT_OSS_20B
from tests.common.utils import cleanup_memory, gpu_mem_gb, log

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 2
NUM_TRAIN_SAMPLES = 64
NUM_EVAL_SAMPLES = 16
MAX_LENGTH = 4096
MAX_PROMPT_LENGTH = 2048
NUM_TRAIN_STEPS = 5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 5e-6
SEED = 42


def run_smpo_ep_test():
    """Run SMPO trainer test with Expert Parallelism (EP=2)."""
    rank, world_size, local_rank = init_distributed()

    log(f"\n{'=' * 70}")
    log("SMPO EP TEST: Expert Parallelism (EP=2) with GptOss-20B")
    log(f"{'=' * 70}")
    log(f"World size: {world_size}")
    log(f"EP size: {EP_SIZE}")
    log(f"Model: {MODEL_NAME}")
    log(f"Train samples: {NUM_TRAIN_SAMPLES}, Eval samples: {NUM_EVAL_SAMPLES}")
    log(f"Max length: {MAX_LENGTH}, Max prompt length: {MAX_PROMPT_LENGTH}")
    log(f"Batch size: {BATCH_SIZE}, Grad accum: {GRADIENT_ACCUMULATION}")
    log(f"Training steps: {NUM_TRAIN_STEPS}")
    log(f"GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"GPU memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f}GB")

    if world_size != EP_SIZE:
        log(f"\nERROR: This test requires exactly {EP_SIZE} GPUs, got {world_size}")
        teardown_distributed()
        return False

    output_dir, cache_dir = setup_cache_dirs("smpo_ep", rank)
    log(f"Output directory: {output_dir}")
    log(f"Cache directory: {cache_dir}")

    try:
        log("\n--- Ensuring model is downloaded ---")
        ensure_model_downloaded(MODEL_NAME, rank)

        # trainers require an initialized PartialState
        PartialState()

        log("\n--- Loading tokenizer ---")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        log(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

        log("\n--- Creating synthetic preference datasets ---")
        train_dataset = create_preference_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
        eval_dataset = create_preference_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)
        log(f"Train dataset: {len(train_dataset)} samples")
        log(f"Eval dataset: {len(eval_dataset)} samples")

        if rank == 0:
            sample = train_dataset[0]
            log(f"\nSample prompt (truncated): {sample['prompt'][:120]}...")
            log(f"Sample chosen (truncated): {sample['chosen'][:120]}...")
            log(f"Sample rejected (truncated): {sample['rejected'][:120]}...")

        log("\n--- Loading model with Expert Parallelism ---")
        log(f"GPU memory before model load: {gpu_mem_gb():.1f}GB")

        parallelism_config = ParallelismConfig(ep_size=EP_SIZE)

        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flex_attention",
            use_liger_kernel=True,
        )

        log(f"Model loaded: {model.config.model_type}")
        log(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        log(f"GPU memory after model load: {gpu_mem_gb():.1f}GB")

        log("\n--- Creating SMPO config ---")
        config = SmoothMarginPOConfig(
            output_dir=output_dir,
            max_steps=NUM_TRAIN_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,  # already applied by load_distributed_model
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=MAX_LENGTH,
            max_prompt_length=MAX_PROMPT_LENGTH,
            dataloader_drop_last=True,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=True,  # Required for EP (inactive experts)
            fsdp="",  # Mixin handles FSDP wrapping
        )

        log(f"Config: beta={config.beta}, target_margin={config.target_margin}, loss_type={config.loss_type}")

        log("\n--- Creating SmoothMarginPOTrainer ---")
        trainer = SmoothMarginPOTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )
        log("Trainer created successfully")

        log("\n--- Starting training ---")
        log(f"GPU memory before training: {gpu_mem_gb():.1f}GB")
        from src.distributed.runtime import barrier

        barrier()

        train_result = trainer.train()

        log("\n--- Training completed ---")
        log(f"Training loss: {train_result.training_loss:.6f}")
        log(f"Training steps: {train_result.global_step}")
        log(f"GPU memory after training: {gpu_mem_gb():.1f}GB")

        log("\n--- Validating results ---")
        log_history = trainer.state.log_history

        step_losses = []
        grad_norms = []
        for entry in log_history:
            if "loss" in entry and "eval_loss" not in entry:
                step_losses.append(entry["loss"])
            if "grad_norm" in entry:
                grad_norms.append(entry["grad_norm"])

        log(f"Per-step losses: {[f'{l:.4f}' for l in step_losses]}")
        if grad_norms:
            log(f"Per-step grad norms: {[f'{g:.2f}' for g in grad_norms]}")

        success = True
        issues = []

        if not torch.isfinite(torch.tensor(train_result.training_loss)):
            issues.append(f"Training loss is not finite: {train_result.training_loss}")
            success = False

        has_nan = any(not torch.isfinite(torch.tensor(l)) for l in step_losses)
        if has_nan:
            issues.append("NaN/Inf detected in step losses")
            success = False

        has_bad_grad = any(not torch.isfinite(torch.tensor(g)) for g in grad_norms) if grad_norms else False
        if has_bad_grad:
            issues.append("NaN/Inf detected in gradient norms")
            success = False

        if train_result.global_step != NUM_TRAIN_STEPS:
            issues.append(f"Expected {NUM_TRAIN_STEPS} steps, got {train_result.global_step}")
            success = False

        if train_result.training_loss > 100:
            issues.append(f"Training loss unreasonably high: {train_result.training_loss}")
            success = False

        if success:
            log(f"\n{'=' * 70}")
            log("SMPO EP TEST PASSED")
            log(f"  - Training loss: {train_result.training_loss:.6f}")
            log(f"  - Steps completed: {train_result.global_step}")
            log("  - All losses finite")
            log(f"{'=' * 70}")
        else:
            log(f"\n{'=' * 70}")
            log("SMPO EP TEST FAILED")
            for issue in issues:
                log(f"  - {issue}")
            log(f"{'=' * 70}")

        return success

    except Exception as e:
        log(f"\nTEST FAILED WITH ERROR: {e}")
        if rank == 0:
            traceback.print_exc()
        return False

    finally:
        from src.distributed.runtime import barrier

        barrier()
        cleanup_dirs(output_dir, cache_dir)
        cleanup_memory()


def main():
    success = run_smpo_ep_test()
    teardown_distributed()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
