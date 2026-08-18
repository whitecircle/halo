#!/usr/bin/env python
"""
Test script for DistributedSFTTrainer with GLM-4.7-Flash MoE model and EP=2.

Validates that SFT training works correctly with the GLM-4.7-Flash (GLM4 MoE Lite)
model under expert parallelism. GLM-4.7-Flash is a Mixture-of-Experts model,
making it a good target for EP testing alongside Qwen3 and GptOss MoE models.

Test validates:
1. EP model loading for GLM4 MoE via load_distributed_model
2. SFT training completes without errors
3. Loss values are finite (no NaN/Inf)
4. Training produces reasonable metrics

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_glm4_moe.py

Requirements:
    - 2x GPUs with >=80GB memory each
    - DeepEP installed
    - Model: zai-org/GLM-4.7-Flash (auto-downloaded)
"""

import sys
import traceback

import torch
from accelerate import PartialState
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.env import env_int, env_str
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import VERBOSE_MATH_TEMPLATES, create_single_turn_sft_dataset
from tests.common.distributed import (
    cleanup_dirs,
    ensure_model_downloaded,
    init_distributed,
    setup_cache_dirs,
    teardown_distributed,
)
from tests.common.models import GLM4_FLASH
from tests.common.utils import cleanup_memory, gpu_mem_gb, log

# Test Configuration

MODEL_NAME = env_str("HALO_TEST_GLM4_MODEL", GLM4_FLASH)
# EP size defaults to world_size (set inside main); HALO_TEST_EP forces a specific value.
EP_SIZE_OVERRIDE = env_int("HALO_TEST_EP", None)
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
MAX_SEQ_LENGTH = 4096
NUM_TRAIN_STEPS = 5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 1e-5
SEED = 42


# Dataset Creation (custom for GLM4 MoE -- kept inline)


# Main Test


def run_sft_glm4_moe_test():
    """Run SFT trainer test with GLM-4.7-Flash MoE under EP."""
    # Initialize distributed
    rank, world_size, local_rank = init_distributed()
    ep_size = EP_SIZE_OVERRIDE if EP_SIZE_OVERRIDE is not None else world_size

    log(f"\n{'=' * 70}")
    log(f"SFT GLM4 MoE TEST: EP={ep_size} with GLM-4.7-Flash (world_size={world_size})")
    log(f"{'=' * 70}")
    log(f"World size: {world_size}")
    log(f"EP size: {ep_size}")
    log(f"Model: {MODEL_NAME}")
    log(f"Train samples: {NUM_TRAIN_SAMPLES}, Eval samples: {NUM_EVAL_SAMPLES}")
    log(f"Max seq length: {MAX_SEQ_LENGTH}")
    log(f"Batch size: {BATCH_SIZE}, Grad accum: {GRADIENT_ACCUMULATION}")
    log(f"Training steps: {NUM_TRAIN_STEPS}")
    log(f"GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"GPU memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f}GB")

    if world_size % ep_size != 0:
        log(f"\nERROR: world_size={world_size} must be divisible by ep_size={ep_size}")
        teardown_distributed()
        return False

    # Isolate HF datasets cache
    output_dir, cache_dir = setup_cache_dirs("sft_glm4_moe_test", rank)
    log(f"Output directory: {output_dir}")
    log(f"Cache directory: {cache_dir}")

    try:
        # --- Ensure model is cached (download on rank 0 first) ---
        log("\n--- Ensuring model is downloaded ---")
        ensure_model_downloaded(MODEL_NAME, rank)

        # Initialize PartialState for accelerate compatibility
        PartialState()

        # --- Load tokenizer ---
        log("\n--- Loading tokenizer ---")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        log(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

        # --- Create synthetic datasets ---
        log("\n--- Creating synthetic SFT datasets ---")
        train_dataset = create_single_turn_sft_dataset(
            NUM_TRAIN_SAMPLES, tokenizer, seed=SEED, templates=VERBOSE_MATH_TEMPLATES
        )
        eval_dataset = create_single_turn_sft_dataset(
            NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100, templates=VERBOSE_MATH_TEMPLATES
        )
        log(f"Train dataset: {len(train_dataset)} samples")
        log(f"Eval dataset: {len(eval_dataset)} samples")

        if rank == 0:
            sample = train_dataset[0]
            log(f"\nSample text (truncated): {sample['text'][:200]}...")

        # --- Load model with EP ---
        log("\n--- Loading GLM4 MoE model with Expert Parallelism ---")
        log(f"GPU memory before model load: {gpu_mem_gb():.1f}GB")

        parallelism_config = ParallelismConfig(ep_size=ep_size)

        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            use_liger_kernel=True,
        )

        log(f"Model loaded: {model.config.model_type}")
        log(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        log(f"GPU memory after model load: {gpu_mem_gb():.1f}GB")

        # Count EP-patched MoE layers
        ep_layers = sum(1 for m in model.modules() if hasattr(m, "ep_config"))
        log(f"EP MoE layers detected: {ep_layers}")

        # --- Create SFT config ---
        log("\n--- Creating SFT config ---")
        config = SFTConfig(
            output_dir=output_dir,
            max_steps=NUM_TRAIN_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            learning_rate=LEARNING_RATE,
            warmup_steps=1,
            max_length=MAX_SEQ_LENGTH,
            bf16=True,
            gradient_checkpointing=True,
            # FLCE-only model: Liger is already applied by load_distributed_model and its forward
            # returns no logits, so keep the flag on — else TRL's metric path slices the None logits.
            use_liger_kernel=True,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=NUM_TRAIN_STEPS,
            save_strategy="no",
            report_to=[],
            dataloader_num_workers=0,
            ddp_find_unused_parameters=True,  # Required for EP (inactive experts)
            dataloader_drop_last=True,
            remove_unused_columns=False,
            fsdp="",  # Mixin handles FSDP wrapping
        )

        # --- Create trainer ---
        log("\n--- Creating DistributedSFTTrainer ---")
        trainer = DistributedSFTTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )
        log("Trainer created successfully")

        # --- Initial evaluation ---
        log("\n--- Running initial evaluation ---")
        barrier()
        eval_results = trainer.evaluate()
        initial_loss = eval_results.get("eval_loss", float("inf"))
        log(f"Initial eval loss: {initial_loss:.4f}")

        # --- Run training ---
        log("\n--- Starting training ---")
        log(f"GPU memory before training: {gpu_mem_gb():.1f}GB")
        barrier()

        train_result = trainer.train()

        log("\n--- Training completed ---")
        log(f"Training loss: {train_result.training_loss:.6f}")
        log(f"Training steps: {train_result.global_step}")
        log(f"GPU memory after training: {gpu_mem_gb():.1f}GB")

        # --- Final evaluation ---
        log("\n--- Running final evaluation ---")
        barrier()
        final_eval = trainer.evaluate()
        final_loss = final_eval.get("eval_loss", float("inf"))
        log(f"Final eval loss: {final_loss:.4f}")
        log(f"Loss: {initial_loss:.4f} -> {final_loss:.4f}")

        # --- Collect and validate metrics ---
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

        # Validation checks
        success = True
        issues = []

        # Check 1: Training loss is finite
        if not torch.isfinite(torch.tensor(train_result.training_loss)):
            issues.append(f"Training loss is not finite: {train_result.training_loss}")
            success = False

        # Check 2: No NaN/Inf in step losses
        has_nan = any(not torch.isfinite(torch.tensor(l)) for l in step_losses)
        if has_nan:
            issues.append("NaN/Inf detected in step losses")
            success = False

        # Check 3: No NaN/Inf in gradient norms
        has_bad_grad = any(not torch.isfinite(torch.tensor(g)) for g in grad_norms) if grad_norms else False
        if has_bad_grad:
            issues.append("NaN/Inf detected in gradient norms")
            success = False

        # Check 4: Training completed expected number of steps
        if train_result.global_step != NUM_TRAIN_STEPS:
            issues.append(f"Expected {NUM_TRAIN_STEPS} steps, got {train_result.global_step}")
            success = False

        # Check 5: Loss is not unreasonably high
        if train_result.training_loss > 100:
            issues.append(f"Training loss unreasonably high: {train_result.training_loss}")
            success = False

        # Check 6: Final eval loss is finite
        if not torch.isfinite(torch.tensor(final_loss)):
            issues.append(f"Final eval loss is not finite: {final_loss}")
            success = False

        # --- Report ---
        if success:
            log(f"\n{'=' * 70}")
            log("SFT GLM4 MoE EP TEST PASSED")
            log(f"  - Model: {MODEL_NAME}")
            log(f"  - EP size: {ep_size}")
            log(f"  - EP MoE layers: {ep_layers}")
            log(f"  - Training loss: {train_result.training_loss:.6f}")
            log(f"  - Eval loss: {initial_loss:.4f} -> {final_loss:.4f}")
            log(f"  - Steps completed: {train_result.global_step}")
            log("  - All losses finite")
            log(f"{'=' * 70}")
        else:
            log(f"\n{'=' * 70}")
            log("SFT GLM4 MoE EP TEST FAILED")
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
        if "trainer" in locals():
            trainer.cleanup_ep()
        barrier()
        cleanup_dirs(output_dir, cache_dir)
        cleanup_memory()


def main():
    success = run_sft_glm4_moe_test()

    teardown_distributed()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
