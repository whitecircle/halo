#!/usr/bin/env python
"""
SMPO Trainer test with padding-free mode (Flash Attention 2).

Validates that SmoothMarginPOTrainer works correctly in padding-free mode
on Qwen3-0.6B with:
- Padding-free flattening (eliminates padding waste)
- Flash Attention 2 for document boundary handling
- Gradient checkpointing
- Liger kernels (fused RMSNorm + cross-entropy)
- BF16 mixed precision
- Synthetic preference data with math Q&A (correct vs incorrect answers)

Test Phases:
1. Synthetic preference dataset creation (prompt/chosen/rejected)
2. SMPO training for 10 steps with padding_free=True
3. Validation: loss is finite and reasonable

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/preference/test_smpo_padding_free.py
"""

import math

import torch

from src.configs.smpo_config import SmoothMarginPOConfig
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.preference.smpo import SmoothMarginPOTrainer
from tests.common.datasets import create_preference_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.utils import log

# Configuration

MODEL_NAME = QWEN3_0_6B
MAX_STEPS = 10
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 5e-6
MAX_LENGTH = 4096
MAX_PROMPT_LENGTH = 2048
NUM_TRAIN_SAMPLES = 128
NUM_EVAL_SAMPLES = 16
SEED = 42


# Main Test


def run(ctx):
    """Run SMPO padding-free training and validate the losses it produced."""
    log(f"\n{'=' * 70}")
    log("  SMPO Padding-Free Training Test")
    log(f"  World size: {ctx.world_size}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  Max steps: {MAX_STEPS}, Batch size: {BATCH_SIZE}")
    log(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    log(f"  Max length: {MAX_LENGTH}, Max prompt length: {MAX_PROMPT_LENGTH}")
    log("  padding_free: True")
    log(f"{'=' * 70}")

    # -- Load model --
    log("\n[1/4] Loading model via load_distributed_model...")
    parallelism_config = ParallelismConfig()

    model, tokenizer = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_liger_kernel=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    param_count = sum(p.numel() for p in model.parameters())
    log(f"  Model loaded: {param_count / 1e6:.1f}M parameters")
    log(f"  Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # -- Create datasets --
    log("\n[2/4] Creating synthetic preference datasets...")
    train_dataset = create_preference_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_preference_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1)
    log(f"  Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    # -- Configure trainer --
    log("\n[3/4] Configuring SMPO trainer with padding_free=True...")
    config = SmoothMarginPOConfig(
        output_dir=ctx.output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,  # Already applied in load_distributed_model
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        dataloader_drop_last=True,
        fsdp="",  # Mixin handles FSDP wrapping
        padding_free=True,
    )

    trainer = SmoothMarginPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    log(f"  Trainer created: {type(trainer).__name__}")
    log(f"  Parallelism: {parallelism_config.mode_string or 'Standard FSDP'}")
    log(f"  padding_free: {trainer.padding_free}")
    log(f"  Loss type: {config.loss_type}, Beta: {config.beta}")
    log(f"  Target margin: {config.target_margin}")

    # -- Train --
    log("\n[4/4] Training...")
    train_result = trainer.train()

    # -- Collect metrics --
    training_loss = train_result.training_loss
    log_history = trainer.state.log_history
    step_losses = [entry["loss"] for entry in log_history if "loss" in entry and "eval_loss" not in entry]

    log("\n  --- Training Results ---")
    log(f"  Final training loss: {training_loss:.6f}")
    log(f"  Per-step losses: {[f'{l:.4f}' for l in step_losses]}")

    # -- Assertions --
    log("\n  --- Assertions ---")
    checks = {}

    # Check 1: Training completed (loss is finite)
    loss_finite = math.isfinite(training_loss)
    checks["loss_finite"] = loss_finite
    log(f"  Loss is finite: {'PASS' if loss_finite else 'FAIL'} (loss={training_loss:.6f})")

    # Check 2: All step losses are finite (no NaN/Inf)
    all_finite = all(math.isfinite(l) for l in step_losses)
    checks["all_steps_finite"] = all_finite
    log(f"  All step losses finite: {'PASS' if all_finite else 'FAIL'}")

    # Check 3: Training loss is reasonable (not diverged)
    loss_reasonable = training_loss < 100.0
    checks["loss_reasonable"] = loss_reasonable
    log(f"  Loss reasonable (<100): {'PASS' if loss_reasonable else 'FAIL'}")

    return {"checks": checks, "metrics": {"final_train_loss": training_loss}}


main = gpu_test_main(min_world_size=1, prefix="smpo_padding_free")(run)

if __name__ == "__main__":
    main()
