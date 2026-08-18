#!/usr/bin/env python
"""
SMPO Trainer test with standard FSDP (no EP/CP/TP).

Validates that SmoothMarginPOTrainer works correctly in standard FSDP mode
on Qwen3-0.6B with:
- Gradient checkpointing
- Liger kernels (fused RMSNorm + cross-entropy)
- BF16 mixed precision
- Synthetic preference data with math Q&A (correct vs incorrect answers)
- 30% multi-turn prompts

Test Phases:
1. Synthetic preference dataset creation (prompt/chosen/rejected)
2. SMPO training for 10 steps
3. Validation: loss is finite

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/preference/test_smpo_fsdp.py
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


def run(ctx):
    log(f"\n{'=' * 70}")
    log("  SMPO FSDP Training Test (Standard FSDP, no EP/CP/TP)")
    log(f"  World size: {ctx.world_size}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  Max steps: {MAX_STEPS}, Batch size: {BATCH_SIZE}")
    log(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    log(f"  Max length: {MAX_LENGTH}, Max prompt length: {MAX_PROMPT_LENGTH}")
    log(f"{'=' * 70}")

    # ── Load model ──────────────────────────────────────────────────
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

    # ── Create datasets ─────────────────────────────────────────────
    log("\n[2/4] Creating synthetic preference datasets...")
    train_dataset = create_preference_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_preference_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1)
    log(f"  Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    # Log a sample
    if ctx.rank == 0:
        sample = train_dataset[0]
        log(f"  Sample prompt (truncated): {sample['prompt'][:150]}...")
        log(f"  Sample chosen: {sample['chosen']}")
        log(f"  Sample rejected: {sample['rejected']}")

    # Count multi-turn samples
    multi_turn_count = sum(1 for s in train_dataset if s["prompt"].count("user") > 1)
    log(
        f"  Multi-turn samples: {multi_turn_count}/{len(train_dataset)} "
        f"({100 * multi_turn_count / len(train_dataset):.0f}%)"
    )

    # ── Configure trainer ───────────────────────────────────────────
    log("\n[3/4] Configuring SMPO trainer...")
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
    log(f"  Loss type: {config.loss_type}, Beta: {config.beta}")
    log(f"  Target margin: {config.target_margin}")

    # ── Train ───────────────────────────────────────────────────────
    log("\n[4/4] Training...")
    train_result = trainer.train()

    # ── Collect metrics ─────────────────────────────────────────────
    training_loss = train_result.training_loss
    log_history = trainer.state.log_history
    step_losses = [entry["loss"] for entry in log_history if "loss" in entry and "eval_loss" not in entry]

    log("\n  --- Training Results ---")
    log(f"  Final training loss: {training_loss:.6f}")
    log(f"  Per-step losses: {[f'{l:.4f}' for l in step_losses]}")

    # Log SMPO-specific metrics if available
    margin_metrics = [entry.get("margin", None) for entry in log_history if "margin" in entry]
    if margin_metrics:
        log(f"  Margins: {[f'{m:.4f}' for m in margin_metrics if m is not None]}")

    # ── Assertions ──────────────────────────────────────────────────
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

    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="smpo_fsdp")(run)

if __name__ == "__main__":
    main()
