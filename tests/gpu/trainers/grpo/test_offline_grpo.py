#!/usr/bin/env python
"""
Offline GRPO Trainer test with standard FSDP (no EP/CP/TP).

Validates that OfflineGRPOTrainer works correctly in standard FSDP mode
on Qwen3-0.6B with:
- Gradient checkpointing
- Liger kernels (fused RMSNorm + cross-entropy)
- BF16 mixed precision
- Synthetic offline GRPO data (4 completions per prompt with graded rewards)
- 30% multi-turn prompts

Test Phases:
1. Synthetic dataset creation (prompt/completions/rewards)
2. Offline GRPO training for 10 steps
3. Validation: loss is finite

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/grpo/test_offline_grpo.py
"""

import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.grpo.offline import OfflineGRPOTrainer
from tests.common.datasets import create_offline_grpo_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.utils import log

# Configuration

MODEL_NAME = QWEN3_0_6B
MAX_STEPS = 10
BATCH_SIZE = 1
LEARNING_RATE = 5e-6
MAX_PROMPT_LENGTH = 2048
MAX_COMPLETION_LENGTH = 2048
NUM_COMPLETIONS = 4
NUM_TRAIN_SAMPLES = 64
NUM_EVAL_SAMPLES = 16
SEED = 42
MULTI_TURN_RATIO = 0.3


# Main Test


def run(ctx) -> dict:
    """Run Offline GRPO test."""
    log(f"\n{'=' * 70}")
    log("  Offline GRPO Training Test (Standard FSDP, no EP/CP/TP)")
    log(f"  World size: {ctx.world_size}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  Max steps: {MAX_STEPS}, Batch size: {BATCH_SIZE}")
    log(f"  Completions per prompt: {NUM_COMPLETIONS}")
    log(f"  Max prompt length: {MAX_PROMPT_LENGTH}")
    log(f"  Max completion length: {MAX_COMPLETION_LENGTH}")
    log(f"{'=' * 70}")

    # ── Load tokenizer ──────────────────────────────────────────────
    log("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"  Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # ── Create datasets ─────────────────────────────────────────────
    log("\n[2/5] Creating synthetic offline GRPO datasets...")
    train_dataset = create_offline_grpo_dataset(
        tokenizer, NUM_TRAIN_SAMPLES, seed=SEED, num_completions=NUM_COMPLETIONS, multi_turn_ratio=MULTI_TURN_RATIO
    )
    eval_dataset = create_offline_grpo_dataset(
        tokenizer,
        NUM_EVAL_SAMPLES,
        seed=SEED + 1,
        num_completions=NUM_COMPLETIONS,
        multi_turn_ratio=MULTI_TURN_RATIO,
    )
    log(f"  Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    # Log a sample
    if ctx.rank == 0:
        sample = train_dataset[0]
        log(f"  Sample prompt (truncated): {sample['prompt'][:150]}...")
        log(f"  Number of completions: {len(sample['completions'])}")
        log(f"  Rewards: {sample['rewards']}")
        log(f"  Completion 0 (truncated): {sample['completions'][0][:100]}...")

    # Count multi-turn samples
    multi_turn_count = sum(1 for s in train_dataset if s["prompt"].count("user") > 1)
    log(
        f"  Multi-turn samples: {multi_turn_count}/{len(train_dataset)} "
        f"({100 * multi_turn_count / len(train_dataset):.0f}%)"
    )

    # ── Load model ──────────────────────────────────────────────────
    log("\n[3/5] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    param_count = sum(p.numel() for p in model.parameters())
    log(f"  Model loaded: {param_count / 1e6:.1f}M parameters")
    log(f"  Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ── Configure trainer ───────────────────────────────────────────
    log("\n[4/5] Configuring Offline GRPO trainer...")
    config = OfflineGRPOConfig(
        output_dir=ctx.output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        dataloader_drop_last=True,
        fsdp="",  # Mixin handles FSDP wrapping
    )

    parallelism_config = ParallelismConfig()

    trainer = OfflineGRPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    log(f"  Trainer created: {type(trainer).__name__}")
    log(f"  Parallelism: {parallelism_config.mode_string or 'Standard FSDP'}")
    log(f"  Loss type: {config.loss_type}")
    log(f"  Advantage method: {config.advantage_method}")
    log(f"  PG formulation: {config.policy_gradient_formulation}")

    # ── Train ───────────────────────────────────────────────────────
    log("\n[5/5] Training...")
    train_result = trainer.train()

    # ── Collect metrics ─────────────────────────────────────────────
    training_loss = train_result.training_loss
    log_history = trainer.state.log_history
    step_losses = [entry["loss"] for entry in log_history if "loss" in entry and "eval_loss" not in entry]

    log("\n  --- Training Results ---")
    log(f"  Final training loss: {training_loss:.6f}")
    log(f"  Per-step losses: {[f'{l:.4f}' for l in step_losses]}")

    # Log GRPO-specific metrics if available
    reward_metrics = [entry.get("rewards/mean", None) for entry in log_history if "rewards/mean" in entry]
    if reward_metrics:
        log(f"  Mean rewards: {[f'{r:.4f}' for r in reward_metrics if r is not None]}")

    advantage_metrics = [entry.get("advantages/mean", None) for entry in log_history if "advantages/mean" in entry]
    if advantage_metrics:
        log(f"  Mean advantages: {[f'{a:.4f}' for a in advantage_metrics if a is not None]}")

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


main = gpu_test_main(min_world_size=2, prefix="offline_grpo")(run)

if __name__ == "__main__":
    main()
