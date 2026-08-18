#!/usr/bin/env python
"""
SMPO with Tensor Parallelism (TP=2) Training Test.

Validates that SmoothMarginPOTrainer works correctly with Tensor Parallelism
enabled (tp_size=2). TP shards attention/embedding/lm_head weights across GPUs
using DTensor, requiring FSDP2 for data parallel gradient sync when DP > 1.

Test validates:
1. Training completes without errors for 10 steps
2. Final training loss is finite (no NaN/Inf)
3. trainer.is_tp_mode is True (TP properly configured)

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/preference/test_smpo_tp.py

Requirements:
    - 2x GPUs
    - Model: Qwen/Qwen3-0.6B (auto-downloaded)
"""

import torch
from transformers import AutoTokenizer

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
TP_SIZE = 2
NUM_TRAIN_STEPS = 10
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 5e-6
MAX_LENGTH = 4096
MAX_PROMPT_LENGTH = 2048
NUM_TRAIN_SAMPLES = 128
NUM_EVAL_SAMPLES = 16
SEED = 42


# Main Test


def run(ctx) -> dict:
    """Run SMPO + TP=2 training test."""
    log(f"\n{'#' * 70}")
    log(f"  SMPO + TP={TP_SIZE} Training Test")
    log(f"  World size: {ctx.world_size}, TP size: {TP_SIZE}")
    log(f"  Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"{'#' * 70}")
    log(f"Output dir: {ctx.output_dir}")

    # ---- Load Tokenizer ----
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # ---- Create Datasets ----
    log("Creating synthetic preference datasets...")
    train_dataset = create_preference_dataset(
        NUM_TRAIN_SAMPLES,
        tokenizer,
        seed=SEED,
    )
    eval_dataset = create_preference_dataset(
        NUM_EVAL_SAMPLES,
        tokenizer,
        seed=SEED + 1000,
    )
    log(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    # ---- Parallelism Config ----
    parallelism_config = ParallelismConfig(tp_size=TP_SIZE)
    log(f"Parallelism config: {parallelism_config.summary()}")

    # ---- Load Model with TP ----
    log("Loading model with Tensor Parallelism...")
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        use_liger_kernel=True,
        # Pin SDPA for padded preference. SMPO's concatenated chosen+rejected batch carries a
        # padding mask, which routes FA4 (flash_attn.cute) through its varlen kernel —
        # pathologically slow to compile on this stack (~700s/step). FA4 dense is fast
        # everywhere; for padded preference SDPA handles the mask natively with no compile
        # storm (preferred over FA2).
        attn_implementation="sdpa",
    )
    log(f"Model loaded. Type: {type(model).__name__}")

    # ---- Training Config ----
    log("Creating SMPO training config...")
    training_config = SmoothMarginPOConfig(
        output_dir=ctx.output_dir,
        max_steps=NUM_TRAIN_STEPS,
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

    # ---- Create Trainer ----
    log("Creating SmoothMarginPOTrainer with TP...")
    trainer = SmoothMarginPOTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )

    # ---- Check 1: TP mode is active ----
    checks = {"tp_mode_active": bool(trainer.is_tp_mode)}

    # ---- Train ----
    log(f"\nStarting training for {NUM_TRAIN_STEPS} steps...")
    train_result = trainer.train()
    log("Training complete!")

    # ---- Check 2: Training completed with a loss reported ----
    final_loss = train_result.metrics.get("train_loss", None)
    log(f"Final training loss: {final_loss}")
    checks["train_loss_reported"] = final_loss is not None

    # ---- Check 3: Loss is finite ----
    checks["loss_finite"] = final_loss is not None and bool(torch.isfinite(torch.tensor(final_loss)))

    log(f"\n{'=' * 70}")
    log(f"  SMPO + TP={TP_SIZE} training test")
    log(f"  Final loss: {final_loss}")
    log(f"  Steps completed: {trainer.state.global_step}")
    log(f"{'=' * 70}")

    return {"checks": checks}


main = gpu_test_main(min_world_size=TP_SIZE, prefix="smpo_tp")(run)

if __name__ == "__main__":
    main()
