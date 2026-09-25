#!/usr/bin/env python
"""
Test script for SMPO trainer with EP+CP orthogonal (EP=2, CP=2) on GptOss-20B MoE.

Smoke test: runs SmoothMarginPOTrainer with expert parallelism and context parallelism
enabled simultaneously. EP distributes MoE experts across GPUs while CP splits sequences via
Ulysses attention.

Test validates:
1. EP+CP model loading via load_distributed_model
2. SMPO training completes without errors under combined parallelism
3. Loss values and logged gradient norms are finite (no NaN/Inf); every step runs
4. Sequence lengths are compatible with cp_size=2

It does not compare the SMPO loss or gradients against a non-parallel reference.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/preference/test_smpo_ep_cp.py

Requirements:
    - 2x GPUs with >=80GB memory each
    - DeepEP installed
    - Model: unsloth/gpt-oss-20b-BF16 (auto-downloaded)

Note:
    SMPO supports CP. When CP is enabled, sequences are split across CP ranks
    using Ulysses attention. The max_length and max_prompt_length must be
    divisible by cp_size=2.
"""

import torch
from transformers import AutoTokenizer

from src.configs.smpo_config import SmoothMarginPOConfig
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.preference.smpo import SmoothMarginPOTrainer
from tests.common.datasets import create_preference_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.utils import gpu_mem_gb, log

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 2
CP_SIZE = 2
NUM_TRAIN_SAMPLES = 64
NUM_EVAL_SAMPLES = 16
# both lengths must be divisible by cp_size
MAX_LENGTH = 4096
MAX_PROMPT_LENGTH = 2048
NUM_TRAIN_STEPS = 5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 5e-6
SEED = 42


def run(ctx) -> dict:
    """Run SMPO trainer test with EP+CP (EP=2, CP=2)."""
    log(f"\n{'=' * 70}")
    log("SMPO EP+CP TEST: EP=2 + CP=2 with GptOss-20B")
    log(f"{'=' * 70}")
    log(f"World size: {ctx.world_size}")
    log(f"EP size: {EP_SIZE}, CP size: {CP_SIZE}")
    log(f"Model: {MODEL_NAME}")
    log(f"Train samples: {NUM_TRAIN_SAMPLES}, Eval samples: {NUM_EVAL_SAMPLES}")
    log(f"Max length: {MAX_LENGTH} (divisible by cp_size={CP_SIZE})")
    log(f"Max prompt length: {MAX_PROMPT_LENGTH} (divisible by cp_size={CP_SIZE})")
    log(f"Batch size: {BATCH_SIZE}, Grad accum: {GRADIENT_ACCUMULATION}")
    log(f"Training steps: {NUM_TRAIN_STEPS}")
    log(f"GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"GPU memory: {torch.cuda.get_device_properties(ctx.local_rank).total_memory / 1e9:.1f}GB")

    assert MAX_LENGTH % CP_SIZE == 0, f"max_length={MAX_LENGTH} must be divisible by cp_size={CP_SIZE}"
    assert MAX_PROMPT_LENGTH % CP_SIZE == 0, (
        f"max_prompt_length={MAX_PROMPT_LENGTH} must be divisible by cp_size={CP_SIZE}"
    )

    log(f"Output directory: {ctx.output_dir}")
    log(f"Cache directory: {ctx.cache_dir}")

    log("\n--- Ensuring model is downloaded ---")
    ensure_model_downloaded(MODEL_NAME, ctx.rank)

    log("\n--- Loading tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

    log("\n--- Creating synthetic preference datasets ---")
    train_dataset = create_preference_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_preference_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)
    log(f"Train dataset: {len(train_dataset)} samples")
    log(f"Eval dataset: {len(eval_dataset)} samples")

    if ctx.rank == 0:
        sample = train_dataset[0]
        log(f"\nSample prompt (truncated): {sample['prompt'][:120]}...")
        log(f"Sample chosen (truncated): {sample['chosen'][:120]}...")
        log(f"Sample rejected (truncated): {sample['rejected'][:120]}...")

    log("\n--- Loading model with EP+CP ---")
    log(f"GPU memory before model load: {gpu_mem_gb():.1f}GB")

    parallelism_config = ParallelismConfig(ep_size=EP_SIZE, cp_size=CP_SIZE)

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
        output_dir=ctx.output_dir,
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

    log("\n--- Creating SmoothMarginPOTrainer with EP+CP ---")
    trainer = SmoothMarginPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    ctx.on_teardown(trainer.cleanup_ep)
    log("Trainer created successfully with EP+CP")

    log("\n--- Starting training ---")
    log(f"GPU memory before training: {gpu_mem_gb():.1f}GB")
    ctx.barrier()

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

    checks = {
        "loss_finite": bool(torch.isfinite(torch.tensor(train_result.training_loss))),
        "step_losses_finite": all(torch.isfinite(torch.tensor(l)) for l in step_losses),
        "grad_norms_finite": bool(grad_norms) and all(torch.isfinite(torch.tensor(g)) for g in grad_norms),
        "steps_completed": train_result.global_step == NUM_TRAIN_STEPS,
    }
    return {"checks": checks}


main = gpu_test_main(exact_world_size=EP_SIZE, prefix="smpo_ep_cp")(run)

if __name__ == "__main__":
    main()
