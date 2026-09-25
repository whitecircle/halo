#!/usr/bin/env python
"""
Test script for DistributedSFTTrainer with GLM-4.7-Flash MoE model and EP=2.

Validates that SFT training works correctly with the GLM-4.7-Flash (GLM4 MoE Lite)
model under expert parallelism. GLM-4.7-Flash is a Mixture-of-Experts model,
making it a good target for EP testing alongside Qwen3 and GptOss MoE models.

Test validates:
1. EP model loading for GLM4 MoE via load_distributed_model
2. SFT training completes every configured step
3. Loss values and gradient norms are finite (no NaN/Inf)

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_glm4_moe.py

Requirements:
    - 2x GPUs with >=80GB memory each
    - DeepEP installed
    - Model: zai-org/GLM-4.7-Flash (auto-downloaded)
"""

import torch
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.env import env_int, env_str
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import VERBOSE_MATH_TEMPLATES, create_single_turn_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import GLM4_FLASH
from tests.common.utils import gpu_mem_gb, log, step_losses

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


@gpu_test_main(min_world_size=1, prefix="sft_glm4_moe_test")
def run(ctx):
    """Run SFT trainer test with GLM-4.7-Flash MoE under EP."""
    ep_size = EP_SIZE_OVERRIDE if EP_SIZE_OVERRIDE is not None else ctx.world_size

    log(f"\n{'=' * 70}")
    log(f"SFT GLM4 MoE TEST: EP={ep_size} with GLM-4.7-Flash (world_size={ctx.world_size})")
    log(f"{'=' * 70}")
    log(f"World size: {ctx.world_size}")
    log(f"EP size: {ep_size}")
    log(f"Model: {MODEL_NAME}")
    log(f"Train samples: {NUM_TRAIN_SAMPLES}, Eval samples: {NUM_EVAL_SAMPLES}")
    log(f"Max seq length: {MAX_SEQ_LENGTH}")
    log(f"Batch size: {BATCH_SIZE}, Grad accum: {GRADIENT_ACCUMULATION}")
    log(f"Training steps: {NUM_TRAIN_STEPS}")
    log(f"GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"GPU memory: {torch.cuda.get_device_properties(ctx.local_rank).total_memory / 1e9:.1f}GB")

    if ctx.world_size % ep_size != 0:
        raise ValueError(f"world_size={ctx.world_size} must be divisible by ep_size={ep_size}")

    log(f"Output directory: {ctx.output_dir}")
    log(f"Cache directory: {ctx.cache_dir}")

    # --- Ensure model is cached (download on rank 0 first) ---
    log("\n--- Ensuring model is downloaded ---")
    ensure_model_downloaded(MODEL_NAME, ctx.rank)

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

    if ctx.rank == 0:
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
        output_dir=ctx.output_dir,
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
    ctx.on_teardown(trainer.cleanup_ep)
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
    losses = step_losses(trainer)
    grad_norms = [e["grad_norm"] for e in trainer.state.log_history if "grad_norm" in e]

    log(f"Per-step losses: {[f'{l:.4f}' for l in losses]}")
    if grad_norms:
        log(f"Per-step grad norms: {[f'{g:.2f}' for g in grad_norms]}")

    return {
        "checks": {
            "training_loss_finite": bool(torch.isfinite(torch.tensor(train_result.training_loss))),
            "step_losses_finite": all(torch.isfinite(torch.tensor(l)) for l in losses),
            "grad_norms_finite": all(torch.isfinite(torch.tensor(g)) for g in grad_norms),
            "completed_all_steps": train_result.global_step == NUM_TRAIN_STEPS,
            "ep_layers_wrapped": ep_layers > 0,
            "final_eval_loss_finite": bool(torch.isfinite(torch.tensor(final_loss))),
        }
    }


if __name__ == "__main__":
    run()
