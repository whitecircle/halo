#!/usr/bin/env python
"""
Test script for DistributedSFTTrainer with Gemma4 MoE model and EP=2.

Smoke test: SFT training runs on Gemma4-26B-A4B (a VLM-wrapped MoE architecture:
Gemma4ForConditionalGeneration → Gemma4Model → Gemma4TextModel → Gemma4TextDecoderLayer with
inlined router/experts blocks) with EP.

Checks:
1. EP model loading wraps Gemma4TextExperts as EPGemma4MoELayer
2. SFT training completes every step on text-only inputs, with the vision/audio submodules unused
3. Train, step, grad-norm and final eval losses are finite (no NaN/Inf)

No loss is compared against a reference.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_gemma4_moe.py

Requirements:
    - 2x B200/B300 GPUs (>=80GB)
    - DeepEP installed
    - Local checkpoint at $HALO_DATA_ROOT/models/gemma-4-26B-A4B-it-patched
      (override via HALO_TEST_GEMMA4_MODEL env var)
"""

import os
import sys

import torch
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.expert_parallel.layers.gemma4 import EPGemma4MoELayer
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.env import env_int, env_str
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_single_turn_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import GEMMA4_26B_A4B_PATCHED
from tests.common.utils import gpu_mem_gb, log

MODEL_NAME = env_str("HALO_TEST_GEMMA4_MODEL", GEMMA4_26B_A4B_PATCHED)
# EP size defaults to world_size (set inside main); HALO_TEST_EP forces a specific value.
EP_SIZE_OVERRIDE = env_int("HALO_TEST_EP", None)
NUM_TRAIN_SAMPLES = 16
NUM_EVAL_SAMPLES = 4
MAX_SEQ_LENGTH = 1024
NUM_TRAIN_STEPS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 1e-5
SEED = 42


@gpu_test_main(min_world_size=1, prefix="sft_gemma4_moe_test")
def run(ctx):
    ep_size = EP_SIZE_OVERRIDE if EP_SIZE_OVERRIDE is not None else ctx.world_size

    log(f"\n{'=' * 70}")
    log(f"SFT GEMMA4 MoE TEST: EP={ep_size} with Gemma4-26B-A4B (world_size={ctx.world_size})")
    log(f"{'=' * 70}")
    log(f"World size: {ctx.world_size}")
    log(f"EP size: {ep_size}")
    log(f"Model path: {MODEL_NAME}")
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

    # Local path: nothing to download, but still load config/tokenizer rank-0 first.
    log("\n--- Loading config/tokenizer (rank-0 first) ---")
    ensure_model_downloaded(MODEL_NAME, ctx.rank)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"Tokenizer: {tokenizer.__class__.__name__}")

    log("\n--- Creating synthetic SFT datasets ---")
    train_dataset = create_single_turn_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_single_turn_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)
    log(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")
    if ctx.rank == 0:
        log(f"\nSample (first 200 chars): {train_dataset[0]['text'][:200]}...")

    log("\n--- Loading Gemma4 with Expert Parallelism ---")
    log(f"GPU memory before load: {gpu_mem_gb():.1f}GB")

    parallelism_config = ParallelismConfig(ep_size=ep_size)

    # Gemma4 global heads use head_dim=512, past FA2's 256 limit; sdpa takes any head dim.
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        use_liger_kernel=False,  # Liger doesn't have Gemma4-specific kernels
    )

    log(f"Model loaded: {model.config.model_type} ({type(model).__name__})")
    log(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    log(f"GPU memory after load: {gpu_mem_gb():.1f}GB")

    ep_layers = [m for m in model.modules() if isinstance(m, EPGemma4MoELayer)]
    log(f"EPGemma4MoELayer instances: {len(ep_layers)}")
    checks = {"ep_layers_wrapped": bool(ep_layers)}
    if not ep_layers:
        log("ERROR: No EPGemma4MoELayer modules found — EP patching missed Gemma4TextExperts.")
        return {"checks": checks}

    first = ep_layers[0]
    log(
        f"  First EP layer: rank owns experts "
        f"[{first.expert_start}, {first.expert_end}) of {first.num_experts}; "
        f"hidden_dim={first.hidden_dim}, "
        f"gate_up_proj={tuple(first.gate_up_proj.shape)}, "
        f"down_proj={tuple(first.down_proj.shape)}"
    )

    log("\n--- Creating SFTConfig ---")
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
        gradient_checkpointing=False,
        use_liger_kernel=False,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=NUM_TRAIN_STEPS,
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        ddp_find_unused_parameters=True,  # Vision/audio params unused on text-only batches
        dataloader_drop_last=True,
        remove_unused_columns=False,
        fsdp="",
    )

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

    log("\n--- Initial evaluation ---")
    barrier()
    eval_results = trainer.evaluate()
    initial_loss = eval_results.get("eval_loss", float("inf"))
    log(f"Initial eval loss: {initial_loss:.4f}")

    log("\n--- Training ---")
    log(f"GPU memory before training: {gpu_mem_gb():.1f}GB")
    barrier()
    train_result = trainer.train()
    log(f"Training loss: {train_result.training_loss:.6f}")
    log(f"Steps: {train_result.global_step}")
    log(f"GPU memory after training: {gpu_mem_gb():.1f}GB")

    log("\n--- Final evaluation ---")
    barrier()
    final_eval = trainer.evaluate()
    final_loss = final_eval.get("eval_loss", float("inf"))
    log(f"Final eval loss: {final_loss:.4f}")
    log(f"Loss: {initial_loss:.4f} -> {final_loss:.4f}")

    step_losses = [e["loss"] for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]
    grad_norms = [e["grad_norm"] for e in trainer.state.log_history if "grad_norm" in e]
    log(f"Per-step losses: {[f'{l:.4f}' for l in step_losses]}")
    if grad_norms:
        log(f"Per-step grad norms: {[f'{g:.2f}' for g in grad_norms]}")

    checks["train_loss_finite"] = bool(torch.isfinite(torch.tensor(train_result.training_loss)))
    checks["step_losses_finite"] = all(torch.isfinite(torch.tensor(l)) for l in step_losses)
    if grad_norms:
        checks["grad_norms_finite"] = all(torch.isfinite(torch.tensor(g)) for g in grad_norms)
    checks["trained_all_steps"] = train_result.global_step == NUM_TRAIN_STEPS
    checks["final_eval_loss_finite"] = bool(torch.isfinite(torch.tensor(final_loss)))
    return {"checks": checks}


if __name__ == "__main__":
    # A local-checkpoint test declines to run before the harness starts: the launcher reports a
    # ``SKIP:`` line with exit 0 and no result line as a skip.
    if not os.path.isdir(MODEL_NAME):
        log(f"SKIP: local model path missing: {MODEL_NAME} (set HALO_TEST_GEMMA4_MODEL to a present checkpoint)")
        sys.exit(0)
    run()
