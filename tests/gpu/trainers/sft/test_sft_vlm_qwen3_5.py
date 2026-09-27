#!/usr/bin/env python
"""
VLM SFT Training Test for Qwen3.5-4B (natively multimodal dense model).

Qwen3.5 models are natively multimodal (Image-Text-to-Text) with a unified
vision-language architecture, unlike Qwen3 which has separate "-VL" variants.

Architecture:
  - Vision encoder: 24-layer ViT (1024 hidden, patch_size=16)
  - Language model: 32 decoder layers with hybrid attention
    - 24 linear attention layers (Qwen3_5GatedDeltaNet)
    - 8 full attention layers (Qwen3_5Attention)
  - Dense MLP (no MoE)
  - Double-width q_proj (query + sigmoid gate)

Note: Flash Attention 2 is incompatible with Qwen3.5's M-RoPE varlen path
(crashes with cudaErrorIllegalAddress). Use attn_implementation="sdpa".

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_vlm_qwen3_5.py

Requirements:
    - 2x GPUs (tested on B200)
    - causal-conv1d and flash-linear-attention installed (for GatedDeltaNet kernels)
    - Model: Qwen/Qwen3.5-4B (auto-downloaded)
"""

import math

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from trl import SFTConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_5_VLM_4B
from tests.common.utils import gpu_mem_gb, log, step_losses

MODEL_NAME = QWEN3_5_VLM_4B
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
MAX_SEQ_LENGTH = 2048
NUM_TRAIN_STEPS = 5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 2e-5
SEED = 42


@gpu_test_main(min_world_size=1, prefix="test_sft_vlm_qwen3_5")
def run(ctx):
    log(f"\n{'=' * 70}")
    log("VLM SFT Training Test: Qwen3.5-4B (natively multimodal)")
    log(f"{'=' * 70}")
    log(f"World size: {ctx.world_size}")
    log(f"Model: {MODEL_NAME}")
    log(f"GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"GPU memory: {torch.cuda.get_device_properties(ctx.local_rank).total_memory / 1e9:.1f}GB")

    log("\n--- Ensuring model is downloaded ---")
    ensure_model_downloaded(MODEL_NAME, ctx.rank)

    log("\n--- Loading processor and tokenizer ---")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = (
        processor.tokenizer
        if hasattr(processor, "tokenizer")
        else AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"Processor type: {type(processor).__name__}")
    log(f"Tokenizer: vocab_size={tokenizer.vocab_size}")

    # Text-only samples still exercise the VLM code path.
    log("\n--- Creating datasets ---")
    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)
    log(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    log("\n--- Loading VLM model ---")
    log(f"GPU memory before load: {gpu_mem_gb():.1f}GB")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",  # FA2 crashes on Qwen3.5's M-RoPE varlen path
    )

    param_count = sum(p.numel() for p in model.parameters())
    log(f"Model loaded: {type(model).__name__}")
    log(f"Parameters: {param_count / 1e9:.2f}B")
    log(f"GPU memory after load: {gpu_mem_gb():.1f}GB")

    text_model = getattr(model, "model", model)
    if hasattr(text_model, "layers"):
        layers = text_model.layers
    elif hasattr(text_model, "text_model") and hasattr(text_model.text_model, "layers"):
        layers = text_model.text_model.layers
    else:
        layers = []

    layer_types = {}
    for layer in layers:
        attn = getattr(layer, "self_attn", None) or getattr(layer, "linear_attn", None)
        cls_name = type(attn).__name__ if attn else "unknown"
        layer_types[cls_name] = layer_types.get(cls_name, 0) + 1
    log(f"Layer types: {layer_types}")

    log("\n--- Configuring trainer ---")
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
        use_liger_kernel=False,  # Liger has no qwen3_5 support
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=NUM_TRAIN_STEPS,
        save_strategy="no",
        report_to=[],
        logging_nan_inf_filter=False,
        dataloader_num_workers=0,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        fsdp="",  # Mixin handles FSDP wrapping
        ddp_find_unused_parameters=True,  # Vision encoder unused with text-only data
    )

    parallelism_config = ParallelismConfig()

    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    log(f"Trainer created: {type(trainer).__name__}")

    log("\n--- Running initial evaluation ---")
    eval_results = trainer.evaluate()
    initial_loss = eval_results.get("eval_loss", float("inf"))
    log(f"Initial eval loss: {initial_loss:.4f}")

    log("\n--- Starting training ---")
    train_result = trainer.train()

    log("\n--- Training completed ---")
    log(f"Training loss: {train_result.training_loss:.6f}")
    log(f"Steps completed: {train_result.global_step}")
    log(f"GPU memory after training: {gpu_mem_gb():.1f}GB")

    log("\n--- Running final evaluation ---")
    final_eval = trainer.evaluate()
    final_loss = final_eval.get("eval_loss", float("inf"))
    log(f"Loss: {initial_loss:.4f} -> {final_loss:.4f}")

    log_history = trainer.state.log_history
    losses = step_losses(trainer)
    grad_norms = [e["grad_norm"] for e in log_history if "grad_norm" in e]
    log(f"Per-step losses: {[f'{l:.4f}' for l in losses]}")
    if grad_norms:
        log(f"Per-step grad norms: {[f'{g:.2f}' for g in grad_norms]}")

    checks = {
        "train_loss_finite": math.isfinite(train_result.training_loss),
        "step_losses_finite": all(math.isfinite(l) for l in losses),
        "trained_all_steps": train_result.global_step == NUM_TRAIN_STEPS,
        "final_eval_loss_finite": math.isfinite(final_loss),
    }
    if grad_norms:
        checks["grad_norms_finite"] = all(math.isfinite(g) for g in grad_norms)
    return {"checks": checks}


if __name__ == "__main__":
    run()
