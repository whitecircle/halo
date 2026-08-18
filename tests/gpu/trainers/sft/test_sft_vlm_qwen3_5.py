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
import sys
import traceback

import torch
from accelerate import PartialState
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from trl import SFTConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import (
    cleanup_dirs,
    ensure_model_downloaded,
    init_distributed,
    setup_cache_dirs,
    teardown_distributed,
)
from tests.common.models import QWEN3_5_VLM_4B
from tests.common.utils import cleanup_memory, gpu_mem_gb, log

MODEL_NAME = QWEN3_5_VLM_4B
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
MAX_SEQ_LENGTH = 2048
NUM_TRAIN_STEPS = 5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 2e-5
SEED = 42


def main() -> int:
    """Run VLM SFT test for Qwen3.5-4B. Returns 0 on success, 1 on failure."""
    rank, world_size, local_rank = init_distributed()
    PartialState()

    output_dir, cache_dir = setup_cache_dirs("test_sft_vlm_qwen3_5", rank)

    log(f"\n{'=' * 70}")
    log("VLM SFT Training Test: Qwen3.5-4B (natively multimodal)")
    log(f"{'=' * 70}")
    log(f"World size: {world_size}")
    log(f"Model: {MODEL_NAME}")
    log(f"GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"GPU memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f}GB")

    success = False

    try:
        log("\n--- Ensuring model is downloaded ---")
        ensure_model_downloaded(MODEL_NAME, rank)

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
            use_liger_kernel=False,  # Liger has no qwen3_5 support
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=NUM_TRAIN_STEPS,
            save_strategy="no",
            report_to=[],
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
        step_losses = [e["loss"] for e in log_history if "loss" in e and "eval_loss" not in e]
        grad_norms = [e["grad_norm"] for e in log_history if "grad_norm" in e]
        log(f"Per-step losses: {[f'{l:.4f}' for l in step_losses]}")
        if grad_norms:
            log(f"Per-step grad norms: {[f'{g:.2f}' for g in grad_norms]}")

        issues = []

        if not math.isfinite(train_result.training_loss):
            issues.append(f"Training loss not finite: {train_result.training_loss}")

        if any(not math.isfinite(l) for l in step_losses):
            issues.append("NaN/Inf in step losses")

        if grad_norms and any(not math.isfinite(g) for g in grad_norms):
            issues.append("NaN/Inf in gradient norms")

        if train_result.global_step != NUM_TRAIN_STEPS:
            issues.append(f"Expected {NUM_TRAIN_STEPS} steps, got {train_result.global_step}")

        if train_result.training_loss > 100:
            issues.append(f"Training loss unreasonably high: {train_result.training_loss}")

        if not math.isfinite(final_loss):
            issues.append(f"Final eval loss not finite: {final_loss}")

        success = len(issues) == 0

        log(f"\n{'=' * 70}")
        if success:
            log("VLM SFT TEST PASSED: Qwen3.5-4B")
            log(f"  - Training loss: {train_result.training_loss:.6f}")
            log(f"  - Eval loss: {initial_loss:.4f} -> {final_loss:.4f}")
            log(f"  - Steps completed: {train_result.global_step}")
        else:
            log("VLM SFT TEST FAILED: Qwen3.5-4B")
            for issue in issues:
                log(f"  - {issue}")
        log(f"{'=' * 70}")

    except Exception as e:
        log(f"\nVLM SFT TEST FAILED WITH ERROR: {e}")
        if rank == 0:
            traceback.print_exc()
        success = False

    finally:
        cleanup_memory()
        cleanup_dirs(output_dir, cache_dir)
        teardown_distributed()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
