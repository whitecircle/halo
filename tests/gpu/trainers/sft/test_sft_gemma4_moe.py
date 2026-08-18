#!/usr/bin/env python
"""
Test script for DistributedSFTTrainer with Gemma4 MoE model and EP=2.

Validates that SFT training works correctly with Gemma4-26B-A4B (a VLM-wrapped
MoE architecture: Gemma4ForConditionalGeneration → Gemma4Model → Gemma4TextModel
→ Gemma4TextDecoderLayer with inlined router/experts blocks).

Test validates:
1. EP model loading for Gemma4 (Gemma4TextExperts → EPGemma4MoELayer)
2. SFT training completes without errors on text-only inputs
3. Loss values are finite (no NaN/Inf)
4. Vision/audio submodules sit unused but do not break the forward path

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_gemma4_moe.py

Requirements:
    - 2x B200/B300 GPUs (>=80GB)
    - DeepEP installed
    - Local checkpoint at /mnt/models/gemma-4-26B-A4B-it-patched
      (override via HALO_TEST_GEMMA4_MODEL env var)
"""

import os
import sys
import traceback

import torch
from accelerate import PartialState
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.expert_parallel.layers.gemma4 import EPGemma4MoELayer
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.env import env_int, env_str
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_single_turn_sft_dataset
from tests.common.distributed import (
    cleanup_dirs,
    ensure_model_downloaded,
    init_distributed,
    setup_cache_dirs,
    teardown_distributed,
)
from tests.common.models import GEMMA4_26B_A4B_PATCHED
from tests.common.utils import cleanup_memory, gpu_mem_gb, log

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


def run_sft_gemma4_moe_test():
    rank, world_size, local_rank = init_distributed()
    ep_size = EP_SIZE_OVERRIDE if EP_SIZE_OVERRIDE is not None else world_size

    log(f"\n{'=' * 70}")
    log(f"SFT GEMMA4 MoE TEST: EP={ep_size} with Gemma4-26B-A4B (world_size={world_size})")
    log(f"{'=' * 70}")
    log(f"World size: {world_size}")
    log(f"EP size: {ep_size}")
    log(f"Model path: {MODEL_NAME}")
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

    if not os.path.isdir(MODEL_NAME):
        log(f"SKIP: local model path missing: {MODEL_NAME} (set HALO_TEST_GEMMA4_MODEL to a present checkpoint)")
        teardown_distributed()
        return True  # skip, not fail — local-checkpoint-dependent test

    output_dir, cache_dir = setup_cache_dirs("sft_gemma4_moe_test", rank)
    log(f"Output directory: {output_dir}")
    log(f"Cache directory: {cache_dir}")

    trainer = None
    try:
        # Local path: nothing to download, but still load config/tokenizer rank-0 first.
        log("\n--- Loading config/tokenizer (rank-0 first) ---")
        ensure_model_downloaded(MODEL_NAME, rank)
        PartialState()

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        log(f"Tokenizer: {tokenizer.__class__.__name__}")

        log("\n--- Creating synthetic SFT datasets ---")
        train_dataset = create_single_turn_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
        eval_dataset = create_single_turn_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)
        log(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")
        if rank == 0:
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
        if not ep_layers:
            log("ERROR: No EPGemma4MoELayer modules found — EP patching missed Gemma4TextExperts.")
            return False

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
            output_dir=output_dir,
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

        success = True
        issues = []
        if not torch.isfinite(torch.tensor(train_result.training_loss)):
            issues.append(f"Training loss not finite: {train_result.training_loss}")
            success = False
        if any(not torch.isfinite(torch.tensor(l)) for l in step_losses):
            issues.append("NaN/Inf in step losses")
            success = False
        if grad_norms and any(not torch.isfinite(torch.tensor(g)) for g in grad_norms):
            issues.append("NaN/Inf in grad norms")
            success = False
        if train_result.global_step != NUM_TRAIN_STEPS:
            issues.append(f"Expected {NUM_TRAIN_STEPS} steps, got {train_result.global_step}")
            success = False
        if train_result.training_loss > 100:
            issues.append(f"Training loss unreasonably high: {train_result.training_loss}")
            success = False
        if not torch.isfinite(torch.tensor(final_loss)):
            issues.append(f"Final eval loss not finite: {final_loss}")
            success = False

        if success:
            log(f"\n{'=' * 70}")
            log("SFT GEMMA4 MoE EP TEST PASSED")
            log(f"  - Model: {MODEL_NAME}")
            log(f"  - EP size: {ep_size}, EP layers: {len(ep_layers)}")
            log(f"  - Training loss: {train_result.training_loss:.6f}")
            log(f"  - Eval loss: {initial_loss:.4f} -> {final_loss:.4f}")
            log(f"  - All losses finite, {train_result.global_step} steps")
            log(f"{'=' * 70}")
        else:
            log(f"\n{'=' * 70}")
            log("SFT GEMMA4 MoE EP TEST FAILED")
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
        if trainer is not None:
            trainer.cleanup_ep()
        barrier()
        cleanup_dirs(output_dir, cache_dir)
        cleanup_memory()


def main():
    success = run_sft_gemma4_moe_test()
    teardown_distributed()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
