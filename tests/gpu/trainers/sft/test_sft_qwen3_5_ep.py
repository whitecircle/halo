#!/usr/bin/env python
"""
Focused SFT test for Qwen3.5/Qwen3.6 MoE with Expert Parallelism.

Single mode (EP only), env-configurable model path and EP size, 4-GPU
friendly. Smoke-tests the local Qwen3.6-35B-A3B checkpoint produced by patching.

Note: Qwen3.5/3.6 attention uses M-RoPE whose varlen path crashes Flash
Attention 2 (cudaErrorIllegalAddress). Use attn_implementation=sdpa.

Usage:
    torchrun --nproc_per_node=4 \\
        tests/gpu/trainers/sft/test_sft_qwen3_5_ep.py

Requirements:
    - 2-8 B200/B300 GPUs
    - DeepEP installed
    - Local checkpoint at /mnt/models/Qwen3.6-35B-A3B-patched
      (override via HALO_TEST_QWEN3_5_MODEL env var; falls back to HF Hub name)
"""

import sys
import traceback

import torch
from accelerate import PartialState
from transformers import AutoTokenizer
from trl import SFTConfig

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
from tests.common.models import QWEN3_5_MOE_35B
from tests.common.utils import cleanup_memory, gpu_mem_gb, log

# Test Configuration

MODEL_NAME = env_str("HALO_TEST_QWEN3_5_MODEL", QWEN3_5_MOE_35B)
EP_SIZE_OVERRIDE = env_int("HALO_TEST_EP", None)
NUM_TRAIN_SAMPLES = 16
NUM_EVAL_SAMPLES = 4
MAX_SEQ_LENGTH = 1024
NUM_TRAIN_STEPS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 1e-5
SEED = 42


def run_test():
    rank, world_size, local_rank = init_distributed()
    ep_size = EP_SIZE_OVERRIDE if EP_SIZE_OVERRIDE is not None else world_size

    log(f"\n{'=' * 70}")
    log(f"SFT QWEN3.5 MoE TEST: EP={ep_size} with {MODEL_NAME} (world_size={world_size})")
    log(f"{'=' * 70}")
    log(f"GPU: {torch.cuda.get_device_name(local_rank)}")

    if world_size % ep_size != 0:
        log(f"\nERROR: world_size={world_size} must be divisible by ep_size={ep_size}")
        teardown_distributed()
        return False

    output_dir, cache_dir = setup_cache_dirs("sft_qwen3_5_ep_test", rank)

    trainer = None
    try:
        ensure_model_downloaded(MODEL_NAME, rank)
        PartialState()

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        log(f"Tokenizer: {tokenizer.__class__.__name__}")

        train_dataset = create_single_turn_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
        eval_dataset = create_single_turn_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)
        log(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

        log(f"\n--- Loading Qwen3.5 with EP={ep_size} ---")
        log(f"GPU memory before load: {gpu_mem_gb():.1f}GB")

        parallelism_config = ParallelismConfig(ep_size=ep_size)

        # FA2 crashes on Qwen3.5 M-RoPE varlen path; use sdpa.
        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
            use_liger_kernel=True,
        )
        log(f"Model loaded: {model.config.model_type} ({type(model).__name__})")
        log(f"GPU memory after load: {gpu_mem_gb():.1f}GB")

        ep_layers = sum(1 for m in model.modules() if hasattr(m, "ep_config"))
        log(f"EP MoE layers: {ep_layers}")

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
            use_liger_kernel=False,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=NUM_TRAIN_STEPS,
            save_strategy="no",
            report_to=[],
            dataloader_num_workers=0,
            ddp_find_unused_parameters=True,
            dataloader_drop_last=True,
            remove_unused_columns=False,
            fsdp="",
        )

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
        barrier()
        train_result = trainer.train()
        log(f"Training loss: {train_result.training_loss:.6f}")
        log(f"Steps: {train_result.global_step}")
        log(f"GPU memory after training: {gpu_mem_gb():.1f}GB")

        log("\n--- Final evaluation ---")
        barrier()
        final_eval = trainer.evaluate()
        final_loss = final_eval.get("eval_loss", float("inf"))
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
        if not torch.isfinite(torch.tensor(final_loss)):
            issues.append(f"Final eval loss not finite: {final_loss}")
            success = False

        if success:
            log(f"\n{'=' * 70}")
            log("SFT QWEN3.5 MoE EP TEST PASSED")
            log(f"  - Model: {MODEL_NAME}")
            log(f"  - EP size: {ep_size}, EP layers: {ep_layers}")
            log(f"  - Training loss: {train_result.training_loss:.6f}")
            log(f"  - Eval loss: {initial_loss:.4f} -> {final_loss:.4f}")
            log(f"{'=' * 70}")
        else:
            log(f"\n{'=' * 70}")
            log("SFT QWEN3.5 MoE EP TEST FAILED")
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
    success = run_test()
    teardown_distributed()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
