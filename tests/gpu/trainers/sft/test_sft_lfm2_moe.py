#!/usr/bin/env python
"""
Test script for SFT training with LiquidAI/LFM2-24B-A2B MoE model.

Validates two parallelism modes:
1. fsdp  — Default FSDP data parallelism (no EP), all experts on every GPU
2. ep    — EP=2, experts distributed across GPUs via DeepEP

LFM2-24B-A2B architecture:
  - 40 hidden layers (hybrid: conv + full attention)
  - MoE: 64 experts, top-k=4, sigmoid routing with expert bias
  - 2 dense layers (non-routed)
  - 2048 hidden size, 1536 MoE intermediate size
  - ~24B total params, ~2B active per token

Usage:
    # FSDP-only (no EP)
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_lfm2_moe.py --mode fsdp

    # EP=2
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_lfm2_moe.py --mode ep

Requirements:
    - 2x GPUs with >=80GB memory each
    - DeepEP installed (for EP mode)
    - Model: LiquidAI/LFM2-24B-A2B (auto-downloaded)
"""

import sys
import traceback

import torch
from accelerate import PartialState
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import (
    cleanup_dirs,
    ensure_model_downloaded,
    init_distributed,
    setup_cache_dirs,
    teardown_distributed,
)
from tests.common.models import LFM2_24B_A2B
from tests.common.utils import cleanup_memory, gpu_mem_gb, log

MODEL_NAME = LFM2_24B_A2B
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
MAX_SEQ_LENGTH = 4096
NUM_TRAIN_STEPS = 5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 1e-5
SEED = 42

MODES = {
    "fsdp": {
        "name": "FSDP (no EP)",
        "parallelism": {},
        "sft_extra": {},
    },
    "ep": {
        "name": "EP=2",
        "parallelism": {"ep_size": 2},
        "sft_extra": {"ddp_find_unused_parameters": True},
    },
}


def validate_results(trainer, train_result, mode: str) -> tuple[bool, list[str]]:
    """Validate training results. Returns (success, issues)."""
    issues = []

    if not torch.isfinite(torch.tensor(train_result.training_loss)):
        issues.append(f"Training loss is not finite: {train_result.training_loss}")

    log_history = trainer.state.log_history
    step_losses = [e["loss"] for e in log_history if "loss" in e and "eval_loss" not in e]
    grad_norms = [e["grad_norm"] for e in log_history if "grad_norm" in e]

    log(f"Per-step losses: {[f'{l:.4f}' for l in step_losses]}")
    if grad_norms:
        log(f"Per-step grad norms: {[f'{g:.2f}' for g in grad_norms]}")

    if any(not torch.isfinite(torch.tensor(l)) for l in step_losses):
        issues.append("NaN/Inf detected in step losses")

    if grad_norms and any(not torch.isfinite(torch.tensor(g)) for g in grad_norms):
        issues.append("NaN/Inf detected in gradient norms")

    if train_result.global_step != NUM_TRAIN_STEPS:
        issues.append(f"Expected {NUM_TRAIN_STEPS} steps, got {train_result.global_step}")

    if train_result.training_loss > 100:
        issues.append(f"Training loss unreasonably high: {train_result.training_loss}")

    return len(issues) == 0, issues


def run_mode(rank, world_size, local_rank, tokenizer, mode_config) -> bool:
    """Run SFT test for a given parallelism mode."""
    mode_name = mode_config["name"]
    log(f"\n{'=' * 70}")
    log(f"SFT LFM2 MoE TEST: {mode_name}")
    log(f"{'=' * 70}")

    output_dir, cache_dir = setup_cache_dirs(f"sft_lfm2_{mode_name.replace(' ', '_')}", rank)

    try:
        train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
        eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)
        log(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

        log(f"\n--- Loading model ({mode_name}) ---")
        log(f"GPU memory before load: {gpu_mem_gb():.1f}GB")

        parallelism_config = ParallelismConfig(**mode_config["parallelism"])

        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            use_liger_kernel=True,
        )

        log(f"Model loaded: {model.config.model_type}")
        log(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        log(f"GPU memory after load: {gpu_mem_gb():.1f}GB")

        if parallelism_config.is_ep_mode or parallelism_config.needs_ep_wrappers:
            ep_layers = sum(1 for m in model.modules() if hasattr(m, "ep_config"))
            log(f"EP MoE layers detected: {ep_layers}")

        sft_kwargs = {
            "output_dir": output_dir,
            "max_steps": NUM_TRAIN_STEPS,
            "per_device_train_batch_size": BATCH_SIZE,
            "per_device_eval_batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
            "learning_rate": LEARNING_RATE,
            "warmup_steps": 1,
            "max_length": MAX_SEQ_LENGTH,
            "bf16": True,
            "gradient_checkpointing": True,
            "use_liger_kernel": False,  # already applied by load_distributed_model
            "logging_steps": 1,
            "eval_strategy": "steps",
            "eval_steps": NUM_TRAIN_STEPS,
            "save_strategy": "no",
            "report_to": [],
            "dataloader_num_workers": 0,
            "dataloader_drop_last": True,
            "remove_unused_columns": False,
            "fsdp": "",  # Mixin handles FSDP wrapping
        }
        sft_kwargs.update(mode_config["sft_extra"])
        config = SFTConfig(**sft_kwargs)

        log(f"\n--- Creating DistributedSFTTrainer ({mode_name}) ---")
        trainer = DistributedSFTTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )
        log("Trainer created successfully")

        if parallelism_config.is_ep_mode:
            assert trainer.is_ep_mode, "trainer.is_ep_mode should be True"
        else:
            assert not trainer.is_ep_mode, "trainer.is_ep_mode should be False"

        log("\n--- Running initial evaluation ---")
        barrier()
        eval_results = trainer.evaluate()
        initial_loss = eval_results.get("eval_loss", float("inf"))
        log(f"Initial eval loss: {initial_loss:.4f}")

        log(f"\n--- Starting training ({mode_name}) ---")
        barrier()
        train_result = trainer.train()

        log("\n--- Training completed ---")
        log(f"Training loss: {train_result.training_loss:.6f}")
        log(f"Steps completed: {train_result.global_step}")
        log(f"GPU memory after training: {gpu_mem_gb():.1f}GB")

        log("\n--- Running final evaluation ---")
        barrier()
        final_eval = trainer.evaluate()
        final_loss = final_eval.get("eval_loss", float("inf"))
        log(f"Loss: {initial_loss:.4f} -> {final_loss:.4f}")

        success, issues = validate_results(trainer, train_result, mode_name)

        if not torch.isfinite(torch.tensor(final_loss)):
            issues.append(f"Final eval loss not finite: {final_loss}")
            success = False

        if success:
            log(f"\n{'=' * 70}")
            log(f"TEST PASSED: {mode_name}")
            log(f"  - Training loss: {train_result.training_loss:.6f}")
            log(f"  - Eval loss: {initial_loss:.4f} -> {final_loss:.4f}")
            log(f"  - Steps completed: {train_result.global_step}")
            log(f"{'=' * 70}")
        else:
            log(f"\n{'=' * 70}")
            log(f"TEST FAILED: {mode_name}")
            for issue in issues:
                log(f"  - {issue}")
            log(f"{'=' * 70}")

        return success

    except Exception as e:
        log(f"\nTEST FAILED WITH ERROR ({mode_name}): {e}")
        if rank == 0:
            traceback.print_exc()
        return False

    finally:
        if "trainer" in locals():
            trainer.cleanup_ep()
        barrier()
        cleanup_dirs(output_dir, cache_dir)
        cleanup_memory()


def _parse_mode() -> list[dict]:
    """Parse --mode argument. Returns list of mode configs to run."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=list(MODES.keys()) + ["all"],
        default="fsdp",
        help="Parallelism mode to test (default: fsdp). Use 'all' to run all modes "
        "sequentially (may fail for consecutive EP modes due to DeepEP buffer cleanup).",
    )
    args, _ = parser.parse_known_args()

    if args.mode == "all":
        return list(MODES.values())
    return [MODES[args.mode]]


def main():
    rank, world_size, local_rank = init_distributed()
    modes_to_run = _parse_mode()

    log(f"\n{'=' * 70}")
    log("SFT LFM2 MoE TEST SUITE")
    log(f"{'=' * 70}")
    log(f"World size: {world_size}")
    log(f"Model: {MODEL_NAME}")
    log(f"GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"GPU memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f}GB")
    log(f"Modes: {[m['name'] for m in modes_to_run]}")

    if world_size != 2:
        log(f"\nERROR: This test requires exactly 2 GPUs, got {world_size}")
        teardown_distributed()
        sys.exit(1)

    log("\n--- Ensuring model is downloaded ---")
    ensure_model_downloaded(MODEL_NAME, rank)

    # trainers require an initialized PartialState
    PartialState()

    log("\n--- Loading tokenizer ---")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

    results = {}
    for mode in modes_to_run:
        results[mode["name"]] = run_mode(rank, world_size, local_rank, tokenizer, mode)
        cleanup_memory()

    log(f"\n{'=' * 70}")
    log("TEST SUITE SUMMARY")
    for name, success in results.items():
        log(f"  {name}: {'PASSED' if success else 'FAILED'}")
    log(f"{'=' * 70}")

    teardown_distributed()

    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
