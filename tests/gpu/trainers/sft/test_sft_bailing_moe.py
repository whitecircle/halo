#!/usr/bin/env python
"""
Test script for SFT training with inclusionAI/Ring-mini-linear-2.0 MoE model.

Validates parallelism modes and overfitting capability:
1. fsdp       -- Default FSDP data parallelism (no EP), all experts on every GPU
2. ep         -- EP=2, experts distributed across GPUs via DeepEP (grouped GEMM auto-enabled)
3. ep_no_gmm  -- EP=2, without grouped GEMM
4. overfit    -- EP=2, 4 train samples x 40 steps, verifies 99%+ token accuracy

Ring-mini-linear-2.0 architecture (BailingMoeLinearV2):
  - 20 hidden layers (hybrid: 16 linear attention + 4 full attention)
  - Layer 0 is dense, layers 1-19 are MoE
  - MoE: 256 experts (moe_intermediate_size=512), top-k=8, 1 shared expert
  - Sigmoid routing with group-limited top-k (n_group=8, topk_group=4)
  - Full model: ~16.4B params, active per token: ~1.6B
  - Uses trust_remote_code=True (auto_map in config)

Usage:
    # FSDP-only (no EP)
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_bailing_moe.py --mode fsdp

    # EP=2 (grouped GEMM auto-enabled on SM90+)
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_bailing_moe.py --mode ep

    # EP=2 (no grouped GEMM)
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_bailing_moe.py --mode ep_no_gmm

    # Overfit test (EP=2, 99%+ token accuracy)
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_bailing_moe.py --mode overfit

Requirements:
    - 2x GPUs with >=40GB memory each
    - DeepEP installed (for EP modes)
    - flash-linear-attention installed (for linear attention layers)
    - Model: inclusionAI/Ring-mini-linear-2.0 (auto-downloaded)
"""

import sys
import traceback

from src.models.patches.remote_code_compat import apply_remote_code_compat_shims

apply_remote_code_compat_shims()

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
from tests.common.models import BAILING_MOE_RING_MINI
from tests.common.utils import cleanup_memory, gpu_mem_gb, log

# Test Configuration

MODEL_NAME = BAILING_MOE_RING_MINI
MAX_SEQ_LENGTH = 4096
SEED = 42

# Mode definitions: parallelism kwargs, extra SFT kwargs, training params
MODES = {
    "fsdp": {
        "name": "FSDP (no EP)",
        "parallelism": {},
        "sft_extra": {},
        "num_train_samples": 32,
        "num_eval_samples": 8,
        "max_steps": 5,
        "batch_size": 1,
        "gradient_accumulation": 2,
        "learning_rate": 1e-5,
    },
    "ep": {
        "name": "EP=2 (grouped GEMM)",
        "parallelism": {"ep_size": 2},
        "sft_extra": {"ddp_find_unused_parameters": True},
        "num_train_samples": 32,
        "num_eval_samples": 8,
        "max_steps": 5,
        "batch_size": 1,
        "gradient_accumulation": 2,
        "learning_rate": 1e-5,
    },
    "ep_no_gmm": {
        "name": "EP=2 (no grouped GEMM)",
        "parallelism": {"ep_size": 2, "use_grouped_gemm": False},
        "sft_extra": {"ddp_find_unused_parameters": True},
        "num_train_samples": 32,
        "num_eval_samples": 8,
        "max_steps": 5,
        "batch_size": 1,
        "gradient_accumulation": 2,
        "learning_rate": 1e-5,
    },
    "overfit": {
        "name": "EP=2 overfit (token accuracy)",
        "parallelism": {"ep_size": 2},
        "sft_extra": {"ddp_find_unused_parameters": True},
        "num_train_samples": 4,
        "num_eval_samples": 4,
        "max_steps": 40,
        "batch_size": 1,
        "gradient_accumulation": 1,
        "learning_rate": 5e-5,
    },
}


# Token Accuracy Measurement


def get_logged_token_accuracy(trainer) -> tuple[float, float]:
    """Extract peak and mean token accuracy from training log history.

    Uses the trainer's logged metrics which are computed during training
    on each batch. For overfit tests on tiny datasets, this gives an
    accurate picture of memorization quality without requiring a separate
    inference pass (which would need special EP coordination).

    Returns (peak_accuracy, mean_last_10) — peak proves the model CAN
    overfit, mean shows sustained quality.
    """
    accuracies = [
        e["mean_token_accuracy"]
        for e in trainer.state.log_history
        if "mean_token_accuracy" in e and "eval" not in e.get("prefix", "")
    ]
    if not accuracies:
        return 0.0, 0.0
    peak = max(accuracies)
    tail = accuracies[-10:] if len(accuracies) >= 10 else accuracies
    return peak, sum(tail) / len(tail)


# Validation


def validate_results(trainer, train_result, mode_config: dict) -> tuple[bool, list[str]]:
    """Validate training results. Returns (success, issues)."""
    issues = []
    max_steps = mode_config["max_steps"]

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

    if train_result.global_step != max_steps:
        issues.append(f"Expected {max_steps} steps, got {train_result.global_step}")

    if train_result.training_loss > 100:
        issues.append(f"Training loss unreasonably high: {train_result.training_loss}")

    return len(issues) == 0, issues


# Generic Mode Runner


def run_mode(rank, world_size, local_rank, tokenizer, mode_config) -> bool:
    """Run SFT test for a given parallelism mode."""
    mode_name = mode_config["name"]
    is_overfit = "overfit" in mode_name.lower()

    log(f"\n{'=' * 70}")
    log(f"SFT BAILING MoE TEST: {mode_name}")
    log(f"{'=' * 70}")

    output_dir, cache_dir = setup_cache_dirs(f"sft_bailing_{mode_name.replace(' ', '_')}", rank)

    try:
        # Create datasets
        num_train = mode_config["num_train_samples"]
        num_eval = mode_config["num_eval_samples"]
        train_dataset = create_sft_dataset(num_train, tokenizer, seed=SEED)
        eval_dataset = create_sft_dataset(num_eval, tokenizer, seed=SEED + 100)
        log(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

        # Load model
        log(f"\n--- Loading model ({mode_name}) ---")
        log(f"GPU memory before load: {gpu_mem_gb():.1f}GB")

        parallelism_config = ParallelismConfig(**mode_config["parallelism"])

        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
            use_liger_kernel=True,
        )

        log(f"Model loaded: {model.config.model_type}")
        log(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        log(f"GPU memory after load: {gpu_mem_gb():.1f}GB")

        if parallelism_config.is_ep_mode or parallelism_config.needs_ep_wrappers:
            ep_layers = sum(1 for m in model.modules() if hasattr(m, "ep_config"))
            log(f"EP MoE layers detected: {ep_layers}")

        # Create SFT config
        sft_kwargs = {
            "output_dir": output_dir,
            "max_steps": mode_config["max_steps"],
            "per_device_train_batch_size": mode_config["batch_size"],
            "per_device_eval_batch_size": mode_config["batch_size"],
            "gradient_accumulation_steps": mode_config["gradient_accumulation"],
            "learning_rate": mode_config["learning_rate"],
            "warmup_steps": 1,
            "max_length": MAX_SEQ_LENGTH,
            "bf16": True,
            "gradient_checkpointing": True,
            "use_liger_kernel": False,  # Already applied in load_distributed_model
            "logging_steps": 1,
            "eval_strategy": "steps",
            "eval_steps": mode_config["max_steps"],
            "save_strategy": "no",
            "report_to": [],
            "dataloader_num_workers": 0,
            "dataloader_drop_last": True,
            "remove_unused_columns": False,
            "fsdp": "",  # Mixin handles FSDP wrapping
        }
        sft_kwargs.update(mode_config["sft_extra"])
        config = SFTConfig(**sft_kwargs)

        # Create trainer
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

        # Mode-specific assertions
        if parallelism_config.is_ep_mode:
            assert trainer.is_ep_mode, "trainer.is_ep_mode should be True"
        else:
            assert not trainer.is_ep_mode, "trainer.is_ep_mode should be False"

        # Run initial evaluation
        log("\n--- Running initial evaluation ---")
        barrier()
        eval_results = trainer.evaluate()
        initial_loss = eval_results.get("eval_loss", float("inf"))
        log(f"Initial eval loss: {initial_loss:.4f}")

        # Run training
        log(f"\n--- Starting training ({mode_name}) ---")
        barrier()
        train_result = trainer.train()

        log("\n--- Training completed ---")
        log(f"Training loss: {train_result.training_loss:.6f}")
        log(f"Steps completed: {train_result.global_step}")
        log(f"GPU memory after training: {gpu_mem_gb():.1f}GB")

        # Final evaluation
        log("\n--- Running final evaluation ---")
        barrier()
        final_eval = trainer.evaluate()
        final_loss = final_eval.get("eval_loss", float("inf"))
        log(f"Eval loss: {initial_loss:.4f} -> {final_loss:.4f}")

        # Validate
        success, issues = validate_results(trainer, train_result, mode_config)

        if not torch.isfinite(torch.tensor(final_loss)):
            issues.append(f"Final eval loss not finite: {final_loss}")
            success = False

        # Overfit verification: check peak token accuracy from training logs
        accuracy = None
        if is_overfit:
            peak, mean = get_logged_token_accuracy(trainer)
            accuracy = peak
            log(f"Token accuracy — peak: {peak:.4%}, mean (last 10): {mean:.4%}")
            if peak < 0.99:
                issues.append(f"Peak token accuracy {peak:.4%} < 99% (overfit failed)")
                success = False

        if success:
            log(f"\n{'=' * 70}")
            log(f"TEST PASSED: {mode_name}")
            log(f"  - Training loss: {train_result.training_loss:.6f}")
            log(f"  - Eval loss: {initial_loss:.4f} -> {final_loss:.4f}")
            log(f"  - Steps completed: {train_result.global_step}")
            if is_overfit and accuracy is not None:
                log(f"  - Peak token accuracy: {accuracy:.4%}")
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


# Main


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
    log("SFT BAILING MoE TEST SUITE")
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

    # Ensure model is downloaded before tests
    log("\n--- Ensuring model is downloaded ---")
    ensure_model_downloaded(MODEL_NAME, rank)

    # Initialize PartialState for accelerate compatibility
    PartialState()

    # Load tokenizer (shared across tests)
    log("\n--- Loading tokenizer ---")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

    # Run selected test modes
    results = {}
    for mode in modes_to_run:
        results[mode["name"]] = run_mode(rank, world_size, local_rank, tokenizer, mode)
        cleanup_memory()

    # Summary
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
