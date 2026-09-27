#!/usr/bin/env python
"""
Test script for SFT training with inclusionAI/Ring-mini-linear-2.0 MoE model.

Validates parallelism modes and overfitting capability:
1. fsdp       -- Default FSDP data parallelism (no EP), all experts on every GPU
2. ep         -- EP=2, experts distributed across GPUs via DeepEP (grouped GEMM auto-enabled)
3. ep_no_gmm  -- EP=2, without grouped GEMM
4. overfit    -- EP=2, 4 train samples x 40 steps, verifies 99%+ peak logged token accuracy

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

    # Overfit test (EP=2, 99%+ peak token accuracy)
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_bailing_moe.py --mode overfit

Requirements:
    - 2x GPUs with >=40GB memory each
    - DeepEP installed (for EP modes)
    - flash-linear-attention installed (for linear attention layers)
    - Model: inclusionAI/Ring-mini-linear-2.0 (auto-downloaded)
"""

import argparse
import os

from src.models.patches.remote_code_compat import apply_remote_code_compat_shims

apply_remote_code_compat_shims()

import torch
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.expert_parallel.dispatcher import destroy_all_dispatchers
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.models import BAILING_MOE_RING_MINI
from tests.common.utils import cleanup_memory, gpu_mem_gb, log, training_run_checks

# Test Configuration

MODEL_NAME = BAILING_MOE_RING_MINI
MAX_SEQ_LENGTH = 4096
SEED = 42
OVERFIT_MIN_PEAK_TOKEN_ACCURACY = 0.99

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


# Generic Mode Runner


def run_mode(ctx, tokenizer, mode_key: str) -> dict[str, bool]:
    """Run SFT test for one parallelism mode; its checks are keyed ``<mode>_<check>``."""
    mode_config = MODES[mode_key]
    mode_name = mode_config["name"]
    is_overfit = "overfit" in mode_name.lower()

    log(f"\n{'=' * 70}")
    log(f"SFT BAILING MoE TEST: {mode_name}")
    log(f"{'=' * 70}")

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

    checks: dict[str, bool] = {}
    if parallelism_config.needs_ep_wrappers:
        wrapped = len(ep_layers(model))
        log(f"EP MoE layers detected: {wrapped}")
        checks["ep_layers_wrapped"] = wrapped > 0

    # Create SFT config
    sft_kwargs = {
        "output_dir": os.path.join(ctx.output_dir, mode_key),
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
        "logging_nan_inf_filter": False,
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

    checks.update(training_run_checks(train_result, trainer, mode_config["max_steps"], grad_norms=True))
    checks["final_eval_loss_finite"] = bool(torch.isfinite(torch.tensor(final_loss)))

    # Overfit verification: check peak token accuracy from training logs
    if is_overfit:
        peak, mean = get_logged_token_accuracy(trainer)
        log(f"Token accuracy — peak: {peak:.4%}, mean (last 10): {mean:.4%}")
        checks["peak_token_accuracy"] = peak >= OVERFIT_MIN_PEAK_TOKEN_ACCURACY

    trainer.cleanup_ep()
    return {f"{mode_key}_{name}": ok for name, ok in checks.items()}


# Main


def _parse_mode() -> list[str]:
    """Parse --mode argument. Returns the keys of the modes to run."""
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
        return list(MODES)
    return [args.mode]


@gpu_test_main(exact_world_size=2, prefix="sft_bailing")
def run(ctx):
    modes_to_run = _parse_mode()

    log(f"\n{'=' * 70}")
    log("SFT BAILING MoE TEST SUITE")
    log(f"{'=' * 70}")
    log(f"World size: {ctx.world_size}")
    log(f"Model: {MODEL_NAME}")
    log(f"GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"GPU memory: {torch.cuda.get_device_properties(ctx.local_rank).total_memory / 1e9:.1f}GB")
    log(f"Modes: {[MODES[m]['name'] for m in modes_to_run]}")

    # Registry-wide rather than per trainer: a finalizer bound to a trainer would keep each mode's
    # model alive into the next mode.
    ctx.on_teardown(destroy_all_dispatchers)

    # Ensure model is downloaded before tests
    log("\n--- Ensuring model is downloaded ---")
    ensure_model_downloaded(MODEL_NAME, ctx.rank)

    # Load tokenizer (shared across tests)
    log("\n--- Loading tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

    # Run selected test modes
    checks: dict[str, bool] = {}
    for mode_key in modes_to_run:
        checks.update(run_mode(ctx, tokenizer, mode_key))
        cleanup_memory()

    return {"checks": checks}


if __name__ == "__main__":
    run()
