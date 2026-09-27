#!/usr/bin/env python
"""
Test script for SFT training with Qwen3.5-35B-A3B MoE model.

Trains one parallelism mode per launch (``--mode``, default ``ep``):
1. ep        -- EP=2 (grouped GEMM auto-enabled on SM90+)
2. ep_no_gmm -- EP=2 (grouped GEMM disabled)
3. tp        -- TP=2 (Tensor Parallelism on full attention layers)
4. etp       -- ETP=2 (Expert Tensor Parallelism on MoE FFN weights)
and checks that it runs every step with finite losses, gradient norms and final eval loss.

Qwen3.5-35B-A3B has a hybrid attention architecture:
  - 30 linear attention layers (Qwen3_5MoeGatedDeltaNet)
  - 10 full attention layers (Qwen3_5MoeAttention)
  - All 40 layers have MoE MLP (Qwen3_5MoeSparseMoeBlock, 256 experts, top-k=8)

Note: Flash Attention 2 is incompatible with Qwen3.5's M-RoPE varlen path
(crashes with cudaErrorIllegalAddress). Use attn_implementation="sdpa".

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_qwen3_5_moe.py --mode ep

Requirements:
    - 2x GPUs with >=80GB memory each (tested on B200)
    - DeepEP installed (for EP modes)
    - causal-conv1d and flash-linear-attention installed (for GatedDeltaNet kernels)
    - Model: Qwen/Qwen3.5-35B-A3B (auto-downloaded)
"""

import argparse
import os

import torch
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.expert_parallel.dispatcher import destroy_all_dispatchers
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.mesh import has_tp_dim
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_5_MOE_35B
from tests.common.utils import cleanup_memory, gpu_mem_gb, log, step_losses

# Test Configuration

MODEL_NAME = QWEN3_5_MOE_35B
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
MAX_SEQ_LENGTH = 4096
NUM_TRAIN_STEPS = 5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 1e-5
SEED = 42

# Mode definitions: name, ParallelismConfig kwargs, extra SFTConfig kwargs
# Run a single mode per process to avoid DeepEP buffer cleanup issues:
#   torchrun --nproc_per_node=2 \
#       tests/gpu/trainers/sft/test_sft_qwen3_5_moe.py --mode ep
MODES = {
    "ep": {
        "name": "EP=2 (grouped GEMM)",
        "parallelism": {"ep_size": 2},
        "sft_extra": {"ddp_find_unused_parameters": True},
    },
    "ep_no_gmm": {
        "name": "EP=2 (no grouped GEMM)",
        "parallelism": {"ep_size": 2, "use_grouped_gemm": False},
        "sft_extra": {"ddp_find_unused_parameters": True},
    },
    "tp": {
        "name": "TP=2",
        "parallelism": {"tp_size": 2},
        "sft_extra": {},
    },
    "etp": {
        "name": "ETP=2",
        "parallelism": {"ep_size": 1, "expert_tp_size": 2},
        "sft_extra": {"ddp_find_unused_parameters": True},
    },
}


# Validation


def validate_results(trainer, train_result) -> dict[str, bool]:
    """Finiteness and step-count checks on a finished run."""
    losses = step_losses(trainer)
    grad_norms = [e["grad_norm"] for e in trainer.state.log_history if "grad_norm" in e]

    log(f"Per-step losses: {[f'{l:.4f}' for l in losses]}")
    if grad_norms:
        log(f"Per-step grad norms: {[f'{g:.2f}' for g in grad_norms]}")

    return {
        "training_loss_finite": bool(torch.isfinite(torch.tensor(train_result.training_loss))),
        "step_losses_finite": all(torch.isfinite(torch.tensor(l)) for l in losses),
        "grad_norms_finite": all(torch.isfinite(torch.tensor(g)) for g in grad_norms),
        "completed_all_steps": train_result.global_step == NUM_TRAIN_STEPS,
    }


# Generic Mode Runner


def run_mode(ctx, tokenizer, mode_key: str) -> dict[str, bool]:
    """Run SFT test for one parallelism mode; its checks are keyed ``<mode>_<check>``."""
    mode_config = MODES[mode_key]
    mode_name = mode_config["name"]
    log(f"\n{'=' * 70}")
    log(f"SFT QWEN3.5 MoE TEST: {mode_name}")
    log(f"{'=' * 70}")

    # Create datasets
    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)
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
        attn_implementation="sdpa",  # FA2 crashes with Qwen3.5's M-RoPE varlen path
        use_liger_kernel=True,
    )

    log(f"Model loaded: {model.config.model_type}")
    log(f"GPU memory after load: {gpu_mem_gb():.1f}GB")

    checks: dict[str, bool] = {}
    if parallelism_config.needs_ep_wrappers:
        ep_layers = sum(1 for m in model.modules() if hasattr(m, "ep_config"))
        log(f"EP MoE layers detected: {ep_layers}")
        checks["ep_layers_wrapped"] = ep_layers > 0
    if parallelism_config.is_tp_mode:
        checks["tp_sharded_params"] = any(
            isinstance(p.data, DTensor) and has_tp_dim(p.data.device_mesh) for p in model.parameters()
        )

    # Create SFT config
    sft_kwargs = {
        "output_dir": os.path.join(ctx.output_dir, mode_key),
        "max_steps": NUM_TRAIN_STEPS,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": 1,
        "max_length": MAX_SEQ_LENGTH,
        "bf16": True,
        "gradient_checkpointing": True,
        "use_liger_kernel": False,  # Already applied in load_distributed_model
        "logging_steps": 1,
        "eval_strategy": "steps",
        "eval_steps": NUM_TRAIN_STEPS,
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
    log(f"Loss: {initial_loss:.4f} -> {final_loss:.4f}")

    checks.update(validate_results(trainer, train_result))
    checks["final_eval_loss_finite"] = bool(torch.isfinite(torch.tensor(final_loss)))

    trainer.cleanup_ep()
    return {f"{mode_key}_{name}": ok for name, ok in checks.items()}


# Main


def _parse_mode() -> list[str]:
    """Parse --mode argument. Returns the keys of the modes to run."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=list(MODES.keys()) + ["all"],
        default="ep",
        help="Parallelism mode to test (default: ep). Use 'all' to run all modes "
        "sequentially (may fail for consecutive EP modes due to DeepEP buffer cleanup).",
    )
    # Ignore torchrun args
    args, _ = parser.parse_known_args()

    if args.mode == "all":
        return list(MODES)
    return [args.mode]


@gpu_test_main(exact_world_size=2, prefix="sft_qwen3_5")
def run(ctx):
    modes_to_run = _parse_mode()

    log(f"\n{'=' * 70}")
    log("SFT QWEN3.5 MoE TEST SUITE")
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
