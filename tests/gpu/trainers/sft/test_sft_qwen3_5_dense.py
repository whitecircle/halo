#!/usr/bin/env python
"""
SFT test for Qwen3.5-2B (dense) across FSDP and TP=2 modes.

Validates that DistributedSFTTrainer produces correct training metrics for a
Qwen3.5 dense model with hybrid attention (linear + full attention).

Qwen3.5-2B Architecture:
  - 24 decoder layers: 18 linear attention (GatedDeltaNet) + 6 full attention
  - Dense MLP (no MoE)
  - Double-width q_proj (query + sigmoid gate) — incompatible with CP
  - Natively multimodal (Image-Text-to-Text), but text backbone loads via AutoModelForCausalLM

Key constraints:
  - attn_implementation="sdpa" required (FA2 crashes with M-RoPE varlen path)
  - Liger kernel has no qwen3_5 support — disabled
  - CP (Ulysses) not supported (double-width q_proj needs custom wrapper)
  - Requires causal-conv1d + flash-linear-attention for GatedDeltaNet kernels

Modes tested:
  1. FSDP (standard data parallelism, dp=2)
  2. TP=2 (Tensor Parallelism, dp=1)

Usage:
    # Both modes sequentially:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_qwen3_5_dense.py

    # Single mode:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_qwen3_5_dense.py --mode fsdp
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_qwen3_5_dense.py --mode tp

Requirements:
    - 2x GPUs
    - causal-conv1d and flash-linear-attention installed (for GatedDeltaNet kernels)
    - Model: Qwen/Qwen3.5-2B (auto-downloaded)
"""

import argparse
import math
import os

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_5_2B
from tests.common.utils import cleanup_memory, gpu_mem_gb, log

# Configuration

MODEL_NAME = QWEN3_5_2B
MAX_STEPS = 10
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 2e-5
NUM_TRAIN_SAMPLES = 64
NUM_EVAL_SAMPLES = 16
SEED = 42

# Metric thresholds (per-mode)
MODE_THRESHOLDS = {
    "FSDP": {"max_initial_loss": 12.0, "min_initial_loss": 0.5, "max_grad_norm": 500.0},
    "TP=2": {"max_initial_loss": 12.0, "min_initial_loss": 0.5, "max_grad_norm": 500.0},
}


# Helpers


def merge_checks(checks: dict[str, bool], mode_checks: dict[str, bool]) -> None:
    """AND one mode's results into the run's checks — both modes report the same check names."""
    for name, ok in mode_checks.items():
        checks[name] = ok and checks.get(name, True)


def extract_metrics(trainer) -> dict[str, list[float]]:
    """Extract per-step training metrics from trainer log history."""
    log_history = trainer.state.log_history
    metrics = {
        "loss": [],
        "grad_norm": [],
        "mean_token_accuracy": [],
    }
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            metrics["loss"].append(entry["loss"])
        if "grad_norm" in entry:
            metrics["grad_norm"].append(entry["grad_norm"])
        if "mean_token_accuracy" in entry:
            metrics["mean_token_accuracy"].append(entry["mean_token_accuracy"])
    return metrics


def validate_metrics(
    mode_name: str,
    training_loss: float,
    metrics: dict[str, list[float]],
    rank: int,
    world_size: int,
    local_rank: int,
) -> dict[str, bool]:
    """Run all metric validations and return a dict of check_name -> passed."""
    checks = {}
    thresholds = MODE_THRESHOLDS[mode_name]
    max_initial_loss = thresholds["max_initial_loss"]
    min_initial_loss = thresholds["min_initial_loss"]
    max_grad_norm = thresholds["max_grad_norm"]

    losses = metrics["loss"]

    # 1. Training loss is finite
    loss_finite = math.isfinite(training_loss)
    checks["loss_finite"] = loss_finite
    log(f"  Loss is finite: {'PASS' if loss_finite else 'FAIL'} ({training_loss:.6f})")

    # 2. All step losses are finite
    all_losses_finite = all(math.isfinite(l) for l in losses)
    checks["all_losses_finite"] = all_losses_finite
    log(f"  All step losses finite: {'PASS' if all_losses_finite else 'FAIL'}")

    # 3. Initial loss in reasonable range
    if losses:
        first_loss = losses[0]
        initial_ok = min_initial_loss <= first_loss <= max_initial_loss
        checks["initial_loss_reasonable"] = initial_ok
        log(
            f"  Initial loss reasonable ({min_initial_loss}-{max_initial_loss}): "
            f"{'PASS' if initial_ok else 'FAIL'} ({first_loss:.4f})"
        )

    # 4. Loss decreased (first vs last)
    if len(losses) >= 2:
        loss_decreased = losses[-1] < losses[0]
        checks["loss_decreased"] = loss_decreased
        log(f"  Loss decreased: {'PASS' if loss_decreased else 'FAIL'} ({losses[0]:.4f} -> {losses[-1]:.4f})")
    else:
        checks["loss_decreased"] = False
        log("  Loss decreased: FAIL (not enough steps)")

    # 5. Grad norm checks
    grad_norms = metrics["grad_norm"]
    if grad_norms:
        grads_finite = all(math.isfinite(g) for g in grad_norms)
        checks["grad_norms_finite"] = grads_finite
        log(f"  Grad norms finite: {'PASS' if grads_finite else 'FAIL'}")

        max_gn = max(grad_norms)
        grads_reasonable = max_gn < max_grad_norm
        checks["grad_norms_reasonable"] = grads_reasonable
        log(f"  Grad norms reasonable (<{max_grad_norm}): {'PASS' if grads_reasonable else 'FAIL'} (max={max_gn:.2f})")
        log(f"  Grad norm range: [{min(grad_norms):.2f}, {max_gn:.2f}]")

    # 6. Token accuracy checks
    token_accs = metrics["mean_token_accuracy"]
    if token_accs:
        ta_valid = all(0.0 <= ta <= 1.0 for ta in token_accs)
        checks["token_accuracy_valid"] = ta_valid
        log(f"  Token accuracy in [0,1]: {'PASS' if ta_valid else 'FAIL'}")
        log(f"  Token accuracy range: [{min(token_accs):.4f}, {max(token_accs):.4f}]")

    # 7. Cross-rank consistency
    loss_tensor = torch.tensor([training_loss], device=f"cuda:{local_rank}")
    all_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
    dist.all_gather(all_losses, loss_tensor)
    if rank == 0:
        loss_values = [l.item() for l in all_losses]
        spread = max(loss_values) - min(loss_values)
        consistent = spread < 0.01
        checks["loss_consistent_across_ranks"] = consistent
        log(f"  Loss consistent across ranks (spread={spread:.6f}): {'PASS' if consistent else 'FAIL'}")
    else:
        checks["loss_consistent_across_ranks"] = True

    return checks


# Mode runner


def run_mode(
    mode_name: str,
    parallelism_config: ParallelismConfig,
    tokenizer,
    train_dataset,
    eval_dataset,
    base_output_dir: str,
    rank: int,
    world_size: int,
    local_rank: int,
) -> tuple[dict[str, bool], str]:
    """Run SFT training for one parallelism mode with metric validation."""
    output_dir = os.path.join(base_output_dir, mode_name.lower().replace("=", ""))

    log(f"\n  Loading model for {mode_name}...")
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",  # FA2 crashes with Qwen3.5's M-RoPE varlen path
        use_liger_kernel=False,  # Liger has no qwen3_5 support
    )

    param_count = sum(p.numel() for p in model.parameters())
    log(f"  Model loaded: {type(model).__name__}, {param_count / 1e9:.2f}B params")
    log(f"  GPU memory: {gpu_mem_gb():.1f}GB")

    # Log architecture details
    layers = model.model.layers if hasattr(model.model, "layers") else []
    layer_types = {}
    for layer in layers:
        attn = getattr(layer, "self_attn", None) or getattr(layer, "linear_attn", None)
        cls_name = type(attn).__name__ if attn else "unknown"
        layer_types[cls_name] = layer_types.get(cls_name, 0) + 1
    log(f"  Layer types: {layer_types}")

    # SFT config
    config = SFTConfig(
        output_dir=output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        fsdp="",  # Mixin handles FSDP wrapping
    )

    # Create trainer
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )

    # Pre-training assertions
    if parallelism_config.is_tp_mode:
        assert trainer.is_tp_mode, f"Expected is_tp_mode=True for {mode_name}"

    # Train
    log(f"  Training {MAX_STEPS} steps...")
    dist.barrier()
    train_result = trainer.train()

    # Extract and validate metrics
    training_loss = train_result.training_loss
    metrics = extract_metrics(trainer)

    log(f"\n  --- {mode_name} Metrics ---")
    log(f"  Final loss: {training_loss:.6f}")
    log(f"  Step losses: {[f'{l:.4f}' for l in metrics['loss']]}")
    if metrics["grad_norm"]:
        log(f"  Grad norms: {[f'{g:.2f}' for g in metrics['grad_norm']]}")

    log(f"\n  --- {mode_name} Checks ---")
    checks = validate_metrics(
        mode_name,
        training_loss,
        metrics,
        rank,
        world_size,
        local_rank,
    )

    failed = [k for k, v in checks.items() if not v]
    detail = f"loss={training_loss:.4f}"
    if failed:
        detail += f", failed={failed}"

    # Cleanup
    del trainer, model
    cleanup_memory()
    dist.barrier()

    return checks, detail


# Main


def _parse_mode() -> list[str]:
    """Parse --mode argument. Returns list of mode names to run."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["fsdp", "tp", "all"],
        default="all",
        help="Parallelism mode to test (default: all).",
    )
    args, _ = parser.parse_known_args()

    if args.mode == "all":
        return ["fsdp", "tp"]
    return [args.mode]


def run(ctx) -> dict:
    """Run SFT test for Qwen3.5-2B across FSDP and TP modes."""
    log(f"\n{'#' * 70}")
    log("  SFT Qwen3.5-2B Dense — FSDP & TP Validation")
    log(f"  World size: {ctx.world_size}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  Steps: {MAX_STEPS}, Batch: {BATCH_SIZE}, Seq: {MAX_SEQ_LENGTH}")
    log(f"{'#' * 70}")

    modes_to_run = _parse_mode()
    log(f"  Modes: {modes_to_run}")

    checks: dict[str, bool] = {}
    details: dict[str, str] = {}

    # Download model on rank 0 first
    log("\n--- Ensuring model is downloaded ---")
    ensure_model_downloaded(MODEL_NAME, ctx.rank)

    # Shared tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"  Tokenizer: {tokenizer.__class__.__name__}, vocab={tokenizer.vocab_size}")

    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1)
    log(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)} samples")

    # Run modes
    if "fsdp" in modes_to_run:
        log(f"\n{'=' * 60}")
        log(f"  MODE: FSDP (Standard Data Parallelism, dp={ctx.world_size})")
        log(f"{'=' * 60}")

        mode_checks, details["FSDP"] = run_mode(
            "FSDP",
            ParallelismConfig(),
            tokenizer,
            train_dataset,
            eval_dataset,
            ctx.output_dir,
            ctx.rank,
            ctx.world_size,
            ctx.local_rank,
        )
        merge_checks(checks, mode_checks)

    if "tp" in modes_to_run:
        log(f"\n{'=' * 60}")
        log("  MODE: TP=2 (Tensor Parallelism, dp=1)")
        log(f"{'=' * 60}")

        mode_checks, details["TP=2"] = run_mode(
            "TP=2",
            ParallelismConfig(tp_size=2),
            tokenizer,
            train_dataset,
            eval_dataset,
            ctx.output_dir,
            ctx.rank,
            ctx.world_size,
            ctx.local_rank,
        )
        merge_checks(checks, mode_checks)

    log(f"\n{'#' * 70}")
    log("  SFT Qwen3.5-2B Dense — Results")
    log(f"{'#' * 70}")
    for mode_name, detail in details.items():
        log(f"  {mode_name:>8s}: {detail}")

    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="test_sft_qwen3_5_dense")(run)

if __name__ == "__main__":
    main()
