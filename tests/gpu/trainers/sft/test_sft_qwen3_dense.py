#!/usr/bin/env python
"""
Comprehensive SFT test for Qwen3-0.6B (dense) across FSDP, TP=2, and CP=2.

Validates that DistributedSFTTrainer produces correct training metrics in all
three distributed modes. Beyond basic "loss is finite" checks, this test
verifies grad_norm ranges, mean_token_accuracy, loss convergence, the eval leg,
and cross-mode consistency.

Each invocation runs ONE mode (``--mode fsdp|tp|cp``) so every mode owns its own
verdict; the manifest's args_matrix chains the three. ``--mode all`` runs them
sequentially in one process for a standalone sweep.

Test Phases (per mode):
1. Load model via load_distributed_model with appropriate parallelism config
2. Train for 10 steps on synthetic math conversations, evaluating every 5
3. Validate:
   - Loss is finite and decreases
   - Loss starts in a reasonable range (2-8 for Qwen3-0.6B on math data)
   - Grad norms are finite and in a reasonable range (no inflation)
   - mean_token_accuracy is reported and in [0, 1]
   - Token accuracy increases (model is learning)
   - Eval loss is reported and finite
   - The requested mode actually engaged (trainer.is_tp_mode / is_cp_mode)
   - Loss consistent across ranks (TP/CP: all ranks see same loss)

Modes:
1. fsdp — standard data parallelism, dp=2
2. tp   — Tensor Parallelism (TP=2), dp=1
3. cp   — Context Parallelism (CP=2), dp=1

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_qwen3_dense.py --mode fsdp
"""

import argparse
import math
import os

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from trl import SFTConfig

from src.data.collators.factory import select_data_collator
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.tolerances import TOL
from tests.common.utils import cleanup_memory, log

# Configuration

MODEL_NAME = QWEN3_0_6B
MAX_STEPS = 10
EVAL_STEPS = 5
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 4096
LEARNING_RATE = 2e-5
NUM_TRAIN_SAMPLES = 64
NUM_EVAL_SAMPLES = 16
SEED = 42
TP_SIZE = 2
CP_SIZE = 2

# Mode key -> ParallelismConfig kwargs. One mode per invocation (``--mode``); the manifest's
# args_matrix gives each mode its own pytest node, so a regression names the mode that broke.
MODE_CONFIGS: dict[str, dict[str, int]] = {
    "fsdp": {},
    "tp": {"tp_size": TP_SIZE},
    "cp": {"cp_size": CP_SIZE},
}

# Metric thresholds
MAX_INITIAL_LOSS = 8.0  # Qwen3-0.6B on math data typically starts ~4-5
MIN_INITIAL_LOSS = 1.0  # Sanity: loss shouldn't be suspiciously low
MAX_GRAD_NORM = 500.0  # Healthy range; inflated TP was ~16,000+
MIN_TOKEN_ACCURACY = 0.0  # Must be non-negative
MAX_TOKEN_ACCURACY = 1.0  # Must be at most 1.0


# Helpers


def extract_metrics(trainer) -> dict[str, list[float]]:
    """Extract per-step training metrics from trainer log history."""
    log_history = trainer.state.log_history
    metrics = {
        "loss": [],
        "grad_norm": [],
        "mean_token_accuracy": [],
        "eval_loss": [],
    }
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            metrics["loss"].append(entry["loss"])
        if "grad_norm" in entry:
            metrics["grad_norm"].append(entry["grad_norm"])
        if "mean_token_accuracy" in entry:
            metrics["mean_token_accuracy"].append(entry["mean_token_accuracy"])
        if "eval_loss" in entry:
            metrics["eval_loss"].append(entry["eval_loss"])
    return metrics


def validate_metrics(training_loss: float, metrics: dict[str, list[float]], ctx) -> dict[str, bool]:
    """Run all metric validations and return a dict of check_name -> passed."""
    checks = {}

    # ── Loss checks ──────────────────────────────────────────────────────
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
        initial_ok = MIN_INITIAL_LOSS <= first_loss <= MAX_INITIAL_LOSS
        checks["initial_loss_reasonable"] = initial_ok
        log(
            f"  Initial loss reasonable ({MIN_INITIAL_LOSS}-{MAX_INITIAL_LOSS}): "
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

    # ── Grad norm checks ─────────────────────────────────────────────────
    grad_norms = metrics["grad_norm"]
    if grad_norms:
        # 5. All grad norms finite
        grads_finite = all(math.isfinite(g) for g in grad_norms)
        checks["grad_norms_finite"] = grads_finite
        log(f"  Grad norms finite: {'PASS' if grads_finite else 'FAIL'}")

        # 6. Grad norms not inflated (key TP regression check)
        max_gn = max(grad_norms)
        grads_reasonable = max_gn < MAX_GRAD_NORM
        checks["grad_norms_reasonable"] = grads_reasonable
        log(f"  Grad norms reasonable (<{MAX_GRAD_NORM}): {'PASS' if grads_reasonable else 'FAIL'} (max={max_gn:.2f})")

        log(f"  Grad norm range: [{min(grad_norms):.2f}, {max_gn:.2f}]")

    # ── Token accuracy checks ────────────────────────────────────────────
    token_accs = metrics["mean_token_accuracy"]
    if token_accs:
        # 7. Token accuracy in valid range [0, 1]
        ta_valid = all(MIN_TOKEN_ACCURACY <= ta <= MAX_TOKEN_ACCURACY for ta in token_accs)
        checks["token_accuracy_valid"] = ta_valid
        log(f"  Token accuracy in [0,1]: {'PASS' if ta_valid else 'FAIL'}")

        # 8. Token accuracy increases (learning signal)
        if len(token_accs) >= 2:
            ta_increased = token_accs[-1] > token_accs[0]
            checks["token_accuracy_increased"] = ta_increased
            log(
                f"  Token accuracy increased: {'PASS' if ta_increased else 'FAIL'} "
                f"({token_accs[0]:.4f} -> {token_accs[-1]:.4f})"
            )

        log(f"  Token accuracy range: [{min(token_accs):.4f}, {max(token_accs):.4f}]")
    else:
        log("  Token accuracy: not reported (skipping checks)")

    # ── Eval checks ──────────────────────────────────────────────────────
    # 9. Eval loss reported and finite. An empty eval history FAILS rather than skipping:
    #    the eval leg exercises the trainer's evaluation path under TP/CP, and a silently
    #    disabled eval would certify it without running it.
    eval_losses = metrics["eval_loss"]
    eval_finite = bool(eval_losses) and all(math.isfinite(e) for e in eval_losses)
    checks["eval_loss_finite"] = eval_finite
    log(f"  Eval loss finite ({len(eval_losses)} evals): {'PASS' if eval_finite else 'FAIL'}")

    # ── Cross-rank consistency ───────────────────────────────────────────
    # 10. All ranks should report same training loss
    loss_tensor = torch.tensor([training_loss], device=ctx.device)
    all_losses = [torch.zeros_like(loss_tensor) for _ in range(ctx.world_size)]
    dist.all_gather(all_losses, loss_tensor)
    if ctx.rank == 0:
        loss_values = [l.item() for l in all_losses]
        spread = max(loss_values) - min(loss_values)
        consistent = spread < TOL.rank_loss_consistency_abs
        checks["loss_consistent_across_ranks"] = consistent
        log(f"  Loss consistent across ranks (spread={spread:.6f}): {'PASS' if consistent else 'FAIL'}")
    else:
        checks["loss_consistent_across_ranks"] = True

    return checks


# Mode runner


def run_mode(ctx, mode: str, tokenizer, train_dataset, eval_dataset) -> dict[str, bool]:
    """Run SFT training for one parallelism mode with metric validation."""
    parallelism_config = ParallelismConfig(**MODE_CONFIGS[mode])
    output_dir = os.path.join(ctx.output_dir, mode)

    log(f"\n  Loading model for {mode} ({parallelism_config.mode_string or 'standard FSDP'})...")
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_liger_kernel=True,
    )
    log(f"  Model loaded, GPU: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Build SFT config
    sft_kwargs = {
        "output_dir": output_dir,
        "max_steps": MAX_STEPS,
        "per_device_train_batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "bf16": True,
        "gradient_checkpointing": True,
        "use_liger_kernel": False,  # Already applied in load_distributed_model
        "logging_steps": 1,
        "save_strategy": "no",
        "eval_strategy": "steps",
        "eval_steps": EVAL_STEPS,
        "per_device_eval_batch_size": BATCH_SIZE,
        "report_to": "none",
        "max_length": MAX_SEQ_LENGTH,
        "dataloader_drop_last": True,
        "dataloader_num_workers": 0,
        "fsdp": "",
    }

    # CP splits the sequence across ranks, so every batch — train and eval alike — must come out
    # divisible by cp_size. The production script builds that collator through the same factory,
    # which refuses packing under CP (the dense causal kernel carries no document boundaries).
    trainer_kwargs = {}
    if parallelism_config.is_cp_mode:
        trainer_kwargs["data_collator"] = select_data_collator(
            tokenizer=tokenizer,
            pad_to_multiple_of=parallelism_config.cp_size,
            use_context_parallel=True,
        )

    config = SFTConfig(**sft_kwargs)

    # Create trainer
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
        **trainer_kwargs,
    )

    # Pre-training checks
    if parallelism_config.is_tp_mode:
        assert trainer.is_tp_mode, f"Expected is_tp_mode=True for {mode}"
        expected_dp = ctx.world_size // TP_SIZE
        assert parallelism_config.data_parallel_size == expected_dp, (
            f"Expected data_parallel_size={expected_dp}, got {parallelism_config.data_parallel_size}"
        )
    if parallelism_config.is_cp_mode:
        assert trainer.is_cp_mode, f"Expected is_cp_mode=True for {mode}"
        assert trainer.cp_size == CP_SIZE, f"Expected cp_size={CP_SIZE}, got {trainer.cp_size}"

    # Train
    log(f"  Training {MAX_STEPS} steps...")
    dist.barrier()
    train_result = trainer.train()

    # Extract and validate metrics
    training_loss = train_result.training_loss
    metrics = extract_metrics(trainer)

    log(f"\n  --- {mode} Metrics ---")
    log(f"  Final loss: {training_loss:.6f}")
    log(f"  Step losses: {[f'{l:.4f}' for l in metrics['loss']]}")
    if metrics["grad_norm"]:
        log(f"  Grad norms: {[f'{g:.2f}' for g in metrics['grad_norm']]}")
    if metrics["mean_token_accuracy"]:
        log(f"  Token accuracy: {[f'{t:.4f}' for t in metrics['mean_token_accuracy']]}")
    if metrics["eval_loss"]:
        log(f"  Eval losses: {[f'{e:.4f}' for e in metrics['eval_loss']]}")

    log(f"\n  --- {mode} Checks ---")
    checks = validate_metrics(training_loss, metrics, ctx)

    # Mode pin in the verdict: a tp/cp row that quietly fell back to plain FSDP would otherwise
    # pass on the metric checks alone, reporting coverage the run never had.
    if parallelism_config.is_tp_mode:
        checks["tp_mode_active"] = trainer.is_tp_mode
        log(f"  TP mode active: {'PASS' if trainer.is_tp_mode else 'FAIL'}")
    if parallelism_config.is_cp_mode:
        checks["cp_mode_active"] = trainer.is_cp_mode
        log(f"  CP mode active: {'PASS' if trainer.is_cp_mode else 'FAIL'}")

    # Cleanup
    del trainer, model
    cleanup_memory()
    dist.barrier()

    return checks


# Main


def run(ctx) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="all",
        choices=[*MODE_CONFIGS, "all"],
        help="Parallelism mode to test; 'all' runs every mode sequentially in one process.",
    )
    args = parser.parse_args()
    modes = list(MODE_CONFIGS) if args.mode == "all" else [args.mode]

    log(f"\n{'#' * 70}")
    log("  SFT Qwen3 Dense — Comprehensive Metric Validation")
    log(f"  World size: {ctx.world_size}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  Modes: {', '.join(modes)}")
    log(f"  Steps: {MAX_STEPS}, Batch: {BATCH_SIZE}, Seq: {MAX_SEQ_LENGTH}")
    log(f"{'#' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1)
    log(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)} samples")

    checks: dict[str, bool] = {}
    for mode in modes:
        log(f"\n{'=' * 60}")
        log(f"  MODE: {mode}")
        log(f"{'=' * 60}")

        mode_checks = run_mode(ctx, mode, tokenizer, train_dataset, eval_dataset)
        failed = [k for k, v in mode_checks.items() if not v]
        log(f"  {mode:>8s}: {'FAIL' if failed else 'PASS'}" + (f" -- failed={failed}" if failed else ""))
        # AND-merge: with several modes in one process (``--mode all``) a later mode's PASS must not
        # overwrite an earlier mode's FAIL on the same check name.
        for name, ok in mode_checks.items():
            checks[name] = ok and checks.get(name, True)

    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="test_sft_qwen3_dense")(run)

if __name__ == "__main__":
    main()
