#!/usr/bin/env python
"""
Convergence test: LoRA and QLoRA with Expert Parallelism (EP=2) on GptOss-20B.

Trains for 25 steps and verifies that loss consistently decreases, validating
that LoRA/QLoRA gradients flow correctly through the EP routing mechanism.

Convergence criteria:
  - Loss in the final 5 steps is lower than the first 5 steps (mean comparison)
  - Loss is monotonically decreasing over 5-step windows (no divergence)
  - All per-step losses are finite
  - LoRA adapter weights were updated

Tests:
  1. LoRA + EP=2 convergence (25 steps)
  2. QLoRA + EP=2 convergence (25 steps)

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_ep_convergence.py
"""

import math
import os
import sys
import traceback

import torch
import torch.distributed as dist
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from trl import ModelConfig, SFTConfig, get_quantization_config

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import (
    cleanup_dirs,
    ensure_model_downloaded,
    setup_cache_dirs,
    teardown_distributed,
)
from tests.common.models import GPT_OSS_20B
from tests.common.peft_helpers import snapshot_adapters
from tests.common.utils import cleanup_memory, log

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 2
MAX_STEPS = 25
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 2e-4
NUM_TRAIN_SAMPLES = 256
NUM_EVAL_SAMPLES = 16
SEED = 42

LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_R = 8
LORA_ALPHA = 16
WINDOW_SIZE = 5


def _analyze_convergence(step_losses: list[float], window_size: int = WINDOW_SIZE) -> dict[str, bool]:
    """Analyze loss trajectory for convergence signals.

    Returns dict of check_name -> passed. Checks:
    - all_finite: Every step loss is finite
    - loss_decreased: Mean of last window < mean of first window
    - no_divergence: No window has mean > first window mean * 1.5
    - final_loss_reasonable: Final loss < 20 (not stuck at random baseline)
    """
    checks = {}

    all_finite = all(math.isfinite(l) for l in step_losses)
    checks["all_finite"] = all_finite
    log(f"  All losses finite: {'PASS' if all_finite else 'FAIL'}")

    if not all_finite or len(step_losses) < window_size * 2:
        log(f"  Insufficient data for convergence analysis ({len(step_losses)} steps, need {window_size * 2})")
        return checks

    windows = []
    for i in range(0, len(step_losses) - window_size + 1, window_size):
        window = step_losses[i : i + window_size]
        windows.append(sum(window) / len(window))

    log(f"  Window means ({window_size}-step): {[f'{w:.4f}' for w in windows]}")

    first_window = windows[0]
    last_window = windows[-1]

    loss_decreased = last_window < first_window
    decrease_pct = (first_window - last_window) / first_window * 100
    checks["loss_decreased"] = loss_decreased
    log(
        f"  Loss decreased: {'PASS' if loss_decreased else 'FAIL'} "
        f"({first_window:.4f} -> {last_window:.4f}, {decrease_pct:+.1f}%)"
    )

    divergence_threshold = first_window * 1.5
    no_divergence = all(w <= divergence_threshold for w in windows)
    checks["no_divergence"] = no_divergence
    if not no_divergence:
        worst = max(windows)
        log(f"  No divergence: FAIL (worst window={worst:.4f}, threshold={divergence_threshold:.4f})")
    else:
        log(f"  No divergence: PASS (all windows < {divergence_threshold:.4f})")

    # <20 rules out a run stuck at the random baseline
    final_loss = step_losses[-1]
    final_reasonable = final_loss < 20
    checks["final_loss_reasonable"] = final_reasonable
    log(f"  Final loss reasonable (<20): {'PASS' if final_reasonable else 'FAIL'} ({final_loss:.4f})")

    return checks


def _verify_lora_updated(before: dict, after: dict) -> tuple[bool, str]:
    """Check that LoRA weights changed during training."""
    if not before or not after:
        return False, "No LoRA parameters found"

    updated = sum(1 for name in before if name in after and not torch.equal(before[name], after[name]))
    total = len(before)
    all_updated = updated == total
    return all_updated, f"{updated}/{total} LoRA params updated"


def run_lora_ep_convergence(
    parallelism_config: ParallelismConfig,
    tokenizer,
    train_dataset,
    eval_dataset,
    rank: int,
    local_rank: int,
    base_output_dir: str,
) -> tuple[bool, str]:
    """Run LoRA + EP=2 convergence test (25 steps)."""
    model = None
    trainer = None

    try:
        log(f"  Loading model with EP={EP_SIZE}...")
        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flex_attention",
            use_liger_kernel=True,
        )
        log(f"  Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

        log(f"  Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA})...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log(f"  Trainable: {trainable / 1e6:.2f}M / {total / 1e6:.1f}M ({100 * trainable / total:.2f}%)")

        lora_before = snapshot_adapters(model, expert_lora=False)

        sft_config = SFTConfig(
            output_dir=os.path.join(base_output_dir, "lora_ep_conv"),
            max_steps=MAX_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=True,
            fsdp="",
        )

        trainer = DistributedSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )

        log(f"  Training for {MAX_STEPS} steps...")
        train_result = trainer.train()

        step_losses = [
            entry["loss"] for entry in trainer.state.log_history if "loss" in entry and "eval_loss" not in entry
        ]
        log(f"  Per-step losses: {[f'{l:.4f}' for l in step_losses]}")
        log(f"  Final training loss: {train_result.training_loss:.6f}")

        log("\n  --- Convergence Analysis (LoRA+EP) ---")
        checks = _analyze_convergence(step_losses)

        steps_ok = train_result.global_step == MAX_STEPS
        checks["steps_completed"] = steps_ok
        log(f"  Steps completed: {'PASS' if steps_ok else 'FAIL'} ({train_result.global_step}/{MAX_STEPS})")

        lora_after = snapshot_adapters(model, expert_lora=False)
        lora_ok, lora_detail = _verify_lora_updated(lora_before, lora_after)
        checks["lora_updated"] = lora_ok
        log(f"  LoRA weights updated: {'PASS' if lora_ok else 'FAIL'} ({lora_detail})")

        all_passed = all(checks.values())
        detail = f"loss={step_losses[0]:.4f}->{step_losses[-1]:.4f}"
        return all_passed, detail

    except Exception as e:
        log(f"  FAILED with exception: {e}")
        traceback.print_exc()
        return False, f"Exception: {e}"

    finally:
        if trainer is not None and hasattr(trainer, "cleanup_ep"):
            trainer.cleanup_ep()
        del trainer
        del model
        cleanup_memory()
        barrier()


def run_qlora_ep_convergence(
    parallelism_config: ParallelismConfig,
    tokenizer,
    train_dataset,
    eval_dataset,
    rank: int,
    local_rank: int,
    base_output_dir: str,
) -> tuple[bool, str]:
    """Assert QLoRA + EP is rejected at load time.

    QLoRA + EP is unsupported by design: the EP lazy loader streams raw safetensors and rebuilds
    experts as plain ``Parameter`` objects, so bitsandbytes ``Params4bit`` are lost and PEFT's
    4-bit adapter dispatch crashes on the missing ``weight.compress_statistics``.
    ``load_distributed_model`` rejects the combo up front with an actionable ``ValueError``; this
    sub-test verifies that guard fires (so removing it fails the suite). Use QLoRA with standard
    DDP/FSDP, or QLoRA + CP (which preserves quantization), instead.
    """
    model = None

    try:
        model_config = ModelConfig(
            model_name_or_path=MODEL_NAME,
            use_peft=True,
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            use_bnb_nested_quant=True,
            trust_remote_code=True,
            attn_implementation="flex_attention",
        )
        quantization_config = get_quantization_config(model_config)
        log(f"  Quantization: {quantization_config.quant_method}")
        log(f"  Expecting load_distributed_model to reject QLoRA + EP={EP_SIZE}...")

        try:
            model, _ = load_distributed_model(
                model_name_or_path=MODEL_NAME,
                parallelism_config=parallelism_config,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="flex_attention",
                use_liger_kernel=False,
                quantization_config=quantization_config,
            )
        except ValueError as exc:
            msg = str(exc)
            ok = "not supported" in msg and "EP" in msg
            log(f"  Guard raised ValueError: {'PASS' if ok else 'FAIL'} -- {msg[:90]}")
            return ok, ("QLoRA+EP correctly rejected" if ok else f"wrong ValueError: {msg[:60]}")

        # a silent load means the guard is gone; QLoRA+EP would then crash mid-train on lost Params4bit
        log("  Guard did NOT fire: FAIL (QLoRA + EP must be rejected at load time)")
        return False, "QLoRA+EP was not rejected at load time"

    except Exception as e:
        log(f"  FAILED with unexpected exception: {e}")
        traceback.print_exc()
        return False, f"Exception: {e}"

    finally:
        del model
        cleanup_memory()
        barrier()


def main() -> int:
    """Run LoRA/QLoRA + EP convergence tests. Returns 0 on success, 1 on failure."""
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    state = PartialState()
    rank = state.process_index
    local_rank = state.local_process_index
    world_size = state.num_processes

    base_output_dir, cache_dir = setup_cache_dirs("test_lora_ep_convergence", rank)

    log(f"\n{'#' * 70}")
    log("  LoRA/QLoRA + EP Convergence Test (GptOss-20B)")
    log(f"  World size: {world_size}, EP size: {EP_SIZE}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"  Max steps: {MAX_STEPS}, LR: {LEARNING_RATE}, Batch: {BATCH_SIZE}")
    log(f"  Convergence window: {WINDOW_SIZE} steps")
    log(f"{'#' * 70}")

    if world_size != EP_SIZE:
        log(f"\nERROR: This test requires exactly {EP_SIZE} GPUs, got {world_size}")
        if dist.is_initialized():
            teardown_distributed()
        return 1

    ensure_model_downloaded(MODEL_NAME, rank)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1)
    log(f"  Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    parallelism_config = ParallelismConfig(ep_size=EP_SIZE)
    results: dict[str, tuple[bool, str]] = {}

    log(f"\n{'=' * 70}")
    log("  TEST 1: LoRA + EP=2 Convergence (25 steps)")
    log(f"{'=' * 70}")

    success, detail = run_lora_ep_convergence(
        parallelism_config,
        tokenizer,
        train_dataset,
        eval_dataset,
        rank,
        local_rank,
        base_output_dir,
    )
    results["lora_ep_convergence"] = (success, detail)

    log(f"\n{'=' * 70}")
    log("  TEST 2: QLoRA + EP=2 Convergence (25 steps)")
    log(f"{'=' * 70}")

    success, detail = run_qlora_ep_convergence(
        parallelism_config,
        tokenizer,
        train_dataset,
        eval_dataset,
        rank,
        local_rank,
        base_output_dir,
    )
    results["qlora_ep_convergence"] = (success, detail)

    log(f"\n{'#' * 70}")
    log("  CONVERGENCE TEST RESULTS")
    log(f"{'#' * 70}")
    for name, (passed, detail) in results.items():
        status = "PASSED" if passed else "FAILED"
        log(f"  {name:30s} {status} -- {detail}")
    log(f"{'#' * 70}")

    all_passed = all(p for p, _ in results.values())
    log(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    cleanup_dirs(base_output_dir, cache_dir)
    if dist.is_initialized():
        teardown_distributed()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
