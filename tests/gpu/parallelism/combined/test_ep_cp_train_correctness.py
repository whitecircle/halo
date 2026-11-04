#!/usr/bin/env python
"""EP+CP on GPT-OSS 20B against EP alone: forward loss, attention backends, and full trainer training.

Under CP every attn_implementation must resolve to a CP-valid flash backend. The label is platform-dependent (FA2/3 on
Hopper, FA4 on Blackwell), but Ulysses always calls ``_flash_attn_func``, so losses must match exactly.
"""

import math

import torch
from transformers import AutoTokenizer
from trl import SFTConfig

from halo.distributed.context_parallel.model_rules import SUPPORTED_ATTN_IMPLEMENTATIONS
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.runtime.collectives import barrier
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded, world_mean
from tests.common.ep_reference import fixed_chat_batch
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory, gpu_mem_gb
from tests.common.models import GPT_OSS_20B
from tests.common.numerics import max_or_nan
from tests.common.parallel_shape import parallel_shape_checks
from tests.common.rank_log import log, log_all
from tests.common.training_checks import step_losses, training_run_checks

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 2
CP_SIZE = 2
SEQ_LEN = 128
SEED = 42

LOSS_ABS_TOL = 0.15
LOSS_REL_TOL = 0.05  # Either bound satisfies the match.
NUM_FORWARD_SAMPLES = 5

NUM_TRAIN_SAMPLES = 64
NUM_EVAL_SAMPLES = 8
NUM_TRAIN_STEPS = 15
MAX_SEQ_LENGTH = 4096
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
TREND_WINDOW = 3
LOSS_EXPLOSION_FACTOR = 2.0


def _forward_losses(model, inputs):
    losses = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in inputs:
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            losses.append(out.loss.item())
    return losses


def _compare_losses(ep_losses, cp_losses, device):
    """Under no_grad the CP wrapper returns the group mean on every rank, so the CP side is a full-sequence loss
    too."""
    sample_results = []
    all_match = True
    for i in range(len(ep_losses)):
        ep_avg = world_mean(ep_losses[i], device)
        cp_avg = world_mean(cp_losses[i], device)
        abs_diff = abs(ep_avg - cp_avg)
        rel_diff = abs_diff / ep_avg if ep_avg > 1e-6 else float("inf")
        match = abs_diff < LOSS_ABS_TOL or rel_diff < LOSS_REL_TOL
        if not match:
            all_match = False
        sample_results.append((ep_avg, cp_avg, abs_diff, rel_diff, match))
        log(
            f"      Sample {i}: EP={ep_avg:.6f}, EP+CP={cp_avg:.6f}, "
            f"abs={abs_diff:.6f}, rel={rel_diff:.4%} {'OK' if match else 'MISMATCH'}"
        )
    return all_match, sample_results


def test_loss_equivalence(device):
    log(f"\n{'=' * 70}")
    log("  Phase 1: Forward Loss Equivalence")
    log(f"  EP-only (EP={EP_SIZE}) vs EP+CP (EP={EP_SIZE}, CP={CP_SIZE})")
    log(f"  {NUM_FORWARD_SAMPLES} inputs, abs_tol={LOSS_ABS_TOL}, rel_tol={LOSS_REL_TOL:.0%}")
    log(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = []
    for i in range(NUM_FORWARD_SAMPLES):
        inputs.append(fixed_chat_batch(tokenizer, SEQ_LEN, device, conversation=1 + i, broadcast=True))

    log(f"\n  [A] EP-only forward ({NUM_FORWARD_SAMPLES} samples)...")
    ep_config = ParallelismConfig(ep_size=EP_SIZE, cp_size=1, ep_fp32_router=True)

    model_ep, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=ep_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model_ep.eval()

    ep_losses = _forward_losses(model_ep, inputs)
    for i, l in enumerate(ep_losses):
        log_all(f"      Sample {i}: loss={l:.6f}")

    log(f"      GPU memory: {gpu_mem_gb():.1f} GB")
    del model_ep
    cleanup_memory()
    barrier()

    log(f"\n  [B] EP+CP forward ({NUM_FORWARD_SAMPLES} samples)...")
    ep_cp_config = ParallelismConfig(ep_size=EP_SIZE, cp_size=CP_SIZE, ep_fp32_router=True)

    model_ep_cp, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=ep_cp_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model_ep_cp.eval()

    cp_losses = _forward_losses(model_ep_cp, inputs)
    for i, l in enumerate(cp_losses):
        log_all(f"      Sample {i}: loss={l:.6f}")

    log(f"      GPU memory: {gpu_mem_gb():.1f} GB")
    del model_ep_cp
    cleanup_memory()
    barrier()

    log("\n  --- Results ---")

    all_match, sample_results = _compare_losses(ep_losses, cp_losses, device)

    all_finite = all(math.isfinite(l) for l in ep_losses + cp_losses)
    passed = all_match and all_finite

    max_abs = max_or_nan(r[2] for r in sample_results)
    max_rel = max_or_nan(r[3] for r in sample_results)
    log(f"      Max abs diff: {max_abs:.6f} (tol: {LOSS_ABS_TOL})")
    log(f"      Max rel diff: {max_rel:.4%} (tol: {LOSS_REL_TOL:.0%})")
    log(f"      Phase 1: {'PASS' if passed else 'FAIL'}")

    return {"ep_vs_ep_cp_losses_match": all_match, "forward_losses_finite": all_finite}


def test_attention_backends(device):
    log(f"\n{'=' * 70}")
    log("  Phase 2: Attention Backend Comparison")
    log("  Comparing attn_implementation settings under EP+CP")
    log(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = []
    for i in range(3):
        inputs.append(fixed_chat_batch(tokenizer, SEQ_LEN, device, conversation=1 + i, broadcast=True))

    ep_cp_config = ParallelismConfig(ep_size=EP_SIZE, cp_size=CP_SIZE, ep_fp32_router=True)

    log("\n  [A] attn_implementation='flash_attention_2' (explicit)...")
    model_a, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=ep_cp_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    resolved_a = getattr(model_a.config if hasattr(model_a, "config") else None, "_attn_implementation", "unknown")
    log(f"      Resolved implementation: {resolved_a}")
    model_a.eval()

    losses_a = _forward_losses(model_a, inputs)
    for i, l in enumerate(losses_a):
        log_all(f"      Sample {i}: loss={l:.6f}")

    del model_a
    cleanup_memory()
    barrier()

    log("\n  [B] attn_implementation='flex_attention' (→ auto-switched to flash under CP)...")
    model_b, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=ep_cp_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flex_attention",
    )
    resolved_b = getattr(model_b.config if hasattr(model_b, "config") else None, "_attn_implementation", "unknown")
    log(f"      Resolved implementation: {resolved_b}")
    model_b.eval()

    losses_b = _forward_losses(model_b, inputs)
    for i, l in enumerate(losses_b):
        log_all(f"      Sample {i}: loss={l:.6f}")

    del model_b
    cleanup_memory()
    barrier()

    log("\n  [C] attn_implementation=None (auto-detect)...")
    model_c, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=ep_cp_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    resolved_c = getattr(model_c.config if hasattr(model_c, "config") else None, "_attn_implementation", "unknown")
    log(f"      Resolved implementation: {resolved_c}")
    model_c.eval()

    losses_c = _forward_losses(model_c, inputs)
    for i, l in enumerate(losses_c):
        log_all(f"      Sample {i}: loss={l:.6f}")

    del model_c
    cleanup_memory()
    barrier()

    log("\n  --- Results ---")

    checks = {}

    # The label is platform-dependent, so check membership plus the loss equivalence below.
    resolved = {"A": resolved_a, "B": resolved_b, "C": resolved_c}
    checks["resolved_flash_family"] = all(r in SUPPORTED_ATTN_IMPLEMENTATIONS for r in resolved.values())
    log(
        f"      All resolved to a CP-valid flash backend "
        f"(a={resolved_a}, b={resolved_b}, c={resolved_c}): "
        f"{'PASS' if checks['resolved_flash_family'] else 'FAIL'}"
    )

    # Under CP the kernel comes from _flash_attn_func's arch probe, never the label, so losses match bit for bit.
    checks["fa2_vs_flex_identical"] = all(a == b for a, b in zip(losses_a, losses_b, strict=True))
    checks["flex_vs_auto_identical"] = all(b == c for b, c in zip(losses_b, losses_c, strict=True))
    log_all(f"      Explicit FA2 vs flex→flash identical: {checks['fa2_vs_flex_identical']}")
    log_all(f"      flex→flash vs auto-detect identical: {checks['flex_vs_auto_identical']}")

    all_losses = losses_a + losses_b + losses_c
    checks["backend_losses_finite"] = all(math.isfinite(l) for l in all_losses)

    passed = all(checks.values())
    log(f"\n      Phase 2: {'PASS' if passed else 'FAIL'}")
    if not passed:
        log(f"      Failed checks: {[k for k, v in checks.items() if not v]}")

    return checks


def test_ep_cp_training(ctx):
    log(f"\n{'=' * 70}")
    log(f"  Phase 3: EP+CP Training ({NUM_TRAIN_STEPS} steps)")
    log(f"  Model: {MODEL_NAME}, EP={EP_SIZE}, CP={CP_SIZE}")
    log(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("\n  Creating datasets...")
    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)
    log(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    log(f"\n  Loading model with EP={EP_SIZE}, CP={CP_SIZE}...")
    parallelism_config = ParallelismConfig(ep_size=EP_SIZE, cp_size=CP_SIZE, ep_fp32_router=True)

    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        use_liger_kernel=True,
    )
    log(f"  GPU memory after load: {gpu_mem_gb():.1f} GB")

    sft_config = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,  # Already applied in load_distributed_model.
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
    )

    log("\n  Creating DistributedSFTTrainer...")
    trainer = DistributedSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    ctx.on_teardown(trainer.cleanup_ep)

    checks = parallel_shape_checks(model, parallelism_config)

    log(f"\n  Training ({NUM_TRAIN_STEPS} steps)...")
    train_result = trainer.train()
    checks |= training_run_checks(train_result, trainer, NUM_TRAIN_STEPS, grad_norms=True)

    log_history = trainer.state.log_history
    losses = step_losses(trainer)
    grad_norms = [e["grad_norm"] for e in log_history if "grad_norm" in e]
    token_accuracies = [e["mean_token_accuracy"] for e in log_history if "mean_token_accuracy" in e]
    log(f"  Token accuracies: {[f'{t:.4f}' for t in token_accuracies]}")

    # logging_steps=1: a short series fails here rather than skipping the trend checks below.
    checks["every_step_logged"] = len(losses) == len(grad_norms) == len(token_accuracies) == NUM_TRAIN_STEPS
    log(f"  Every step logged: {'PASS' if checks['every_step_logged'] else 'FAIL'}")

    # So one noisy step cannot decide the trend.
    first_avg = sum(losses[:TREND_WINDOW]) / TREND_WINDOW
    last_avg = sum(losses[-TREND_WINDOW:]) / TREND_WINDOW
    checks["loss_decreased"] = last_avg < first_avg
    log(
        f"  Loss decreased (first{TREND_WINDOW}={first_avg:.4f} -> last{TREND_WINDOW}={last_avg:.4f}): "
        f"{'PASS' if checks['loss_decreased'] else 'FAIL'}"
    )

    max_step = max_or_nan(losses[1:])
    explosion_ceiling = losses[0] * LOSS_EXPLOSION_FACTOR
    checks["no_loss_explosion"] = max_step < explosion_ceiling
    log(
        f"  No loss explosion (max={max_step:.4f} < {LOSS_EXPLOSION_FACTOR}x first={explosion_ceiling:.4f}): "
        f"{'PASS' if checks['no_loss_explosion'] else 'FAIL'}"
    )

    first_acc = sum(token_accuracies[:TREND_WINDOW]) / TREND_WINDOW
    last_acc = sum(token_accuracies[-TREND_WINDOW:]) / TREND_WINDOW
    checks["accuracy_increased"] = last_acc > first_acc
    log(
        f"  Accuracy increased (first{TREND_WINDOW}={first_acc:.4f} -> last{TREND_WINDOW}={last_acc:.4f}): "
        f"{'PASS' if checks['accuracy_increased'] else 'FAIL'}"
    )

    log(f"\n  Phase 3: {'PASS' if all(checks.values()) else 'FAIL'}")
    return checks


@gpu_test_main(min_world_size=EP_SIZE, prefix="ep_cp_train_correctness")
def run(ctx):
    device = f"cuda:{ctx.local_rank}"

    log(f"\n{'#' * 70}")
    log("  EP+CP Training Correctness Test")
    log(f"  World: {ctx.world_size}, EP: {EP_SIZE}, CP: {CP_SIZE}")
    log(f"  Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"{'#' * 70}")

    assert SEQ_LEN % CP_SIZE == 0, f"SEQ_LEN ({SEQ_LEN}) not divisible by CP_SIZE ({CP_SIZE})"

    log("\nEnsuring model is downloaded...")
    ensure_model_downloaded(MODEL_NAME, ctx.rank)

    checks = test_loss_equivalence(device)
    barrier()
    cleanup_memory()

    checks |= test_attention_backends(device)
    barrier()
    cleanup_memory()

    checks |= test_ep_cp_training(ctx)
    return {"checks": checks}


if __name__ == "__main__":
    run()
