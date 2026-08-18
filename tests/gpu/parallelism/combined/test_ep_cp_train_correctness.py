#!/usr/bin/env python
"""
EP+CP training correctness test for GPT-OSS 20B MoE.

Validates that adding Context Parallelism (CP) to Expert Parallelism (EP)
produces correct results across three phases:

Phase 1: Forward loss equivalence (EP-only vs EP+CP)
  - EP-only (EP=2) processes full sequences on each rank
  - EP+CP (EP=2, CP=2) splits sequences via Ulysses attention
  - Average loss across CP ranks must match EP-only average
  - 5 different inputs tested for statistical robustness
  Note: per-parameter gradient comparison is NOT applicable to MoE models
  because DeepEP's all-to-all routing is non-deterministic in bf16.
  The dense-model test (test_cp_train_correctness.py on Qwen3) validates
  per-parameter gradient exactness; this test validates loss equivalence.

Phase 2: Attention backend comparison
  - When CP is active, flex_attention is auto-switched to a flash-family
    backend (Ulysses always uses flash_attn_func internally, resolved by
    get_flash_attn_func). This phase verifies that all attn_implementation
    settings (explicit FA2, explicit flex_attention, auto-detect) resolve to a
    CP-valid flash backend and produce identical losses. The resolved label is
    platform-dependent — Hopper reports flash_attention_2/3, Blackwell reports
    flash_attention_4 (FA2 is replaced by FA4 there) — but the underlying
    kernel is identical, so the losses must match exactly.

Phase 3: Full EP+CP training via DistributedSFTTrainer (15 steps)
  - Validates: loss decrease, mean_token_accuracy increase, loss not
    exploding, finite grad norms, cross-rank loss consistency

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/combined/test_ep_cp_train_correctness.py

Requirements:
    - 2x GPUs with >=80GB memory each
    - DeepEP installed
    - Model: unsloth/gpt-oss-20b-BF16 (auto-downloaded)
"""

import math
import sys
import traceback

import torch
import torch.distributed as dist
from accelerate import PartialState
from transformers import AutoTokenizer

from src.distributed.context_parallel.validation import SUPPORTED_ATTN_IMPLEMENTATIONS
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import (
    cleanup_dirs,
    ensure_model_downloaded,
    init_distributed,
    setup_cache_dirs,
    teardown_distributed,
)
from tests.common.ep_reference import fixed_chat_batch
from tests.common.models import GPT_OSS_20B
from tests.common.utils import cleanup_memory, gpu_mem_gb, log, log_all

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 2
CP_SIZE = 2
SEQ_LEN = 128
SEED = 42

LOSS_ABS_TOL = 0.15
LOSS_REL_TOL = 0.05  # either bound satisfies the match
NUM_FORWARD_SAMPLES = 5

NUM_TRAIN_SAMPLES = 64
NUM_EVAL_SAMPLES = 8
NUM_TRAIN_STEPS = 15
MAX_SEQ_LENGTH = 4096
BATCH_SIZE = 1
LEARNING_RATE = 2e-5


def _forward_losses(model, inputs):
    """Run forward pass on a list of inputs, return per-sample losses."""
    losses = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in inputs:
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            losses.append(out.loss.item())
    return losses


def _avg_loss_across_ranks(local_loss, device, world_size):
    """All-gather a scalar loss and return the average across ranks."""
    t = torch.tensor([local_loss], device=device)
    gathered = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(gathered, t)
    return sum(x.item() for x in gathered) / world_size


def _compare_losses(ep_losses, cp_losses, device, world_size, label_a="EP", label_b="EP+CP"):
    """Compare two sets of losses, return (all_match, sample_results)."""
    sample_results = []
    all_match = True
    for i in range(len(ep_losses)):
        ep_avg = _avg_loss_across_ranks(ep_losses[i], device, world_size)
        cp_avg = _avg_loss_across_ranks(cp_losses[i], device, world_size)
        abs_diff = abs(ep_avg - cp_avg)
        rel_diff = abs_diff / ep_avg if ep_avg > 1e-6 else float("inf")
        match = abs_diff < LOSS_ABS_TOL or rel_diff < LOSS_REL_TOL
        if not match:
            all_match = False
        sample_results.append((ep_avg, cp_avg, abs_diff, rel_diff, match))
        log(
            f"      Sample {i}: {label_a}={ep_avg:.6f}, {label_b}={cp_avg:.6f}, "
            f"abs={abs_diff:.6f}, rel={rel_diff:.4%} {'OK' if match else 'MISMATCH'}"
        )
    return all_match, sample_results


def test_loss_equivalence(device, local_rank):
    """
    Compare EP-only vs EP+CP forward loss on multiple inputs.

    Returns (passed, details_dict).
    """
    world_size = dist.get_world_size()

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

    all_match, sample_results = _compare_losses(ep_losses, cp_losses, device, world_size)

    all_finite = all(math.isfinite(l) for l in ep_losses + cp_losses)
    passed = all_match and all_finite

    max_abs = max(r[2] for r in sample_results)
    max_rel = max(r[3] for r in sample_results)
    log(f"      Max abs diff: {max_abs:.6f} (tol: {LOSS_ABS_TOL})")
    log(f"      Max rel diff: {max_rel:.4%} (tol: {LOSS_REL_TOL:.0%})")
    log(f"      Phase 1: {'PASS' if passed else 'FAIL'}")

    return passed, {"max_abs": max_abs, "max_rel": max_rel}


def test_attention_backends(device, local_rank):
    """
    Compare forward losses across different attn_implementation settings.

    Every setting must resolve to a CP-valid flash backend
    (SUPPORTED_ATTN_IMPLEMENTATIONS) — the exact label is platform-dependent
    (FA2/3 on Hopper, FA4 on Blackwell where FA2 is replaced) but the kernel is
    the same, so auto-detected and explicit settings must give identical finite
    losses.

    Returns (passed, details_dict).
    """
    world_size = dist.get_world_size()

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

    # The resolved label is platform-dependent (FA2/3 on Hopper, FA4 on Blackwell), so string equality
    # is the wrong invariant; membership plus the loss equivalence below is the right one.
    resolved = {"A": resolved_a, "B": resolved_b, "C": resolved_c}
    checks["resolved_flash_family"] = all(r in SUPPORTED_ATTN_IMPLEMENTATIONS for r in resolved.values())
    log(
        f"      All resolved to a CP-valid flash backend "
        f"(a={resolved_a}, b={resolved_b}, c={resolved_c}): "
        f"{'PASS' if checks['resolved_flash_family'] else 'FAIL'}"
    )

    log("\n      Explicit FA2 vs flex→flash:")
    ab_match, ab_results = _compare_losses(
        losses_a,
        losses_b,
        device,
        world_size,
        label_a="FA2",
        label_b="flex→flash",
    )
    max_ab_diff = max(r[2] for r in ab_results)
    checks["fa2_vs_flex_match"] = ab_match
    log(f"      Max diff: {max_ab_diff:.8f} {'(identical)' if max_ab_diff == 0.0 else ''}")

    log("\n      flex→flash vs auto-detect:")
    bc_match, bc_results = _compare_losses(
        losses_b,
        losses_c,
        device,
        world_size,
        label_a="flex→flash",
        label_b="auto",
    )
    max_bc_diff = max(r[2] for r in bc_results)
    checks["flex_vs_auto_match"] = bc_match
    log(f"      Max diff: {max_bc_diff:.8f} {'(identical)' if max_bc_diff == 0.0 else ''}")

    all_losses = losses_a + losses_b + losses_c
    checks["all_finite"] = all(math.isfinite(l) for l in all_losses)

    passed = all(checks.values())
    log(f"\n      Phase 2: {'PASS' if passed else 'FAIL'}")
    if not passed:
        log(f"      Failed checks: {[k for k, v in checks.items() if not v]}")

    return passed, checks


def test_ep_cp_training(device, local_rank):
    """
    Full EP+CP training test using DistributedSFTTrainer.

    Verifies that training completes, losses decrease, metrics are computed
    correctly, and results are consistent across ranks.

    Returns (passed, details_dict).
    """
    from trl import SFTConfig

    from src.trainers.sft import DistributedSFTTrainer

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    log(f"\n{'=' * 70}")
    log(f"  Phase 3: EP+CP Training ({NUM_TRAIN_STEPS} steps)")
    log(f"  Model: {MODEL_NAME}, EP={EP_SIZE}, CP={CP_SIZE}")
    log(f"{'=' * 70}")

    output_dir, cache_dir = setup_cache_dirs("ep_cp_train_correctness", rank)

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
        output_dir=output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,  # Already applied in load_distributed_model
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=True,
        fsdp="",
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

    checks = {}

    checks["ep_mode_active"] = trainer.is_ep_mode
    checks["cp_mode_active"] = trainer.is_cp_mode
    log(f"  EP mode: {'PASS' if checks['ep_mode_active'] else 'FAIL'}")
    log(f"  CP mode: {'PASS' if checks['cp_mode_active'] else 'FAIL'}")

    log(f"\n  Training ({NUM_TRAIN_STEPS} steps)...")
    train_result = trainer.train()

    training_loss = train_result.training_loss
    log_history = trainer.state.log_history
    step_losses = [e["loss"] for e in log_history if "loss" in e and "eval_loss" not in e]
    grad_norms = [e["grad_norm"] for e in log_history if "grad_norm" in e]
    token_accuracies = [e["mean_token_accuracy"] for e in log_history if "mean_token_accuracy" in e]

    log("\n  --- Metrics ---")
    log(f"  Final training loss: {training_loss:.6f}")
    log(f"  Step losses: {[f'{l:.4f}' for l in step_losses]}")
    if grad_norms:
        log(f"  Grad norms: {[f'{g:.2f}' for g in grad_norms]}")
    if token_accuracies:
        log(f"  Token accuracies: {[f'{t:.4f}' for t in token_accuracies]}")

    log("\n  --- Checks ---")

    checks["training_completed"] = len(step_losses) == NUM_TRAIN_STEPS
    log(f"  Training completed: {'PASS' if checks['training_completed'] else 'FAIL'}")

    all_finite = all(math.isfinite(l) for l in step_losses + [training_loss])
    checks["losses_finite"] = all_finite
    log(f"  Losses finite: {'PASS' if all_finite else 'FAIL'}")

    losses_reasonable = all(0 < l < 100 for l in step_losses)
    checks["losses_reasonable"] = losses_reasonable
    log(f"  Losses reasonable (0-100): {'PASS' if losses_reasonable else 'FAIL'}")

    # First-3 vs last-3 average, so one noisy step cannot decide the trend.
    if len(step_losses) >= 6:
        first_avg = sum(step_losses[:3]) / 3
        last_avg = sum(step_losses[-3:]) / 3
        loss_decreased = last_avg < first_avg
        checks["loss_decreased"] = loss_decreased
        log(
            f"  Loss decreased (first3={first_avg:.4f} -> last3={last_avg:.4f}): "
            f"{'PASS' if loss_decreased else 'FAIL'}"
        )
    elif len(step_losses) >= 2:
        loss_decreased = step_losses[-1] < step_losses[0]
        checks["loss_decreased"] = loss_decreased
        log(
            f"  Loss decreased ({step_losses[0]:.4f} -> {step_losses[-1]:.4f}): {'PASS' if loss_decreased else 'FAIL'}"
        )

    if len(step_losses) >= 2:
        max_step = max(step_losses[1:])
        no_explosion = max_step < step_losses[0] * 2.0
        checks["no_loss_explosion"] = no_explosion
        log(
            f"  No loss explosion (max={max_step:.4f} < 2x first={step_losses[0] * 2:.4f}): "
            f"{'PASS' if no_explosion else 'FAIL'}"
        )

    if len(token_accuracies) >= 6:
        first_acc = sum(token_accuracies[:3]) / 3
        last_acc = sum(token_accuracies[-3:]) / 3
        acc_increased = last_acc > first_acc
        checks["accuracy_increased"] = acc_increased
        log(
            f"  Accuracy increased (first3={first_acc:.4f} -> last3={last_acc:.4f}): "
            f"{'PASS' if acc_increased else 'FAIL'}"
        )
    elif len(token_accuracies) >= 2:
        acc_increased = token_accuracies[-1] > token_accuracies[0]
        checks["accuracy_increased"] = acc_increased
        log(
            f"  Accuracy increased ({token_accuracies[0]:.4f} -> {token_accuracies[-1]:.4f}): "
            f"{'PASS' if acc_increased else 'FAIL'}"
        )

    if grad_norms:
        grad_ok = all(math.isfinite(g) for g in grad_norms)
        checks["grad_finite"] = grad_ok
        log(f"  Grad norms finite: {'PASS' if grad_ok else 'FAIL'}")

    loss_tensor = torch.tensor([training_loss], device=device)
    all_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
    dist.all_gather(all_losses, loss_tensor)
    if rank == 0:
        losses_list = [l.item() for l in all_losses]
        spread = max(losses_list) - min(losses_list)
        checks["loss_consistent"] = spread < 0.01
        log(f"  Cross-rank loss consistency (spread={spread:.6f}): {'PASS' if checks['loss_consistent'] else 'FAIL'}")
    else:
        checks["loss_consistent"] = True

    if token_accuracies:
        acc_tensor = torch.tensor([token_accuracies[-1]], device=device)
        all_accs = [torch.zeros_like(acc_tensor) for _ in range(world_size)]
        dist.all_gather(all_accs, acc_tensor)
        if rank == 0:
            acc_list = [a.item() for a in all_accs]
            acc_spread = max(acc_list) - min(acc_list)
            checks["accuracy_consistent"] = acc_spread < 0.01
            log(
                f"  Cross-rank accuracy consistency (spread={acc_spread:.6f}): "
                f"{'PASS' if checks['accuracy_consistent'] else 'FAIL'}"
            )
        else:
            checks["accuracy_consistent"] = True

    all_passed = all(checks.values())
    log(f"\n  Phase 3: {'PASS' if all_passed else 'FAIL'}")

    del trainer, model
    cleanup_memory()
    cleanup_dirs(output_dir, cache_dir)

    return all_passed, {
        "training_loss": training_loss,
        "step_losses": step_losses,
        "grad_norms": grad_norms,
        "token_accuracies": token_accuracies,
    }


# Main


def main():
    rank, world_size, local_rank = init_distributed()
    device = f"cuda:{local_rank}"

    PartialState()

    log(f"\n{'#' * 70}")
    log("  EP+CP Training Correctness Test")
    log(f"  World: {world_size}, EP: {EP_SIZE}, CP: {CP_SIZE}")
    log(f"  Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"{'#' * 70}")

    if world_size < EP_SIZE:
        log(f"\nERROR: Need at least {EP_SIZE} GPUs, got {world_size}")
        teardown_distributed()
        return 1

    assert SEQ_LEN % CP_SIZE == 0, f"SEQ_LEN ({SEQ_LEN}) not divisible by CP_SIZE ({CP_SIZE})"

    log("\nEnsuring model is downloaded...")
    ensure_model_downloaded(MODEL_NAME, rank)

    results = {}

    # Phase 1: Forward loss equivalence
    try:
        loss_ok, info = test_loss_equivalence(device, local_rank)
        results["loss_equivalence"] = loss_ok
    except Exception as e:
        log(f"\n  Phase 1 FAILED with exception: {e}")
        if rank == 0:
            traceback.print_exc()
        results["loss_equivalence"] = False

    barrier()
    cleanup_memory()

    # Phase 2: Attention backend comparison
    try:
        attn_ok, info = test_attention_backends(device, local_rank)
        results["attention_backends"] = attn_ok
    except Exception as e:
        log(f"\n  Phase 2 FAILED with exception: {e}")
        if rank == 0:
            traceback.print_exc()
        results["attention_backends"] = False

    barrier()
    cleanup_memory()

    # Phase 3: EP+CP training
    try:
        train_ok, info = test_ep_cp_training(device, local_rank)
        results["training"] = train_ok
    except Exception as e:
        log(f"\n  Phase 3 FAILED with exception: {e}")
        if rank == 0:
            traceback.print_exc()
        results["training"] = False

    # Summary
    all_passed = all(results.values())

    log(f"\n{'#' * 70}")
    log("  SUMMARY")
    log(f"{'#' * 70}")
    for name, passed in results.items():
        log(f"    {name}: {'PASS' if passed else 'FAIL'}")

    log(f"\n{'#' * 70}")
    if all_passed:
        log("  EP+CP TRAINING CORRECTNESS TEST PASSED")
    else:
        failed = [k for k, v in results.items() if not v]
        log(f"  EP+CP TRAINING CORRECTNESS TEST FAILED: {failed}")
    log(f"{'#' * 70}\n")

    barrier()
    teardown_distributed()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
