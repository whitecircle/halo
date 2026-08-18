#!/usr/bin/env python
"""
Context Parallelism training correctness test for Qwen3.

Validates that CP training produces *equivalent* results to non-CP training by
comparing, step-by-step on the same Qwen3-0.6B model and data:

Phase 1 — Forward pass loss equivalence (tighter than test_cp_correctness.py)
Phase 2 — Gradient equivalence after forward+backward (core correctness proof)
Phase 3 — Multi-step training trajectory and final weight convergence

Non-CP baseline runs each rank independently on the full sequence (no gradient
sync needed since each rank has identical model + data). CP mode uses the
UlyssesCPModelWrapper which splits sequences and coordinates via all-to-all.

After CP backward, gradients are all-reduced (AVG) to simulate FSDP NO_SHARD
averaging. The CP loss scaling (cp_size * local_sum / global_tokens) ensures
that after this averaging, gradients match the non-CP baseline:

    avg_cp_grad = (1/cp_size) * sum_r(cp_size * d(local_sum_r)/d(p) / global_tokens)
                = d(global_sum_CE) / d(p) / global_tokens
                = d(mean_CE_full_seq) / d(p)   ← matches non-CP baseline

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/cp/test_cp_train_correctness.py

Requirements:
    - 2x GPUs
    - flash-attn installed (Ulysses requires Flash Attention 2 or 3)
    - Model: Qwen/Qwen3-0.6B (auto-downloaded)
"""

import math
import os
import sys
import traceback

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.distributed.context_parallel.config import CPConfig
from src.distributed.context_parallel.validation import validate_model_for_ulysses
from src.distributed.context_parallel.wrapper import patch_model_for_cp
from tests.common.distributed import init_distributed, teardown_distributed
from tests.common.models import QWEN3_0_6B
from tests.common.utils import cleanup_memory, log, log_all

# Configuration

MODEL_NAME = QWEN3_0_6B
SEQ_LEN = 128  # Must be divisible by cp_size=2
SEED = 42
NUM_TRAIN_STEPS = 5
LEARNING_RATE = 1e-4

# Tolerances — calibrated for bf16 + Ulysses all-to-all precision
LOSS_ATOL = 0.1  # Forward pass loss (bf16 rounding through all-to-all)
GRAD_COSINE_MIN = 0.99  # Per-parameter gradient cosine similarity
GRAD_NORM_RTOL = 0.10  # Gradient norm ratio tolerance (10%)
TRAIN_LOSS_ATOL = 0.15  # Per-step training loss (compounds over steps)
WEIGHT_COS_MIN = 0.9999  # Final weight cosine similarity after training


# Helpers


def load_model(device: str):
    """Load a fresh Qwen3-0.6B model."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.to(device)
    return model


def create_input(vocab_size: int, device: str, seed: int = SEED):
    """Create deterministic input_ids and labels, broadcast-synchronized."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    input_ids = torch.randint(100, min(30000, vocab_size), (1, SEQ_LEN), device=device)
    labels = input_ids.clone()
    dist.broadcast(input_ids, src=0)
    dist.broadcast(labels, src=0)
    return input_ids, labels


def cosine_similarity_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two tensors (flattened to 1-D)."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    denom = a_flat.norm() * b_flat.norm()
    if denom < 1e-12:
        return 1.0  # Both near-zero → treat as equivalent
    return (a_flat @ b_flat / denom).item()


# Phase 1 + 2: Loss and Gradient Equivalence (combined to minimize loads)


def test_loss_and_gradient_equivalence(device, cp_size):
    """
    Test forward pass loss AND gradient equivalence in a single pass.

    Returns (phase1_passed, phase2_passed, details_dict).
    """
    log(f"\n{'=' * 70}")
    log("  Phase 1+2: Loss and Gradient Equivalence")
    log(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    input_ids, labels = create_input(tokenizer.vocab_size, device)

    # ── Non-CP baseline: forward + backward ─────────────────────────────
    log("\n  [A] Non-CP forward + backward...")
    model_base = load_model(device)
    model_base.train()
    model_base.zero_grad()

    out_base = model_base(input_ids=input_ids, labels=labels)
    base_loss_val = out_base.loss.item()
    out_base.loss.backward()

    base_grads = {name: p.grad.clone().detach() for name, p in model_base.named_parameters() if p.grad is not None}
    base_grad_norm = torch.nn.utils.clip_grad_norm_(model_base.parameters(), float("inf")).item()

    log(f"      Loss: {base_loss_val:.6f}")
    log(f"      Params with grads: {len(base_grads)}")
    log(f"      Grad norm: {base_grad_norm:.6f}")

    del model_base, out_base
    cleanup_memory()

    # ── CP: forward + backward ──────────────────────────────────────────
    log("\n  [B] CP forward + backward...")
    model_cp = load_model(device)

    # No except-and-skip: a refusal on a supported CP family is the regression this file exists to
    # catch, and returning (True, True, {}) would delete the train-step comparison while exiting 0.
    validate_model_for_ulysses(model_cp, cp_size=cp_size)

    cp_config = CPConfig(cp_size=cp_size, world_size=dist.get_world_size(), gpus_per_node=dist.get_world_size())
    model_cp = patch_model_for_cp(model_cp, cp_config)
    model_cp.train()
    model_cp.zero_grad()

    out_cp = model_cp(input_ids=input_ids, labels=labels)
    cp_loss_val = out_cp.loss.item()
    out_cp.loss.backward()

    # Simulate FSDP NO_SHARD: average gradients across CP ranks
    for p in model_cp.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    cp_grads = {name: p.grad.clone().detach() for name, p in model_cp.named_parameters() if p.grad is not None}
    cp_grad_norm = torch.nn.utils.clip_grad_norm_(model_cp.parameters(), float("inf")).item()

    log_all(f"      Local CP loss: {cp_loss_val:.6f}")
    log(f"      Params with grads: {len(cp_grads)}")
    log(f"      Grad norm (after avg): {cp_grad_norm:.6f}")

    del model_cp, out_cp
    cleanup_memory()

    # ── Phase 1: Compare losses ─────────────────────────────────────────
    log("\n  --- Phase 1: Loss Equivalence ---")

    # Average CP losses across ranks
    cp_loss_tensor = torch.tensor([cp_loss_val], device=device)
    gathered = [torch.zeros(1, device=device) for _ in range(cp_size)]
    dist.all_gather(gathered, cp_loss_tensor)
    avg_cp_loss = sum(t.item() for t in gathered) / cp_size

    loss_diff = abs(avg_cp_loss - base_loss_val)
    loss_passed = loss_diff < LOSS_ATOL and math.isfinite(base_loss_val) and math.isfinite(avg_cp_loss)

    log(f"      Baseline loss:    {base_loss_val:.6f}")
    log(f"      Avg CP loss:      {avg_cp_loss:.6f}")
    log(f"      Difference:       {loss_diff:.6f} (tol: {LOSS_ATOL})")
    log(f"      Phase 1 result:   {'PASS' if loss_passed else 'FAIL'}")

    # ── Phase 2: Compare gradients ──────────────────────────────────────
    log("\n  --- Phase 2: Gradient Equivalence ---")

    common_params = sorted(set(base_grads.keys()) & set(cp_grads.keys()))
    log(f"      Common parameters: {len(common_params)}")

    if not common_params:
        log("      ERROR: No common parameters found!")
        log(f"      Base keys (sample): {sorted(base_grads.keys())[:5]}")
        log(f"      CP keys (sample): {sorted(cp_grads.keys())[:5]}")
        return loss_passed, False, {}

    cosine_sims = []
    low_cosine_params = []

    for name in common_params:
        bg = base_grads[name]
        cg = cp_grads[name]
        if bg.shape != cg.shape:
            low_cosine_params.append((name, "shape mismatch"))
            continue
        cos = cosine_similarity_flat(bg, cg)
        cosine_sims.append(cos)
        if cos < GRAD_COSINE_MIN:
            low_cosine_params.append((name, f"cos={cos:.6f}"))

    avg_cosine = sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0.0
    min_cosine = min(cosine_sims) if cosine_sims else 0.0

    # Gradient norm ratio
    norm_ratio = cp_grad_norm / base_grad_norm if base_grad_norm > 1e-12 else float("inf")
    norm_close = abs(norm_ratio - 1.0) < GRAD_NORM_RTOL

    grad_passed = min_cosine >= GRAD_COSINE_MIN and norm_close

    log(f"      Avg cosine similarity:  {avg_cosine:.6f}")
    log(f"      Min cosine similarity:  {min_cosine:.6f} (threshold: {GRAD_COSINE_MIN})")
    log(f"      Grad norm ratio:        {norm_ratio:.6f} (target: 1.0 ± {GRAD_NORM_RTOL})")
    log(f"      Low-cosine params:      {len(low_cosine_params)}")
    if low_cosine_params:
        for name, reason in low_cosine_params[:5]:
            log(f"        {name}: {reason}")
        if len(low_cosine_params) > 5:
            log(f"        ... and {len(low_cosine_params) - 5} more")
    log(f"      Phase 2 result:         {'PASS' if grad_passed else 'FAIL'}")

    details = {
        "base_loss": base_loss_val,
        "avg_cp_loss": avg_cp_loss,
        "loss_diff": loss_diff,
        "avg_cosine": avg_cosine,
        "min_cosine": min_cosine,
        "grad_norm_ratio": norm_ratio,
    }
    return loss_passed, grad_passed, details


# Phase 3: Multi-Step Training Equivalence


def test_training_equivalence(device, cp_size):
    """
    Train for N steps with and without CP, compare loss trajectories and
    final weight similarity.

    Both modes use the same model initialization, same data per step, and
    same optimizer config. CP gradients are averaged after backward to
    simulate FSDP NO_SHARD, making the optimizer updates equivalent.
    """
    log(f"\n{'=' * 70}")
    log(f"  Phase 3: Multi-Step Training Equivalence ({NUM_TRAIN_STEPS} steps)")
    log(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Pre-generate all batches (deterministic, broadcast-synced)
    batches = []
    for step in range(NUM_TRAIN_STEPS):
        inp, lab = create_input(tokenizer.vocab_size, device, seed=SEED + step)
        batches.append((inp, lab))

    # ── Non-CP training ─────────────────────────────────────────────────
    log(f"\n  [A] Non-CP training (lr={LEARNING_RATE})...")
    model_base = load_model(device)
    model_base.train()
    optimizer_base = torch.optim.AdamW(model_base.parameters(), lr=LEARNING_RATE)

    base_losses = []
    for step, (inp, lab) in enumerate(batches):
        optimizer_base.zero_grad()
        out = model_base(input_ids=inp, labels=lab)
        out.loss.backward()
        optimizer_base.step()
        base_losses.append(out.loss.item())

    base_weights = {name: p.data.clone().detach() for name, p in model_base.named_parameters()}
    log(f"      Losses: {[f'{l:.4f}' for l in base_losses]}")

    del model_base, optimizer_base
    cleanup_memory()

    # ── CP training ─────────────────────────────────────────────────────
    log(f"\n  [B] CP training (lr={LEARNING_RATE})...")
    model_cp = load_model(device)
    cp_config = CPConfig(cp_size=cp_size, world_size=dist.get_world_size(), gpus_per_node=dist.get_world_size())
    model_cp = patch_model_for_cp(model_cp, cp_config)
    model_cp.train()
    optimizer_cp = torch.optim.AdamW(model_cp.parameters(), lr=LEARNING_RATE)

    cp_local_losses = []
    for step, (inp, lab) in enumerate(batches):
        optimizer_cp.zero_grad()
        out = model_cp(input_ids=inp, labels=lab)
        out.loss.backward()

        # Simulate FSDP NO_SHARD: average gradients across CP ranks
        for p in model_cp.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        optimizer_cp.step()
        cp_local_losses.append(out.loss.item())

    cp_weights = {name: p.data.clone().detach() for name, p in model_cp.named_parameters()}

    del model_cp, optimizer_cp
    cleanup_memory()

    # Average CP losses across ranks for each step
    cp_avg_losses = []
    for local_loss in cp_local_losses:
        gathered = [torch.zeros(1, device=device) for _ in range(cp_size)]
        dist.all_gather(gathered, torch.tensor([local_loss], device=device))
        cp_avg_losses.append(sum(t.item() for t in gathered) / cp_size)

    log(f"      CP losses (avg): {[f'{l:.4f}' for l in cp_avg_losses]}")

    # ── Compare ─────────────────────────────────────────────────────────
    log("\n  --- Comparing ---")
    checks = {}

    # Check 1: Per-step loss closeness
    step_diffs = [abs(b - c) for b, c in zip(base_losses, cp_avg_losses, strict=False)]
    max_step_diff = max(step_diffs)
    loss_close = max_step_diff < TRAIN_LOSS_ATOL
    checks["loss_trajectory_close"] = loss_close
    log(f"      Per-step diffs: {[f'{d:.4f}' for d in step_diffs]}")
    log(f"      Max step diff:  {max_step_diff:.6f} (tol: {TRAIN_LOSS_ATOL})")
    log(f"      Loss trajectory: {'PASS' if loss_close else 'FAIL'}")

    # Check 2: Loss trend correlation
    if len(base_losses) >= 2:
        base_t = torch.tensor(base_losses, dtype=torch.float)
        cp_t = torch.tensor(cp_avg_losses, dtype=torch.float)
        # Constant losses → same trend (corr = 1.0)
        corr = torch.corrcoef(torch.stack([base_t, cp_t]))[0, 1].item() if base_t.std() > 0 and cp_t.std() > 0 else 1.0
        corr_ok = corr > 0.90
        checks["loss_correlation"] = corr_ok
        log(f"      Loss correlation: {corr:.4f} ({'PASS' if corr_ok else 'FAIL'})")
    else:
        checks["loss_correlation"] = True

    # Check 3: Final weight similarity (cosine per param, then aggregate)
    common_params = sorted(set(base_weights.keys()) & set(cp_weights.keys()))
    weight_cosines = []
    for name in common_params:
        wb = base_weights[name]
        wc = cp_weights[name]
        if wb.shape == wc.shape:
            weight_cosines.append(cosine_similarity_flat(wb, wc))

    if weight_cosines:
        avg_w_cos = sum(weight_cosines) / len(weight_cosines)
        min_w_cos = min(weight_cosines)
        weights_ok = min_w_cos >= WEIGHT_COS_MIN
        checks["weights_close"] = weights_ok
        log(f"      Weight cosine (avg): {avg_w_cos:.8f}")
        log(f"      Weight cosine (min): {min_w_cos:.8f} (threshold: {WEIGHT_COS_MIN})")
        log(f"      Weights close:       {'PASS' if weights_ok else 'FAIL'}")
    else:
        checks["weights_close"] = False
        log("      ERROR: No weights to compare!")

    # Check 4: All losses finite
    all_finite = all(math.isfinite(l) for l in base_losses + cp_avg_losses)
    checks["all_finite"] = all_finite
    log(f"      All losses finite:   {'PASS' if all_finite else 'FAIL'}")

    all_passed = all(checks.values())
    log(f"\n      Phase 3 result: {'PASS' if all_passed else 'FAIL'}")

    return all_passed, {
        "base_losses": base_losses,
        "cp_losses": cp_avg_losses,
        "max_step_diff": max_step_diff,
        "min_weight_cosine": min(weight_cosines) if weight_cosines else 0.0,
    }


# Main


def run_all_tests() -> bool:
    """Run all CP training correctness phases."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local_rank}"
    cp_size = world_size

    log(f"\n{'=' * 70}")
    log("  CP Training Correctness Test (Qwen3)")
    log(f"  World size: {world_size}, CP size: {cp_size}")
    log(f"  Model: {MODEL_NAME}")
    log(f"  Seq length: {SEQ_LEN}, Train steps: {NUM_TRAIN_STEPS}, LR: {LEARNING_RATE}")
    log(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"{'=' * 70}")

    if SEQ_LEN % cp_size != 0:
        log(f"\nERROR: seq_len ({SEQ_LEN}) must be divisible by cp_size ({cp_size})")
        return False

    results = {}

    # Phase 1+2: Loss and gradient equivalence
    try:
        loss_ok, grad_ok, info = test_loss_and_gradient_equivalence(device, cp_size)
        results["loss_equivalence"] = loss_ok
        results["gradient_equivalence"] = grad_ok
    except Exception as e:
        log(f"\n  Phase 1+2 FAILED with exception: {e}")
        if rank == 0:
            traceback.print_exc()
        results["loss_equivalence"] = False
        results["gradient_equivalence"] = False

    # Phase 3: Multi-step training
    try:
        train_ok, info = test_training_equivalence(device, cp_size)
        results["training_equivalence"] = train_ok
    except Exception as e:
        log(f"\n  Phase 3 FAILED with exception: {e}")
        if rank == 0:
            traceback.print_exc()
        results["training_equivalence"] = False

    # Summary
    all_passed = all(results.values())

    log(f"\n{'=' * 70}")
    log("  SUMMARY")
    log(f"{'=' * 70}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        log(f"    {name}: {status}")

    log(f"\n{'=' * 70}")
    if all_passed:
        log("  CP TRAINING CORRECTNESS TEST PASSED")
    else:
        failed = [k for k, v in results.items() if not v]
        log(f"  CP TRAINING CORRECTNESS TEST FAILED: {failed}")
    log(f"{'=' * 70}\n")

    return all_passed


def main() -> int:
    """Run CP training correctness test. Returns 0 on success, 1 on failure."""
    rank, world_size, local_rank = init_distributed()

    try:
        success = run_all_tests()
    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        if rank == 0:
            traceback.print_exc()
        success = False
    finally:
        cleanup_memory()
        if dist.is_initialized():
            dist.barrier()
        teardown_distributed()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
