#!/usr/bin/env python
"""
Context Parallelism forward pass equivalence test.

Validates that CP (Ulysses sequence parallelism) produces equivalent results to a
non-CP baseline: the same Qwen3-0.6B model and a fixed-seed input (seq_len divisible
by cp_size) run first without CP (each rank independently, full sequence), then
through UlyssesCPModelWrapper, which takes the full input_ids/labels, splits by
cp_rank, computes the local loss with boundary handling, and normalizes globally
across CP ranks.

Tolerance: avg CP loss must match the baseline within 0.05 absolute (both finite).
CP's global sum-normalization equals the baseline mean loss in exact arithmetic, so
the only gap is bf16 rounding through the Ulysses all-to-all (~1e-2 at this shape).

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/cp/test_cp_correctness.py

Requirements:
    - 2x GPUs
    - flash-attn installed (Ulysses requires Flash Attention 2 or 3)
    - Model: Qwen/Qwen3-0.6B (auto-downloaded)
"""

import argparse
import math

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.distributed.context_parallel.config import CPConfig
from src.distributed.context_parallel.validation import validate_model_for_ulysses
from src.distributed.context_parallel.wrapper import patch_model_for_cp
from tests.common.distributed import world_mean, world_spread
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.tolerances import TOL
from tests.common.utils import cleanup_memory, log, log_all

# Configuration

MODEL_NAME = QWEN3_0_6B
SEQ_LEN = 128  # default; must be divisible by cp_size. Override with --seq to probe long context.
SEED = 42


# Main Test


def run(ctx) -> dict:
    parser = argparse.ArgumentParser(description="CP forward-pass equivalence vs a non-CP baseline")
    parser.add_argument(
        "--seq",
        type=int,
        default=SEQ_LEN,
        help=f"Sequence length, divisible by cp_size (default {SEQ_LEN}). Raise it to probe long context.",
    )
    seq_len = parser.parse_args().seq

    world_size = ctx.world_size
    device = ctx.device

    log(f"\n{'=' * 70}")
    log("  CP Forward Pass Correctness Test")
    log(f"  World size: {world_size}, Model: {MODEL_NAME}")
    log(f"  Sequence length: {seq_len}, CP size: {world_size}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"{'=' * 70}")

    if seq_len % world_size != 0:
        log(f"\nERROR: seq_len ({seq_len}) must be divisible by world_size ({world_size})")
        return {"checks": {"seq_len_divisible_by_cp_size": False}}

    # ── Step 1: Load tokenizer (for vocab size reference) ────────────
    log("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    log(f"  Vocab size: {vocab_size}")

    # ── Step 2: Create identical input on all ranks ──────────────────
    log("\n[2/6] Creating identical input on all ranks...")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    # Use token IDs in a valid range (avoid special tokens at boundaries)
    input_ids = torch.randint(100, min(30000, vocab_size), (1, seq_len), device=device)
    labels = input_ids.clone()
    log(f"  Input shape: {input_ids.shape}")

    # Ensure all ranks have exactly the same input
    dist.broadcast(input_ids, src=0)
    dist.broadcast(labels, src=0)

    # ── Step 3: Non-CP forward pass (baseline) ───────────────────────
    log("\n[3/6] Running baseline forward pass (no CP)...")
    model_base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model_base.to(device)
    model_base.eval()

    # Second label set with the whole FIRST half masked: under CP that puts every loss token in the
    # last chunk, the shape on which a chunk-partial eval loss is maximally biased.
    masked_labels = labels.clone()
    masked_labels[:, : seq_len // 2] = -100

    with torch.no_grad():
        out_base = model_base(input_ids=input_ids, labels=labels)
        base_loss = out_base.loss.item()
        base_masked_loss = model_base(input_ids=input_ids, labels=masked_labels).loss.item()

    log_all(f"  Baseline loss: {base_loss:.6f} (first-half-masked: {base_masked_loss:.6f})")
    log(f"  Baseline loss is finite: {math.isfinite(base_loss)}")

    # Free baseline model to save memory
    del model_base, out_base
    cleanup_memory()

    # ── Step 4: Validate model compatibility with Ulysses ────────────
    log("\n[4/6] Validating model for Ulysses CP...")
    # Load a fresh model to validate (validate_model_for_ulysses checks config)
    model_cp = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model_cp.to(device)

    # No except-and-skip: this model IS a supported CP family, so a refusal here is the
    # regression the file exists to catch. Swallowing it deletes the CP-vs-single-GPU
    # comparison below from the suite while still exiting 0.
    validate_model_for_ulysses(model_cp, cp_size=world_size)
    log("  Model validated for Ulysses CP")

    # ── Step 5: CP forward pass ──────────────────────────────────────
    log("\n[5/6] Running CP forward pass...")
    cp_config = CPConfig(
        cp_size=world_size,
        world_size=world_size,
        gpus_per_node=world_size,  # single-node test: the whole world is one NVLink domain
    )

    # Wrap model for CP (Ulysses patches attention layers)
    model_cp = patch_model_for_cp(model_cp, cp_config)
    model_cp.eval()

    log(f"  CP config: cp_size={cp_config.cp_size}, cp_rank={cp_config.cp_rank}")

    # The CP wrapper handles splitting internally - pass full sequence
    with torch.no_grad():
        out_cp = model_cp(input_ids=input_ids, labels=labels)
        cp_loss = out_cp.loss.item()
        cp_masked_loss = model_cp(input_ids=input_ids, labels=masked_labels).loss.item()

    log_all(f"  CP loss: {cp_loss:.6f} (first-half-masked: {cp_masked_loss:.6f})")
    log(f"  CP loss is finite: {math.isfinite(cp_loss)}")

    # Free CP model
    del model_cp, out_cp
    cleanup_memory()

    # ── Step 6: Compare losses ───────────────────────────────────────
    log("\n[6/6] Comparing losses...")

    checks = {}

    # Check 1: Baseline loss is finite
    base_finite = math.isfinite(base_loss)
    checks["base_loss_finite"] = base_finite
    log(f"  Baseline loss finite: {'PASS' if base_finite else 'FAIL'}")

    # Check 2: CP loss is finite (each rank reports its own)
    cp_finite = math.isfinite(cp_loss)
    checks["cp_loss_finite"] = cp_finite
    log_all(f"  CP loss finite: {'PASS' if cp_finite else 'FAIL'}")

    # Check 3: CP's global sum-normalization equals the baseline mean loss in exact arithmetic, so
    # the rank average must match the baseline up to bf16 rounding through the Ulysses all-to-all.
    avg_cp_loss = world_mean(cp_loss)
    loss_diff = abs(avg_cp_loss - base_loss)
    loss_close = loss_diff < TOL.parallel_vs_baseline_loss_abs
    checks["loss_close"] = loss_close
    log(f"  Avg CP loss: {avg_cp_loss:.6f}, Baseline loss: {base_loss:.6f}")
    log(f"  Loss difference: {loss_diff:.6f} (tolerance: {TOL.parallel_vs_baseline_loss_abs})")
    log(f"  Loss close: {'PASS' if loss_close else 'FAIL'}")

    # Check 4: under no_grad (the eval loop's path) the CP loss must be rank-UNIFORM — HF's
    # DP-scoped metric gather keeps one CP sibling's copy, so a rank-varying value IS eval_loss bias.
    spread = world_spread(cp_loss)
    checks["eval_loss_rank_uniform"] = spread == 0.0
    log(f"  Eval-path CP loss rank spread: {spread:.6e} ({'PASS' if spread == 0.0 else 'FAIL'})")

    # Check 5: with every loss token in the LAST chunk, each rank's eval-path loss must still equal
    # the baseline masked loss — a chunk-partial value reports ~0 on cp_rank 0 here.
    masked_ok = abs(cp_masked_loss - base_masked_loss) < TOL.parallel_vs_baseline_loss_abs
    checks["eval_loss_unbiased_under_uneven_mask"] = masked_ok
    log_all(
        f"  Masked eval loss {cp_masked_loss:.6f} vs baseline {base_masked_loss:.6f} ({'PASS' if masked_ok else 'FAIL'})"
    )

    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="cp_correctness")(run)

if __name__ == "__main__":
    main()
