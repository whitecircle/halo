#!/usr/bin/env python
"""CP (Ulysses) forward loss equals the non-CP baseline: equal in exact arithmetic, so the gap is bf16 rounding (~1e-2
here)."""

import argparse
import math

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from halo.distributed.context_parallel.config import CPConfig
from halo.distributed.context_parallel.model_rules import reject_unsupported_ulysses_model
from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from tests.common.distributed import world_mean, world_spread
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log, log_all
from tests.common.tolerances import TOL

MODEL_NAME = QWEN3_0_6B
SEQ_LEN = 128  # Must be divisible by cp_size; --seq probes long context.
SEED = 42


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

    log("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    log(f"  Vocab size: {vocab_size}")

    log("\n[2/6] Creating identical input on all ranks...")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    # A valid range, avoiding special tokens at boundaries.
    input_ids = torch.randint(100, min(30000, vocab_size), (1, seq_len), device=device)
    labels = input_ids.clone()
    log(f"  Input shape: {input_ids.shape}")

    dist.broadcast(input_ids, src=0)
    dist.broadcast(labels, src=0)

    log("\n[3/6] Running baseline forward pass (no CP)...")
    model_base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model_base.to(device)
    model_base.eval()

    # The whole first half masked: every loss token in the last chunk, where a chunk-partial eval loss is maximally
    # biased.
    masked_labels = labels.clone()
    masked_labels[:, : seq_len // 2] = -100

    with torch.no_grad():
        out_base = model_base(input_ids=input_ids, labels=labels)
        base_loss = out_base.loss.item()
        base_masked_loss = model_base(input_ids=input_ids, labels=masked_labels).loss.item()

    log_all(f"  Baseline loss: {base_loss:.6f} (first-half-masked: {base_masked_loss:.6f})")
    log(f"  Baseline loss is finite: {math.isfinite(base_loss)}")

    del model_base, out_base
    cleanup_memory()

    log("\n[4/6] Validating model for Ulysses CP...")
    model_cp = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model_cp.to(device)

    # No except-and-skip: a refusal on this supported CP family is the regression here, and swallowing it exits 0.
    reject_unsupported_ulysses_model(model_cp, cp_size=world_size)
    log("  Model validated for Ulysses CP")

    log("\n[5/6] Running CP forward pass...")
    cp_config = CPConfig(
        world_size,
        stage_world_size=world_size,
        stage_base_rank=0,
        nvlink_domain_size=world_size,  # Single-node test: the whole world is one NVLink domain.
    )

    model_cp = UlyssesModel(model_cp, cp_config)
    model_cp.eval()

    log(f"  CP config: cp_size={cp_config.cp_size}, cp_rank={cp_config.cp_rank}")

    # The CP wrapper splits internally, so pass the full sequence.
    with torch.no_grad():
        out_cp = model_cp(input_ids=input_ids, labels=labels)
        cp_loss = out_cp.loss.item()
        cp_masked_loss = model_cp(input_ids=input_ids, labels=masked_labels).loss.item()

    log_all(f"  CP loss: {cp_loss:.6f} (first-half-masked: {cp_masked_loss:.6f})")
    log(f"  CP loss is finite: {math.isfinite(cp_loss)}")

    del model_cp, out_cp
    cleanup_memory()

    log("\n[6/6] Comparing losses...")

    checks = {}

    base_finite = math.isfinite(base_loss)
    checks["base_loss_finite"] = base_finite
    log(f"  Baseline loss finite: {'PASS' if base_finite else 'FAIL'}")

    cp_finite = math.isfinite(cp_loss)
    checks["cp_loss_finite"] = cp_finite
    log_all(f"  CP loss finite: {'PASS' if cp_finite else 'FAIL'}")

    avg_cp_loss = world_mean(cp_loss)
    loss_diff = abs(avg_cp_loss - base_loss)
    loss_close = loss_diff < TOL.parallel_vs_baseline_loss_abs
    checks["loss_close"] = loss_close
    log(f"  Avg CP loss: {avg_cp_loss:.6f}, Baseline loss: {base_loss:.6f}")
    log(f"  Loss difference: {loss_diff:.6f} (tolerance: {TOL.parallel_vs_baseline_loss_abs})")
    log(f"  Loss close: {'PASS' if loss_close else 'FAIL'}")

    # Under no_grad, rank-uniform: HF's DP-scoped metric gather keeps one CP sibling's copy, so a rank-varying value
    # biases eval_loss.
    spread = world_spread(cp_loss)
    checks["eval_loss_rank_uniform"] = spread == 0.0
    log(f"  Eval-path CP loss rank spread: {spread:.6e} ({'PASS' if spread == 0.0 else 'FAIL'})")

    # Every loss token in the last chunk: a chunk-partial value reports ~0 on cp_rank 0.
    masked_ok = abs(cp_masked_loss - base_masked_loss) < TOL.parallel_vs_baseline_loss_abs
    checks["eval_loss_unbiased_under_uneven_mask"] = masked_ok
    log_all(
        f"  Masked eval loss {cp_masked_loss:.6f} vs baseline {base_masked_loss:.6f} ({'PASS' if masked_ok else 'FAIL'})"
    )

    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="cp_correctness")(run)

if __name__ == "__main__":
    main()
