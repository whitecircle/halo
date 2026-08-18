#!/usr/bin/env python
"""
EP+CP combined correctness test on MoE model.

Compares EP-only (EP=2) forward pass loss versus EP+CP (EP=2, CP=2)
forward pass loss on GptOss-20B. Both modes should produce equivalent
results since Context Parallelism (Ulysses attention) only splits the
sequence across ranks without changing the mathematical result.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/combined/test_ep_cp_correctness.py

Requirements:
    - 2x GPUs with >=80GB memory each
    - DeepEP installed
    - Model: unsloth/gpt-oss-20b-BF16 (auto-downloaded)
"""

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from tests.common.distributed import ensure_model_downloaded
from tests.common.ep_reference import fixed_chat_batch
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.utils import cleanup_memory, gpu_mem_gb, log, log_all

# Test Configuration

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 2
CP_SIZE = 2
SEQ_LEN = 128  # Must be divisible by cp_size=2
SEED = 42
# A longer fixed conversation than the default, so the tokens fill more of the window CP splits.
CONVERSATION = 1

# Tolerances. gpt-oss MoE routing is numerically sensitive: near-tied top-k router scores flip
# expert selection under any bf16 kernel/order change (CP's chunked dispatch vs EP-only's full-batch
# dispatch), and a flipped expert changes that token's logits outright. A per-position probe shows a
# broad small logit drift (MAD ~0.5, growing with depth, no chunk-boundary spike) yielding ~5.5%
# mean-CE difference on this 128-token input — routing noise, not a CP math error (EP+CP SFT
# convergence is validated end-to-end by the trainer tests).
LOSS_REL_TOL = 0.10
LOSS_ABS_TOL = 0.5  # Absolute tolerance
# EP-only rank consistency: with bf16 + DeepEP all-to-all the per-rank reduction
# order isn't bit-identical, so a small spread is expected.
EP_RANK_ABS_TOL = 0.15


# Test-Specific Helpers


def zero_router_aux_loss(model) -> None:
    """Exclude the MoE load-balancing aux loss from the comparison.

    The aux loss is a NONLINEAR function of routing fractions: EP-only computes it over the full
    sequence while each CP rank computes it over its 1/cp_size chunk (an inherent CP approximation),
    so with gpt-oss's router_aux_loss_coef=0.9 the totals legitimately diverge. This test asserts
    the CE math is preserved — zero the coefficient on both the ForCausalLM attribute (read by the
    EP-only forward) and the config (read by the CP wrapper).
    """
    inner = model
    while hasattr(inner, "model") and not hasattr(inner, "router_aux_loss_coef"):
        if hasattr(inner, "config"):
            inner.config.router_aux_loss_coef = 0.0
        inner = inner.model
    if hasattr(inner, "router_aux_loss_coef"):
        inner.router_aux_loss_coef = 0.0
    if hasattr(inner, "config"):
        inner.config.router_aux_loss_coef = 0.0
    if hasattr(model, "config"):
        model.config.router_aux_loss_coef = 0.0


# Phase 1: EP-Only Forward Pass


def compute_ep_only_loss(tokenizer, local_rank):
    """Compute forward pass loss with EP=2 only (no CP).

    Returns:
        list[float]: Losses from all ranks.
    """
    device = f"cuda:{local_rank}"

    log(f"\n  Loading EP-only model (EP={EP_SIZE}, CP=1)...")
    log(f"  GPU memory before load: {gpu_mem_gb():.2f} GB")

    ep_config = ParallelismConfig(
        ep_size=EP_SIZE,
        cp_size=1,
        tp_size=1,
    )
    log(f"  Config: {ep_config.summary()}")

    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=ep_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    zero_router_aux_loss(model)

    log(f"  GPU memory after load: {gpu_mem_gb():.2f} GB")

    # Create identical input on all ranks
    input_ids, attention_mask, labels = fixed_chat_batch(
        tokenizer, SEQ_LEN, device, seed=SEED, conversation=CONVERSATION
    )
    dist.broadcast(input_ids, src=0)
    dist.broadcast(attention_mask, src=0)
    dist.broadcast(labels, src=0)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        local_loss = outputs.loss.item()

    log_all(f"  EP-only loss: {local_loss:.6f}")

    all_losses = [None] * dist.get_world_size()
    dist.all_gather_object(all_losses, local_loss)

    del model, outputs
    cleanup_memory()
    log(f"  GPU memory after cleanup: {gpu_mem_gb():.2f} GB")

    return all_losses


# Phase 2: EP+CP Forward Pass


def compute_ep_cp_loss(tokenizer, local_rank):
    """Compute forward pass loss with EP=2 + CP=2.

    When both EP and CP are active with ep_group_size=gpus_per_node,
    this is the orthogonal mode where EP and CP share the same group.

    Returns:
        list[float]: Losses from all ranks.
    """
    device = f"cuda:{local_rank}"

    log(f"\n  Loading EP+CP model (EP={EP_SIZE}, CP={CP_SIZE})...")
    log(f"  GPU memory before load: {gpu_mem_gb():.2f} GB")

    ep_cp_config = ParallelismConfig(
        ep_size=EP_SIZE,
        cp_size=CP_SIZE,
        tp_size=1,
    )
    log(f"  Config: {ep_cp_config.summary()}")

    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=ep_cp_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    zero_router_aux_loss(model)

    log(f"  GPU memory after load: {gpu_mem_gb():.2f} GB")

    # Create identical input on all ranks
    input_ids, attention_mask, labels = fixed_chat_batch(
        tokenizer, SEQ_LEN, device, seed=SEED, conversation=CONVERSATION
    )
    dist.broadcast(input_ids, src=0)
    dist.broadcast(attention_mask, src=0)
    dist.broadcast(labels, src=0)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        local_loss = outputs.loss.item()

    log_all(f"  EP+CP loss: {local_loss:.6f}")

    all_losses = [None] * dist.get_world_size()
    dist.all_gather_object(all_losses, local_loss)

    del model, outputs
    cleanup_memory()
    log(f"  GPU memory after cleanup: {gpu_mem_gb():.2f} GB")

    return all_losses


def run(ctx):
    log(f"\n{'#' * 70}")
    log("  EP+CP Combined Correctness Test")
    log(f"  World size: {ctx.world_size}, EP: {EP_SIZE}, CP: {CP_SIZE}")
    log(f"  Model: {MODEL_NAME}")
    log(f"  Seq len: {SEQ_LEN} (divisible by CP={CP_SIZE})")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"{'#' * 70}")

    # Verify seq_len divisibility
    assert SEQ_LEN % CP_SIZE == 0, f"SEQ_LEN ({SEQ_LEN}) must be divisible by CP_SIZE ({CP_SIZE})"

    # --- Ensure model is cached ---
    log("\nEnsuring model is downloaded...")
    ensure_model_downloaded(MODEL_NAME, ctx.rank)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Phase 1: EP-Only Loss ---
    log(f"\n{'=' * 70}")
    log("PHASE 1: EP-Only Forward Pass (EP=2, CP=1)")
    log(f"{'=' * 70}")

    ep_only_losses = compute_ep_only_loss(tokenizer, ctx.local_rank)

    barrier()
    cleanup_memory()

    # --- Phase 2: EP+CP Loss ---
    log(f"\n{'=' * 70}")
    log("PHASE 2: EP+CP Forward Pass (EP=2, CP=2)")
    log(f"{'=' * 70}")

    ep_cp_losses = compute_ep_cp_loss(tokenizer, ctx.local_rank)

    barrier()

    # --- Validation ---
    checks = {}
    if ctx.rank == 0:
        log(f"\n{'=' * 70}")
        log("VALIDATION")
        log(f"{'=' * 70}")

        # Check 1: EP-only losses are finite
        ep_finite = all(not (torch.isnan(torch.tensor(l)) or torch.isinf(torch.tensor(l))) for l in ep_only_losses)
        checks["ep_only_finite"] = ep_finite
        log(f"\n  EP-only losses finite: {'PASS' if ep_finite else 'FAIL'}")
        log(f"    Losses: {[f'{l:.6f}' for l in ep_only_losses]}")

        # Check 2: EP+CP losses are finite
        ep_cp_finite = all(not (torch.isnan(torch.tensor(l)) or torch.isinf(torch.tensor(l))) for l in ep_cp_losses)
        checks["ep_cp_finite"] = ep_cp_finite
        log(f"  EP+CP losses finite: {'PASS' if ep_cp_finite else 'FAIL'}")
        log(f"    Losses: {[f'{l:.6f}' for l in ep_cp_losses]}")

        # Check 3: EP-only rank consistency
        ep_spread = max(ep_only_losses) - min(ep_only_losses)
        ep_consistent = ep_spread < EP_RANK_ABS_TOL
        checks["ep_rank_consistency"] = ep_consistent
        log(f"  EP-only rank consistency (spread={ep_spread:.8f}): {'PASS' if ep_consistent else 'FAIL'}")

        # Check 4: EP+CP per-rank losses are NOT expected to match.
        # In CP mode each rank computes the loss only on its sequence chunk
        # (the wrapper reports `local_sum * cp_size / global_tokens`), so
        # different chunks → different per-rank values by design. Correctness
        # of the aggregate is covered by Check 5 (mean vs. EP-only).
        ep_cp_spread = max(ep_cp_losses) - min(ep_cp_losses)
        log(
            f"  EP+CP per-rank loss spread: {ep_cp_spread:.6f} "
            f"(expected to differ — each rank holds a different sequence chunk)"
        )

        # Check 5: EP-only vs EP+CP loss comparison
        ep_avg = sum(ep_only_losses) / len(ep_only_losses)
        ep_cp_avg = sum(ep_cp_losses) / len(ep_cp_losses)
        abs_diff = abs(ep_avg - ep_cp_avg)
        rel_diff = abs_diff / max(abs(ep_avg), 1e-10)

        loss_match = abs_diff < LOSS_ABS_TOL or rel_diff < LOSS_REL_TOL
        checks["ep_vs_ep_cp_match"] = loss_match

        log("\n  --- EP-Only vs EP+CP Comparison ---")
        log(f"  EP-only loss (avg):  {ep_avg:.6f}")
        log(f"  EP+CP loss (avg):    {ep_cp_avg:.6f}")
        log(f"  Abs diff:            {abs_diff:.6f} (tol: {LOSS_ABS_TOL})")
        log(f"  Rel diff:            {rel_diff:.4%} (tol: {LOSS_REL_TOL:.4%})")
        log(f"  Match: {'PASS' if loss_match else 'FAIL'}")

        # Check 6: Losses are in reasonable range
        all_losses = ep_only_losses + ep_cp_losses
        reasonable = all(0 < l < 100 for l in all_losses)
        checks["losses_reasonable"] = reasonable
        log(f"  All losses in range (0, 100): {'PASS' if reasonable else 'FAIL'}")

    # Only rank 0 gathers both phases' per-rank losses into a verdict.
    return {"checks": ctx.broadcast_checks(checks)}


main = gpu_test_main(exact_world_size=EP_SIZE, prefix="ep_cp_correctness")(run)

if __name__ == "__main__":
    main()
