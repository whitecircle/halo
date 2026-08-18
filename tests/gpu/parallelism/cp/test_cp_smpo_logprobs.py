#!/usr/bin/env python
"""
Test CP-aware log probability aggregation for SMPO-style training.

Validates that when computing per-token log probabilities with Context Parallelism,
the split-and-gather approach produces the same result as computing on the full
sequence without CP.

On a fixed input sequence (deterministic seed), Qwen3-0.6B is run without CP to get
baseline per-token log probs, then with CP, whose local logits are all-gathered before
the same computation. This is the mathematical equivalence of the logit computation:
under Ulysses each rank sees partial heads but full sequence positions in attention,
so the gathered logits must match the full-sequence ones.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/cp/test_cp_smpo_logprobs.py

Requirements:
    - 2x GPUs
    - flash-attn installed (Ulysses requires Flash Attention 2 or 3)
    - Model: Qwen/Qwen3-0.6B (auto-downloaded)
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.distributed.context_parallel.base_layer import UlyssesAttentionBase
from src.distributed.context_parallel.config import CPConfig
from src.distributed.context_parallel.patching import patch_attention_for_ulysses
from src.distributed.context_parallel.validation import validate_model_for_ulysses
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.utils import cleanup_memory, log, log_all

# Configuration

MODEL_NAME = QWEN3_0_6B
SEQ_LEN = 128  # Must be divisible by cp_size=2
SEED = 42
# Tolerance for log probability comparison (bf16 precision limits)
LOGPROB_ATOL = 0.05  # Absolute tolerance per token
LOGPROB_RTOL = 0.02  # Relative tolerance


# Test-Specific Helpers


def compute_per_token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute per-token log probabilities from logits and labels.

    This mirrors the log probability computation used in SMPO/DPO trainers.

    Args:
        logits: [batch, seq_len, vocab_size]
        labels: [batch, seq_len]

    Returns:
        Per-token log probabilities: [batch, seq_len-1]
        (shifted: logits[:-1] predict labels[1:])
    """
    # Shift: logits[:-1] predicts labels[1:]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Compute log softmax over vocab dimension
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)

    # Gather log probs for the actual labels
    per_token_logprobs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    return per_token_logprobs


# Main Test


def run(ctx) -> dict:
    """Test CP log probability equivalence."""
    world_size = ctx.world_size
    device = f"cuda:{ctx.local_rank}"
    cp_size = world_size

    log(f"\n{'=' * 70}")
    log("  CP Log Probability Aggregation Test")
    log(f"  World size: {world_size}, Model: {MODEL_NAME}")
    log(f"  Sequence length: {SEQ_LEN}, CP size: {cp_size}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"{'=' * 70}")

    if SEQ_LEN % cp_size != 0:
        raise ValueError(f"seq_len ({SEQ_LEN}) must be divisible by cp_size ({cp_size})")

    chunk_size = SEQ_LEN // cp_size

    # ── Step 1: Load tokenizer ───────────────────────────────────────
    log("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    log(f"  Vocab size: {vocab_size}")

    # ── Step 2: Create identical input ───────────────────────────────
    log("\n[2/6] Creating identical input on all ranks...")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    input_ids = torch.randint(100, min(30000, vocab_size), (1, SEQ_LEN), device=device)
    labels = input_ids.clone()

    # Broadcast to ensure all ranks have identical data
    dist.broadcast(input_ids, src=0)
    dist.broadcast(labels, src=0)
    log(f"  Input shape: {input_ids.shape}")

    # ── Step 3: Non-CP baseline logits and log probs ─────────────────
    log("\n[3/6] Computing baseline log probs (no CP)...")
    model_base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model_base.to(device)
    model_base.eval()

    with torch.no_grad():
        out_base = model_base(input_ids=input_ids)
        base_logits = out_base.logits  # [1, SEQ_LEN, vocab]

    base_logprobs = compute_per_token_logprobs(base_logits, labels)
    base_logprobs_sum = base_logprobs.sum().item()

    log(f"  Baseline logits shape: {base_logits.shape}")
    log(f"  Baseline log probs shape: {base_logprobs.shape}")
    log(f"  Baseline log probs sum: {base_logprobs_sum:.6f}")
    log(f"  Baseline log probs mean: {base_logprobs.mean().item():.6f}")

    # Free baseline model
    del model_base, out_base
    cleanup_memory()

    # ── Step 4: Validate model for Ulysses ──────────────────────────
    log("\n[4/6] Validating model for Ulysses CP...")
    model_cp = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model_cp.to(device)

    # No except-and-skip: a refusal on a supported CP family is the regression this file
    # exists to catch, and swallowing it removes the log-prob comparison while exiting 0.
    validate_model_for_ulysses(model_cp, cp_size=cp_size)
    log("  Model validated for Ulysses CP")

    # ── Step 5: CP forward pass - get local logits ───────────────────
    log("\n[5/6] Running CP forward pass and gathering logits...")
    cp_config = CPConfig(
        cp_size=cp_size,
        world_size=world_size,
        gpus_per_node=world_size,  # single-node test: the whole world is one NVLink domain
    )

    # The CP wrapper patches attention for Ulysses but handles
    # sequence splitting in forward(). For logit comparison, we need
    # to get the raw logits from the CP forward pass.
    #
    # Strategy: Use the model WITHOUT the full CPModelWrapper, but WITH
    # Ulysses attention patching. Manually split input and gather output.
    cp_group = cp_config.process_group

    num_patched = patch_attention_for_ulysses(
        model_cp,
        cp_group,
        cp_size,
        validate=False,  # Already validated above
    )
    log(f"  Patched {num_patched} attention layers")

    # Same-kernel comparison: the wrapper's runtime probe would pick FA4 on Blackwell while
    # the baseline above runs HF's FA2, and FA4-vs-FA2 per-token bf16 deltas swamp the atol.
    # This test pins the CP MATH (head-scatter/gather), so hold the kernel constant instead.
    for module in model_cp.modules():
        if isinstance(module, UlyssesAttentionBase):
            module._allow_fa4 = False

    model_cp.eval()

    # Manually split input for CP
    local_start = cp_config.cp_rank * chunk_size
    local_end = local_start + chunk_size
    local_input_ids = input_ids[:, local_start:local_end].contiguous()

    # Create position IDs for the local chunk (global positions)
    batch_size = input_ids.shape[0]
    local_position_ids = (
        torch.arange(
            local_start,
            local_end,
            device=device,
            dtype=torch.long,
        )
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    log_all(f"  Local input shape: {local_input_ids.shape}, positions: [{local_start}, {local_end})")

    with torch.no_grad():
        out_cp = model_cp(
            input_ids=local_input_ids,
            position_ids=local_position_ids,
        )
        local_logits = out_cp.logits  # [1, chunk_size, vocab]

    log_all(f"  Local logits shape: {local_logits.shape}")

    # Gather logits from all CP ranks
    # Each rank has [1, chunk_size, vocab_size]
    gathered_logits_list = [torch.zeros_like(local_logits) for _ in range(cp_size)]
    dist.all_gather(gathered_logits_list, local_logits.contiguous(), group=cp_group)

    # Concatenate along sequence dimension: [1, SEQ_LEN, vocab]
    cp_full_logits = torch.cat(gathered_logits_list, dim=1)
    log(f"  Gathered CP logits shape: {cp_full_logits.shape}")

    # Compute log probs from gathered CP logits
    cp_logprobs = compute_per_token_logprobs(cp_full_logits, labels)
    cp_logprobs_sum = cp_logprobs.sum().item()

    log(f"  CP log probs shape: {cp_logprobs.shape}")
    log(f"  CP log probs sum: {cp_logprobs_sum:.6f}")
    log(f"  CP log probs mean: {cp_logprobs.mean().item():.6f}")

    # Free CP model
    del model_cp, out_cp, local_logits, gathered_logits_list
    cleanup_memory()

    # ── Step 6: Compare log probs ────────────────────────────────────
    log("\n[6/6] Comparing log probabilities...")

    checks = {}

    # Check 1: Both log prob tensors are finite
    base_finite = torch.isfinite(base_logprobs).all().item()
    cp_finite = torch.isfinite(cp_logprobs).all().item()
    checks["base_logprobs_finite"] = base_finite
    checks["cp_logprobs_finite"] = cp_finite
    log(f"  Base log probs all finite: {'PASS' if base_finite else 'FAIL'}")
    log(f"  CP log probs all finite: {'PASS' if cp_finite else 'FAIL'}")

    # Check 2: Log prob sums are close
    sum_diff = abs(cp_logprobs_sum - base_logprobs_sum)
    sum_close = sum_diff < (LOGPROB_ATOL * SEQ_LEN)
    checks["logprob_sum_close"] = sum_close
    log(f"  Log prob sum diff: {sum_diff:.6f} (tolerance: {LOGPROB_ATOL * SEQ_LEN:.2f})")
    log(f"  Log prob sum close: {'PASS' if sum_close else 'FAIL'}")

    # Check 3: Per-token log probs are close (element-wise)
    per_token_diff = (cp_logprobs - base_logprobs).abs()
    max_diff = per_token_diff.max().item()
    mean_diff = per_token_diff.mean().item()
    tokens_close = max_diff < LOGPROB_ATOL
    checks["per_token_close"] = tokens_close
    log(f"  Per-token max diff: {max_diff:.6f}")
    log(f"  Per-token mean diff: {mean_diff:.6f}")
    log(f"  Per-token close (atol={LOGPROB_ATOL}): {'PASS' if tokens_close else 'FAIL'}")

    # Check 4: Log prob means are in reasonable range
    base_mean = base_logprobs.mean().item()
    cp_mean = cp_logprobs.mean().item()
    # Log probs should be negative (log of probability < 1)
    base_neg = base_mean < 0
    cp_neg = cp_mean < 0
    checks["base_mean_negative"] = base_neg
    checks["cp_mean_negative"] = cp_neg
    log(f"  Base mean log prob < 0: {'PASS' if base_neg else 'FAIL'} ({base_mean:.6f})")
    log(f"  CP mean log prob < 0: {'PASS' if cp_neg else 'FAIL'} ({cp_mean:.6f})")

    # Check 5: Correlation between base and CP log probs
    # Even if absolute values differ slightly, the ordering should be preserved
    base_flat = base_logprobs.flatten().float()
    cp_flat = cp_logprobs.flatten().float()
    if base_flat.std() > 0 and cp_flat.std() > 0:
        correlation = torch.corrcoef(torch.stack([base_flat, cp_flat]))[0, 1].item()
        corr_high = correlation > 0.95
        checks["correlation_high"] = corr_high
        log(f"  Pearson correlation: {correlation:.6f}")
        log(f"  Correlation > 0.95: {'PASS' if corr_high else 'FAIL'}")
    else:
        checks["correlation_high"] = True
        log("  Correlation: SKIP (zero variance)")

    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="cp_smpo_logprobs", partial_state=False)(run)

if __name__ == "__main__":
    main()
