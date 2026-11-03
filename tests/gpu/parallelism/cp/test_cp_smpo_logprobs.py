#!/usr/bin/env python
"""CP's gathered per-token log-probs match the full sequence's: under Ulysses each rank sees partial heads but full
positions."""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from halo.distributed.context_parallel.config import CPConfig
from halo.distributed.context_parallel.model_rules import reject_unsupported_ulysses_model
from halo.distributed.context_parallel.ulysses_attention import UlyssesAttention
from halo.distributed.context_parallel.ulysses_model import apply_ulysses_attention
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log, log_all
from tests.common.tolerances import TOL

MODEL_NAME = QWEN3_0_6B
SEQ_LEN = 128  # Must be divisible by cp_size=2.
SEED = 42
# Must also rank the tokens alike, beyond sitting within the atol.
MIN_LOGPROB_CORRELATION = 0.95


def compute_per_token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mirrors the log-prob computation in the SMPO/DPO trainers; logits[:-1] predict labels[1:]."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)

    per_token_logprobs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    return per_token_logprobs


def run(ctx) -> dict:
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

    log("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    log(f"  Vocab size: {vocab_size}")

    log("\n[2/6] Creating identical input on all ranks...")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    input_ids = torch.randint(100, min(30000, vocab_size), (1, SEQ_LEN), device=device)
    labels = input_ids.clone()

    dist.broadcast(input_ids, src=0)
    dist.broadcast(labels, src=0)
    log(f"  Input shape: {input_ids.shape}")

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
        base_logits = out_base.logits

    base_logprobs = compute_per_token_logprobs(base_logits, labels)
    base_logprobs_sum = base_logprobs.sum().item()

    log(f"  Baseline logits shape: {base_logits.shape}")
    log(f"  Baseline log probs shape: {base_logprobs.shape}")
    log(f"  Baseline log probs sum: {base_logprobs_sum:.6f}")
    log(f"  Baseline log probs mean: {base_logprobs.mean().item():.6f}")

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

    # No except-and-skip: a refusal on a supported CP family is the regression here, and swallowing it exits 0.
    reject_unsupported_ulysses_model(model_cp, cp_size=cp_size)
    log("  Model validated for Ulysses CP")

    log("\n[5/6] Running CP forward pass and gathering logits...")
    cp_config = CPConfig(
        cp_size,
        stage_world_size=world_size,
        stage_base_rank=0,
        nvlink_domain_size=world_size,  # Single-node test: the whole world is one NVLink domain.
    )

    # The model with Ulysses attention patching but no UlyssesModel, splitting input and gathering output by hand.
    cp_group = cp_config.process_group

    num_patched = apply_ulysses_attention(model_cp, cp_group, cp_size)
    log(f"  Patched {num_patched} attention layers")

    # Same-kernel comparison: the wrapper's probe would pick FA4 on Blackwell against the baseline's HF FA2, and
    # FA4-vs-FA2 per-token bf16 deltas swamp the atol.
    for module in model_cp.modules():
        if isinstance(module, UlyssesAttention):
            module._allow_fa4 = False

    model_cp.eval()

    local_start = cp_config.cp_rank * chunk_size
    local_end = local_start + chunk_size
    local_input_ids = input_ids[:, local_start:local_end].contiguous()

    # Global positions.
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
        local_logits = out_cp.logits

    log_all(f"  Local logits shape: {local_logits.shape}")

    gathered_logits_list = [torch.zeros_like(local_logits) for _ in range(cp_size)]
    dist.all_gather(gathered_logits_list, local_logits.contiguous(), group=cp_group)

    cp_full_logits = torch.cat(gathered_logits_list, dim=1)
    log(f"  Gathered CP logits shape: {cp_full_logits.shape}")

    cp_logprobs = compute_per_token_logprobs(cp_full_logits, labels)
    cp_logprobs_sum = cp_logprobs.sum().item()

    log(f"  CP log probs shape: {cp_logprobs.shape}")
    log(f"  CP log probs sum: {cp_logprobs_sum:.6f}")
    log(f"  CP log probs mean: {cp_logprobs.mean().item():.6f}")

    del model_cp, out_cp, local_logits, gathered_logits_list
    cleanup_memory()

    log("\n[6/6] Comparing log probabilities...")

    checks = {}

    base_finite = torch.isfinite(base_logprobs).all().item()
    cp_finite = torch.isfinite(cp_logprobs).all().item()
    checks["base_logprobs_finite"] = base_finite
    checks["cp_logprobs_finite"] = cp_finite
    log(f"  Base log probs all finite: {'PASS' if base_finite else 'FAIL'}")
    log(f"  CP log probs all finite: {'PASS' if cp_finite else 'FAIL'}")

    sum_diff = abs(cp_logprobs_sum - base_logprobs_sum)
    sum_close = sum_diff < (TOL.logprob_atol * SEQ_LEN)
    checks["logprob_sum_close"] = sum_close
    log(f"  Log prob sum diff: {sum_diff:.6f} (tolerance: {TOL.logprob_atol * SEQ_LEN:.2f})")
    log(f"  Log prob sum close: {'PASS' if sum_close else 'FAIL'}")

    per_token_diff = (cp_logprobs - base_logprobs).abs()
    max_diff = per_token_diff.max().item()
    mean_diff = per_token_diff.mean().item()
    tokens_close = max_diff < TOL.logprob_atol
    checks["per_token_close"] = tokens_close
    log(f"  Per-token max diff: {max_diff:.6f}")
    log(f"  Per-token mean diff: {mean_diff:.6f}")
    log(f"  Per-token close (atol={TOL.logprob_atol}): {'PASS' if tokens_close else 'FAIL'}")

    # Even if absolute values differ slightly, the ordering should be preserved.
    base_flat = base_logprobs.flatten().float()
    cp_flat = cp_logprobs.flatten().float()
    if base_flat.std() > 0 and cp_flat.std() > 0:
        correlation = torch.corrcoef(torch.stack([base_flat, cp_flat]))[0, 1].item()
        corr_high = correlation > MIN_LOGPROB_CORRELATION
        checks["correlation_high"] = corr_high
        log(f"  Pearson correlation: {correlation:.6f}")
        log(f"  Correlation > {MIN_LOGPROB_CORRELATION}: {'PASS' if corr_high else 'FAIL'}")
    else:
        # Constant log-probs over random tokens mean a degenerate forward.
        checks["correlation_high"] = False
        log("  Correlation: FAIL (zero variance)")

    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="cp_smpo_logprobs", partial_state=False)(run)

if __name__ == "__main__":
    main()
