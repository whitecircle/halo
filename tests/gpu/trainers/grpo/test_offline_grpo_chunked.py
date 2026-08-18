#!/usr/bin/env python
"""Offline GRPO ``use_chunked_grpo_logprobs`` under real FSDP2 + FA4 on 2 GPUs.

Two gates, both of which FAIL if the chunked path drifts:

1. Parity: per-token completion log-probs from the chunked path (backbone hidden + vocab-chunked
   softmax; per-row dense forwards under FA4) must match the full-logits path on the same collated
   batch, on every rank, over the real (non-pad) completion positions.
2. Training: a short chunked run with ``kl_beta > 0`` (policy AND reference both chunked) and the
   default ``min_log_prob`` clamp finishes with finite losses.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/grpo/test_offline_grpo_chunked.py
"""

import math

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.grpo.offline import OfflineGRPOTrainer
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.utils import log
from tests.gpu.trainers.grpo.test_offline_grpo import create_offline_grpo_dataset

MODEL_NAME = QWEN3_0_6B
MAX_STEPS = 4
BATCH_SIZE = 2
NUM_TRAIN_SAMPLES = 32
SEED = 42
# The reference itself is coarse: TRL's bf16 branch returns bf16-ROUNDED log-probs (ULP 0.06-0.25 at
# |logp| 8-32) while the chunked path is fp32, and FA4's padded-varlen vs per-row-dense forwards add
# per-token kernel noise — measured mean ~0.03 / max ~0.18 with the full-logits path against ITSELF
# on trimmed rows already at mean ~0.019 / max 0.125. The gate is therefore the MEAN at 0.05: a
# systematic defect (position shift, off-by-one span) moves every completion token by whole nats.
# Exact-math equivalence is pinned separately in fp32 on CPU (test_offline_grpo_chunked_logprobs.py).
PARITY_MEAN_TOL = 5e-2
PARITY_MAX_TOL = 0.5


def _batch_parity_check(trainer) -> tuple[float, float]:
    """(mean, max) |chunked − full| completion log-prob over this rank's first batch (pad positions
    excluded: the dense per-row path leaves their hidden zero by design while the full path computes
    them)."""
    batch = next(iter(trainer.get_train_dataloader()))
    batch = trainer._prepare_inputs(batch)
    input_ids = torch.cat([batch["prompt_input_ids"], batch["completion_input_ids"]], dim=1)
    attention_mask = torch.cat([batch["prompt_attention_mask"], batch["completion_attention_mask"]], dim=1)
    logits_to_keep = batch["completion_input_ids"].size(1)

    was_chunked = trainer._use_chunked_grpo_logprobs
    with torch.no_grad():
        try:
            trainer._use_chunked_grpo_logprobs = False
            full, _ = trainer._get_per_token_logps(trainer.model, input_ids, attention_mask, logits_to_keep)
            trainer._use_chunked_grpo_logprobs = True
            chunked, _ = trainer._get_per_token_logps(trainer.model, input_ids, attention_mask, logits_to_keep)
        finally:
            trainer._use_chunked_grpo_logprobs = was_chunked

    mask = batch["completion_attention_mask"].bool()
    diff = (chunked - full.float()).abs()[mask]
    return diff.mean().item(), diff.max().item()


def run(ctx) -> dict:
    log(f"\n{'=' * 70}")
    log("  Offline GRPO use_chunked_grpo_logprobs test (FSDP2 + FA4)")
    log(f"  World size: {ctx.world_size}, Model: {MODEL_NAME}")
    log(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = create_offline_grpo_dataset(tokenizer, NUM_TRAIN_SAMPLES, seed=SEED)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_4",
    )

    config = OfflineGRPOConfig(
        output_dir=ctx.output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_prompt_length=2048,
        max_completion_length=2048,
        dataloader_drop_last=True,
        fsdp="",  # Mixin handles FSDP wrapping
        use_chunked_grpo_logprobs=True,
        kl_beta=0.04,  # reference logps take the chunked path too
    )

    trainer = OfflineGRPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        parallelism_config=ParallelismConfig(),
    )

    log("\n[1/2] Chunked vs full log-prob parity on a collated batch...")
    mean_diff, max_diff = _batch_parity_check(trainer)
    diff_tensor = torch.tensor([mean_diff, max_diff], device=ctx.device)
    dist.all_reduce(diff_tensor, op=dist.ReduceOp.MAX)
    mean_diff, max_diff = diff_tensor.tolist()
    parity_ok = mean_diff < PARITY_MEAN_TOL and max_diff < PARITY_MAX_TOL
    log(
        f"  |chunked - full| over ranks: mean {mean_diff:.5f} (tol {PARITY_MEAN_TOL}), "
        f"max {max_diff:.5f} (tol {PARITY_MAX_TOL}) ({'PASS' if parity_ok else 'FAIL'})"
    )

    log("\n[2/2] Training with chunked logprobs (kl_beta > 0)...")
    train_result = trainer.train()
    step_losses = [e["loss"] for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]
    losses_finite = bool(step_losses) and all(math.isfinite(l) for l in step_losses)
    log(f"  Losses: {[f'{l:.4f}' for l in step_losses]} ({'PASS' if losses_finite else 'FAIL'})")
    log(f"  Final training loss: {train_result.training_loss:.6f}")

    return {"checks": {"logprob_parity": parity_ok, "losses_finite": losses_finite}}


main = gpu_test_main(min_world_size=1, prefix="offline_grpo_chunked")(run)

if __name__ == "__main__":
    main()
