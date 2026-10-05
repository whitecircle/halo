#!/usr/bin/env python
"""Offline GRPO ``use_chunked_grpo_logprobs`` under real FSDP2 + FA4 on 2 GPUs.

The chunked path and frozen-reference checkpoint lifecycle must both hold:

1. Parity: per-token completion log-probs from the chunked path (backbone hidden + vocab-chunked
   softmax; per-row dense forwards under FA4) must match the full-logits path on the same collated
   batch, on every rank, over the real (non-pad) completion positions.
2. Training: a short chunked run with ``kl_beta > 0`` and the default ``min_log_prob`` clamp
   finishes with finite losses. The run-start scores are restored without a sweep on resume;
   trained weights, optimizer moments and scheduler restore at the midpoint, then export reloads.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/grpo/test_offline_grpo_chunked.py
"""

import math
import os
from unittest.mock import patch

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.distributed.parallelism_config import ParallelismConfig
from src.kernels.liger.orchestrator import apply_liger_kernel
from src.trainers.grpo.offline import OfflineGRPOTrainer
from tests.common.checkpoint_io import (
    TP_RESUME_PROBE_TEXT,
    RestorePointSnapshot,
    ResumeCapture,
    fixed_batch_loss,
    fixed_text_batch,
    resume_continuity_checks,
)
from tests.common.datasets import create_offline_grpo_dataset
from tests.common.distributed import shared_output_dir
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.tolerances import TOL
from tests.common.utils import cleanup_memory, log, resumed_loss_deltas, step_losses

MODEL_NAME = QWEN3_0_6B
MAX_STEPS = 4
BATCH_SIZE = 2
NUM_TRAIN_SAMPLES = 32
SEED = 42
SAVE_STEP = MAX_STEPS // 2
# The reference itself is coarse: TRL's bf16 branch returns bf16-rounded log-probs (ULP 0.06-0.25 at
# |logp| 8-32) while the chunked path is fp32, and FA4's padded-varlen vs per-row-dense forwards add
# per-token kernel noise the full-logits path shows even against itself. The gate is therefore the
# mean at 0.05, with a looser per-token ceiling: a systematic defect (position shift, off-by-one
# span) moves every completion token by whole nats.
# Exact-math equivalence is pinned separately in fp32 on CPU (test_offline_grpo_chunked_logprobs.py).
PARITY_MEAN_TOL = 5e-2
PARITY_MAX_TOL = 0.5


def _load_model(model_path, *, attn_implementation="flash_attention_4"):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # A late instance patch misses Qwen3's q/k norms, while later loads use the patched classes.
    apply_liger_kernel(config, liger_kernel_config={"cross_entropy": False, "fused_linear_cross_entropy": False})
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )


class _SavedWeights(RestorePointSnapshot):
    def extra(self):
        ids, labels = fixed_text_batch(
            self.trainer.processing_class, torch.cuda.current_device(), TP_RESUME_PROBE_TEXT
        )
        live_loss = fixed_batch_loss(self.trainer.model, ids, labels)
        checkpoint = os.path.join(self.trainer.args.output_dir, f"checkpoint-{self.trainer.state.global_step}")
        # The checkpoint oracle separates a stale live forward from an incorrect resume without
        # replacing the live loss that the resume comparison must still match.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            oracle = _load_model(checkpoint).to(ids.device)
            try:
                checkpoint_loss = fixed_batch_loss(oracle, ids, labels)
            finally:
                del oracle
        return {"loss": live_loss, "checkpoint_loss": checkpoint_loss}


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
    output_dir = shared_output_dir(ctx)
    log(f"\n{'=' * 70}")
    log("  Offline GRPO use_chunked_grpo_logprobs test (FSDP2 + FA4)")
    log(f"  World size: {ctx.world_size}, Model: {MODEL_NAME}")
    log(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = create_offline_grpo_dataset(tokenizer, NUM_TRAIN_SAMPLES, seed=SEED)

    model = _load_model(MODEL_NAME)

    config = OfflineGRPOConfig(
        output_dir=output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=SAVE_STEP,
        save_total_limit=2,
        report_to="none",
        logging_nan_inf_filter=False,
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
    original_reference = trainer._reference_storage_by_split["training"].values.clone()
    saved = _SavedWeights("save", trainer, capture_optimizer=False)
    trainer.add_callback(saved)

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
    losses = step_losses(trainer)
    losses_finite = bool(losses) and all(math.isfinite(l) for l in losses)
    log(f"  Losses: {[f'{l:.4f}' for l in losses]} ({'PASS' if losses_finite else 'FAIL'})")
    log(f"  Final training loss: {train_result.training_loss:.6f}")

    checks = {"logprob_parity": parity_ok, "losses_finite": losses_finite}
    checkpoint = os.path.join(output_dir, f"checkpoint-{SAVE_STEP}")
    checkpoint_loss = saved.captured["loss"]
    checks["save_probe_matches_plain_checkpoint"] = (
        math.isfinite(checkpoint_loss)
        and abs(checkpoint_loss - saved.captured["checkpoint_loss"]) < TOL.resume_fixed_batch_loss_abs
    )
    del trainer, model, saved
    cleanup_memory()
    dist.barrier()
    model = _load_model(checkpoint)
    with patch.object(OfflineGRPOTrainer, "_sweep_reference_logps", side_effect=AssertionError("resume re-swept")):
        resumed = OfflineGRPOTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            parallelism_config=ParallelismConfig(),
            resume_checkpoint=checkpoint,
        )
    checks["reference_restored_bit_exact"] = torch.equal(
        original_reference, resumed._reference_storage_by_split["training"].values
    )
    checks["no_second_reference_model"] = resumed.ref_model is None
    ids, labels = fixed_text_batch(tokenizer, ctx.device, TP_RESUME_PROBE_TEXT)
    capture = ResumeCapture(resumed, ids, labels)
    resumed.add_callback(capture)
    result = resumed.train(resume_from_checkpoint=checkpoint)
    checks.update(
        resume_continuity_checks(
            capture.capture, checkpoint_loss, save_step=SAVE_STEP, loss_tol=TOL.resume_fixed_batch_loss_abs
        )
    )
    deltas = resumed_loss_deltas(losses, step_losses(resumed), save_step=SAVE_STEP, total_steps=MAX_STEPS)
    checks["resumed_losses_match"] = bool(deltas) and max(deltas) < TOL.resume_loss_abs
    checks["resumed_steps_and_loss"] = resumed.state.global_step == MAX_STEPS and math.isfinite(result.training_loss)
    trained_loss = fixed_batch_loss(resumed.model, ids, labels)
    export = os.path.join(output_dir, "export")
    resumed.save_model(export)
    del resumed, model
    cleanup_memory()
    dist.barrier()
    reloaded = _load_model(export).to(ctx.device)
    checks["export_reload_matches"] = (
        abs(fixed_batch_loss(reloaded, ids, labels) - trained_loss) < TOL.resume_fixed_batch_loss_abs
    )
    del reloaded
    cleanup_memory()
    return {"checks": checks}


main = gpu_test_main(min_world_size=1, prefix="offline_grpo_chunked")(run)

if __name__ == "__main__":
    main()
