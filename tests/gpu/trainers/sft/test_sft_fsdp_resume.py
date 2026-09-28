#!/usr/bin/env python
"""
SFT Checkpoint Save + Resume Test (FSDP2 mode).

Validates that DistributedSFTTrainer correctly saves FSDP2 per-rank optimizer
shards during mid-training checkpoints and can resume from them.

Test plan:
  Phase 1 — Train for SAVE_AT_STEP steps with save_strategy="steps":
    • optimizer_shard_XXXXX.pt exists for every rank
    • optimizer_meta.pt present with correct num_ranks
    • scheduler.pt, rng_state_*.pth, trainer_state.json, model weights present
  Phase 2 — Resume from checkpoint, continue to TOTAL_STEPS:
    • global_step == TOTAL_STEPS after resume, every resumed step loss finite
    • at the first resumed step: the trained weights (fixed-batch loss), Adam's moments bit-exact
      against the pre-save shard view, and the LR scheduler at the saved step

Model: Qwen/Qwen3-0.6B  |  GPUs: 2  |  Mode: FSDP2 (standard data parallelism)

Run:
    torchrun --nproc_per_node=2 \\
        tests/gpu/trainers/sft/test_sft_fsdp_resume.py
"""

import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.checkpoint_io import (
    ResumeCapture,
    fixed_batch_loss,
    fixed_text_batch,
    resume_checkpoint_checks,
    resume_continuity_checks,
)
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.tolerances import TOL
from tests.common.utils import cleanup_memory, local_optimizer_state, log, step_losses, training_run_checks

# Configuration

MODEL_NAME = QWEN3_0_6B
TOTAL_STEPS = 6
SAVE_AT_STEP = 3
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 512
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
SEED = 42

# The sequence the fixed-batch loss is scored on before the save and after the resume.
FIXED_TEXT = (
    "User: What is 17 plus 25?\nAssistant: The answer is 42. "
    "Optimizer moments and weights must survive a checkpoint save and resume intact."
)


def _load_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )


def _sft_config(output_dir: str, max_steps: int, **save_args) -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        **save_args,
    )


# Phase 1: Train + Save Checkpoint


def phase1_train_and_save(
    ctx,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
) -> tuple[dict[str, bool], list[float], float, dict]:
    """Train for SAVE_AT_STEP steps and check the checkpoint's files (rank 0 reads, every rank agrees).

    Returns (checks, losses, L_pre, optimizer_state) where L_pre is the forward loss on
    a FIXED deterministic batch computed AFTER training but BEFORE save — the reference for
    the post-resume weight-continuity check in Phase 2 — and optimizer_state is this rank's
    view of the moments the checkpoint's shard holds, the reference for the bit-exact check.
    """
    log(f"\n{'=' * 60}")
    log(f"  Phase 1: Train {SAVE_AT_STEP} steps + verify checkpoint (FSDP2)")
    log(f"{'=' * 60}")

    model = _load_model()
    trainer = DistributedSFTTrainer(
        model=model,
        args=_sft_config(output_dir, SAVE_AT_STEP, save_strategy="steps", save_steps=SAVE_AT_STEP, save_total_limit=1),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=ParallelismConfig(),
    )
    train_result = trainer.train()
    losses = step_losses(trainer)
    checks = training_run_checks(train_result, trainer, SAVE_AT_STEP)

    # Reference forward loss on a FIXED batch with the trained (== saved) weights.
    # The checkpoint was written during train() at step SAVE_AT_STEP, so these are
    # exactly the weights resume must restore.
    ids, labels = fixed_text_batch(tokenizer, torch.cuda.current_device(), FIXED_TEXT)
    l_pre = fixed_batch_loss(trainer.model, ids, labels)
    log(f"  L_pre (fixed-batch forward loss, trained weights): {l_pre:.6f}")

    # The checkpoint was written at step SAVE_AT_STEP, the last step of this phase, and nothing
    # steps the optimizer afterwards — so this snapshot IS the state the shard files carry.
    optimizer_state = local_optimizer_state(trainer.model, trainer.optimizer)
    log(f"  Pre-save optimizer state: {len(optimizer_state['state'])} params with moments")

    ctx.barrier()
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{SAVE_AT_STEP}")
    file_checks = resume_checkpoint_checks(checkpoint_dir, ctx.world_size) if ctx.rank == 0 else {}
    checks |= ctx.broadcast_checks(file_checks)

    del trainer, model
    cleanup_memory()
    ctx.barrier()
    return checks, losses, l_pre, optimizer_state


# Phase 2: Resume from checkpoint


def phase2_resume_and_train(
    ctx,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
    l_pre: float,
    optimizer_state_pre: dict,
) -> tuple[dict[str, bool], list[float]]:
    """Resume from the checkpoint and train to TOTAL_STEPS, grading the state the resume restored at
    its first step (:func:`~tests.common.checkpoint_io.resume_continuity_checks`)."""
    log(f"\n{'=' * 60}")
    log(f"  Phase 2: Resume from checkpoint-{SAVE_AT_STEP} -> step {TOTAL_STEPS}")
    log(f"{'=' * 60}")

    checkpoint_path = os.path.join(output_dir, f"checkpoint-{SAVE_AT_STEP}")
    model = _load_model()
    trainer = DistributedSFTTrainer(
        model=model,
        args=_sft_config(output_dir, TOTAL_STEPS, save_strategy="no"),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=ParallelismConfig(),
    )

    # Capture restored state at on_train_begin (post-resume, pre-first-step).
    ids, labels = fixed_text_batch(tokenizer, torch.cuda.current_device(), FIXED_TEXT)
    resume_capture = ResumeCapture(trainer, ids, labels, optimizer_state=True)
    trainer.add_callback(resume_capture)

    log(f"  Resuming from: {checkpoint_path}")
    train_result = trainer.train(resume_from_checkpoint=checkpoint_path)
    losses = step_losses(trainer)

    checks = {f"resume_{name}": ok for name, ok in training_run_checks(train_result, trainer, TOTAL_STEPS).items()}
    checks |= resume_continuity_checks(
        resume_capture.capture,
        l_pre,
        save_step=SAVE_AT_STEP,
        loss_tol=TOL.resume_fixed_batch_loss_abs,
        optimizer_state=optimizer_state_pre,
    )

    del trainer, model
    cleanup_memory()
    ctx.barrier()
    return checks, losses


# Main


def run(ctx) -> dict:
    log(f"\n{'#' * 70}")
    log("  SFT FSDP2 Checkpoint Save + Resume Test")
    log(f"  Model: {MODEL_NAME}")
    log(f"  World size: {ctx.world_size}, GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  Plan: Train {SAVE_AT_STEP} steps → Save → Resume → Train to {TOTAL_STEPS}")
    log(f"{'#' * 70}")

    # All ranks must use the same output_dir (the checkpoint path phase 2 resumes from).
    output_dir = shared_scratch_dir("sft_fsdp_resume")
    if ctx.rank == 0:
        ctx.on_teardown(lambda: shutil.rmtree(output_dir, ignore_errors=True))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1)

    checks, phase1_losses, l_pre, optimizer_state_pre = phase1_train_and_save(
        ctx, tokenizer, train_dataset, eval_dataset, output_dir
    )
    # Every phase-1 check is rank-uniform (the logged losses are world means, the file checks are
    # rank 0's broadcast), so every rank takes the same branch.
    if not all(checks.values()):
        log("\n  Phase 1 FAILED — skipping Phase 2")
        return {"checks": checks}

    phase2_checks, phase2_losses = phase2_resume_and_train(
        ctx, tokenizer, train_dataset, eval_dataset, output_dir, l_pre, optimizer_state_pre
    )
    checks |= phase2_checks

    log(f"\n  Phase 1 losses: {[f'{l:.4f}' for l in phase1_losses]}")
    log(f"  Phase 2 losses: {[f'{l:.4f}' for l in phase2_losses]}")
    return {"checks": checks}


main = gpu_test_main(min_world_size=1, prefix="sft_fsdp_resume")(run)

if __name__ == "__main__":
    main()
