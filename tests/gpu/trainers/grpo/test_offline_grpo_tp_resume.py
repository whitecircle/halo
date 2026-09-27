#!/usr/bin/env python
"""
Offline GRPO checkpoint resume test with Tensor Parallelism (TP=2) or Expert Parallelism (EP=2).

Validates that OfflineGRPOTrainer can train, save a mid-training checkpoint, resume from it, and
restore the trained weights BY VALUE (|L_post - L_pre| < tol at on_train_begin), plus the mode's
optimizer contract: TP exact-resumes the per-rank Adam shards; EP warm-restarts the optimizer (its
shards reference the EP-fused experts) and restores only the LR scheduler.

Mode via HALO_TEST_OFFGRPO_PARALLEL: "tp" (default, Qwen3-0.6B dense, tp_plan="auto") or "ep" (gpt-oss-20b MoE).

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/grpo/test_offline_grpo_tp_resume.py            # TP
    HALO_TEST_OFFGRPO_PARALLEL=ep torchrun --nproc_per_node=2 \
        tests/gpu/trainers/grpo/test_offline_grpo_tp_resume.py            # EP

Requirements:
    - 2x GPUs
"""

import math
import os
import traceback
from types import SimpleNamespace

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.env import env_str
from src.trainers.grpo.offline import OfflineGRPOTrainer
from src.training.environment import resolve_resume_weights_source
from tests.common.checkpoint_io import ResumeCapture, fixed_batch_loss, fixed_text_batch
from tests.common.datasets import create_offline_grpo_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B, QWEN3_0_6B
from tests.common.utils import cleanup_memory, log, step_losses

# Configuration

# HALO_TEST_OFFGRPO_PARALLEL=ep runs the same resume test under Expert Parallelism (EP=2) on an MoE model
# (warm-restart optimizer by design — EP fuses experts, so the optimizer reinitializes and only the LR
# scheduler restores); default tp runs Tensor Parallelism (TP=2, exact per-rank optimizer resume).
_MODE = env_str("HALO_TEST_OFFGRPO_PARALLEL", "tp")
_IS_EP = _MODE == "ep"
MODEL_NAME = GPT_OSS_20B if _IS_EP else QWEN3_0_6B
TP_SIZE = 2
EP_SIZE = 2
TOTAL_STEPS = 6
SAVE_AT_STEP = 3
BATCH_SIZE = 1
LEARNING_RATE = 5e-6
MAX_PROMPT_LENGTH = 1024
MAX_COMPLETION_LENGTH = 1024
NUM_TRAIN_SAMPLES = 64
NUM_EVAL_SAMPLES = 8
SEED = 42


# By-value weight-restoration probes (catch a silent-base-weights / corrupted-gather resume)

LOSS_TOL = 1e-2  # TP forward is deterministic, so restored weights must reproduce L_pre tightly.
# The sequence the fixed-batch loss is scored on before the save and after the resume.
FIXED_TEXT = (
    "User: What is 17 plus 25?\nAssistant: The answer is 42. "
    "The TP gather and re-shard must survive a checkpoint save and resume intact."
)


# Phase 1: Train + Save Checkpoint


def phase1_train_and_save(
    rank,
    world_size,
    local_rank,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir,
) -> tuple[bool, list[float], float]:
    """Train for SAVE_AT_STEP steps with TP=2, save checkpoint. Returns (ok, losses, l_pre).

    ``l_pre`` is the fixed-batch forward loss on the TRAINED weights — Phase 2 must reproduce it after
    resume (else the checkpoint did not restore the trained weights; the silent-base-weights regression).
    """
    log(f"\n{'=' * 60}")
    log(f"  Phase 1: Train {SAVE_AT_STEP} steps + Save (TP={TP_SIZE})")
    log(f"{'=' * 60}")

    try:
        parallelism_config = ParallelismConfig(ep_size=EP_SIZE) if _IS_EP else ParallelismConfig(tp_size=TP_SIZE)

        log("Loading model with TP...")
        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            use_liger_kernel=True,
        )
        log(f"Model loaded. GPU mem: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        config = OfflineGRPOConfig(
            output_dir=output_dir,
            max_steps=SAVE_AT_STEP,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=1,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="steps",
            save_steps=SAVE_AT_STEP,
            save_total_limit=1,
            report_to="none",
            logging_nan_inf_filter=False,
            max_prompt_length=MAX_PROMPT_LENGTH,
            max_completion_length=MAX_COMPLETION_LENGTH,
            dataloader_drop_last=True,
            fsdp="",
        )

        trainer = OfflineGRPOTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )

        assert trainer._has_ep_layers if _IS_EP else trainer.is_tp_mode, f"expected {_MODE} mode active"

        log(f"Training for {SAVE_AT_STEP} steps...")
        train_result = trainer.train()

        training_loss = train_result.training_loss
        losses = step_losses(trainer)
        log(f"Phase 1 loss: {training_loss:.6f}")
        log(f"Phase 1 step losses: {[f'{l:.4f}' for l in losses]}")

        # Fixed-batch loss on the TRAINED weights (before they're freed) — the by-value anchor for resume.
        ids, labels = fixed_text_batch(tokenizer, torch.cuda.current_device(), FIXED_TEXT)
        l_pre = fixed_batch_loss(trainer.model, ids, labels)
        log(f"Phase 1 L_pre (fixed-batch forward loss, trained weights): {l_pre:.6f}")

        # Check checkpoint
        expected_ckpt = os.path.join(output_dir, f"checkpoint-{SAVE_AT_STEP}")
        barrier()

        if rank == 0:
            if os.path.isdir(expected_ckpt):
                files = os.listdir(expected_ckpt)
                log(f"Checkpoint saved at: {expected_ckpt}")
                log(f"Checkpoint files: {sorted(files)}")
            else:
                log(f"ERROR: Checkpoint not found at {expected_ckpt}")
                return False, losses, l_pre

        loss_ok = math.isfinite(training_loss)
        if not loss_ok:
            log(f"ERROR: Phase 1 loss not finite: {training_loss}")

        del trainer, model
        cleanup_memory()
        barrier()

        return loss_ok, losses, l_pre

    except Exception as e:
        log(f"Phase 1 FAILED: {e}")
        traceback.print_exc()
        return False, [], float("nan")


# Phase 2: Resume from Checkpoint


def phase2_resume_and_train(
    rank,
    world_size,
    local_rank,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir,
    l_pre,
) -> tuple[bool, list[float]]:
    """Resume from checkpoint and train to TOTAL_STEPS, asserting the trained weights were restored.

    TP exact-resumes: the by-value check (``|L_post - L_pre| < LOSS_TOL`` at on_train_begin) catches a
    silent-base-weights / corrupted-gather resume, and the Adam-2nd-moment + scheduler checks lock the
    full-continuity contract.
    """
    log(f"\n{'=' * 60}")
    log(f"  Phase 2: Resume + Train to step {TOTAL_STEPS} (TP={TP_SIZE})")
    log(f"{'=' * 60}")

    checkpoint_path = os.path.join(output_dir, f"checkpoint-{SAVE_AT_STEP}")

    try:
        parallelism_config = ParallelismConfig(ep_size=EP_SIZE) if _IS_EP else ParallelismConfig(tp_size=TP_SIZE)

        # Resolve the weights source via the SAME production helper the training scripts use: EP skips the
        # loader's weight reload, so the trained weights must come from the checkpoint at construction (else
        # the model silently carries BASE weights and the by-value check passes only because few steps barely
        # moved them). TP returns base and the loader reloads the checkpoint.
        weights_source = resolve_resume_weights_source(
            checkpoint_path, SimpleNamespace(model_name_or_path=MODEL_NAME), parallelism_config
        )
        log(
            f"Loading fresh model for resume (weights_source={'checkpoint' if weights_source == checkpoint_path else 'base'})..."
        )
        model, _ = load_distributed_model(
            model_name_or_path=weights_source,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            use_liger_kernel=True,
        )
        log(f"Model loaded. GPU mem: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        config = OfflineGRPOConfig(
            output_dir=output_dir,
            max_steps=TOTAL_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=1,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            logging_nan_inf_filter=False,
            max_prompt_length=MAX_PROMPT_LENGTH,
            max_completion_length=MAX_COMPLETION_LENGTH,
            dataloader_drop_last=True,
            fsdp="",
        )

        trainer = OfflineGRPOTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )

        assert trainer._has_ep_layers if _IS_EP else trainer.is_tp_mode, f"expected {_MODE} mode active"

        # Capture restored state at on_train_begin (post-resume, pre-first-step).
        ids, labels = fixed_text_batch(tokenizer, torch.cuda.current_device(), FIXED_TEXT)
        resume_capture = ResumeCapture(trainer, ids, labels)
        trainer.add_callback(resume_capture)

        log(f"Resuming from: {checkpoint_path}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint_path)

        training_loss = train_result.training_loss
        losses = step_losses(trainer)
        log(f"Phase 2 loss: {training_loss:.6f}")
        log(f"Phase 2 step losses: {[f'{l:.4f}' for l in losses]}")
        log(f"Phase 2 global step: {trainer.state.global_step}")

        loss_ok = math.isfinite(training_loss)
        steps_ok = trainer.state.global_step == TOTAL_STEPS
        all_finite = all(math.isfinite(l) for l in losses)

        if not loss_ok:
            log(f"ERROR: Phase 2 loss not finite: {training_loss}")
        if not steps_ok:
            log(f"ERROR: Expected {TOTAL_STEPS} steps, got {trainer.state.global_step}")
        if not all_finite:
            log("ERROR: NaN/Inf in step losses")

        # ---- By-value resume continuity (post-resume, pre-step snapshot) ----
        cap = resume_capture.capture
        if cap is None:
            log("ERROR: resume-capture callback did not fire (on_train_begin missed)")
            del trainer, model
            cleanup_memory()
            barrier()
            return False, losses

        l_post = cap["l_post"]
        loss_delta = abs(l_post - l_pre)
        # Restored trained weights must reproduce L_pre; a base-weights/corrupted-gather resume shifts the
        # fixed-batch loss by ~O(1). EP forward is non-deterministic (DeepEP all-to-all reordering) → looser
        # band, still far below the not-restored gap; TP forward is deterministic → tight band.
        weight_tol = 0.5 if _IS_EP else LOSS_TOL
        weights_ok = math.isfinite(l_post) and loss_delta < weight_tol
        log(f"  L_pre={l_pre:.6f}  L_post={l_post:.6f}  |delta|={loss_delta:.6f} (tol {weight_tol})")
        if not weights_ok:
            log(f"  ERROR: weights not restored — |L_post - L_pre|={loss_delta:.6f} >= {weight_tol}")

        # TP = exact resume (Adam 2nd moment restored); EP = warm restart (optimizer reinitialized — its
        # shards reference the EP-fused structure), so it must be RESET. Both restore the LR scheduler.
        if _IS_EP:
            optim_ok = not cap["moments_materialized"]
        else:
            optim_ok = cap["moments_materialized"] and cap["moments_nonzero"] and cap["moments_finite"]
        sched_ok = cap["sched_last_epoch"] == SAVE_AT_STEP
        log(
            f"  optimizer exp_avg_sq materialized={cap['moments_materialized']} "
            f"nonzero={cap['moments_nonzero']} finite={cap['moments_finite']}; "
            f"scheduler last_epoch={cap['sched_last_epoch']} (expected {SAVE_AT_STEP})"
        )
        if not optim_ok:
            log(f"  ERROR: optimizer state wrong for {_MODE} (TP wants full continuity; EP wants reset)")
        if not sched_ok:
            log(f"  ERROR: scheduler last_epoch={cap['sched_last_epoch']} != {SAVE_AT_STEP}")

        del trainer, model
        cleanup_memory()
        barrier()

        return loss_ok and steps_ok and all_finite and weights_ok and optim_ok and sched_ok, losses

    except Exception as e:
        log(f"Phase 2 FAILED: {e}")
        traceback.print_exc()
        return False, []


# Main


def run(ctx) -> dict:
    log(f"\n{'#' * 70}")
    log(f"  Offline GRPO + TP={TP_SIZE} Checkpoint Resume Test")
    log(f"  Model: {MODEL_NAME}")
    log(f"  World: {ctx.world_size}, GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  Plan: Train {SAVE_AT_STEP} steps -> Save -> Resume -> Train to {TOTAL_STEPS}")
    log(f"{'#' * 70}")

    # Broadcast output_dir from rank 0 so all ranks use the same checkpoint path
    dirs = [ctx.output_dir]
    dist.broadcast_object_list(dirs, src=0)
    output_dir = dirs[0]

    log("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("Creating synthetic datasets...")
    train_dataset = create_offline_grpo_dataset(tokenizer, NUM_TRAIN_SAMPLES, seed=SEED)
    eval_dataset = create_offline_grpo_dataset(tokenizer, NUM_EVAL_SAMPLES, seed=SEED + 100)
    log(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Phase 1: Train + Save
    phase1_ok, phase1_losses, l_pre = phase1_train_and_save(
        ctx.rank,
        ctx.world_size,
        ctx.local_rank,
        tokenizer,
        train_dataset,
        eval_dataset,
        output_dir,
    )
    checks = {"phase1_train_and_save": phase1_ok}

    if not phase1_ok:
        log("\nPhase 1 FAILED - skipping Phase 2")
        return {"checks": checks}

    # Phase 2: Resume + Train (asserts trained weights restored by value)
    phase2_ok, phase2_losses = phase2_resume_and_train(
        ctx.rank,
        ctx.world_size,
        ctx.local_rank,
        tokenizer,
        train_dataset,
        eval_dataset,
        output_dir,
        l_pre,
    )
    checks["phase2_resume_and_train"] = phase2_ok

    log(f"  Phase 1 losses: {[f'{l:.4f}' for l in phase1_losses]}")
    log(f"  Phase 2 losses: {[f'{l:.4f}' for l in phase2_losses]}")

    return {"checks": checks}


main = gpu_test_main(min_world_size=TP_SIZE, prefix="offline_grpo_tp_resume")(run)

if __name__ == "__main__":
    main()
