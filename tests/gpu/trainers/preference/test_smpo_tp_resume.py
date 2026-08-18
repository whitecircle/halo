#!/usr/bin/env python
"""
SMPO checkpoint resume test with Tensor Parallelism (TP=2).

Validates that SmoothMarginPOTrainer can:
1. Train with TP=2, save a mid-training checkpoint
2. Resume from that checkpoint and continue training
3. Produce finite losses after resume

This specifically tests the checkpoint save/load path under TP mode,
where model parameters are DTensors that need to be gathered for saving
and re-sharded when loading.

Model: Qwen/Qwen3-0.6B (dense, supports tp_plan="auto")

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/preference/test_smpo_tp_resume.py

Requirements:
    - 2x GPUs
"""

import math
import os
import traceback

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from src.configs.smpo_config import SmoothMarginPOConfig
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.preference.smpo import SmoothMarginPOTrainer
from tests.common.datasets import create_preference_dataset
from tests.common.distributed import cleanup_dirs, shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.utils import cleanup_memory, log

# Configuration

MODEL_NAME = QWEN3_0_6B
TP_SIZE = 2
TOTAL_STEPS = 6
SAVE_AT_STEP = 3  # Save checkpoint at step 3, resume and train to step 6
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 5e-6
MAX_LENGTH = 2048
MAX_PROMPT_LENGTH = 1024
NUM_TRAIN_SAMPLES = 64
NUM_EVAL_SAMPLES = 8
SEED = 42

# Weight round-trip tolerance: the TP DTensor gather→re-shard path is the point of this
# test. A correct resume reloads the trained weights, so the pre-save and post-resume
# forward loss on a FIXED batch must match to bf16 round-trip noise; a corrupted gather
# (dropped/mis-sharded weights) shifts it by >>1.
LOSS_TOL = 1e-2


# Checkpoint file + by-value continuity helpers


def verify_tp_checkpoint(checkpoint_dir: str) -> tuple[bool, str]:
    """Per-file checkpoint asserts (vs a bare isdir): TP saves gathered HF weights + scheduler.pt +
    per-rank optimizer shards (optimizer_meta.pt + optimizer_shard_*.pt — TP exact-resumes the
    per-rank Adam shards, same contract as test_offline_grpo_tp_resume)."""
    if not os.path.isdir(checkpoint_dir):
        return False, f"  Checkpoint directory not found: {checkpoint_dir}"
    files = set(os.listdir(checkpoint_dir))
    checks = {
        "model_weights(.safetensors)": any(f.endswith(".safetensors") for f in files),
        "scheduler.pt": "scheduler.pt" in files,
        "optimizer_meta.pt": "optimizer_meta.pt" in files,
        "optimizer_shards(per-rank)": len([f for f in files if f.startswith("optimizer_shard_")]) == TP_SIZE,
        "trainer_state.json": "trainer_state.json" in files,
    }
    lines = [f"  Checkpoint: {checkpoint_dir}", f"  Files: {sorted(files)}"]
    for name, ok in checks.items():
        lines.append(f"    {'OK' if ok else 'MISSING':7s}  {name}")
    failed = [k for k, v in checks.items() if not v]
    if failed:
        lines.append(f"  MISSING: {failed}")
    return all(checks.values()), "\n".join(lines)


def _fixed_batch(tokenizer, device):
    """Deterministic single-sequence batch (identical tokens pre- and post-resume)."""
    text = (
        "User: What is 17 plus 25?\nAssistant: The answer is 42. "
        "The TP gather and re-shard must survive a checkpoint save and resume intact."
    )
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    ids = enc["input_ids"].to(device)
    return ids, ids.clone()


def _forward_loss(trainer, ids, labels) -> float:
    """Plain causal-LM forward loss on a fixed batch (probes weights, not the SMPO loss)."""
    model = trainer.model
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(input_ids=ids, labels=labels, use_cache=False)
    finally:
        if was_training:
            model.train()
    return out.loss.item()


def _optimizer_moments_stats(trainer) -> tuple[bool, bool, bool]:
    """Scan this rank's local optimizer exp_avg_sq. Returns (materialized, nonzero, finite).

    TP stores DTensors; ``.to_local()`` reads this rank's shard with no collective.
    """
    materialized = any_nonzero = False
    all_finite = True
    for state in trainer.optimizer.state.values():
        sq = state.get("exp_avg_sq")
        if sq is None:
            continue
        materialized = True
        local = (sq.to_local() if hasattr(sq, "to_local") else sq).detach()
        if (local != 0).any().item():
            any_nonzero = True
        if not torch.isfinite(local).all().item():
            all_finite = False
    return materialized, any_nonzero, all_finite


def _make_resume_capture_callback(trainer_ref: dict, ids, labels):
    """Callback snapshotting resumed state at on_train_begin (post-resume, pre-step)."""
    from transformers import TrainerCallback

    class _ResumeCaptureCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            trainer = trainer_ref["trainer"]
            materialized, nonzero, finite = _optimizer_moments_stats(trainer)
            trainer_ref["capture"] = {
                "l_post": _forward_loss(trainer, ids, labels),
                "moments_materialized": materialized,
                "moments_nonzero": nonzero,
                "moments_finite": finite,
                "sched_last_epoch": int(trainer.lr_scheduler.last_epoch),
            }
            return control

    return _ResumeCaptureCallback()


# Phase 1: Train + Save Checkpoint


def phase1_train_and_save(
    rank,
    world_size,
    local_rank,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir,
    checkpoint_dir,
) -> tuple[bool, list[float], float]:
    """Train for SAVE_AT_STEP steps with TP=2, save checkpoint.

    Returns (ok, step_losses, L_pre) — L_pre is the fixed-batch causal-LM forward loss
    with the trained (== saved) weights, the reference for the Phase 2 round-trip check.
    """
    log(f"\n{'=' * 60}")
    log(f"  Phase 1: Train {SAVE_AT_STEP} steps + Save Checkpoint (TP={TP_SIZE})")
    log(f"{'=' * 60}")

    try:
        parallelism_config = ParallelismConfig(tp_size=TP_SIZE)

        log("Loading model with TP...")
        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            use_liger_kernel=True,
            attn_implementation="sdpa",  # padded preference: avoid FA4 varlen-bwd cold-compile (see test_smpo_tp)
        )
        log(f"Model loaded. GPU mem: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        config = SmoothMarginPOConfig(
            output_dir=output_dir,
            max_steps=SAVE_AT_STEP,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="steps",
            save_steps=SAVE_AT_STEP,
            save_total_limit=1,
            report_to="none",
            max_length=MAX_LENGTH,
            max_prompt_length=MAX_PROMPT_LENGTH,
            dataloader_drop_last=True,
            fsdp="",
        )

        trainer = SmoothMarginPOTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )

        assert trainer.is_tp_mode, "trainer.is_tp_mode should be True"

        log(f"Training for {SAVE_AT_STEP} steps...")
        train_result = trainer.train()

        training_loss = train_result.training_loss
        step_losses = [e["loss"] for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]
        log(f"Phase 1 training loss: {training_loss:.6f}")
        log(f"Phase 1 step losses: {[f'{l:.4f}' for l in step_losses]}")

        # Reference forward loss on a FIXED batch with the trained (== saved) weights.
        ids, labels = _fixed_batch(tokenizer, torch.cuda.current_device())
        l_pre = _forward_loss(trainer, ids, labels)
        log(f"Phase 1 L_pre (fixed-batch forward loss, trained weights): {l_pre:.6f}")

        # The trainer should have saved a checkpoint at step SAVE_AT_STEP — assert per-file.
        expected_ckpt = os.path.join(output_dir, f"checkpoint-{SAVE_AT_STEP}")
        barrier()

        ckpt_ok = True
        if rank == 0:
            ckpt_ok, ckpt_detail = verify_tp_checkpoint(expected_ckpt)
            log(ckpt_detail)
        result_t = torch.tensor([1 if ckpt_ok else 0], dtype=torch.int64, device=torch.cuda.current_device())
        dist.broadcast(result_t, src=0)
        if result_t.item() == 0:
            log(f"ERROR: checkpoint at {expected_ckpt} missing required files")
            return False, step_losses, l_pre

        loss_ok = math.isfinite(training_loss)
        if not loss_ok:
            log(f"ERROR: Phase 1 training loss not finite: {training_loss}")
            return False, step_losses, l_pre

        del trainer, model
        cleanup_memory()
        barrier()

        return True, step_losses, l_pre

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
    """Resume training from checkpoint and train to TOTAL_STEPS.

    Adds the TP weight round-trip check (|L_post - L_pre| < LOSS_TOL on a fixed batch,
    captured post-resume/pre-step) plus full optimizer continuity (exp_avg_sq restored,
    nonzero, finite) — TP does a full optimizer restore, not a warm restart.
    """
    log(f"\n{'=' * 60}")
    log(f"  Phase 2: Resume from Checkpoint + Train to step {TOTAL_STEPS} (TP={TP_SIZE})")
    log(f"{'=' * 60}")

    checkpoint_path = os.path.join(output_dir, f"checkpoint-{SAVE_AT_STEP}")

    try:
        parallelism_config = ParallelismConfig(tp_size=TP_SIZE)

        log("Loading fresh model with TP for resume...")
        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            use_liger_kernel=True,
            attn_implementation="sdpa",  # padded preference: avoid FA4 varlen-bwd cold-compile (see test_smpo_tp)
        )
        log(f"Model loaded. GPU mem: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        config = SmoothMarginPOConfig(
            output_dir=output_dir,
            max_steps=TOTAL_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=MAX_LENGTH,
            max_prompt_length=MAX_PROMPT_LENGTH,
            dataloader_drop_last=True,
            fsdp="",
        )

        trainer = SmoothMarginPOTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )

        assert trainer.is_tp_mode, "trainer.is_tp_mode should be True"

        # Capture restored state at on_train_begin (post-resume, pre-first-step).
        ids, labels = _fixed_batch(tokenizer, torch.cuda.current_device())
        trainer_ref: dict = {"trainer": trainer, "capture": None}
        trainer.add_callback(_make_resume_capture_callback(trainer_ref, ids, labels))

        log(f"Resuming from checkpoint: {checkpoint_path}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint_path)

        training_loss = train_result.training_loss
        step_losses = [e["loss"] for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]
        log(f"Phase 2 training loss: {training_loss:.6f}")
        log(f"Phase 2 step losses: {[f'{l:.4f}' for l in step_losses]}")
        log(f"Phase 2 global step: {trainer.state.global_step}")

        # Validate
        loss_ok = math.isfinite(training_loss)
        steps_ok = trainer.state.global_step == TOTAL_STEPS
        all_finite = all(math.isfinite(l) for l in step_losses)

        if not loss_ok:
            log(f"ERROR: Phase 2 loss not finite: {training_loss}")
        if not steps_ok:
            log(f"ERROR: Expected {TOTAL_STEPS} steps, got {trainer.state.global_step}")
        if not all_finite:
            log("ERROR: NaN/Inf in step losses")

        # ---- By-value resume continuity (TP DTensor gather→re-shard round-trip) ----
        cap = trainer_ref["capture"]
        if cap is None:
            log("ERROR: resume-capture callback did not fire (on_train_begin missed)")
            weights_ok = optim_ok = False
        else:
            l_post = cap["l_post"]
            loss_delta = abs(l_post - l_pre)
            weights_ok = math.isfinite(l_post) and loss_delta < LOSS_TOL
            optim_ok = cap["moments_materialized"] and cap["moments_nonzero"] and cap["moments_finite"]
            sched_ok = cap["sched_last_epoch"] == SAVE_AT_STEP
            log(f"L_pre={l_pre:.6f}  L_post={l_post:.6f}  |delta|={loss_delta:.6f} (tol {LOSS_TOL})")
            log(
                f"optimizer exp_avg_sq materialized={cap['moments_materialized']} "
                f"nonzero={cap['moments_nonzero']} finite={cap['moments_finite']}"
            )
            log(f"lr_scheduler.last_epoch={cap['sched_last_epoch']} (expected {SAVE_AT_STEP})")
            if not weights_ok:
                log(
                    f"ERROR: weights not restored — |L_post - L_pre|={loss_delta:.6f} >= {LOSS_TOL} "
                    "(TP gather/re-shard corrupted the round-trip)"
                )
            if not optim_ok:
                log("ERROR: optimizer 2nd moment not restored (zero/non-finite/absent after TP resume)")
            if not sched_ok:
                log(
                    f"ERROR: LR scheduler not restored — last_epoch={cap['sched_last_epoch']} "
                    f"(expected {SAVE_AT_STEP})"
                )
            optim_ok = optim_ok and sched_ok

        del trainer, model
        cleanup_memory()
        barrier()

        return loss_ok and steps_ok and all_finite and weights_ok and optim_ok, step_losses

    except Exception as e:
        log(f"Phase 2 FAILED: {e}")
        traceback.print_exc()
        return False, []


def run(ctx):
    log(f"\n{'#' * 70}")
    log(f"  SMPO + TP={TP_SIZE} Checkpoint Resume Test")
    log(f"  Model: {MODEL_NAME}")
    log(f"  World: {ctx.world_size}, GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  Plan: Train {SAVE_AT_STEP} steps -> Save -> Resume -> Train to {TOTAL_STEPS}")
    log(f"{'#' * 70}")

    # Rank-shared: the checkpoint rank 0 writes is the one every rank resumes from. Rank 0 clears it
    # first so a checkpoint left by an earlier run cannot stand in for the one this run must save.
    output_dir = shared_scratch_dir("smpo_tp_resume_out")
    if ctx.rank == 0:
        cleanup_dirs(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    ctx.on_teardown(lambda: cleanup_dirs(output_dir))
    ctx.barrier()
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{SAVE_AT_STEP}")

    # Load tokenizer
    log("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create datasets
    log("Creating synthetic preference datasets...")
    train_dataset = create_preference_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_preference_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)
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
        checkpoint_dir,
    )
    checks = {"phase1_train_and_save": phase1_ok}

    if not phase1_ok:
        log("\nPhase 1 FAILED - skipping Phase 2")
        return {"checks": checks}

    # Phase 2: Resume + Train
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


main = gpu_test_main(min_world_size=TP_SIZE, prefix="smpo_tp_resume")(run)

if __name__ == "__main__":
    main()
