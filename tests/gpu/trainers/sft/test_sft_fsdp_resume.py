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
    • global_step == TOTAL_STEPS after resume
    • All resumed step losses are finite

Model: Qwen/Qwen3-0.6B  |  GPUs: 2  |  Mode: FSDP2 (standard data parallelism)

Run:
    torchrun --nproc_per_node=2 \\
        tests/gpu/trainers/sft/test_sft_fsdp_resume.py
"""

import math
import os
import shutil

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.utils import assert_optimizer_state_bit_exact, cleanup_memory, local_optimizer_state, log

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

# Weight round-trip tolerance: a correct resume reloads the trained weights, so the
# pre-save and post-resume forward loss on a FIXED batch must match to bf16 round-trip
# noise. A resume that leaves the model at fresh init (or corrupts the gather) shifts the
# loss by >>1, so this is comfortably discriminating.
LOSS_TOL = 1e-2


# By-value continuity helpers


def _fixed_batch(tokenizer, device):
    """Deterministic single-sequence batch (identical tokens pre- and post-resume)."""
    text = (
        "User: What is 17 plus 25?\nAssistant: The answer is 42. "
        "Optimizer moments and weights must survive a checkpoint save and resume intact."
    )
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    ids = enc["input_ids"].to(device)
    return ids, ids.clone()


def _forward_loss(trainer, ids, labels) -> float:
    """Forward loss on a fixed batch through the live (FSDP2-wrapped) trainer model.

    Uses no optimizer step, so it reflects only the current weights.
    """
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


def _optimizer_moments_stats(trainer) -> tuple[bool, bool]:
    """Scan this rank's local optimizer state for the 2nd moment (exp_avg_sq).

    Returns (any_nonzero, all_finite) over the per-rank local shards. FSDP2/TP store
    DTensors; ``.to_local()`` reads this rank's shard with no collective (so it is
    safe to call independently per rank). A resume that reinitialised Adam to zero
    leaves exp_avg_sq all-zero on every rank → any_nonzero is False.
    """
    any_nonzero = False
    all_finite = True
    seen = False
    for state in trainer.optimizer.state.values():
        sq = state.get("exp_avg_sq")
        if sq is None:
            continue
        seen = True
        local = sq.to_local() if hasattr(sq, "to_local") else sq
        local = local.detach()
        if (local != 0).any().item():
            any_nonzero = True
        if not torch.isfinite(local).all().item():
            all_finite = False
    if not seen:
        # No exp_avg_sq materialised — cannot claim restored state; treat as failure.
        return False, False
    return any_nonzero, all_finite


def _make_resume_capture_callback(trainer_ref: dict, ids, labels):
    """Build a callback that snapshots resumed state at on_train_begin.

    on_train_begin fires AFTER the resume hooks (model + optimizer + scheduler
    restored) but BEFORE the first optimizer step, so it reflects the restored
    state, not a freshly-stepped one. Captures into ``trainer_ref["capture"]``:
      l_post           — fixed-batch forward loss with restored weights
      moments_nonzero  — exp_avg_sq has a nonzero entry on this rank
      moments_finite   — exp_avg_sq is finite on this rank
      optimizer_state  — this rank's restored shard view, for the bit-exact compare
      sched_last_epoch — lr_scheduler.last_epoch after resume
    """

    class _ResumeCaptureCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            trainer = trainer_ref["trainer"]
            l_post = _forward_loss(trainer, ids, labels)
            nonzero, finite = _optimizer_moments_stats(trainer)
            trainer_ref["capture"] = {
                "l_post": l_post,
                "moments_nonzero": nonzero,
                "moments_finite": finite,
                "optimizer_state": local_optimizer_state(trainer.model, trainer.optimizer),
                "sched_last_epoch": int(trainer.lr_scheduler.last_epoch),
            }
            return control

    return _ResumeCaptureCallback()


# Checkpoint file verification


def verify_optimizer_checkpoint(checkpoint_dir: str, world_size: int) -> tuple[bool, str]:
    """Verify optimizer state and supporting files exist in checkpoint_dir.

    FSDP2 mode (world_size > 1): checks per-rank optimizer_shard_XXXXX.pt + optimizer_meta.pt.
    Single GPU (world_size == 1): FSDP2 is skipped, checks standard optimizer.pt instead.
    """
    checks = {}
    fsdp2_mode = world_size > 1
    lines = [f"  Checkpoint: {checkpoint_dir}"]
    lines.append(f"  Mode: {'FSDP2 sharded' if fsdp2_mode else 'single-GPU standard'}")

    if not os.path.isdir(checkpoint_dir):
        return False, f"  Checkpoint directory not found: {checkpoint_dir}"

    files = set(os.listdir(checkpoint_dir))
    lines.append(f"  Files: {sorted(files)}")

    if fsdp2_mode:
        # FSDP2: per-rank optimizer shards + meta
        for rank_idx in range(world_size):
            shard_name = f"optimizer_shard_{rank_idx:05d}.pt"
            checks[shard_name] = shard_name in files

        checks["optimizer_meta.pt"] = "optimizer_meta.pt" in files
        if checks["optimizer_meta.pt"]:
            meta = torch.load(
                os.path.join(checkpoint_dir, "optimizer_meta.pt"),
                map_location="cpu",
                weights_only=False,
            )
            saved_ranks = meta.get("num_ranks", -1)
            checks["meta_num_ranks_correct"] = saved_ranks == world_size
            lines.append(f"  optimizer_meta.pt: num_ranks={saved_ranks} (expected {world_size})")

        # rng states — one per rank (multi-GPU naming: rng_state_0.pth, rng_state_1.pth, ...)
        for rank_idx in range(world_size):
            rng_name = f"rng_state_{rank_idx}.pth"
            checks[rng_name] = rng_name in files
    else:
        # Single GPU: standard optimizer.pt + rng_state.pth (no rank suffix)
        checks["optimizer.pt"] = "optimizer.pt" in files
        checks["rng_state.pth"] = "rng_state.pth" in files

    # scheduler and trainer state — present in all modes
    checks["scheduler.pt"] = "scheduler.pt" in files
    checks["trainer_state.json"] = "trainer_state.json" in files

    # model weights
    has_safetensors = any(f.endswith(".safetensors") for f in files)
    has_pytorch = "pytorch_model.bin" in files or any(
        f.startswith("pytorch_model") and f.endswith(".bin") for f in files
    )
    checks["model_weights"] = has_safetensors or has_pytorch

    # log results
    for name, ok in checks.items():
        lines.append(f"    {'OK' if ok else 'MISSING':7s}  {name}")

    failed = [k for k, v in checks.items() if not v]
    passed = all(checks.values())
    if failed:
        lines.append(f"  MISSING files: {failed}")

    return passed, "\n".join(lines)


# Phase 1: Train + Save Checkpoint


def phase1_train_and_save(
    ctx,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
) -> tuple[dict[str, bool], list[float], float, dict]:
    """Train for SAVE_AT_STEP steps, verify checkpoint files on rank 0.

    Returns (checks, step_losses, L_pre, optimizer_state) where L_pre is the forward loss on
    a FIXED deterministic batch computed AFTER training but BEFORE save — the reference for
    the post-resume weight-continuity check in Phase 2 — and optimizer_state is this rank's
    view of the moments the checkpoint's shard holds, the reference for the bit-exact check.
    """
    log(f"\n{'=' * 60}")
    log(f"  Phase 1: Train {SAVE_AT_STEP} steps + verify checkpoint (FSDP2)")
    log(f"{'=' * 60}")

    log("  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    log(f"  Model loaded. GPU mem: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    config = SFTConfig(
        output_dir=output_dir,
        max_steps=SAVE_AT_STEP,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=SAVE_AT_STEP,
        save_total_limit=1,
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
    )

    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=ParallelismConfig(),
    )

    log(f"  Training for {SAVE_AT_STEP} steps...")
    train_result = trainer.train()
    training_loss = train_result.training_loss
    step_losses = [e["loss"] for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]
    log(f"  Training loss: {training_loss:.6f}")
    log(f"  Step losses:   {[f'{l:.4f}' for l in step_losses]}")

    # Reference forward loss on a FIXED batch with the trained (== saved) weights.
    # The checkpoint was written during train() at step SAVE_AT_STEP, so these are
    # exactly the weights resume must restore.
    ids, labels = _fixed_batch(tokenizer, torch.cuda.current_device())
    l_pre = _forward_loss(trainer, ids, labels)
    log(f"  L_pre (fixed-batch forward loss, trained weights): {l_pre:.6f}")

    # The checkpoint was written at step SAVE_AT_STEP, the last step of this phase, and nothing
    # steps the optimizer afterwards — so this snapshot IS the state the shard files carry.
    optimizer_state = local_optimizer_state(trainer.model, trainer.optimizer)
    log(f"  Pre-save optimizer state: {len(optimizer_state['state'])} params with moments")

    checks = {"train_loss_finite": math.isfinite(training_loss)}
    if not checks["train_loss_finite"]:
        log(f"  ERROR: Loss not finite: {training_loss}")
    else:
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{SAVE_AT_STEP}")
        ctx.barrier()

        ckpt_ok = True
        if ctx.rank == 0:
            ckpt_ok, ckpt_detail = verify_optimizer_checkpoint(checkpoint_dir, ctx.world_size)
            log("\n  Checkpoint verification:")
            log(ckpt_detail)

        # Broadcast verification result to all ranks
        result_t = torch.tensor(
            [1 if ckpt_ok else 0],
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        dist.broadcast(result_t, src=0)
        checks["checkpoint_files_complete"] = result_t.item() == 1

    del trainer, model
    cleanup_memory()
    ctx.barrier()
    return checks, step_losses, l_pre, optimizer_state


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
    """Resume from the checkpoint and train to TOTAL_STEPS.

    Adds by-value continuity checks captured at on_train_begin (post-resume,
    pre-step): weights restored (|L_post - L_pre| < LOSS_TOL), Adam 2nd moment
    nonzero+finite (not reinitialised to zero), every moment bit-identical to the
    pre-save snapshot, and the LR scheduler advanced to the saved step (not reset
    to warmup-start)."""
    log(f"\n{'=' * 60}")
    log(f"  Phase 2: Resume from checkpoint-{SAVE_AT_STEP} -> step {TOTAL_STEPS}")
    log(f"{'=' * 60}")

    checkpoint_path = os.path.join(output_dir, f"checkpoint-{SAVE_AT_STEP}")

    log("  Loading fresh model for resume...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    log(f"  Model loaded. GPU mem: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    config = SFTConfig(
        output_dir=output_dir,
        max_steps=TOTAL_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
    )

    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=ParallelismConfig(),
    )

    # Capture restored state at on_train_begin (post-resume, pre-first-step).
    ids, labels = _fixed_batch(tokenizer, torch.cuda.current_device())
    trainer_ref: dict = {"trainer": trainer, "capture": None}
    trainer.add_callback(_make_resume_capture_callback(trainer_ref, ids, labels))

    log(f"  Resuming from: {checkpoint_path}")
    train_result = trainer.train(resume_from_checkpoint=checkpoint_path)

    training_loss = train_result.training_loss
    step_losses = [e["loss"] for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]
    global_step = trainer.state.global_step

    log(f"  Training loss:  {training_loss:.6f}")
    log(f"  Step losses:    {[f'{l:.4f}' for l in step_losses]}")
    log(f"  Final step:     {global_step} (expected {TOTAL_STEPS})")

    loss_ok = math.isfinite(training_loss)
    steps_ok = global_step == TOTAL_STEPS
    all_finite = all(math.isfinite(l) for l in step_losses) if step_losses else True

    if not loss_ok:
        log(f"  ERROR: Loss not finite: {training_loss}")
    if not steps_ok:
        log(f"  ERROR: Expected {TOTAL_STEPS} global steps, got {global_step}")
    if not all_finite:
        log(f"  ERROR: NaN/Inf in resumed step losses: {step_losses}")

    checks = {
        "resume_loss_finite": loss_ok,
        "resume_reached_total_steps": steps_ok,
        "resume_step_losses_finite": all_finite,
    }

    # ---- By-value resume continuity (post-resume, pre-step snapshot) ----
    cap = trainer_ref["capture"]
    checks["resume_capture_fired"] = cap is not None
    if cap is None:
        log("  ERROR: resume-capture callback did not fire (on_train_begin missed)")
    else:
        l_post = cap["l_post"]
        loss_delta = abs(l_post - l_pre)
        # SAVE_AT_STEP scheduler steps occurred before the checkpoint; the LR scheduler
        # is stepped once per optimizer step, so last_epoch must be == SAVE_AT_STEP, not 0.
        sched_ok = cap["sched_last_epoch"] == SAVE_AT_STEP
        weights_ok = math.isfinite(l_post) and loss_delta < LOSS_TOL
        moments_ok = cap["moments_nonzero"] and cap["moments_finite"]

        log(f"  L_pre={l_pre:.6f}  L_post={l_post:.6f}  |delta|={loss_delta:.6f} (tol {LOSS_TOL})")
        log(f"  optimizer exp_avg_sq nonzero={cap['moments_nonzero']} finite={cap['moments_finite']}")
        log(f"  lr_scheduler.last_epoch={cap['sched_last_epoch']} (expected {SAVE_AT_STEP})")

        if not weights_ok:
            log(f"  ERROR: weights not restored — |L_post - L_pre|={loss_delta:.6f} >= {LOSS_TOL}")
        if not moments_ok:
            log("  ERROR: optimizer 2nd moment (exp_avg_sq) zero/non-finite after resume (Adam state not restored)")
        if not sched_ok:
            log(f"  ERROR: LR scheduler not restored — last_epoch={cap['sched_last_epoch']} (expected {SAVE_AT_STEP})")

        # Nonzero+finite only says SOMETHING was restored; the shard must round-trip EXACTLY.
        # A cast, a dropped param or a re-derived moment leaves the trajectory silently off.
        try:
            assert_optimizer_state_bit_exact(optimizer_state_pre, cap["optimizer_state"])
            bit_exact = True
        except AssertionError as e:
            bit_exact = False
            log(f"  ERROR: optimizer state not bit-exact after resume: {e}")
        log(f"  optimizer state bit-exact vs pre-save shard view: {bit_exact}")

        checks["resume_weights_restored"] = weights_ok
        checks["resume_optimizer_moments_restored"] = moments_ok
        checks["resume_optimizer_state_bit_exact"] = bit_exact
        checks["resume_scheduler_restored"] = sched_ok

    del trainer, model
    cleanup_memory()
    ctx.barrier()
    return checks, step_losses


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

    log("\n[Setup] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"  Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    log("[Setup] Creating synthetic datasets...")
    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1)
    log(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    checks, phase1_losses, l_pre, optimizer_state_pre = phase1_train_and_save(
        ctx,
        tokenizer,
        train_dataset,
        eval_dataset,
        output_dir,
    )
    if not all(checks.values()):
        log("\n  Phase 1 FAILED — skipping Phase 2")
        return {"checks": checks}

    phase2_checks, phase2_losses = phase2_resume_and_train(
        ctx,
        tokenizer,
        train_dataset,
        eval_dataset,
        output_dir,
        l_pre,
        optimizer_state_pre,
    )
    checks.update(phase2_checks)

    log(f"\n  Phase 1 losses: {[f'{l:.4f}' for l in phase1_losses]}")
    log(f"  Phase 2 losses: {[f'{l:.4f}' for l in phase2_losses]}")
    return {"checks": checks}


main = gpu_test_main(min_world_size=1, prefix="sft_fsdp_resume")(run)

if __name__ == "__main__":
    main()
