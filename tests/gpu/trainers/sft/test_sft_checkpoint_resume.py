#!/usr/bin/env python
"""
SFT Checkpoint Save + Resume Test across parallelism modes.

Tests that DistributedSFTTrainer can save checkpoints and resume correctly
in all supported parallelism modes on 2 GPUs:

  --mode fsdp    Standard FSDP2 data parallelism (full optimizer resume)
  --mode cp      Context Parallelism CP=2 (full optimizer resume)
  --mode tp      Tensor Parallelism TP=2 (full optimizer resume)
  --mode ep      Expert Parallelism EP=2 on MoE model (full optimizer resume, gather + reload)
  --mode all     Run all modes sequentially (default)

Test plan per mode:
  Phase 1 — Train for SAVE_AT_STEP steps with save_strategy="steps":
    • Model weights saved (mode-appropriate gathering/remapping)
    • trainer_state.json saved (global_step, epoch, etc.)
  Phase 2 — Resume from checkpoint, continue to TOTAL_STEPS:
    • global_step == TOTAL_STEPS after resume
    • All resumed step losses are finite
    • Training loss is finite

All sharded modes (fsdp/tp/cp/ep) persist per-rank optimizer shards
(optimizer_shard_XXXXX.pt + optimizer_meta.pt) and restore full optimizer
continuity on resume, gated by the topology fingerprint. EP/CP additionally save
gathered HF-format weights and reload the model from scratch via
load_distributed_model() (re-applying EP/CP transformations); their optimizer
moments then restore from the shards keyed by param FQN.

Run:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_checkpoint_resume.py --mode all
"""

import argparse
import math
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.env import env_flag, env_str
from src.trainers.sft import DistributedSFTTrainer
from src.training.environment import resolve_resume_weights_source
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import cleanup_dirs, shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B, QWEN3_0_6B
from tests.common.utils import cleanup_memory, log

# Overridable so the same harness can validate resume across model families.
DEFAULT_MODEL = env_str("HALO_TEST_RESUME_MODEL", QWEN3_0_6B)
EP_MODEL = env_str("HALO_TEST_RESUME_EP_MODEL", GPT_OSS_20B)  # EP requires an MoE model
TOTAL_STEPS = 6
SAVE_AT_STEP = 3
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 512
MAX_SEQ_LENGTH_CP = 4096  # CP requires longer sequences (divisible by cp_size)
MAX_SEQ_LENGTH_EP = 2048
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
SEED = 42

# bf16 round-trip noise on a fixed-batch forward loss: a correct resume reloads the trained weights,
# so pre-save and post-resume must match to this in every mode.
LOSS_TOL = 1e-2


def _fixed_batch(tokenizer, device, seq_len: int = 64):
    """Deterministic single-sequence batch (identical tokens pre- and post-resume).

    Padded/truncated to a FIXED even ``seq_len`` so the same batch is comparable across
    a resume and is valid under CP (sequence length divisible by cp_size=2). The compared
    quantity is the per-rank forward loss computed identically pre/post resume — under CP
    each rank sees its own chunk, but L_pre and L_post use the same deterministic split, so
    a weight-corrupting reload still shifts the value.
    """
    text = (
        "User: What is 17 plus 25?\nAssistant: The answer is 42. "
        "Weights must survive a checkpoint save and resume intact across parallelism modes."
    )
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=seq_len)
    ids = enc["input_ids"].to(device)
    labels = ids.clone()
    # Keep some active labels in BOTH halves so every CP rank's chunk has supervised tokens.
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        labels[ids == pad_id] = -100
    return ids, labels


def _forward_loss(trainer, ids, labels) -> float:
    """Forward loss on a fixed batch through the live (wrapped) trainer model, no step."""
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
    """Scan this rank's local optimizer 2nd moment (exp_avg_sq).

    Returns (materialized, any_nonzero, all_finite). ``materialized`` is True if any
    exp_avg_sq exists at all. Every sharded mode (fsdp/tp/cp/ep) restores its saved
    optimizer moments on resume, so exp_avg_sq is populated → materialized True and
    any_nonzero True before the first step. FSDP/TP/CP store DTensors; ``.to_local()``
    reads this rank's shard with no collective (plain EP expert tensors need none).
    """
    materialized = False
    any_nonzero = False
    all_finite = True
    for state in trainer.optimizer.state.values():
        sq = state.get("exp_avg_sq")
        if sq is None:
            continue
        materialized = True
        local = sq.to_local() if hasattr(sq, "to_local") else sq
        local = local.detach()
        if (local != 0).any().item():
            any_nonzero = True
        if not torch.isfinite(local).all().item():
            all_finite = False
    return materialized, any_nonzero, all_finite


def _make_resume_capture_callback(trainer_ref: dict, ids, labels):
    """Callback snapshotting resumed state at on_train_begin (post-resume, pre-step)."""

    class _ResumeCaptureCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            trainer = trainer_ref["trainer"]
            materialized, nonzero, finite = _optimizer_moments_stats(trainer)
            sched = getattr(trainer, "lr_scheduler", None)
            trainer_ref["capture"] = {
                "l_post": _forward_loss(trainer, ids, labels),
                "moments_materialized": materialized,
                "moments_nonzero": nonzero,
                "moments_finite": finite,
                "sched_last_epoch": int(sched.last_epoch) if sched is not None else None,
            }
            return control

    return _ResumeCaptureCallback()


def model_for_mode(mode: str) -> str:
    """Pick an appropriate model for the parallelism mode (EP needs MoE)."""
    return EP_MODEL if mode == "ep" else DEFAULT_MODEL


def max_seq_length_for_mode(mode: str) -> int:
    if mode == "cp":
        return MAX_SEQ_LENGTH_CP
    if mode == "ep":
        return MAX_SEQ_LENGTH_EP
    return MAX_SEQ_LENGTH


def verify_checkpoint(checkpoint_dir: str, mode: str, world_size: int) -> tuple[bool, str]:
    """Verify checkpoint files exist."""
    checks = {}
    lines = [f"  Checkpoint: {checkpoint_dir}", f"  Mode: {mode}"]

    if not os.path.isdir(checkpoint_dir):
        return False, f"  Checkpoint directory not found: {checkpoint_dir}"

    files = set(os.listdir(checkpoint_dir))
    lines.append(f"  Files: {sorted(files)}")

    checks["trainer_state.json"] = "trainer_state.json" in files

    has_safetensors = any(f.endswith(".safetensors") for f in files)
    has_pytorch = any(f.startswith("pytorch_model") and f.endswith(".bin") for f in files)
    checks["model_weights"] = has_safetensors or has_pytorch

    # Every sharded mode persists per-rank optimizer shards plus the fingerprint meta.
    for rank_idx in range(world_size):
        shard_name = f"optimizer_shard_{rank_idx:05d}.pt"
        checks[shard_name] = shard_name in files
    checks["optimizer_meta.pt"] = "optimizer_meta.pt" in files
    checks["scheduler.pt"] = "scheduler.pt" in files

    for name, ok in checks.items():
        lines.append(f"    {'OK' if ok else 'MISSING':7s}  {name}")

    failed = [k for k, v in checks.items() if not v]
    if failed:
        lines.append(f"  MISSING: {failed}")

    return all(checks.values()), "\n".join(lines)


def load_model_for_mode(mode: str, parallelism_config: ParallelismConfig, model_path: str | None = None):
    """Load model appropriate for the parallelism mode.

    ``model_path`` overrides the source weights. For EP/CP resume it must be the
    checkpoint dir: the CheckpointLoader deliberately skips base-weight reload for
    EP/CP (loader.py docstring — "saved HF-format weights are reloaded by
    load_distributed_model, not here"), so the trained weights are restored ONLY by
    loading the model from the checkpoint here. fsdp/tp instead reload weights via
    the loader's set_model_state_dict path, so they load from base.
    """
    model_name = model_path or model_for_mode(mode)
    if mode == "fsdp":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return model, tokenizer
    else:
        # CP/TP/EP need load_distributed_model to apply Ulysses, DTensor or DeepEP patching.
        return load_distributed_model(
            model_name_or_path=model_name,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            use_liger_kernel=True,
        )


def phase1_train_and_save(
    rank: int,
    world_size: int,
    mode: str,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
    parallelism_config: ParallelismConfig,
) -> tuple[bool, list[float], float]:
    """Train for SAVE_AT_STEP steps, verify checkpoint on rank 0.

    Returns (ok, step_losses, L_pre) — L_pre is the fixed-batch forward loss with the
    trained (== saved) weights, the reference for the Phase 2 weight-continuity check.
    """
    log(f"\n  Phase 1: Train {SAVE_AT_STEP} steps + save checkpoint ({mode})")

    model = None
    trainer = None
    try:
        log("  Loading model...")
        model, tok = load_model_for_mode(mode, parallelism_config)

        max_len = max_seq_length_for_mode(mode)
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
            max_length=max_len,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
        )

        trainer = DistributedSFTTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )

        log(f"  Training for {SAVE_AT_STEP} steps...")
        train_result = trainer.train()
        training_loss = train_result.training_loss
        step_losses = [e["loss"] for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]
        log(f"  Training loss: {training_loss:.6f}")
        log(f"  Step losses:   {[f'{l:.4f}' for l in step_losses]}")

        ids, labels = _fixed_batch(tokenizer, torch.cuda.current_device())
        l_pre = _forward_loss(trainer, ids, labels)
        log(f"  L_pre (fixed-batch forward loss, trained weights): {l_pre:.6f}")

        if not math.isfinite(training_loss):
            log(f"  ERROR: Loss not finite: {training_loss}")
            return False, step_losses, l_pre

        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{SAVE_AT_STEP}")
        barrier()

        if rank == 0:
            ckpt_ok, ckpt_detail = verify_checkpoint(checkpoint_dir, mode, world_size)
            log("\n  Checkpoint verification:")
            log(ckpt_detail)
        else:
            ckpt_ok = True

        result_t = torch.tensor(
            [1 if ckpt_ok else 0],
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        dist.broadcast(result_t, src=0)
        if result_t.item() == 0:
            return False, step_losses, l_pre

        return True, step_losses, l_pre

    finally:
        del trainer, model
        cleanup_memory()
        barrier()


def phase2_resume_and_train(
    mode: str,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
    parallelism_config: ParallelismConfig,
    l_pre: float,
) -> tuple[bool, list[float]]:
    """Resume from checkpoint and train to TOTAL_STEPS.

    All modes: assert weights restored by value (|L_post - L_pre| < LOSS_TOL) at
    on_train_begin (post-resume, pre-step), and assert the optimizer 2nd moment was
    restored (materialized, nonzero, finite) — full optimizer continuity from the
    per-rank shards, uniform across fsdp/tp/cp/ep.
    """
    log(f"\n  Phase 2: Resume from checkpoint-{SAVE_AT_STEP} -> step {TOTAL_STEPS} ({mode})")

    checkpoint_path = os.path.join(output_dir, f"checkpoint-{SAVE_AT_STEP}")
    model = None
    trainer = None
    try:
        log("  Loading fresh model for resume...")
        # Through the production helper, not a hardcoded path: EP/CP must get the checkpoint dir
        # (the loader skips their base-weight reload), fsdp/tp base. Returning base for EP/CP loads
        # untrained weights silently, which is what the by-value check below catches.
        model_cfg = SimpleNamespace(model_name_or_path=model_for_mode(mode))
        resume_model_path = resolve_resume_weights_source(checkpoint_path, model_cfg, parallelism_config)
        model, tok = load_model_for_mode(mode, parallelism_config, model_path=resume_model_path)

        max_len = max_seq_length_for_mode(mode)
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
            max_length=max_len,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
        )

        trainer = DistributedSFTTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )

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

        cap = trainer_ref["capture"]
        if cap is None:
            log("  ERROR: resume-capture callback did not fire (on_train_begin missed)")
            return False, step_losses

        l_post = cap["l_post"]
        loss_delta = abs(l_post - l_pre)
        # EP's forward is non-deterministic (DeepEP all-to-all reordering), so it gets the looser
        # band — still far under the ~2+ gap an unrestored (untrained) weight set produces.
        weight_tol = 0.5 if mode == "ep" else LOSS_TOL
        weights_ok = math.isfinite(l_post) and loss_delta < weight_tol
        log(f"  L_pre={l_pre:.6f}  L_post={l_post:.6f}  |delta|={loss_delta:.6f} (tol {weight_tol})")
        if not weights_ok:
            log(f"  ERROR: weights not restored — |L_post - L_pre|={loss_delta:.6f} >= {weight_tol}")

        optim_ok = cap["moments_materialized"] and cap["moments_nonzero"] and cap["moments_finite"]
        log(
            f"  optimizer exp_avg_sq materialized={cap['moments_materialized']} "
            f"nonzero={cap['moments_nonzero']} finite={cap['moments_finite']}"
        )
        if not optim_ok:
            log(
                "  ERROR: optimizer 2nd moment not restored (expected full continuity for "
                f"{mode}); Adam state zero/non-finite/absent after resume"
            )

        ok = loss_ok and steps_ok and all_finite and weights_ok and optim_ok
        return ok, step_losses

    finally:
        del trainer, model
        cleanup_memory()
        barrier()


def run_mode(mode: str, rank: int, world_size: int) -> bool:
    """Run checkpoint save + resume test for a single mode."""
    log(f"\n{'=' * 60}")
    log(f"  Mode: {mode.upper()}")
    log(f"{'=' * 60}")

    # Rank 0 writes the checkpoint every rank reads back, so the dir must be identical world-wide.
    output_dir = shared_scratch_dir(f"sft_resume_{mode}")

    # ZeRO-3 on the DP axis, default ZeRO-2. Only fsdp/cp honour it: ParallelismConfig rejects
    # TP+DP+FULL_SHARD (no DTensor all-gather strategy for the backward re-gather), and EP too.
    reshard = env_flag("HALO_TEST_RESUME_FSDP_RESHARD")
    if mode == "fsdp":
        parallelism_config = ParallelismConfig(fsdp_reshard_after_forward=reshard)
    elif mode == "cp":
        parallelism_config = ParallelismConfig(cp_size=2, fsdp_reshard_after_forward=reshard)
    elif mode == "tp":
        parallelism_config = ParallelismConfig(tp_size=2)  # TP+DP+FULL_SHARD config-rejected → ZeRO-2 only
    elif mode == "ep":
        parallelism_config = ParallelismConfig(ep_size=2)
    else:
        log(f"  Unknown mode: {mode}")
        return False

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_for_mode(mode), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
        eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1)

        phase1_ok, phase1_losses, l_pre = phase1_train_and_save(
            rank,
            world_size,
            mode,
            tokenizer,
            train_dataset,
            eval_dataset,
            output_dir,
            parallelism_config,
        )
        if not phase1_ok:
            log(f"  Phase 1 FAILED for {mode} — skipping Phase 2")
            return False

        phase2_ok, phase2_losses = phase2_resume_and_train(
            mode,
            tokenizer,
            train_dataset,
            eval_dataset,
            output_dir,
            parallelism_config,
            l_pre,
        )

        if phase1_ok and phase2_ok:
            log(f"  PASSED: {mode.upper()}")
            log(f"    Phase 1 losses: {[f'{l:.4f}' for l in phase1_losses]}")
            log(f"    Phase 2 losses: {[f'{l:.4f}' for l in phase2_losses]}")
        else:
            log(f"  FAILED: {mode.upper()}")

        return phase1_ok and phase2_ok

    finally:
        cleanup_memory()
        cleanup_dirs(output_dir)
        barrier()


def run(ctx) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["fsdp", "cp", "tp", "ep", "all"])
    args, _ = parser.parse_known_args()

    modes = ["fsdp", "cp", "tp", "ep"] if args.mode == "all" else [args.mode]

    log(f"\n{'#' * 70}")
    log("  SFT Checkpoint Save + Resume Test")
    log(f"  Default model: {DEFAULT_MODEL} (EP mode uses {EP_MODEL})")
    log(f"  World size: {ctx.world_size}, GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  Modes: {modes}")
    log(f"  Plan: Train {SAVE_AT_STEP} steps -> Save -> Resume -> Train to {TOTAL_STEPS}")
    log(f"{'#' * 70}")

    # One check per mode — exactly the per-mode verdict the script summarised.
    checks: dict[str, bool] = {}
    for mode in modes:
        checks[mode] = run_mode(mode, ctx.rank, ctx.world_size)

    log(f"\n{'#' * 70}")
    log("  Results:")
    for mode, passed in checks.items():
        log(f"    {mode.upper():6s}: {'PASS' if passed else 'FAIL'}")
    log(f"{'#' * 70}")

    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="sft_resume")(run)

if __name__ == "__main__":
    main()
