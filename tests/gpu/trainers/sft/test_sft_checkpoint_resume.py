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
    • every step finite, and the checkpoint holds the trainer state, the scheduler, the weights, one
      optimizer shard and RNG state per rank and the fingerprint meta
  Phase 2 — Resume from checkpoint, continue to TOTAL_STEPS:
    • global_step == TOTAL_STEPS after resume, every resumed step loss finite
    • at the first resumed step: the trained weights (fixed-batch loss), Adam's moments and the LR
      scheduler restored

All sharded modes (fsdp/tp/cp/ep) persist per-rank optimizer shards
(optimizer_shard_XXXXX.pt + optimizer_meta.pt) and restore full optimizer
continuity on resume, gated by the topology fingerprint. EP/CP additionally save
gathered HF-format weights and reload the model from scratch via
load_distributed_model() (re-applying EP/CP transformations); their optimizer
moments then restore from the shards keyed by param FQN.

Run:
    torchrun --nproc_per_node=2 \\
        tests/gpu/trainers/sft/test_sft_checkpoint_resume.py --mode all
"""

import argparse
import os
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.env import env_flag, env_str
from src.trainers.sft import DistributedSFTTrainer
from src.training.environment import resolve_resume_weights_source
from tests.common.checkpoint_io import (
    ResumeCapture,
    fixed_batch_loss,
    resume_checkpoint_checks,
    resume_continuity_checks,
)
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import cleanup_dirs, shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B, QWEN3_0_6B
from tests.common.tolerances import TOL
from tests.common.utils import cleanup_memory, log, step_losses, training_run_checks

# Overridable so the same harness can validate resume across model families.
DEFAULT_MODEL = env_str("HALO_TEST_RESUME_MODEL", QWEN3_0_6B)
EP_MODEL = env_str("HALO_TEST_RESUME_EP_MODEL", GPT_OSS_20B)  # EP requires an MoE model
TOTAL_STEPS = 6
SAVE_AT_STEP = 3
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
SEED = 42
# ZeRO-3 on the DP axis, default ZeRO-2. Only fsdp/cp honour it: ParallelismConfig rejects
# TP+DP+FULL_SHARD (no DTensor all-gather strategy for the backward re-gather), and EP too.
RESHARD = env_flag("HALO_TEST_RESUME_FSDP_RESHARD")


@dataclass(frozen=True)
class ResumeMode:
    """One mode's ParallelismConfig kwargs and training sequence cap (CP needs one divisible by
    cp_size)."""

    parallelism: dict
    max_length: int
    model: str = DEFAULT_MODEL


MODES = {
    "fsdp": ResumeMode({"fsdp_reshard_after_forward": RESHARD}, 512),
    "cp": ResumeMode({"cp_size": 2, "fsdp_reshard_after_forward": RESHARD}, 4096),
    "tp": ResumeMode({"tp_size": 2}, 512),
    "ep": ResumeMode({"ep_size": 2}, 2048, model=EP_MODEL),
}


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


def load_model_for_mode(mode: str, parallelism_config: ParallelismConfig, model_path: str):
    """Load the model for ``mode`` from ``model_path``.

    For EP/CP resume ``model_path`` must be the checkpoint dir: the CheckpointLoader deliberately skips
    base-weight reload for EP/CP (its docstring: their "weights are reloaded by
    ``load_distributed_model`` (not here)"), so the trained weights are restored ONLY by loading the model
    from the checkpoint here. fsdp/tp instead reload weights via the loader's set_model_state_dict
    path, so they load from base.
    """
    if mode == "fsdp":
        return AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa"
        )
    # CP/TP/EP need load_distributed_model to apply Ulysses, DTensor or DeepEP patching.
    model, _ = load_distributed_model(
        model_name_or_path=model_path,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        use_liger_kernel=True,
    )
    return model


def _sft_config(output_dir: str, max_steps: int, max_length: int, **save_args) -> SFTConfig:
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
        max_length=max_length,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        **save_args,
    )


def phase1_train_and_save(
    ctx, mode: str, tokenizer, datasets, output_dir: str, parallelism_config: ParallelismConfig
) -> tuple[dict[str, bool], list[float], float]:
    """Train for SAVE_AT_STEP steps and check the checkpoint's files (rank 0 reads, every rank agrees).

    Returns (checks, losses, L_pre) — L_pre is the fixed-batch forward loss with the
    trained (== saved) weights, the reference for the Phase 2 weight-continuity check.
    """
    log(f"\n  Phase 1: Train {SAVE_AT_STEP} steps + save checkpoint ({mode})")
    model = load_model_for_mode(mode, parallelism_config, MODES[mode].model)
    trainer = DistributedSFTTrainer(
        model=model,
        args=_sft_config(
            output_dir,
            SAVE_AT_STEP,
            MODES[mode].max_length,
            save_strategy="steps",
            save_steps=SAVE_AT_STEP,
            save_total_limit=1,
        ),
        train_dataset=datasets[0],
        eval_dataset=datasets[1],
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    train_result = trainer.train()
    losses = step_losses(trainer)
    checks = training_run_checks(train_result, trainer, SAVE_AT_STEP)

    ids, labels = _fixed_batch(tokenizer, torch.cuda.current_device())
    l_pre = fixed_batch_loss(trainer.model, ids, labels)
    log(f"  L_pre (fixed-batch forward loss, trained weights): {l_pre:.6f}")

    barrier()
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{SAVE_AT_STEP}")
    checks |= ctx.broadcast_checks(resume_checkpoint_checks(checkpoint_dir, ctx.world_size) if ctx.rank == 0 else {})

    del trainer, model
    cleanup_memory()
    barrier()
    return checks, losses, l_pre


def phase2_resume_and_train(
    mode: str, tokenizer, datasets, output_dir: str, parallelism_config: ParallelismConfig, l_pre: float
) -> tuple[dict[str, bool], list[float]]:
    """Resume from the checkpoint and train to TOTAL_STEPS, grading the state the resume restored at
    its first step (:func:`~tests.common.checkpoint_io.resume_continuity_checks`)."""
    log(f"\n  Phase 2: Resume from checkpoint-{SAVE_AT_STEP} -> step {TOTAL_STEPS} ({mode})")
    checkpoint_path = os.path.join(output_dir, f"checkpoint-{SAVE_AT_STEP}")
    # Through the production helper, not a hardcoded path: EP/CP must get the checkpoint dir (the
    # loader skips their base-weight reload), fsdp/tp base. Returning base for EP/CP loads untrained
    # weights silently, which is what the by-value check catches.
    model_cfg = SimpleNamespace(model_name_or_path=MODES[mode].model)
    resume_model_path = resolve_resume_weights_source(checkpoint_path, model_cfg, parallelism_config)
    model = load_model_for_mode(mode, parallelism_config, resume_model_path)
    trainer = DistributedSFTTrainer(
        model=model,
        args=_sft_config(output_dir, TOTAL_STEPS, MODES[mode].max_length, save_strategy="no"),
        train_dataset=datasets[0],
        eval_dataset=datasets[1],
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    ids, labels = _fixed_batch(tokenizer, torch.cuda.current_device())
    resume_capture = ResumeCapture(trainer, ids, labels)
    trainer.add_callback(resume_capture)

    log(f"  Resuming from: {checkpoint_path}")
    train_result = trainer.train(resume_from_checkpoint=checkpoint_path)
    losses = step_losses(trainer)
    checks = {f"resume_{name}": ok for name, ok in training_run_checks(train_result, trainer, TOTAL_STEPS).items()}
    checks |= resume_continuity_checks(
        resume_capture.capture, l_pre, save_step=SAVE_AT_STEP, loss_tol=TOL.resume_fixed_batch_loss_abs
    )

    del trainer, model
    cleanup_memory()
    barrier()
    return checks, losses


def run_mode(ctx, mode: str) -> dict[str, bool]:
    """Checkpoint save + resume for one mode; its checks are keyed ``<mode>_<check>``."""
    log(f"\n{'=' * 60}")
    log(f"  Mode: {mode.upper()}")
    log(f"{'=' * 60}")
    parallelism_config = ParallelismConfig(**MODES[mode].parallelism)
    # Rank 0 writes the checkpoint every rank reads back, so the dir must be identical world-wide.
    output_dir = shared_scratch_dir(f"sft_resume_{mode}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODES[mode].model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        datasets = (
            create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED),
            create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1),
        )
        checks, phase1_losses, l_pre = phase1_train_and_save(
            ctx, mode, tokenizer, datasets, output_dir, parallelism_config
        )
        # Rank-uniform: the logged losses are world means and the file checks rank 0's broadcast.
        if all(checks.values()):
            phase2_checks, phase2_losses = phase2_resume_and_train(
                mode, tokenizer, datasets, output_dir, parallelism_config, l_pre
            )
            checks |= phase2_checks
            log(f"    Phase 1 losses: {[f'{l:.4f}' for l in phase1_losses]}")
            log(f"    Phase 2 losses: {[f'{l:.4f}' for l in phase2_losses]}")
        else:
            log(f"  Phase 1 FAILED for {mode} — skipping Phase 2")
        return {f"{mode}_{name}": ok for name, ok in checks.items()}
    finally:
        cleanup_memory()
        cleanup_dirs(output_dir)
        barrier()


def run(ctx) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=[*MODES, "all"])
    mode = parser.parse_args().mode
    modes = list(MODES) if mode == "all" else [mode]

    log(f"\n{'#' * 70}")
    log("  SFT Checkpoint Save + Resume Test")
    log(f"  Default model: {DEFAULT_MODEL} (EP mode uses {EP_MODEL})")
    log(f"  World size: {ctx.world_size}, GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  Modes: {modes}")
    log(f"  Plan: Train {SAVE_AT_STEP} steps -> Save -> Resume -> Train to {TOTAL_STEPS}")
    log(f"{'#' * 70}")

    checks: dict[str, bool] = {}
    for mode in modes:
        checks |= run_mode(ctx, mode)
    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="sft_resume")(run)

if __name__ == "__main__":
    main()
