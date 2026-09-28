#!/usr/bin/env python
"""
SMPO with Context Parallelism (CP=2): the first step is the unsplit sequence's step.

Each CP rank holds one chunk of every sequence and all-reduces its partial log-prob and NLL sums, so
every rank computes the whole-sequence loss. Those reduces have to be autograd-aware: their backward
sums the gradient over the CP group, which FSDP2's world-wide average divides back. An in-place
``dist.all_reduce`` reaches autograd only through PyTorch's c10d fallback (identity backward, a
warning), and a loss multiplied by ``cp_size`` to make up the gradient logs ``cp_size`` times the true
loss. Both train to finite losses, so the run is pinned by value: step 1's microbatches, as
``compute_loss`` saw them, are scored again by a CP=1 SmoothMarginPOTrainer on the initial weights, and

  1. each microbatch loss and the logged step-1 loss equal the reference's;
  2. the step-1 gradient the optimizer is handed matches the reference's in direction and norm
     (clipping off, so it is the backward's own gradient);
  3. no backward reached the c10d autograd fallback;

besides the smoke checks: CP mode active, every step run, the training loss finite.
tests/cpu/trainers/test_smpo_cp_gradient.py pins the same objective exactly, in float64 on gloo.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/preference/test_smpo_cp.py

Requirements:
    - 2x GPUs
    - Model: Qwen/Qwen3-0.6B (auto-downloaded)
"""

import math

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from src.configs.smpo_config import SmoothMarginPOConfig
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.preference.smpo import SmoothMarginPOTrainer
from tests.common.datasets import create_preference_dataset
from tests.common.first_step import (
    FirstStep,
    first_step_checks,
    first_step_gradient_checks,
    score_first_step,
    train_recording_first_step,
)
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.utils import cleanup_memory, gpu_mem_gb, log

# Configuration

MODEL_NAME = QWEN3_0_6B
CP_SIZE = 2
NUM_TRAIN_STEPS = 10
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 5e-6
MAX_LENGTH = 4096
MAX_PROMPT_LENGTH = 2048
NUM_TRAIN_SAMPLES = 128
NUM_EVAL_SAMPLES = 16
SEED = 42
# Step 1 against the CP=1 trainer, relative. The squared-hinge margin term amplifies bf16 log-prob noise:
# measured over four seeds at 2 to 16 microbatches, a microbatch loss is at most 3.1% off and the logged
# step loss 2.0%. A loss counted once per CP rank is 100% off.
LOSS_RTOL = 0.1


def _smpo_config(output_dir: str) -> SmoothMarginPOConfig:
    return SmoothMarginPOConfig(
        output_dir=output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        # Off, so the recorded step-1 gradient is the backward's own; HF still logs its norm.
        max_grad_norm=0.0,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,  # Already applied in load_distributed_model
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        dataloader_drop_last=True,
        fsdp="",  # Mixin handles FSDP wrapping
    )


def _load_model(parallelism_config: ParallelismConfig, attn_implementation: str | None = None):
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        use_liger_kernel=True,
    )
    return model


def _train_cp(output_dir: str, tokenizer) -> tuple[dict[str, bool], FirstStep, str]:
    """Train under CP; return the smoke checks, the recorded first step and the attention kernel."""
    train_dataset = create_preference_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_preference_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1000)
    log(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    parallelism_config = ParallelismConfig(cp_size=CP_SIZE)
    log(f"Parallelism config: {parallelism_config.summary()}")
    model = _load_model(parallelism_config)
    # The flash kernel CP resolved, so the reference scores with the same one.
    attn_implementation = model.config._attn_implementation
    log(f"Model loaded. Type: {type(model).__name__}, attention: {attn_implementation}")

    trainer = SmoothMarginPOTrainer(
        model=model,
        args=_smpo_config(output_dir),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    checks = {"cp_mode_active": bool(trainer.is_cp_mode)}
    log(f"trainer.is_cp_mode: {trainer.is_cp_mode}")

    log(f"\nStarting training for {NUM_TRAIN_STEPS} steps...")
    train_result, first_step = train_recording_first_step(trainer)
    log("Training complete!")

    final_loss = train_result.metrics["train_loss"]
    log(f"Final training loss: {final_loss}")
    checks["train_loss_finite"] = math.isfinite(final_loss)
    checks["steps_completed"] = trainer.state.global_step == NUM_TRAIN_STEPS
    log(f"Steps completed: {trainer.state.global_step}")

    del trainer, model
    cleanup_memory()
    return checks, first_step, attn_implementation


def _score_reference(output_dir: str, tokenizer, first_step: FirstStep, attn_implementation: str) -> FirstStep:
    """Score the recorded first step with a CP=1 trainer on the initial weights."""
    parallelism_config = ParallelismConfig()
    trainer = SmoothMarginPOTrainer(
        model=_load_model(parallelism_config, attn_implementation),
        args=_smpo_config(output_dir),
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    return score_first_step(trainer, first_step)


# Main Test


def run(ctx) -> dict:
    """Run SMPO + CP=2 training test."""
    log(f"\n{'#' * 70}")
    log(f"  SMPO + CP={CP_SIZE} Training Test")
    log(f"  World size: {ctx.world_size}, CP size: {CP_SIZE}")
    log(f"  Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"{'#' * 70}")

    # ---- Output Directory (broadcast from rank 0) ----
    output_dir = ctx.output_dir
    if ctx.world_size > 1:
        output_list = [output_dir]
        dist.broadcast_object_list(output_list, src=0)
        output_dir = output_list[0]
    log(f"Output dir: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    checks, first_step, attn_implementation = _train_cp(output_dir, tokenizer)
    log(f"\nScoring step 1 with a CP=1 trainer (GPU memory after freeing the CP run: {gpu_mem_gb():.1f}GB)")
    reference = _score_reference(output_dir, tokenizer, first_step, attn_implementation)
    loss_checks, metrics = first_step_checks(first_step, reference, miscount_factor=CP_SIZE, loss_rtol=LOSS_RTOL)
    grad_checks, grad_metrics = first_step_gradient_checks(first_step, reference)
    return {"checks": checks | loss_checks | grad_checks, "metrics": metrics | grad_metrics}


main = gpu_test_main(min_world_size=CP_SIZE, prefix="smpo_cp")(run)

if __name__ == "__main__":
    main()
