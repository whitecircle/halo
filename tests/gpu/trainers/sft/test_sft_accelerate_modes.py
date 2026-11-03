#!/usr/bin/env python
"""DistributedSFTTrainer smoke under torchrun FSDP2 and accelerate DDP / FSDP2, auto-detecting its launch. Compares no
loss against a reference."""

import math
import os

import torch
from trl import SFTConfig

from halo.distributed.parallelism_config import ParallelismConfig
from halo.env import is_accelerate_fsdp_launch, is_accelerate_launch
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.checkpoint_io import model_save_checks
from tests.common.datasets import create_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log
from tests.common.training_checks import training_run_checks

MODEL_NAME = QWEN3_0_6B
MAX_STEPS = 5
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 512
LEARNING_RATE = 2e-5
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
SEED = 42


def detect_launch_mode() -> str:
    if is_accelerate_fsdp_launch():
        # accelerate's own var, parsed the way accelerate writes it.
        reshard = os.environ.get("FSDP_RESHARD_AFTER_FORWARD", "false").lower()
        if reshard in ("true", "1"):
            return "accelerate_fsdp2_full_shard"
        return "accelerate_fsdp2_shard_grad_op"
    elif is_accelerate_launch():
        return "accelerate_ddp"
    else:
        return "torchrun_fsdp2_shard_grad_op"


@gpu_test_main(min_world_size=1, prefix="test_sft_accel")
def run(ctx):
    launch_mode = detect_launch_mode()
    save_dir = os.path.join(ctx.output_dir, "saved_model")

    log(f"\n{'=' * 70}")
    log("  SFT Accelerate/Torchrun Test")
    log(f"  Launch mode: {launch_mode}")
    log(f"  World size: {ctx.world_size}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  is_fsdp_enabled env: {os.environ.get('ACCELERATE_USE_FSDP', 'not set')}")
    log(f"  mixed_precision env: {os.environ.get('ACCELERATE_MIXED_PRECISION', 'not set')}")
    log(f"{'=' * 70}")

    log("\n[1/5] Loading model...")
    parallelism_config = ParallelismConfig()

    model, tokenizer = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_liger_kernel=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    param_count = sum(p.numel() for p in model.parameters())
    log(f"  Model: {param_count / 1e6:.1f}M params, GPU: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    log("\n[2/5] Creating datasets...")
    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1)
    log(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    log("\n[3/5] Creating trainer...")
    config = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        # With drop_last, a per-device eval batch past NUM_EVAL_SAMPLES / world size yields no batch and no logged
        # loss.
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,  # Already applied at load.
        logging_steps=1,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=3,
        report_to="none",
        logging_nan_inf_filter=False,
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
        parallelism_config=parallelism_config,
    )

    log(f"  Trainer: {type(trainer).__name__}")
    log(f"  is_fsdp_enabled: {trainer.is_fsdp_enabled}")
    log(f"  _fsdp_wrapped: {trainer._fsdp_wrapped}")
    log(f"  _accelerate_manages_fsdp: {trainer._accelerate_manages_fsdp}")
    log(f"  _accelerate_manages_ddp: {trainer._accelerate_manages_ddp}")

    log("\n[4/5] Training...")
    train_result = trainer.train()
    checks = training_run_checks(train_result, trainer, MAX_STEPS, loss_decreased=True)
    eval_losses = [entry["eval_loss"] for entry in trainer.state.log_history if "eval_loss" in entry]
    checks["eval_finite"] = bool(eval_losses) and all(math.isfinite(loss) for loss in eval_losses)
    log(f"  Eval losses: {[f'{loss:.6f}' for loss in eval_losses]}")

    log("\n[5/5] Saving model...")
    trainer.save_model(save_dir)
    ctx.barrier()
    checks |= model_save_checks(save_dir, ctx.rank)
    return {"checks": checks}


if __name__ == "__main__":
    run()
