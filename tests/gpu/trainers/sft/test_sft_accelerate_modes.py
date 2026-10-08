#!/usr/bin/env python
"""
SFT Trainer test for accelerate launch configurations and torchrun standard DP.

Smoke test of DistributedSFTTrainer under different launch methods and data parallel strategies:

1. torchrun (FSDP2 SHARD_GRAD_OP): standard DP via mixin, safe checkpoints
2. accelerate MULTI_GPU (DDP): accelerate manages DDP wrapping
3. accelerate FSDP2 SHARD_GRAD_OP: accelerate manages FSDP v2 (fully_shard)
4. accelerate FSDP2 FULL_SHARD: accelerate manages FSDP v2 with resharding

The test auto-detects its launch method. It checks that training runs every step with finite train
and eval losses and a lower last-step loss than first, and that ``save_model`` writes ``config.json``
and weight files (rank 0). No loss is compared against a reference and the saved checkpoint is not
reloaded.

Run with torchrun (tests the mixin's FSDP2 SHARD_GRAD_OP path):
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_accelerate_modes.py

Run with accelerate DDP (tests MULTI_GPU path):
    accelerate launch --config_file launcher-configs/accelerate/multigpu_dp_config.yaml \
        --num_processes=2 tests/gpu/trainers/sft/test_sft_accelerate_modes.py

Run with accelerate FSDP2 SHARD_GRAD_OP:
    accelerate launch --config_file launcher-configs/accelerate/fsdp2_gradop_config.yaml \
        --num_processes=2 tests/gpu/trainers/sft/test_sft_accelerate_modes.py

Run with accelerate FSDP2 FULL_SHARD:
    accelerate launch --config_file launcher-configs/accelerate/fsdp2_full_config.yaml \
        --num_processes=2 tests/gpu/trainers/sft/test_sft_accelerate_modes.py
"""

import math
import os

import torch
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.env import is_accelerate_fsdp_launch, is_accelerate_launch
from src.trainers.sft import DistributedSFTTrainer
from tests.common.checkpoint_io import model_save_checks
from tests.common.datasets import create_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.utils import log, training_run_checks

MODEL_NAME = QWEN3_0_6B
MAX_STEPS = 5
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 512
LEARNING_RATE = 2e-5
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
SEED = 42


def detect_launch_mode() -> str:
    """Detect how this script was launched."""
    if is_accelerate_fsdp_launch():
        # accelerate's own var, parsed the way accelerate writes it — not a toolkit knob.
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
    """Run SFT test under current launch configuration."""
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
        # With drop_last, a per-device eval batch past NUM_EVAL_SAMPLES / world size yields no batch,
        # and the eval leg then logs no loss at all.
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,  # already applied at load
        logging_steps=1,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=3,
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        fsdp="",  # Mixin handles wrapping
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
