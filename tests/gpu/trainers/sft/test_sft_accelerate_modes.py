#!/usr/bin/env python
"""
SFT Trainer test for accelerate launch configurations and torchrun standard DP.

Validates DistributedSFTTrainer trains correctly and saves checkpoints under
different launch methods and data parallel strategies:

1. torchrun (FSDP NO_SHARD): standard DP via mixin, safe checkpoints
2. accelerate MULTI_GPU (DDP): accelerate manages DDP wrapping
3. accelerate FSDP2 SHARD_GRAD_OP: accelerate manages FSDP v2 (fully_shard)
4. accelerate FSDP2 FULL_SHARD: accelerate manages FSDP v2 with resharding

The test auto-detects its launch method and validates accordingly.

Run with torchrun (tests FSDP NO_SHARD path):
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
import sys
import traceback

import torch
import torch.distributed as dist
from accelerate import PartialState
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.env import is_accelerate_fsdp_launch, is_accelerate_launch
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import cleanup_dirs, setup_cache_dirs
from tests.common.models import QWEN3_0_6B
from tests.common.utils import cleanup_memory, log

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
        return "torchrun_fsdp_no_shard"


def verify_checkpoint(save_dir: str, rank: int) -> dict:
    """Verify checkpoint files on rank 0."""
    if rank != 0:
        return {}

    checks = {}
    dir_exists = os.path.isdir(save_dir)
    checks["save_dir_exists"] = dir_exists
    log(f"  Save dir exists: {'PASS' if dir_exists else 'FAIL'} ({save_dir})")

    if not dir_exists:
        return checks

    contents = os.listdir(save_dir)
    log(f"  Contents: {sorted(contents)}")

    has_config = "config.json" in contents
    checks["has_config"] = has_config
    log(f"  config.json: {'PASS' if has_config else 'FAIL'}")

    has_model = any(f.startswith("model") and f.endswith(".safetensors") for f in contents) or any(
        f.startswith("pytorch_model") and f.endswith(".bin") for f in contents
    )
    checks["has_model_weights"] = has_model
    log(f"  Model weights: {'PASS' if has_model else 'FAIL'}")

    return checks


def main() -> int:
    """Run SFT test under current launch configuration."""
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    state = PartialState()
    rank = state.process_index
    local_rank = state.local_process_index
    world_size = state.num_processes

    launch_mode = detect_launch_mode()

    base_output_dir, cache_dir = setup_cache_dirs("test_sft_accel", rank)
    save_dir = os.path.join(base_output_dir, "saved_model")

    log(f"\n{'=' * 70}")
    log("  SFT Accelerate/Torchrun Test")
    log(f"  Launch mode: {launch_mode}")
    log(f"  World size: {world_size}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"  is_fsdp_enabled env: {os.environ.get('ACCELERATE_USE_FSDP', 'not set')}")
    log(f"  mixed_precision env: {os.environ.get('ACCELERATE_MIXED_PRECISION', 'not set')}")
    log(f"{'=' * 70}")

    success = False

    try:
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
            output_dir=base_output_dir,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,  # already applied at load
            logging_steps=1,
            save_strategy="no",
            eval_strategy="steps",
            eval_steps=3,
            report_to="none",
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

        training_loss = train_result.training_loss
        log_history = trainer.state.log_history
        step_losses = [e["loss"] for e in log_history if "loss" in e and "eval_loss" not in e]
        eval_entries = [e for e in log_history if "eval_loss" in e]

        log(f"\n  Training loss: {training_loss:.6f}")
        log(f"  Step losses: {[f'{l:.4f}' for l in step_losses]}")
        for e in eval_entries:
            log(f"  Eval loss (step {e.get('step', '?')}): {e['eval_loss']:.6f}")

        log("\n[5/5] Saving model...")
        trainer.save_model(save_dir)
        if dist.is_initialized():
            dist.barrier()

        log("\n  --- Assertions ---")
        checks = {}

        loss_finite = math.isfinite(training_loss)
        checks["loss_finite"] = loss_finite
        log(f"  Loss finite: {'PASS' if loss_finite else 'FAIL'} ({training_loss:.6f})")

        all_finite = all(math.isfinite(l) for l in step_losses)
        checks["all_steps_finite"] = all_finite
        log(f"  All steps finite: {'PASS' if all_finite else 'FAIL'}")

        if len(step_losses) >= 2:
            decreased = step_losses[-1] < step_losses[0]
            checks["loss_decreased"] = decreased
            log(f"  Loss decreased: {'PASS' if decreased else 'FAIL'} ({step_losses[0]:.4f} -> {step_losses[-1]:.4f})")

        if eval_entries:
            eval_finite = all(math.isfinite(e["eval_loss"]) for e in eval_entries)
            checks["eval_finite"] = eval_finite
            log(f"  Eval finite: {'PASS' if eval_finite else 'FAIL'}")

        steps_ok = train_result.global_step == MAX_STEPS
        checks["steps_completed"] = steps_ok
        log(f"  Steps completed: {'PASS' if steps_ok else 'FAIL'} ({train_result.global_step}/{MAX_STEPS})")

        ckpt_checks = verify_checkpoint(save_dir, rank)
        checks.update(ckpt_checks)

        all_passed = all(checks.values())

        log(f"\n{'=' * 70}")
        if all_passed:
            log(f"  SFT TEST PASSED ({launch_mode})")
        else:
            failed = [k for k, v in checks.items() if not v]
            log(f"  SFT TEST FAILED ({launch_mode}): {failed}")
        log(f"{'=' * 70}\n")

        success = all_passed

    except Exception as e:
        log(f"\n  TEST FAILED with exception: {e}")
        traceback.print_exc()
        success = False

    finally:
        cleanup_memory()
        cleanup_dirs(base_output_dir, cache_dir)
        if dist.is_initialized():
            dist.destroy_process_group()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
