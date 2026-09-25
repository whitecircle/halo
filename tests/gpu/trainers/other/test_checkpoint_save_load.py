#!/usr/bin/env python
"""
Checkpoint Save Test across Parallelism Modes.

Validates that DistributedSFTTrainer can train and save model checkpoints
correctly under different parallelism configurations. For each mode, the
test trains for 3 steps, saves the model via trainer.save_model(), and
verifies that the expected checkpoint files exist on disk.

Modes tested (sequentially):
1. FSDP mode: ParallelismConfig() -- standard data parallelism
2. CP=2 mode: ParallelismConfig(cp_size=2) -- Context Parallelism
3. TP=2 mode: ParallelismConfig(tp_size=2) -- Tensor Parallelism

Model: Qwen/Qwen3-0.6B (dense, supports tp_plan="auto")

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/other/test_checkpoint_save_load.py
"""

import math
import os
import traceback

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.utils import cleanup_memory, log

MODEL_NAME = QWEN3_0_6B
MAX_STEPS = 3
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 512
LEARNING_RATE = 2e-5
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
SEED = 42


def verify_checkpoint(save_dir: str, mode_name: str) -> tuple[bool, str]:
    """Verify that a saved checkpoint has the expected files.

    Checks for config.json and model weights (safetensors or pytorch format).

    Returns:
        (success: bool, detail_message: str)
    """
    if not os.path.exists(save_dir):
        return False, f"Save directory does not exist: {save_dir}"

    files = os.listdir(save_dir)

    has_config = os.path.exists(os.path.join(save_dir, "config.json"))
    has_safetensors = os.path.exists(os.path.join(save_dir, "model.safetensors")) or any(
        f.startswith("model") and f.endswith(".safetensors") for f in files
    )
    has_pytorch = os.path.exists(os.path.join(save_dir, "pytorch_model.bin")) or any(
        f.startswith("pytorch_model") and f.endswith(".bin") for f in files
    )
    has_model = has_safetensors or has_pytorch

    details = []
    details.append(f"  Files found: {sorted(files)}")
    details.append(f"  config.json: {'YES' if has_config else 'NO'}")
    details.append(f"  Model weights: {'YES' if has_model else 'NO'}")

    if has_config and has_model:
        return True, "\n".join(details)
    else:
        missing = []
        if not has_config:
            missing.append("config.json")
        if not has_model:
            missing.append("model weights (safetensors or pytorch)")
        return False, "\n".join(details) + f"\n  MISSING: {missing}"


def run_mode_test(
    mode_name: str,
    parallelism_config: ParallelismConfig,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    base_output_dir: str,
    rank: int,
    local_rank: int,
) -> tuple[bool, str]:
    """Run a training + checkpoint save test for a given parallelism mode.

    Args:
        mode_name: Human-readable name (e.g., "FSDP", "CP=2", "TP=2")
        parallelism_config: ParallelismConfig for this mode
        tokenizer: Pretrained tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        base_output_dir: Base directory for checkpoints
        rank: Global rank
        local_rank: Local rank

    Returns:
        (success: bool, detail_message: str)
    """
    save_dir = os.path.join(base_output_dir, mode_name.replace("=", "").replace(" ", "_").lower())
    log(f"\n  --- Mode: {mode_name} ---")
    log(f"  Config: {parallelism_config.mode_string or 'standard'}")
    log(f"  Save dir: {save_dir}")

    model = None
    trainer = None

    try:
        log("  Loading model...")
        if parallelism_config.is_tp_mode or parallelism_config.is_cp_mode:
            model, _ = load_distributed_model(
                model_name_or_path=MODEL_NAME,
                parallelism_config=parallelism_config,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )

        mem_gb = torch.cuda.memory_allocated() / 1e9
        log(f"  Model loaded, GPU memory: {mem_gb:.2f} GB")

        config = SFTConfig(
            output_dir=os.path.join(base_output_dir, f"trainer_{mode_name.lower()}"),
            max_steps=MAX_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            dataloader_drop_last=True,
        )

        trainer = DistributedSFTTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )

        log(f"  Training for {MAX_STEPS} steps...")
        train_result = trainer.train()
        training_loss = train_result.training_loss
        loss_finite = math.isfinite(training_loss)
        log(f"  Training loss: {training_loss:.6f} (finite={loss_finite})")

        if not loss_finite:
            return False, f"Training loss is not finite: {training_loss}"

        log("  Saving checkpoint...")
        trainer.save_model(save_dir)
        dist.barrier()

        if rank == 0:
            ckpt_ok, ckpt_detail = verify_checkpoint(save_dir, mode_name)
            log("  Checkpoint verification:")
            log(ckpt_detail)
        else:
            ckpt_ok = True
            ckpt_detail = "(verified on rank 0)"

        # every rank must return the same verdict or they diverge on the next mode
        result_tensor = torch.tensor([1 if ckpt_ok else 0], dtype=torch.int64, device=f"cuda:{local_rank}")
        dist.broadcast(result_tensor, src=0)
        ckpt_ok = result_tensor.item() == 1

        success = loss_finite and ckpt_ok
        status = "PASS" if success else "FAIL"
        detail = f"loss={training_loss:.6f}, checkpoint={'valid' if ckpt_ok else 'INVALID'}"
        log(f"  Result: {status} ({detail})")
        return success, detail

    except Exception as e:
        log(f"  FAILED with exception: {e}")
        traceback.print_exc()
        return False, f"Exception: {e}"

    finally:
        # free GPU memory before the next mode loads its own model
        del trainer
        del model
        cleanup_memory()
        dist.barrier()


@gpu_test_main(min_world_size=2, prefix="test_ckpt_save_load")
def run(ctx) -> dict:
    rank, local_rank, base_output_dir = ctx.rank, ctx.local_rank, ctx.output_dir

    log(f"\n{'#' * 70}")
    log("  Checkpoint Save Test")
    log(f"  World size: {ctx.world_size}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    log("  Modes: FSDP, CP=2, TP=2")
    log(f"  Steps per mode: {MAX_STEPS}")
    log(f"{'#' * 70}")

    results: dict[str, tuple[bool, str]] = {}

    log("\n[Setup] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"  Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    log("[Setup] Creating synthetic datasets...")
    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1)
    log(f"  Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    log(f"\n{'=' * 60}")
    log("  MODE 1: FSDP (Standard Data Parallelism)")
    log(f"{'=' * 60}")

    results["FSDP"] = run_mode_test(
        mode_name="FSDP",
        parallelism_config=ParallelismConfig(),
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        base_output_dir=base_output_dir,
        rank=rank,
        local_rank=local_rank,
    )

    log(f"\n{'=' * 60}")
    log("  MODE 2: CP=2 (Context Parallelism)")
    log(f"{'=' * 60}")

    results["CP=2"] = run_mode_test(
        mode_name="CP2",
        parallelism_config=ParallelismConfig(cp_size=2),
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        base_output_dir=base_output_dir,
        rank=rank,
        local_rank=local_rank,
    )

    log(f"\n{'=' * 60}")
    log("  MODE 3: TP=2 (Tensor Parallelism)")
    log(f"{'=' * 60}")

    results["TP=2"] = run_mode_test(
        mode_name="TP2",
        parallelism_config=ParallelismConfig(tp_size=2),
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        base_output_dir=base_output_dir,
        rank=rank,
        local_rank=local_rank,
    )

    for mode_name, (passed, detail) in results.items():
        log(f"  {mode_name:>8s}: {'PASS' if passed else 'FAIL'} -- {detail}")

    return {"checks": {mode_name: passed for mode_name, (passed, _) in results.items()}}


if __name__ == "__main__":
    run()
