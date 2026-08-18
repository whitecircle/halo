#!/usr/bin/env python
"""
Test: LoRA adapters with Context Parallelism (CP=2), plus LoRA+TP rejection,
on a dense model (Qwen3-0.6B).

Two sub-tests:
1. Sub-test A: LoRA + CP=2 (Ulysses attention) — supported: train 5 steps, verify adapter
   weights updated. CP shards the sequence, not the weights, so LoRA stays replicated/correct.
2. Sub-test B: LoRA + TP=2 — must be REJECTED at trainer construction. TP shards the attention
   weights as DTensors; PEFT adapters are plain tensors outside the TP graph, so the adapter
   would train rank-inconsistent. Asserts DistributedSFTTrainer raises ValueError.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_cp_tp.py
"""

import math
import sys
import traceback

import torch
import torch.distributed as dist
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import (
    cleanup_dirs,
    ensure_model_downloaded,
    init_distributed,
    setup_cache_dirs,
    teardown_distributed,
)
from tests.common.models import QWEN3_0_6B
from tests.common.utils import cleanup_memory, gpu_mem_gb, log

# Configuration

MODEL_NAME = QWEN3_0_6B
MAX_STEPS = 5
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 2e-4
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
SEED = 42


# LoRA Helpers


def create_lora_config() -> LoraConfig:
    """Create a standard LoRA configuration for testing."""
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )


def snapshot_lora_weights(model) -> dict:
    """Capture a snapshot of all LoRA adapter weights."""
    snapshot = {}
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            data = param.data
            # Handle FSDP2 DTensor params — gather to full tensor
            if hasattr(data, "full_tensor"):
                data = data.full_tensor()
            snapshot[name] = data.clone().cpu()
    return snapshot


def verify_lora_updated(before: dict, after: dict) -> tuple:
    """Check that LoRA weights changed during training. Returns (all_updated, details)."""
    if not before or not after:
        return False, "No LoRA parameters found"

    updated_count = 0
    total_count = 0
    unchanged = []

    for name in before:
        if name in after:
            total_count += 1
            if not torch.equal(before[name], after[name]):
                updated_count += 1
            else:
                unchanged.append(name)

    all_updated = updated_count > 0 and len(unchanged) == 0
    details = f"{updated_count}/{total_count} LoRA params updated" + (
        f", unchanged: {unchanged[:3]}..." if unchanged else ""
    )
    return all_updated, details


# Sub-test A: LoRA + CP=2


def run_lora_cp_test(rank: int, local_rank: int, world_size: int) -> bool:
    """Test LoRA adapters with Context Parallelism (CP=2)."""
    log("\n" + "=" * 70)
    log("  SUB-TEST A: LoRA + CP=2 (Context Parallelism)")
    log("=" * 70)

    output_dir, cache_dir = setup_cache_dirs("lora_cp", rank)

    try:
        # Load model via load_distributed_model
        log("[A.1] Loading model with CP=2...")
        parallelism_config = ParallelismConfig(cp_size=2)

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
        log(f"  Base params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

        # Create datasets
        log("[A.2] Creating synthetic datasets...")
        train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
        eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1)
        log(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        # Apply LoRA
        log("[A.4] Applying LoRA adapters...")
        lora_config = create_lora_config()
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log(f"  Trainable: {trainable / 1e6:.2f}M / {total / 1e6:.1f}M ({100 * trainable / total:.2f}%)")

        # Snapshot LoRA weights before training
        lora_before = snapshot_lora_weights(model)
        log(f"  LoRA params tracked: {len(lora_before)}")

        # Configure trainer with CP=2
        log("[A.5] Configuring trainer with CP=2...")
        config = SFTConfig(
            output_dir=output_dir,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=1,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,  # Already applied in load_distributed_model
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            fsdp="",  # Mixin handles FSDP wrapping
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
        log(f"  Parallelism: {parallelism_config.mode_string}")

        # Train
        log("[A.6] Training (5 steps)...")
        train_result = trainer.train()

        # Validate
        log("[A.7] Validating results...")
        training_loss = train_result.training_loss
        log(f"  Training loss: {training_loss:.6f}")

        checks = {}

        # Check 1: Loss is finite
        loss_finite = math.isfinite(training_loss)
        checks["loss_finite"] = loss_finite
        log(f"  Loss is finite: {'PASS' if loss_finite else 'FAIL'}")

        # Check 2: LoRA weights updated
        lora_after = snapshot_lora_weights(model)
        lora_updated, lora_details = verify_lora_updated(lora_before, lora_after)
        checks["lora_updated"] = lora_updated
        log(f"  LoRA weights updated: {'PASS' if lora_updated else 'FAIL'} ({lora_details})")

        # Check 3: Completed expected steps
        steps_ok = train_result.global_step == MAX_STEPS
        checks["steps_completed"] = steps_ok
        log(f"  Steps completed: {'PASS' if steps_ok else 'FAIL'} ({train_result.global_step}/{MAX_STEPS})")

        all_passed = all(checks.values())
        log(f"\n  Sub-test A (LoRA+CP): {'PASSED' if all_passed else 'FAILED'}")
        return all_passed

    except Exception as e:
        log(f"\n  Sub-test A FAILED with exception: {e}")
        if rank == 0:
            traceback.print_exc()
        return False

    finally:
        cleanup_memory()
        cleanup_dirs(output_dir, cache_dir)


# Sub-test B: LoRA + TP=2


def run_lora_tp_test(rank: int, local_rank: int, world_size: int) -> bool:
    """LoRA + TP=2 must be REJECTED at trainer construction.

    TP shards the attention base layers as DTensors; PEFT adapters are plain tensors outside the
    TP graph, so the adapter would train rank-inconsistent (the replicated matrix diverges across
    ranks and the sharded one is corrupted by the TP replicated-grad sync). The trainer fails fast
    instead. This test FAILS if the guard is removed (the trainer would construct successfully).
    """
    log("\n" + "=" * 70)
    log("  SUB-TEST B: LoRA + TP=2 must be REJECTED")
    log("=" * 70)

    output_dir, cache_dir = setup_cache_dirs("lora_tp", rank)

    try:
        # Load tokenizer
        log("[B.1] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create datasets
        log("[B.2] Creating synthetic datasets...")
        train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED + 10)
        eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 11)
        log(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        # Load model with TP via load_distributed_model
        log("[B.3] Loading model with TP=2...")

        # Ensure model is downloaded on rank 0 first
        ensure_model_downloaded(MODEL_NAME, rank)

        parallelism_config = ParallelismConfig(tp_size=2)

        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            use_liger_kernel=True,
        )
        log(f"  Model loaded: {model.config.model_type}")
        log(f"  Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        log(f"  GPU memory: {gpu_mem_gb():.2f} GB")

        # Apply LoRA
        log("[B.4] Applying LoRA adapters...")
        lora_config = create_lora_config()
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log(f"  Trainable: {trainable / 1e6:.2f}M / {total / 1e6:.1f}M ({100 * trainable / total:.2f}%)")

        # Assert the trainer rejects LoRA+TP at construction. The guard fires in
        # _setup_distributed_modes (after super().__init__) deterministically on every rank, so
        # all ranks raise together — no collective desync.
        log("[B.5] Asserting DistributedSFTTrainer(LoRA, TP=2) raises ValueError...")
        config = SFTConfig(
            output_dir=output_dir,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            bf16=True,
            use_liger_kernel=False,
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            save_strategy="no",
            dataloader_num_workers=0,
            fsdp="",
        )
        raised, err = False, ""
        try:
            DistributedSFTTrainer(
                model=model,
                args=config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
                parallelism_config=parallelism_config,
            )
        except ValueError as e:
            raised, err = True, str(e)

        checks = {
            "construction_rejected": raised,
            "error_explains_tp": ("Tensor Parallelism" in err) or ("tp_size" in err),
        }
        log(f"  LoRA+TP rejected at construction: {'PASS' if checks['construction_rejected'] else 'FAIL'}")
        log(f"  Error explains TP incompatibility: {'PASS' if checks['error_explains_tp'] else 'FAIL'}")

        all_passed = all(checks.values())
        log(f"\n  Sub-test B (LoRA+TP rejection): {'PASSED' if all_passed else 'FAILED'}")
        return all_passed

    except Exception as e:
        log(f"\n  Sub-test B FAILED with exception: {e}")
        if rank == 0:
            traceback.print_exc()
        return False

    finally:
        cleanup_memory()
        cleanup_dirs(output_dir, cache_dir)


# Main


def main() -> int:
    """Run both LoRA + CP and LoRA + TP tests. Returns 0 on success, 1 on failure."""
    # Initialize distributed
    rank, world_size, local_rank = init_distributed()

    log(f"\n{'=' * 70}")
    log("  LoRA + CP/TP Parallelism Test")
    log(f"  World size: {world_size}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"{'=' * 70}")

    if world_size != 2:
        log(f"ERROR: This test requires exactly 2 GPUs, got {world_size}")
        teardown_distributed()
        return 1

    # Ensure model is downloaded on rank 0 first
    ensure_model_downloaded(MODEL_NAME, rank)
    PartialState()

    results = {}

    # Sub-test A: LoRA + CP=2
    results["lora_cp"] = run_lora_cp_test(rank, local_rank, world_size)
    dist.barrier()
    cleanup_memory()

    # Sub-test B: LoRA + TP=2
    results["lora_tp"] = run_lora_tp_test(rank, local_rank, world_size)
    dist.barrier()
    cleanup_memory()

    # Summary
    log(f"\n{'=' * 70}")
    log("  RESULTS SUMMARY")
    log(f"{'=' * 70}")
    for name, passed in results.items():
        log(f"  {name}: {'PASSED' if passed else 'FAILED'}")

    all_passed = all(results.values())
    log(f"\n  OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    log(f"{'=' * 70}\n")

    if dist.is_initialized():
        dist.barrier()
    teardown_distributed()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
