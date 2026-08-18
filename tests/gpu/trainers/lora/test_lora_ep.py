#!/usr/bin/env python
"""
Test: LoRA adapters with Expert Parallelism (EP=2) on a MoE model (GptOss-20B).

Validates that PEFT LoRA adapters work correctly with Expert Parallelism:
1. Loads GptOss-20B MoE model with EP=2 via load_distributed_model
2. Applies LoRA (r=8, alpha=16) to attention q_proj and v_proj
3. Trains for 3 steps with DistributedSFTTrainer
4. Verifies only LoRA parameters have gradients (base model frozen)
5. Verifies training completes without errors and loss is finite

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_ep.py

Requirements:
    - 2x GPUs with >=80GB memory each
    - DeepEP installed
    - Model: unsloth/gpt-oss-20b-BF16 (auto-downloaded)
"""

import math
import sys
import traceback

import torch
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
from tests.common.models import GPT_OSS_20B
from tests.common.utils import cleanup_memory, gpu_mem_gb, log

# Configuration

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 2
MAX_STEPS = 3
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 5e-5
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
SEED = 42


# Main Test


def run_lora_ep_test() -> bool:
    """Run LoRA + EP=2 test on GptOss-20B MoE model."""
    # Initialize distributed
    rank, world_size, local_rank = init_distributed()

    log(f"\n{'=' * 70}")
    log("  LoRA + EP Test: Expert Parallelism (EP=2) with GptOss-20B")
    log(f"{'=' * 70}")
    log(f"  World size: {world_size}")
    log(f"  EP size: {EP_SIZE}")
    log(f"  Model: {MODEL_NAME}")
    log(f"  Train samples: {NUM_TRAIN_SAMPLES}, Eval samples: {NUM_EVAL_SAMPLES}")
    log(f"  Max seq length: {MAX_SEQ_LENGTH}")
    log(f"  Batch size: {BATCH_SIZE}, Steps: {MAX_STEPS}")
    log(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"  GPU memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f}GB")

    if world_size != EP_SIZE:
        log(f"\nERROR: This test requires exactly {EP_SIZE} GPUs, got {world_size}")
        teardown_distributed()
        return False

    # Isolate caches
    output_dir, cache_dir = setup_cache_dirs("lora_ep", rank)
    log(f"  Output dir: {output_dir}")

    try:
        # --- Download model on rank 0 ---
        log("\n[1/7] Ensuring model is downloaded...")
        ensure_model_downloaded(MODEL_NAME, rank)

        # Initialize PartialState for accelerate compatibility
        PartialState()

        # --- Load tokenizer ---
        log("\n[2/7] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        log(f"  Tokenizer: {tokenizer.__class__.__name__}, vocab_size={tokenizer.vocab_size}")

        # --- Create datasets ---
        log("\n[3/7] Creating synthetic datasets...")
        train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
        eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)
        log(f"  Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

        if rank == 0:
            sample = train_dataset[0]["text"]
            log(f"  Sample (truncated): {sample[:150]}...")

        # --- Load model with EP ---
        log("\n[4/7] Loading model with Expert Parallelism (EP=2)...")
        log(f"  GPU memory before model load: {gpu_mem_gb():.1f}GB")

        parallelism_config = ParallelismConfig(ep_size=EP_SIZE)

        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flex_attention",
            use_liger_kernel=True,
        )
        log(f"  Model type: {model.config.model_type}")
        log(f"  Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        log(f"  GPU memory after load: {gpu_mem_gb():.1f}GB")

        # --- Apply LoRA ---
        log("\n[5/7] Applying LoRA adapters (q_proj, v_proj)...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        frozen_params = total_params - trainable_params
        log(f"  Trainable: {trainable_params / 1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")
        log(f"  Frozen: {frozen_params / 1e6:.1f}M")

        # Verify only LoRA params are trainable
        non_lora_trainable = []
        lora_param_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "lora_" in name:
                    lora_param_count += 1
                else:
                    non_lora_trainable.append(name)

        log(f"  LoRA trainable param groups: {lora_param_count}")
        if non_lora_trainable:
            log(f"  WARNING: Non-LoRA trainable params found: {non_lora_trainable[:5]}...")

        # Snapshot LoRA weights before training
        lora_before = {}
        for name, param in model.named_parameters():
            if "lora_" in name and param.requires_grad:
                data = param.data
                if hasattr(data, "full_tensor"):
                    data = data.full_tensor()
                lora_before[name] = data.clone().cpu()

        # --- Create trainer (mirrors sft.py settings) ---
        log("\n[6/7] Creating SFT trainer...")
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
            remove_unused_columns=False,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=True,  # Required for EP (inactive experts)
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
        log(f"  Parallelism: EP={parallelism_config.ep_size}")

        # --- Train ---
        log(f"\n[7/7] Training ({MAX_STEPS} steps)...")
        log(f"  GPU memory before training: {gpu_mem_gb():.1f}GB")
        from src.distributed.runtime import barrier

        barrier()

        train_result = trainer.train()

        log(f"  Training loss: {train_result.training_loss:.6f}")
        log(f"  Steps completed: {train_result.global_step}")
        log(f"  GPU memory after training: {gpu_mem_gb():.1f}GB")

        # --- Validate ---
        log("\n--- Validating results ---")
        log_history = trainer.state.log_history
        step_losses = [entry["loss"] for entry in log_history if "loss" in entry and "eval_loss" not in entry]
        log(f"  Per-step losses: {[f'{l:.4f}' for l in step_losses]}")

        checks = {}

        # Check 1: Training loss is finite
        loss_finite = math.isfinite(train_result.training_loss)
        checks["loss_finite"] = loss_finite
        log(f"  Loss is finite: {'PASS' if loss_finite else 'FAIL'} ({train_result.training_loss:.6f})")

        # Check 2: No NaN/Inf in step losses
        all_finite = all(math.isfinite(l) for l in step_losses)
        checks["all_steps_finite"] = all_finite
        log(f"  All step losses finite: {'PASS' if all_finite else 'FAIL'}")

        # Check 3: Completed expected number of steps
        steps_ok = train_result.global_step == MAX_STEPS
        checks["steps_completed"] = steps_ok
        log(f"  Steps completed: {'PASS' if steps_ok else 'FAIL'} ({train_result.global_step}/{MAX_STEPS})")

        # Check 4: LoRA weights were updated
        lora_after = {}
        for name, param in model.named_parameters():
            if "lora_" in name and param.requires_grad:
                data = param.data
                if hasattr(data, "full_tensor"):
                    data = data.full_tensor()
                lora_after[name] = data.clone().cpu()

        updated_count = 0
        unchanged_params = []
        for name in lora_before:
            if name in lora_after:
                if not torch.equal(lora_before[name], lora_after[name]):
                    updated_count += 1
                else:
                    unchanged_params.append(name)

        lora_updated = updated_count > 0
        checks["lora_updated"] = lora_updated
        log(
            f"  LoRA weights updated: {'PASS' if lora_updated else 'FAIL'} "
            f"({updated_count}/{len(lora_before)} params changed)"
        )
        if unchanged_params:
            log(f"    Unchanged: {unchanged_params[:3]}{'...' if len(unchanged_params) > 3 else ''}")

        # Check 5: Only LoRA params have gradients (base frozen)
        only_lora_grads = len(non_lora_trainable) == 0
        checks["only_lora_grads"] = only_lora_grads
        log(f"  Only LoRA params trainable: {'PASS' if only_lora_grads else 'FAIL'}")

        # Check 6: Loss is not unreasonably high
        loss_reasonable = train_result.training_loss < 100
        checks["loss_reasonable"] = loss_reasonable
        log(f"  Loss reasonable (<100): {'PASS' if loss_reasonable else 'FAIL'}")

        # --- Report ---
        all_passed = all(checks.values())
        log(f"\n{'=' * 70}")
        if all_passed:
            log("  LoRA EP TEST PASSED")
            log(f"  - Training loss: {train_result.training_loss:.6f}")
            log(f"  - Steps completed: {train_result.global_step}")
            log(f"  - LoRA params updated: {updated_count}/{len(lora_before)}")
        else:
            log("  LoRA EP TEST FAILED")
            failed = [k for k, v in checks.items() if not v]
            for f in failed:
                log(f"  - FAILED: {f}")
        log(f"{'=' * 70}")

        return all_passed

    except Exception as e:
        log(f"\n  LoRA EP TEST FAILED WITH ERROR: {e}")
        if rank == 0:
            traceback.print_exc()
        return False

    finally:
        from src.distributed.runtime import barrier

        barrier()
        cleanup_dirs(output_dir, cache_dir)
        cleanup_memory()


# Main


def main():
    success = run_lora_ep_test()
    teardown_distributed()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
