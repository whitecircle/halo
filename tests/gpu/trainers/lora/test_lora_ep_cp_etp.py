#!/usr/bin/env python
"""
Test: LoRA adapters with EP+CP and ETP parallelism modes on GptOss-20B.

Trains PEFT LoRA adapters in combined modes and checks the loss stays finite, the adapters (a
lora_B included) move, and the saved adapter reloads to its trained values; no loss or gradient is
compared against a non-parallel reference:
  1. LoRA + EP=2 + CP=2 -> train 5 steps -> save -> verify checkpoint -> reload
  2. LoRA + ETP (expert_tp_size=2) -> train 5 steps -> save -> verify checkpoint -> reload

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_ep_cp_etp.py [--mode ep_cp|etp|all]
"""

import argparse
import math
import os
import traceback

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.peft_helpers import (
    adapter_save_checks,
    assert_adapters_moved,
    snapshot_adapters,
    verify_adapter_reload,
)
from tests.common.utils import cleanup_memory, gpu_mem_gb, log, step_losses

MODEL_NAME = GPT_OSS_20B
MAX_STEPS = 5
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 2e-4
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
SEED = 42

LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_R = 8
LORA_ALPHA = 16


def run_lora_ep_cp(
    tokenizer,
    train_dataset,
    eval_dataset,
    rank: int,
    local_rank: int,
    base_output_dir: str,
) -> tuple[bool, str]:
    """Run LoRA + EP=2 + CP=2 on GptOss-20B MoE."""
    model = None
    trainer = None

    try:
        parallelism_config = ParallelismConfig(ep_size=2, cp_size=2)
        log(f"  Config: {parallelism_config.mode_string}")

        log("  Loading model with EP+CP...")
        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flex_attention",
            use_liger_kernel=True,
        )
        total_params = sum(p.numel() for p in model.parameters())
        log(f"  Model loaded: {total_params / 1e6:.1f}M params, GPU: {gpu_mem_gb():.2f} GB")

        log(f"  Applying LoRA (r={LORA_R}, targets={LORA_TARGET_MODULES})...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"  Trainable: {trainable / 1e6:.2f}M ({100 * trainable / total_params:.2f}%)")

        lora_before = snapshot_adapters(model, expert_lora=False)

        sft_config = SFTConfig(
            output_dir=os.path.join(base_output_dir, "lora_ep_cp_train"),
            max_steps=MAX_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            logging_nan_inf_filter=False,
            max_length=MAX_SEQ_LENGTH,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=True,
            fsdp="",
        )

        trainer = DistributedSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )
        log(f"  EP mode: {trainer.is_ep_mode}, CP mode: {trainer.is_cp_mode}")

        log(f"  Training for {MAX_STEPS} steps...")
        train_result = trainer.train()

        checks = {}
        training_loss = train_result.training_loss
        losses = step_losses(trainer)

        loss_finite = math.isfinite(training_loss)
        checks["loss_finite"] = loss_finite
        log(f"  Loss is finite: {'PASS' if loss_finite else 'FAIL'} ({training_loss:.6f})")
        log(f"  Per-step losses: {[f'{l:.4f}' for l in losses]}")

        steps_ok = train_result.global_step == MAX_STEPS
        checks["steps_completed"] = steps_ok
        log(f"  Steps completed: {'PASS' if steps_ok else 'FAIL'}")

        lora_after = snapshot_adapters(model, expert_lora=False)
        # A zero-init lora_B must move: lora_before predates the trainer's bf16 cast of the fp32
        # adapters, which alone changes every lora_A.
        lora_ok, detail = assert_adapters_moved(lora_before, lora_after)
        checks["lora_updated"] = lora_ok
        log(f"  LoRA weights updated: {'PASS' if lora_ok else 'FAIL'} ({detail})")

        save_dir = os.path.join(base_output_dir, "lora_ep_cp_save")
        log("\n  --- Checkpoint Save (LoRA+EP+CP) ---")
        trainer.save_model(save_dir)
        barrier()

        save_checks = adapter_save_checks(save_dir, rank)
        checks.update(save_checks)
        reload_checks = verify_adapter_reload(
            save_dir, lora_after, model_name=MODEL_NAME, tokenizer=tokenizer, rank=rank, local_rank=local_rank
        )
        checks.update(reload_checks)
        barrier()

        all_passed = all(checks.values())
        return all_passed, f"loss={training_loss:.6f}"

    except Exception as e:
        log(f"  FAILED with exception: {e}")
        traceback.print_exc()
        return False, f"Exception: {e}"

    finally:
        if trainer is not None and hasattr(trainer, "cleanup_ep"):
            trainer.cleanup_ep()
        del trainer
        del model
        cleanup_memory()
        barrier()


def run_lora_etp(
    tokenizer,
    train_dataset,
    eval_dataset,
    rank: int,
    local_rank: int,
    base_output_dir: str,
) -> tuple[bool, str]:
    """Run LoRA + ETP (expert_tp_size=2) on GptOss-20B MoE."""
    model = None
    trainer = None

    try:
        parallelism_config = ParallelismConfig(ep_size=1, expert_tp_size=2)
        log(f"  Config: {parallelism_config.mode_string}")

        log("  Loading model with ETP...")
        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flex_attention",
            use_liger_kernel=True,
        )
        total_params = sum(p.numel() for p in model.parameters())
        log(f"  Model loaded: {total_params / 1e6:.1f}M params, GPU: {gpu_mem_gb():.2f} GB")

        log(f"  Applying LoRA (r={LORA_R}, targets={LORA_TARGET_MODULES})...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"  Trainable: {trainable / 1e6:.2f}M ({100 * trainable / total_params:.2f}%)")

        lora_before = snapshot_adapters(model, expert_lora=False)

        sft_config = SFTConfig(
            output_dir=os.path.join(base_output_dir, "lora_etp_train"),
            max_steps=MAX_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            logging_nan_inf_filter=False,
            max_length=MAX_SEQ_LENGTH,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=True,
            fsdp="",
        )

        trainer = DistributedSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )

        log(f"  Training for {MAX_STEPS} steps...")
        train_result = trainer.train()

        checks = {}
        training_loss = train_result.training_loss
        losses = step_losses(trainer)

        loss_finite = math.isfinite(training_loss)
        checks["loss_finite"] = loss_finite
        log(f"  Loss is finite: {'PASS' if loss_finite else 'FAIL'} ({training_loss:.6f})")
        log(f"  Per-step losses: {[f'{l:.4f}' for l in losses]}")

        steps_ok = train_result.global_step == MAX_STEPS
        checks["steps_completed"] = steps_ok
        log(f"  Steps completed: {'PASS' if steps_ok else 'FAIL'}")

        lora_after = snapshot_adapters(model, expert_lora=False)
        # A zero-init lora_B must move: lora_before predates the trainer's bf16 cast of the fp32
        # adapters, which alone changes every lora_A.
        lora_ok, detail = assert_adapters_moved(lora_before, lora_after)
        checks["lora_updated"] = lora_ok
        log(f"  LoRA weights updated: {'PASS' if lora_ok else 'FAIL'} ({detail})")

        save_dir = os.path.join(base_output_dir, "lora_etp_save")
        log("\n  --- Checkpoint Save (LoRA+ETP) ---")
        trainer.save_model(save_dir)
        barrier()

        save_checks = adapter_save_checks(save_dir, rank)
        checks.update(save_checks)
        reload_checks = verify_adapter_reload(
            save_dir, lora_after, model_name=MODEL_NAME, tokenizer=tokenizer, rank=rank, local_rank=local_rank
        )
        checks.update(reload_checks)
        barrier()

        all_passed = all(checks.values())
        return all_passed, f"loss={training_loss:.6f}"

    except Exception as e:
        log(f"  FAILED with exception: {e}")
        traceback.print_exc()
        return False, f"Exception: {e}"

    finally:
        if trainer is not None and hasattr(trainer, "cleanup_ep"):
            trainer.cleanup_ep()
        del trainer
        del model
        cleanup_memory()
        barrier()


def run(ctx) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ep_cp", "etp", "all"], default="all")
    args, _ = parser.parse_known_args()

    log(f"\n{'#' * 70}")
    log("  LoRA + EP+CP / ETP Parallelism Test")
    log(f"  World size: {ctx.world_size}, Mode: {args.mode}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"{'#' * 70}")

    ensure_model_downloaded(MODEL_NAME, ctx.rank)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1)

    results: dict[str, tuple[bool, str]] = {}

    if args.mode in ("ep_cp", "all"):
        log(f"\n{'=' * 70}")
        log("  TEST 1: LoRA + EP=2 + CP=2 (GptOss-20B)")
        log(f"{'=' * 70}")
        results["lora_ep_cp"] = run_lora_ep_cp(
            tokenizer, train_dataset, eval_dataset, ctx.rank, ctx.local_rank, ctx.output_dir
        )

    if args.mode in ("etp", "all"):
        log(f"\n{'=' * 70}")
        log("  TEST 2: LoRA + ETP (expert_tp_size=2) (GptOss-20B)")
        log(f"{'=' * 70}")
        results["lora_etp"] = run_lora_etp(
            tokenizer, train_dataset, eval_dataset, ctx.rank, ctx.local_rank, ctx.output_dir
        )

    for name, (passed, detail) in results.items():
        log(f"  {name:20s} {'PASSED' if passed else 'FAILED'} -- {detail}")

    return {"checks": {name: passed for name, (passed, _) in results.items()}}


main = gpu_test_main(exact_world_size=2, prefix="test_lora_ep_cp_etp")(run)

if __name__ == "__main__":
    main()
