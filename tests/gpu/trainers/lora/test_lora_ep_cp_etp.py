#!/usr/bin/env python
"""
Test: LoRA adapters with EP+CP and ETP parallelism modes on GptOss-20B.

Validates that PEFT LoRA adapters work correctly in combined modes:
  1. LoRA + EP=2 + CP=2 -> train 5 steps -> save -> verify checkpoint -> reload
  2. LoRA + ETP (expert_tp_size=2) -> train 5 steps -> save -> verify checkpoint -> reload

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_ep_cp_etp.py [--mode ep_cp|etp|all]
"""

import argparse
import math
import os
import sys
import traceback

import torch
import torch.distributed as dist
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
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
from tests.common.peft_helpers import adapter_save_checks, snapshot_adapters
from tests.common.utils import cleanup_memory, gpu_mem_gb, log

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


def _verify_checkpoint_reload(
    save_dir: str,
    tokenizer,
    rank: int,
    local_rank: int,
) -> dict[str, bool]:
    if rank != 0:
        return {}
    checks = {}
    try:
        log("  Reloading base model for checkpoint verification...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": local_rank},
        )
        log(f"  Loading adapter from {save_dir}...")
        reloaded = PeftModel.from_pretrained(base_model, save_dir)
        checks["adapter_reload"] = True
        log("  Adapter reload: PASS")

        reloaded.eval()
        test_input = tokenizer("What is 2 + 2?", return_tensors="pt").to(f"cuda:{local_rank}")
        with torch.no_grad():
            output = reloaded(**test_input)
        logits_finite = torch.isfinite(output.logits).all().item()
        checks["reload_logits_finite"] = logits_finite
        log(f"  Reload logits finite: {'PASS' if logits_finite else 'FAIL'}")

        del reloaded, base_model
        cleanup_memory()
    except Exception as e:
        log(f"  Checkpoint reload FAILED: {e}")
        traceback.print_exc()
        checks["adapter_reload"] = False
    return checks


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
        step_losses = [e["loss"] for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]

        loss_finite = math.isfinite(training_loss)
        checks["loss_finite"] = loss_finite
        log(f"  Loss is finite: {'PASS' if loss_finite else 'FAIL'} ({training_loss:.6f})")
        log(f"  Per-step losses: {[f'{l:.4f}' for l in step_losses]}")

        steps_ok = train_result.global_step == MAX_STEPS
        checks["steps_completed"] = steps_ok
        log(f"  Steps completed: {'PASS' if steps_ok else 'FAIL'}")

        lora_after = snapshot_adapters(model, expert_lora=False)
        updated = sum(1 for n in lora_before if n in lora_after and not torch.equal(lora_before[n], lora_after[n]))
        lora_ok = updated > 0
        checks["lora_updated"] = lora_ok
        log(f"  LoRA weights updated: {'PASS' if lora_ok else 'FAIL'} ({updated}/{len(lora_before)})")

        save_dir = os.path.join(base_output_dir, "lora_ep_cp_save")
        log("\n  --- Checkpoint Save (LoRA+EP+CP) ---")
        trainer.save_model(save_dir)
        barrier()

        save_checks = adapter_save_checks(save_dir, rank)
        checks.update(save_checks)
        reload_checks = _verify_checkpoint_reload(save_dir, tokenizer, rank, local_rank)
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
        step_losses = [e["loss"] for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]

        loss_finite = math.isfinite(training_loss)
        checks["loss_finite"] = loss_finite
        log(f"  Loss is finite: {'PASS' if loss_finite else 'FAIL'} ({training_loss:.6f})")
        log(f"  Per-step losses: {[f'{l:.4f}' for l in step_losses]}")

        steps_ok = train_result.global_step == MAX_STEPS
        checks["steps_completed"] = steps_ok
        log(f"  Steps completed: {'PASS' if steps_ok else 'FAIL'}")

        lora_after = snapshot_adapters(model, expert_lora=False)
        updated = sum(1 for n in lora_before if n in lora_after and not torch.equal(lora_before[n], lora_after[n]))
        lora_ok = updated > 0
        checks["lora_updated"] = lora_ok
        log(f"  LoRA weights updated: {'PASS' if lora_ok else 'FAIL'} ({updated}/{len(lora_before)})")

        save_dir = os.path.join(base_output_dir, "lora_etp_save")
        log("\n  --- Checkpoint Save (LoRA+ETP) ---")
        trainer.save_model(save_dir)
        barrier()

        save_checks = adapter_save_checks(save_dir, rank)
        checks.update(save_checks)
        reload_checks = _verify_checkpoint_reload(save_dir, tokenizer, rank, local_rank)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ep_cp", "etp", "all"], default="all")
    args, _ = parser.parse_known_args()

    rank, world_size, local_rank = init_distributed()
    PartialState()

    base_output_dir, cache_dir = setup_cache_dirs("test_lora_ep_cp_etp", rank)

    log(f"\n{'#' * 70}")
    log("  LoRA + EP+CP / ETP Parallelism Test")
    log(f"  World size: {world_size}, Mode: {args.mode}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"{'#' * 70}")

    if world_size != 2:
        log(f"\nERROR: This test requires exactly 2 GPUs, got {world_size}")
        teardown_distributed()
        return 1

    ensure_model_downloaded(MODEL_NAME, rank)

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
        success, detail = run_lora_ep_cp(
            tokenizer,
            train_dataset,
            eval_dataset,
            rank,
            local_rank,
            base_output_dir,
        )
        results["lora_ep_cp"] = (success, detail)

    if args.mode in ("etp", "all"):
        log(f"\n{'=' * 70}")
        log("  TEST 2: LoRA + ETP (expert_tp_size=2) (GptOss-20B)")
        log(f"{'=' * 70}")
        success, detail = run_lora_etp(
            tokenizer,
            train_dataset,
            eval_dataset,
            rank,
            local_rank,
            base_output_dir,
        )
        results["lora_etp"] = (success, detail)

    log(f"\n{'#' * 70}")
    log("  FINAL RESULTS")
    log(f"{'#' * 70}")
    for name, (passed, detail) in results.items():
        status = "PASSED" if passed else "FAILED"
        log(f"  {name:20s} {status} -- {detail}")
    log(f"{'#' * 70}")

    all_passed = all(p for p, _ in results.values())
    log(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    cleanup_dirs(base_output_dir, cache_dir)
    if dist.is_initialized():
        teardown_distributed()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
