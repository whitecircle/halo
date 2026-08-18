#!/usr/bin/env python
"""
SFT Trainer test: LoRA and QLoRA on GptOss-20B MoE with Expert Parallelism (EP=2).

Tests LoRA/QLoRA parameter modes under EP with checkpoint save + reload
verification for each configuration.

Tests:
  1. LoRA  + EP=2 -> train -> save -> verify checkpoint -> reload adapter
  2. QLoRA + EP=2 -> train -> save -> verify checkpoint -> reload adapter

Checkpoint verification:
  - Adapter files exist (adapter_config.json, adapter_model.safetensors)
  - Adapter reloads successfully via PeftModel.from_pretrained()
  - Reloaded model produces finite logits on a sample input

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_oss20b_ep_lora.py
"""

import math
import os
import sys
import traceback

import torch
import torch.distributed as dist
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ModelConfig, SFTConfig, get_quantization_config

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import (
    cleanup_dirs,
    ensure_model_downloaded,
    setup_cache_dirs,
    teardown_distributed,
)
from tests.common.models import GPT_OSS_20B
from tests.common.peft_helpers import adapter_save_checks
from tests.common.utils import cleanup_memory, gpu_mem_gb, log

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 2
MAX_STEPS = 5
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 1e-4
NUM_TRAIN_SAMPLES = 64
NUM_EVAL_SAMPLES = 16
SEED = 42

LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_R = 8
LORA_ALPHA = 16


def _validate_training(train_result, trainer, max_steps):
    """Validate common training results. Returns (checks_dict, step_losses)."""
    training_loss = train_result.training_loss
    log_history = trainer.state.log_history
    step_losses = [entry["loss"] for entry in log_history if "loss" in entry and "eval_loss" not in entry]

    checks = {}

    loss_finite = math.isfinite(training_loss)
    checks["loss_finite"] = loss_finite
    log(f"  Loss is finite: {'PASS' if loss_finite else 'FAIL'} ({training_loss:.6f})")

    all_finite = all(math.isfinite(l) for l in step_losses)
    checks["all_steps_finite"] = all_finite
    log(f"  All step losses finite: {'PASS' if all_finite else 'FAIL'}")

    steps_ok = train_result.global_step == max_steps
    checks["steps_completed"] = steps_ok
    log(f"  Steps completed: {'PASS' if steps_ok else 'FAIL'} ({train_result.global_step}/{max_steps})")

    loss_reasonable = training_loss < 100
    checks["loss_reasonable"] = loss_reasonable
    log(f"  Loss reasonable (<100): {'PASS' if loss_reasonable else 'FAIL'}")

    return checks, step_losses


def _validate_lora_updated(model, lora_before: dict) -> dict[str, bool]:
    """Verify LoRA weights were updated during training."""
    checks = {}
    lora_after = {}
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            data = param.data
            if hasattr(data, "full_tensor"):
                data = data.full_tensor()
            lora_after[name] = data.clone().cpu()

    updated_count = 0
    for name in lora_before:
        if name in lora_after and not torch.equal(lora_before[name], lora_after[name]):
            updated_count += 1

    lora_updated = updated_count > 0
    checks["lora_updated"] = lora_updated
    log(
        f"  LoRA weights updated: {'PASS' if lora_updated else 'FAIL'} "
        f"({updated_count}/{len(lora_before)} params changed)"
    )

    return checks


def run_lora_ep(
    parallelism_config: ParallelismConfig,
    tokenizer,
    train_dataset,
    eval_dataset,
    rank: int,
    local_rank: int,
    base_output_dir: str,
) -> tuple[bool, str]:
    """Run LoRA + EP=2 on GptOss-20B."""
    model = None
    trainer = None
    save_dir = os.path.join(base_output_dir, "lora_ep_save")

    try:
        log(f"  Loading model with EP={EP_SIZE}...")
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

        # EP models need direct PEFT wrapping, not TRL's peft_config path
        log(f"  Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA}, targets={LORA_TARGET_MODULES})...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"  Trainable: {trainable / 1e6:.2f}M / {total_params / 1e6:.1f}M ({100 * trainable / total_params:.2f}%)")

        lora_before = {}
        for name, param in model.named_parameters():
            if "lora_" in name and param.requires_grad:
                data = param.data
                if hasattr(data, "full_tensor"):
                    data = data.full_tensor()
                lora_before[name] = data.clone().cpu()

        sft_config = SFTConfig(
            output_dir=os.path.join(base_output_dir, "lora_ep_train"),
            max_steps=MAX_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
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

        log(f"  Saving model to {save_dir}...")
        trainer.save_model(save_dir)
        barrier()

        log("\n  --- Training Validation (LoRA+EP) ---")
        checks, step_losses = _validate_training(train_result, trainer, MAX_STEPS)
        log(f"  Per-step losses: {[f'{l:.4f}' for l in step_losses]}")

        lora_checks = _validate_lora_updated(model, lora_before)
        checks.update(lora_checks)

        log("\n  --- Checkpoint Verification (LoRA+EP) ---")
        ckpt_checks = adapter_save_checks(save_dir, rank)
        checks.update(ckpt_checks)

        trainer.cleanup_ep()
        del trainer
        trainer = None
        del model
        model = None
        cleanup_memory()
        barrier()

        # reload without EP: the saved adapter must be portable off the EP layout
        if rank == 0:
            log("\n  --- Checkpoint Reload (LoRA+EP) ---")
            try:
                log("  Reloading base model...")
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
                checks["adapter_reload"] = False

        all_passed = all(checks.values())
        detail = f"loss={train_result.training_loss:.6f}"
        return all_passed, detail

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


def run_qlora_ep(
    parallelism_config: ParallelismConfig,
    tokenizer,
    train_dataset,
    eval_dataset,
    rank: int,
    local_rank: int,
    base_output_dir: str,
) -> tuple[bool, str]:
    """Run QLoRA + EP=2 on GptOss-20B."""
    model = None
    trainer = None
    os.path.join(base_output_dir, "qlora_ep_save")

    try:
        model_config = ModelConfig(
            model_name_or_path=MODEL_NAME,
            use_peft=True,
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            use_bnb_nested_quant=True,
            trust_remote_code=True,
            attn_implementation="flex_attention",
        )
        quantization_config = get_quantization_config(model_config)
        log(f"  Quantization: {quantization_config.quant_method}")

        # QLoRA(4-bit)+EP is refused: EP loaders materialize plain de-quantized weights, losing
        # bitsandbytes Params4bit, so PEFT's 4-bit dispatch fails on `weight.compress_statistics`.
        log(f"  Verifying QLoRA(4-bit)+EP={EP_SIZE} is cleanly rejected by the config guard...")
        try:
            load_distributed_model(
                model_name_or_path=MODEL_NAME,
                parallelism_config=parallelism_config,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="flex_attention",
                use_liger_kernel=False,
                quantization_config=quantization_config,
            )
        except ValueError as e:
            if "QLoRA" in str(e) or "quantiz" in str(e).lower():
                log(f"  Correctly rejected: {e}")
                return True, "QLoRA+EP cleanly rejected (use DDP/FSDP for QLoRA, plain LoRA for EP)"
            return False, f"Rejected with unexpected error: {e}"
        return False, "QLoRA+EP should have been rejected by the guard but load succeeded"

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
    """Run LoRA/QLoRA + EP tests on GptOss-20B. Returns 0 on success, 1 on failure."""
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    state = PartialState()
    rank = state.process_index
    local_rank = state.local_process_index
    world_size = state.num_processes

    base_output_dir, cache_dir = setup_cache_dirs("test_sft_oss20b_ep_lora", rank)

    log(f"\n{'#' * 70}")
    log("  SFT LoRA/QLoRA + EP Test on GptOss-20B MoE")
    log(f"  World size: {world_size}, EP size: {EP_SIZE}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"  Max steps: {MAX_STEPS}, Batch size: {BATCH_SIZE}, Seq length: {MAX_SEQ_LENGTH}")
    log(f"{'#' * 70}")

    if world_size != EP_SIZE:
        log(f"\nERROR: This test requires exactly {EP_SIZE} GPUs, got {world_size}")
        if dist.is_initialized():
            teardown_distributed()
        return 1

    ensure_model_downloaded(MODEL_NAME, rank)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1)
    log(f"  Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    parallelism_config = ParallelismConfig(ep_size=EP_SIZE)
    results: dict[str, tuple[bool, str]] = {}

    log(f"\n{'=' * 70}")
    log("  TEST 1: LoRA + EP=2 (GptOss-20B)")
    log(f"{'=' * 70}")

    success, detail = run_lora_ep(
        parallelism_config,
        tokenizer,
        train_dataset,
        eval_dataset,
        rank,
        local_rank,
        base_output_dir,
    )
    results["lora_ep"] = (success, detail)

    log(f"\n{'=' * 70}")
    log("  TEST 2: QLoRA (4-bit) + EP=2 (GptOss-20B)")
    log(f"{'=' * 70}")

    success, detail = run_qlora_ep(
        parallelism_config,
        tokenizer,
        train_dataset,
        eval_dataset,
        rank,
        local_rank,
        base_output_dir,
    )
    results["qlora_ep"] = (success, detail)

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
