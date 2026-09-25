#!/usr/bin/env python
"""
SFT Trainer test: LoRA and QLoRA on GptOss-20B MoE with Expert Parallelism (EP=2).

Trains LoRA under EP with checkpoint save + reload verification, and asserts QLoRA+EP is
rejected at load time.

Tests:
  1. LoRA  + EP=2 -> train -> save -> verify checkpoint -> reload adapter
  2. QLoRA + EP=2 -> REJECTED by the load-time guard (no training)

Checkpoint verification:
  - Adapter files exist (adapter_config.json, adapter_model.safetensors)
  - PeftModel.from_pretrained() on a fresh non-EP base restores every trained LoRA tensor
  - Reloaded model produces finite logits on a sample input

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_sft_oss20b_ep_lora.py
"""

import math
import os
import traceback

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from trl import ModelConfig, SFTConfig, get_quantization_config

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

    return checks, step_losses


def _validate_lora_updated(lora_before: dict, lora_after: dict) -> dict[str, bool]:
    """Verify training moved the adapters: a zero-init lora_B must change, since ``lora_before`` is
    taken ahead of the trainer's bf16 cast of the fp32 adapters, which alone changes every lora_A."""
    lora_updated, detail = assert_adapters_moved(lora_before, lora_after)
    log(f"  LoRA weights updated: {'PASS' if lora_updated else 'FAIL'} ({detail})")
    return {"lora_updated": lora_updated}


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

        lora_before = snapshot_adapters(model, expert_lora=False)

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

        log(f"  Saving model to {save_dir}...")
        trainer.save_model(save_dir)
        barrier()

        log("\n  --- Training Validation (LoRA+EP) ---")
        checks, step_losses = _validate_training(train_result, trainer, MAX_STEPS)
        log(f"  Per-step losses: {[f'{l:.4f}' for l in step_losses]}")

        lora_after = snapshot_adapters(model, expert_lora=False)
        checks.update(_validate_lora_updated(lora_before, lora_after))

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
        log("\n  --- Checkpoint Reload (LoRA+EP) ---")
        checks.update(
            verify_adapter_reload(
                save_dir, lora_after, model_name=MODEL_NAME, tokenizer=tokenizer, rank=rank, local_rank=local_rank
            )
        )

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
) -> tuple[bool, str]:
    """Run QLoRA + EP=2 on GptOss-20B."""
    model = None
    trainer = None

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


def run(ctx) -> dict:
    log(f"\n{'#' * 70}")
    log("  SFT LoRA/QLoRA + EP Test on GptOss-20B MoE")
    log(f"  World size: {ctx.world_size}, EP size: {EP_SIZE}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  Max steps: {MAX_STEPS}, Batch size: {BATCH_SIZE}, Seq length: {MAX_SEQ_LENGTH}")
    log(f"{'#' * 70}")

    ensure_model_downloaded(MODEL_NAME, ctx.rank)

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

    results["lora_ep"] = run_lora_ep(
        parallelism_config,
        tokenizer,
        train_dataset,
        eval_dataset,
        ctx.rank,
        ctx.local_rank,
        ctx.output_dir,
    )

    log(f"\n{'=' * 70}")
    log("  TEST 2: QLoRA (4-bit) + EP=2 (GptOss-20B)")
    log(f"{'=' * 70}")

    results["qlora_ep"] = run_qlora_ep(
        parallelism_config,
        tokenizer,
        train_dataset,
        eval_dataset,
        ctx.rank,
        ctx.local_rank,
    )

    for name, (passed, detail) in results.items():
        log(f"  {name:20s} {'PASSED' if passed else 'FAILED'} -- {detail}")

    return {"checks": {name: passed for name, (passed, _) in results.items()}}


main = gpu_test_main(exact_world_size=EP_SIZE, prefix="test_sft_oss20b_ep_lora")(run)

if __name__ == "__main__":
    main()
