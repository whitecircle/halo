#!/usr/bin/env python
"""
SFT + EP training test on GptOss-20B with Flash Attention 2: Full, LoRA, QLoRA.

Validates DistributedSFTTrainer with Expert Parallelism (EP=2) on a MoE model
(32 experts across 2 GPUs via DeepEP) using flash_attention_2. Supports three
parameter modes, selected via --mode flag:

  full  — all parameters trainable
  lora  — attention-only LoRA adapters (q_proj, v_proj); EP-safe
  qlora — 4-bit NF4 quantized base + LoRA; unsupported with EP, asserts the load-time guard fires

Each mode is run as a separate torchrun invocation to avoid DeepEP buffer
cleanup issues when reloading EP models in the same process.

Model: unsloth/gpt-oss-20b-BF16 (MoE, 32 experts)

Usage (single mode):
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_ep_fa2_modes.py --mode full

    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_ep_fa2_modes.py --mode lora

    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_ep_fa2_modes.py --mode qlora

Requirements:
    - 2x GPUs with >=80GB memory each
    - DeepEP installed
    - flash-attn installed (FA2)
    - Model: unsloth/gpt-oss-20b-BF16 (auto-downloaded)
"""

import argparse
import math
import os
import sys
import traceback

import torch
import torch.distributed as dist
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from trl import ModelConfig, SFTConfig, get_quantization_config

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
from tests.common.utils import gpu_mem_gb, log

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 2
MAX_STEPS = 5
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 2e-5
LORA_LEARNING_RATE = 5e-5
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
SEED = 42
ATTN_IMPL = "flash_attention_2"

# Expert projections are EP-sharded across ranks, so LoRA on them breaks grad sync.
LORA_TARGET_MODULES = ["q_proj", "v_proj"]


def _to_full_cpu(tensor):
    """Materialize a (possibly FSDP2 DTensor-sharded) tensor to a plain CPU tensor.

    LoRA params are snapshotted before FSDP2 wrapping (plain) but read back after training
    (DTensor); ``torch.equal`` rejects a plain-vs-DTensor mix, so normalize both sides here.
    ``full_tensor()`` is a collective — all ranks must call it.
    """
    if hasattr(tensor, "full_tensor"):
        tensor = tensor.full_tensor()
    return tensor.detach().cpu()


def _validate_training(train_result, trainer, max_steps, rank, local_rank, world_size):
    """Validate common training results. Returns checks dict."""
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
    log(f"  Per-step losses: {[f'{l:.4f}' for l in step_losses]}")

    steps_ok = train_result.global_step == max_steps
    checks["steps_completed"] = steps_ok
    log(f"  Steps completed: {'PASS' if steps_ok else 'FAIL'} ({train_result.global_step}/{max_steps})")

    loss_reasonable = training_loss < 100
    checks["loss_reasonable"] = loss_reasonable
    log(f"  Loss reasonable (<100): {'PASS' if loss_reasonable else 'FAIL'}")

    ep_active = trainer.is_ep_mode
    checks["ep_mode"] = ep_active
    log(f"  EP mode active: {'PASS' if ep_active else 'FAIL'}")

    loss_tensor = torch.tensor([training_loss], device=f"cuda:{local_rank}")
    all_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
    dist.all_gather(all_losses, loss_tensor)
    if rank == 0:
        losses_list = [l.item() for l in all_losses]
        spread = max(losses_list) - min(losses_list)
        checks["loss_consistent"] = spread < 0.05
        log(f"  Loss consistent (spread={spread:.6f}): {'PASS' if checks['loss_consistent'] else 'FAIL'}")
    else:
        checks["loss_consistent"] = True

    return checks


def _validate_lora(model, lora_before):
    """Validate LoRA-specific checks: weights updated, only LoRA trainable."""
    checks = {}

    non_lora_trainable = [
        name for name, param in model.named_parameters() if param.requires_grad and "lora_" not in name
    ]
    only_lora = len(non_lora_trainable) == 0
    checks["only_lora_trainable"] = only_lora
    log(f"  Only LoRA params trainable: {'PASS' if only_lora else 'FAIL'}")
    if non_lora_trainable:
        log(f"    Non-LoRA trainable: {non_lora_trainable[:5]}")

    lora_after = {
        name: _to_full_cpu(param.data)
        for name, param in model.named_parameters()
        if "lora_" in name and param.requires_grad
    }
    updated_count = sum(
        1 for name in lora_before if name in lora_after and not torch.equal(lora_before[name], lora_after[name])
    )
    lora_updated = updated_count > 0
    checks["lora_updated"] = lora_updated
    log(
        f"  LoRA weights updated: {'PASS' if lora_updated else 'FAIL'} "
        f"({updated_count}/{len(lora_before)} params changed)"
    )

    return checks


def run_full_ft(parallelism_config, tokenizer, train_dataset, eval_dataset, output_dir, rank, local_rank, world_size):
    """Full fine-tune with EP + FA2."""
    log(f"  Loading model (full FT, attn={ATTN_IMPL})...")
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=ATTN_IMPL,
        use_liger_kernel=True,
    )
    total_params = sum(p.numel() for p in model.parameters())
    log(f"  Model loaded: {total_params / 1e9:.2f}B params, GPU: {gpu_mem_gb():.1f}GB")

    sft_config = SFTConfig(
        output_dir=os.path.join(output_dir, "full_train"),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,  # already applied at load
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=True,  # Required for EP (inactive experts)
        fsdp="",  # Mixin handles FSDP wrapping
    )

    trainer = DistributedSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )

    log(f"  Training ({MAX_STEPS} steps)...")
    train_result = trainer.train()

    log("\n  --- Validation (full FT) ---")
    checks = _validate_training(train_result, trainer, MAX_STEPS, rank, local_rank, world_size)

    trainer.cleanup_ep()
    return checks, train_result.training_loss


def run_lora(parallelism_config, tokenizer, train_dataset, eval_dataset, output_dir, rank, local_rank, world_size):
    """LoRA with EP + FA2 (attention-only targets)."""
    log(f"  Loading model (LoRA, attn={ATTN_IMPL})...")
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=ATTN_IMPL,
        use_liger_kernel=True,
    )
    log(f"  GPU after load: {gpu_mem_gb():.1f}GB")

    log(f"  Applying LoRA (r=8, alpha=16, targets={LORA_TARGET_MODULES})...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"  Trainable: {trainable / 1e6:.2f}M ({100 * trainable / total:.2f}%)")

    lora_before = {
        name: _to_full_cpu(param.data)
        for name, param in model.named_parameters()
        if "lora_" in name and param.requires_grad
    }

    sft_config = SFTConfig(
        output_dir=os.path.join(output_dir, "lora_train"),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LORA_LEARNING_RATE,
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

    log(f"  Training ({MAX_STEPS} steps)...")
    train_result = trainer.train()

    log("\n  --- Validation (LoRA) ---")
    checks = _validate_training(train_result, trainer, MAX_STEPS, rank, local_rank, world_size)
    lora_checks = _validate_lora(model, lora_before)
    checks.update(lora_checks)

    trainer.cleanup_ep()
    return checks, train_result.training_loss


def run_qlora(parallelism_config, tokenizer, train_dataset, eval_dataset, output_dir, rank, local_rank, world_size):
    """Assert QLoRA + EP is rejected at load time.

    QLoRA + EP is unsupported: the EP lazy loader streams raw safetensors and rebuilds experts as
    plain ``Parameter`` objects, so bitsandbytes ``Params4bit`` are lost and PEFT's 4-bit adapter
    dispatch crashes on the missing ``weight.compress_statistics``. ``load_distributed_model``
    rejects the combo up front; verify that guard fires (removing it fails this sub-test). Use
    QLoRA with standard DDP/FSDP, or QLoRA + CP (which preserves quantization), instead.
    """
    log(f"  Expecting load_distributed_model to reject QLoRA + EP={EP_SIZE} (attn={ATTN_IMPL})...")

    model_config = ModelConfig(
        model_name_or_path=MODEL_NAME,
        use_peft=True,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        use_bnb_nested_quant=True,
        trust_remote_code=True,
        attn_implementation=ATTN_IMPL,
    )
    quantization_config = get_quantization_config(model_config)
    log(f"  Quantization: {quantization_config.quant_method}")

    try:
        load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=ATTN_IMPL,
            use_liger_kernel=False,  # Liger incompatible with quantized models
            quantization_config=quantization_config,
        )
    except ValueError as exc:
        msg = str(exc)
        ok = "not supported" in msg and "EP" in msg
        log(f"  Guard raised ValueError: {'PASS' if ok else 'FAIL'} -- {msg[:90]}")
        return {"qlora_ep_rejected": ok}, 0.0

    log("  Guard did NOT fire: FAIL (QLoRA + EP must be rejected at load time)")
    return {"qlora_ep_rejected": False}, 0.0


MODE_RUNNERS = {
    "full": ("Full Fine-Tune + EP + FA2", run_full_ft),
    "lora": ("LoRA + EP + FA2", run_lora),
    "qlora": ("QLoRA (4-bit NF4) + EP + FA2", run_qlora),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=list(MODE_RUNNERS.keys()), required=True, help="Training mode: full, lora, or qlora"
    )
    args, _ = parser.parse_known_args()

    rank, world_size, local_rank = init_distributed()
    PartialState()

    mode_label, mode_fn = MODE_RUNNERS[args.mode]

    log(f"\n{'#' * 70}")
    log(f"  SFT EP FA2 Test: {mode_label}")
    log(f"  World: {world_size}, EP: {EP_SIZE}, Attn: {ATTN_IMPL}")
    log(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"  Steps: {MAX_STEPS}, Batch: {BATCH_SIZE}, Seq len: {MAX_SEQ_LENGTH}")
    log(f"{'#' * 70}")

    if world_size < EP_SIZE:
        log(f"\nERROR: Need at least {EP_SIZE} GPUs, got {world_size}")
        teardown_distributed()
        return 1

    output_dir, cache_dir = setup_cache_dirs(f"sft_ep_fa2_{args.mode}", rank)

    try:
        log("\nEnsuring model is downloaded...")
        ensure_model_downloaded(MODEL_NAME, rank)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        log("\nCreating synthetic datasets...")
        train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
        eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)
        log(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        parallelism_config = ParallelismConfig(ep_size=EP_SIZE)
        log(f"Parallelism: {parallelism_config.summary()}")

        checks, loss = mode_fn(
            parallelism_config,
            tokenizer,
            train_dataset,
            eval_dataset,
            output_dir,
            rank,
            local_rank,
            world_size,
        )

        all_passed = all(checks.values())
        log(f"\n{'#' * 70}")
        log(f"  {mode_label}: {'PASSED' if all_passed else 'FAILED'} (loss={loss:.6f})")
        if not all_passed:
            log(f"  Failed: {[k for k, v in checks.items() if not v]}")
        log(f"{'#' * 70}\n")

        cleanup_dirs(output_dir, cache_dir)
        barrier()
        teardown_distributed()
        return 0 if all_passed else 1

    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        if rank == 0:
            traceback.print_exc()
        cleanup_dirs(output_dir, cache_dir)
        teardown_distributed()
        return 1


if __name__ == "__main__":
    sys.exit(main())
