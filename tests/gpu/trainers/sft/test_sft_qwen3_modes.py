#!/usr/bin/env python
"""
SFT Trainer test: Full Fine-Tune, LoRA, QLoRA, TP=2, CP=2 on Qwen3-0.6B.

Mirrors the sft.py training pipeline: load_distributed_model → setup_peft_model
→ DistributedSFTTrainer → train → save_model. Tests all parameter modes
and parallelism configurations with packing and checkpoint verification.

Run with torchrun (all modes including TP/CP):
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_qwen3_modes.py

Run with accelerate (FSDP, skips TP/CP/QLoRA):
    accelerate launch --config_file launcher-configs/accelerate/multigpu_dp_config.yaml \
        --num_processes=2 tests/gpu/trainers/sft/test_sft_qwen3_modes.py

Run single-GPU (skips TP/CP):
    torchrun --nproc_per_node=1 tests/gpu/trainers/sft/test_sft_qwen3_modes.py
"""

import math
import os
import traceback
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer
from trl import ModelConfig, SFTConfig, get_quantization_config

from src.data.collators.factory import select_data_collator
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.loading.peft_setup import setup_peft_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.env import is_accelerate_fsdp_launch
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.utils import cleanup_memory, gpu_mem_gb, log

# Configuration

MODEL_NAME = QWEN3_0_6B
MAX_STEPS = 5
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 8192
LEARNING_RATE = 2e-5
LORA_LEARNING_RATE = 1e-4
NUM_TRAIN_SAMPLES = 1024
NUM_EVAL_SAMPLES = 128
SEED = 42

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Minimal args stub for setup_peft_model (only needs freeze/unfreeze patterns)
PEFT_ARGS = SimpleNamespace(
    unfreeze_layers_patterns=None,
    freeze_layers_patterns=None,
)


# Helpers


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

    # A band, not `< 100`: no finite LM loss on this model reaches 100, so that bound was strictly
    # weaker than the finiteness check beside it and could not fail. The floor catches a collapsed
    # or masked-away objective, the ceiling an untrained/garbage forward.
    loss_reasonable = 0.05 < training_loss < 20.0
    checks["loss_reasonable"] = loss_reasonable
    log(f"  Loss in band (0.05, 20): {'PASS' if loss_reasonable else 'FAIL'} ({training_loss:.4f})")

    return checks, step_losses


def _verify_checkpoint(save_dir: str, mode: str, rank: int) -> dict[str, bool]:
    """Verify saved model files exist. Only rank 0 checks."""
    if rank != 0:
        return {}

    checks = {}
    dir_exists = os.path.isdir(save_dir)
    checks["save_dir_exists"] = dir_exists
    log(f"  Save dir exists: {'PASS' if dir_exists else 'FAIL'} ({save_dir})")

    if not dir_exists:
        return checks

    contents = os.listdir(save_dir)
    log(f"  Save contents: {sorted(contents)}")

    if mode == "full":
        has_model = any(f.startswith("model") and f.endswith(".safetensors") for f in contents) or any(
            f.startswith("pytorch_model") and f.endswith(".bin") for f in contents
        )
        checks["has_model_weights"] = has_model
        log(f"  Has model weights: {'PASS' if has_model else 'FAIL'}")
    else:
        has_adapter = "adapter_config.json" in contents or any("adapter" in f for f in contents)
        checks["has_adapter"] = has_adapter
        log(f"  Has adapter files: {'PASS' if has_adapter else 'FAIL'}")

    if mode == "full":
        has_config = "config.json" in contents
        checks["has_config"] = has_config
        log(f"  Has config.json: {'PASS' if has_config else 'FAIL'}")
    else:
        # PEFT saves adapter_config.json, not config.json
        has_adapter_config = "adapter_config.json" in contents
        checks["has_adapter_config"] = has_adapter_config
        log(f"  Has adapter_config.json: {'PASS' if has_adapter_config else 'FAIL'}")

    return checks


# Mode runner — mirrors sft.py: load_distributed_model → setup_peft_model
#                                → DistributedSFTTrainer → train → save_model


def run_mode(
    mode_name: str,
    model_config: ModelConfig,
    sft_config: SFTConfig,
    parallelism_config: ParallelismConfig,
    tokenizer,
    train_dataset,
    eval_dataset,
    rank: int,
    save_dir: str,
    data_collator=None,
) -> tuple[bool, str]:
    """Run one SFT training mode following the sft.py pipeline."""
    model = None
    trainer = None

    try:
        # --- Step 1: Quantization config (mirrors sft.py line 148) ---
        quantization_config = get_quantization_config(model_config)
        if quantization_config is not None:
            log(f"  Quantization: {quantization_config.quant_method}")

        # --- Step 2: Load model (mirrors sft.py line 151) ---
        log("  Loading model via load_distributed_model...")
        model, _ = load_distributed_model(
            model_name_or_path=model_config.model_name_or_path,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=model_config.trust_remote_code,
            attn_implementation=model_config.attn_implementation,
            use_liger_kernel=sft_config.use_liger_kernel,
            quantization_config=quantization_config,
        )
        total_params = sum(p.numel() for p in model.parameters())
        log(f"  Model loaded: {total_params / 1e6:.1f}M params, GPU: {gpu_mem_gb():.2f} GB")

        # Disable Liger in SFTConfig — already applied during model loading.
        # Prevents TRL from re-applying with different defaults (sft.py line 179).
        if sft_config.use_liger_kernel:
            sft_config.use_liger_kernel = False

        # --- Step 3: Setup PEFT (mirrors sft.py line 186) ---
        peft_config = setup_peft_model(PEFT_ARGS, model, model_config, "CAUSAL_LM")
        if peft_config is not None:
            log(f"  PEFT config: r={peft_config.r}, alpha={peft_config.lora_alpha}")
        else:
            log("  PEFT: disabled (full fine-tune)")

        # --- Step 4: Create trainer (mirrors sft.py line 365) ---
        trainer_kwargs = {}
        if data_collator is not None:
            trainer_kwargs["data_collator"] = data_collator
        trainer = DistributedSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            parallelism_config=parallelism_config,
            **trainer_kwargs,
        )

        # --- Step 5: Train (mirrors sft.py line 394) ---
        log(f"  Training for {MAX_STEPS} steps...")
        train_result = trainer.train()

        # --- Step 6: Save model (mirrors sft.py line 399) ---
        log(f"  Saving model to {save_dir}...")
        trainer.save_model(save_dir)
        barrier()

        # --- Validate ---
        log(f"\n  --- Validation ({mode_name}) ---")
        checks, step_losses = _validate_training(train_result, trainer, MAX_STEPS)
        log(f"  Per-step losses: {[f'{l:.4f}' for l in step_losses]}")

        ckpt_checks = _verify_checkpoint(save_dir, mode_name, rank)
        checks.update(ckpt_checks)

        all_passed = all(checks.values())
        detail = f"loss={train_result.training_loss:.6f}"
        return all_passed, detail

    except Exception as e:
        log(f"  FAILED with exception: {e}")
        traceback.print_exc()
        return False, f"Exception: {e}"

    finally:
        del trainer
        del model
        cleanup_memory()
        barrier()


# Main


def run(ctx) -> dict:
    """Run all SFT mode tests."""
    rank = ctx.rank
    world_size = ctx.world_size
    base_output_dir = ctx.output_dir

    log(f"\n{'#' * 70}")
    log("  SFT Training Modes Test (Full / LoRA / QLoRA)")
    log(f"  World size: {world_size}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  Max steps: {MAX_STEPS}, Batch size: {BATCH_SIZE}, Seq length: {MAX_SEQ_LENGTH}")
    log("  Packing: enabled")
    log(f"{'#' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1)
    log(f"  Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    parallelism_config = ParallelismConfig()
    results: dict[str, tuple[bool, str]] = {}

    # Shared SFTConfig kwargs — mirrors sft.py settings
    shared_sft_kwargs = {
        "max_steps": MAX_STEPS,
        "per_device_train_batch_size": BATCH_SIZE,
        "bf16": True,
        "gradient_checkpointing": True,
        "logging_steps": 1,
        "save_strategy": "no",  # Manual save after training (avoids FSDP auto-save issues)
        "report_to": "none",
        "max_length": MAX_SEQ_LENGTH,
        "packing": True,
        "dataloader_drop_last": True,
        "fsdp": "",  # Mixin handles FSDP wrapping (sft.py line 353)
    }

    # ── Test 1: Full Fine-Tune ──────────────────────────────────────────
    log(f"\n{'=' * 70}")
    log("  TEST 1: Full Fine-Tune")
    log(f"{'=' * 70}")

    full_model_config = ModelConfig(
        model_name_or_path=MODEL_NAME,
        use_peft=False,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    full_sft_config = SFTConfig(
        output_dir=os.path.join(base_output_dir, "full_train"),
        learning_rate=LEARNING_RATE,
        use_liger_kernel=True,
        **shared_sft_kwargs,
    )

    success, detail = run_mode(
        "full",
        full_model_config,
        full_sft_config,
        parallelism_config,
        tokenizer,
        train_dataset,
        eval_dataset,
        rank,
        save_dir=os.path.join(base_output_dir, "full_save"),
    )
    results["full_finetune"] = (success, detail)

    # ── Test 2: LoRA ────────────────────────────────────────────────────
    log(f"\n{'=' * 70}")
    log("  TEST 2: LoRA")
    log(f"{'=' * 70}")

    lora_model_config = ModelConfig(
        model_name_or_path=MODEL_NAME,
        use_peft=True,
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        lora_target_modules=LORA_TARGET_MODULES,
        lora_task_type="CAUSAL_LM",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    lora_sft_config = SFTConfig(
        output_dir=os.path.join(base_output_dir, "lora_train"),
        learning_rate=LORA_LEARNING_RATE,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=True,
        **shared_sft_kwargs,
    )

    success, detail = run_mode(
        "lora",
        lora_model_config,
        lora_sft_config,
        parallelism_config,
        tokenizer,
        train_dataset,
        eval_dataset,
        rank,
        save_dir=os.path.join(base_output_dir, "lora_save"),
    )
    results["lora"] = (success, detail)

    # ── Test 3: QLoRA ───────────────────────────────────────────────────
    # QLoRA is incompatible with accelerate-managed FSDP (integer dtype tensors
    # from 4-bit quantization can't be flattened). Skip when FSDP is active.
    accelerate_fsdp = is_accelerate_fsdp_launch()
    if accelerate_fsdp:
        log(f"\n{'=' * 70}")
        log("  TEST 3: QLoRA (4-bit) -- SKIPPED (incompatible with accelerate FSDP)")
        log(f"{'=' * 70}")
        results["qlora"] = (None, "skipped (accelerate FSDP)")
    else:
        log(f"\n{'=' * 70}")
        log("  TEST 3: QLoRA (4-bit)")
        log(f"{'=' * 70}")

        qlora_model_config = ModelConfig(
            model_name_or_path=MODEL_NAME,
            use_peft=True,
            lora_r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            lora_target_modules=LORA_TARGET_MODULES,
            lora_task_type="CAUSAL_LM",
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            use_bnb_nested_quant=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        qlora_sft_config = SFTConfig(
            output_dir=os.path.join(base_output_dir, "qlora_train"),
            learning_rate=LORA_LEARNING_RATE,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            use_liger_kernel=False,  # Liger incompatible with quantized models
            **shared_sft_kwargs,
        )

        success, detail = run_mode(
            "qlora",
            qlora_model_config,
            qlora_sft_config,
            parallelism_config,
            tokenizer,
            train_dataset,
            eval_dataset,
            rank,
            save_dir=os.path.join(base_output_dir, "qlora_save"),
        )
        results["qlora"] = (success, detail)

    # ── Test 4: TP=2 (Tensor Parallelism) ────────────────────────────────
    # TP and CP require torchrun (not accelerate) and world_size >= 2
    can_run_tp_cp = world_size >= 2 and not accelerate_fsdp
    if not can_run_tp_cp:
        log(f"\n{'=' * 70}")
        reason = "accelerate FSDP" if accelerate_fsdp else f"world_size={world_size}"
        log(f"  TEST 4: TP=2 -- SKIPPED ({reason})")
        log(f"{'=' * 70}")
        results["tp2"] = (None, f"skipped ({reason})")
    else:
        log(f"\n{'=' * 70}")
        log("  TEST 4: TP=2 (Tensor Parallelism)")
        log(f"{'=' * 70}")

        tp_parallelism = ParallelismConfig(tp_size=2)

        tp_model_config = ModelConfig(
            model_name_or_path=MODEL_NAME,
            use_peft=False,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        tp_sft_config = SFTConfig(
            output_dir=os.path.join(base_output_dir, "tp2_train"),
            learning_rate=LEARNING_RATE,
            use_liger_kernel=True,
            **shared_sft_kwargs,
        )

        success, detail = run_mode(
            "full",
            tp_model_config,
            tp_sft_config,
            tp_parallelism,
            tokenizer,
            train_dataset,
            eval_dataset,
            rank,
            save_dir=os.path.join(base_output_dir, "tp2_save"),
        )
        results["tp2"] = (success, detail)

    # ── Test 5: CP=2 (Context Parallelism) ─────────────────────────────
    if not can_run_tp_cp:
        log(f"\n{'=' * 70}")
        reason = "accelerate FSDP" if accelerate_fsdp else f"world_size={world_size}"
        log(f"  TEST 5: CP=2 -- SKIPPED ({reason})")
        log(f"{'=' * 70}")
        results["cp2"] = (None, f"skipped ({reason})")
    else:
        log(f"\n{'=' * 70}")
        log("  TEST 5: CP=2 (Context Parallelism)")
        log(f"{'=' * 70}")

        cp_parallelism = ParallelismConfig(cp_size=2)

        cp_model_config = ModelConfig(
            model_name_or_path=MODEL_NAME,
            use_peft=False,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        # Ulysses has no per-document boundaries, so packing is rejected under CP; CP instead needs
        # padded sequences whose length divides cp_size. Both come from the production factory.
        cp_sft_config = SFTConfig(
            output_dir=os.path.join(base_output_dir, "cp2_train"),
            learning_rate=LEARNING_RATE,
            use_liger_kernel=True,
            **{**shared_sft_kwargs, "packing": False},
        )

        cp_collator = select_data_collator(
            tokenizer=tokenizer,
            use_context_parallel=True,
            pad_to_multiple_of=2,
        )

        success, detail = run_mode(
            "full",
            cp_model_config,
            cp_sft_config,
            cp_parallelism,
            tokenizer,
            train_dataset,
            eval_dataset,
            rank,
            save_dir=os.path.join(base_output_dir, "cp2_save"),
            data_collator=cp_collator,
        )
        results["cp2"] = (success, detail)

    # ── Summary ─────────────────────────────────────────────────────────
    log(f"\n{'#' * 70}")
    log("  FINAL RESULTS")
    log(f"{'#' * 70}")
    # None == did not run. Reporting a skipped mode as PASSED claims coverage the run never had.
    for name, (passed, detail) in results.items():
        status = "SKIPPED" if passed is None else ("PASSED" if passed else "FAILED")
        log(f"  {name:20s} {status} -- {detail}")
    log(f"{'#' * 70}")

    # A mode that did not run contributes no check — reporting it as PASSED would claim
    # coverage the run never had.
    return {"checks": {name: passed for name, (passed, _) in results.items() if passed is not None}}


main = gpu_test_main(min_world_size=1, prefix="test_sft_modes")(run)

if __name__ == "__main__":
    main()
