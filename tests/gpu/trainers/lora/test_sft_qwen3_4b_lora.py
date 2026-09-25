#!/usr/bin/env python
"""
SFT Trainer test: LoRA and QLoRA on Qwen3-4B with FSDP (+ LoRA/QLoRA TP rejection).

Tests LoRA/QLoRA under FSDP (data parallel) with checkpoint save + reload, and asserts that the
unsupported TP combinations are rejected at trainer construction.

Tests:
  1. LoRA  + FSDP  -> train -> save -> verify checkpoint -> reload adapter
  2. QLoRA + FSDP  -> train -> save -> verify checkpoint -> reload adapter
  3. LoRA  + TP=2  -> REJECTED at construction (adapters not in the TP DTensor graph)
  4. QLoRA + TP=2  -> SKIPPED (4-bit quantization incompatible with DTensor TP)

Checkpoint verification:
  - Adapter files exist (adapter_config.json, adapter_model.safetensors)
  - PeftModel.from_pretrained() on a fresh base restores every trained LoRA tensor
  - Reloaded model produces finite logits on a sample input

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_sft_qwen3_4b_lora.py
"""

import math
import os
import traceback
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer
from trl import ModelConfig, SFTConfig, get_quantization_config

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.loading.peft_setup import setup_peft_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_4B_INSTRUCT
from tests.common.peft_helpers import adapter_save_checks, snapshot_adapters, unwrap, verify_adapter_reload
from tests.common.utils import cleanup_memory, gpu_mem_gb, log

# Configuration

MODEL_NAME = QWEN3_4B_INSTRUCT
MAX_STEPS = 5
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 4096
LEARNING_RATE = 1e-4
NUM_TRAIN_SAMPLES = 256
NUM_EVAL_SAMPLES = 32
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

    all_finite = all(math.isfinite(sl) for sl in step_losses)
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


# Mode runner


def run_mode(
    mode_name: str,
    model_config: ModelConfig,
    sft_config: SFTConfig,
    parallelism_config: ParallelismConfig,
    tokenizer,
    train_dataset,
    eval_dataset,
    rank: int,
    local_rank: int,
    save_dir: str,
    expect_rejection: bool = False,
) -> tuple[bool, str]:
    """Run one SFT training mode following the sft.py pipeline.

    ``expect_rejection=True`` flips the mode into a guard test: instead of training, it asserts
    that ``DistributedSFTTrainer`` construction raises ``ValueError`` (used for the unsupported
    LoRA+TP combination — the adapter would train rank-inconsistent).
    """
    model = None
    trainer = None

    try:
        # Step 1: Quantization config
        quantization_config = get_quantization_config(model_config)
        if quantization_config is not None:
            log(f"  Quantization: {quantization_config.quant_method}")

        # Step 2: Load model
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

        # Disable Liger in SFTConfig — already applied during model loading
        if sft_config.use_liger_kernel:
            sft_config.use_liger_kernel = False

        # Step 3: Setup PEFT
        peft_config = setup_peft_model(PEFT_ARGS, model, model_config, "CAUSAL_LM")
        if peft_config is not None:
            log(f"  PEFT config: r={peft_config.r}, alpha={peft_config.lora_alpha}")
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log(
                f"  Trainable params: {trainable / 1e6:.2f}M / {total_params / 1e6:.1f}M "
                f"({100 * trainable / total_params:.2f}%)"
            )

        # Step 4: Create trainer (or assert rejection for unsupported LoRA+TP)
        if expect_rejection:
            raised, err = False, ""
            try:
                DistributedSFTTrainer(
                    model=model,
                    args=sft_config,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    processing_class=tokenizer,
                    peft_config=peft_config,
                    parallelism_config=parallelism_config,
                )
            except ValueError as e:
                raised, err = True, str(e)
            ok = raised and (("Tensor Parallelism" in err) or ("tp_size" in err))
            log(f"  LoRA+TP rejected at construction: {'PASS' if ok else 'FAIL'}")
            return ok, ("rejected at construction" if ok else "NOT rejected — guard missing?")

        trainer = DistributedSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            parallelism_config=parallelism_config,
        )

        # Step 5: Train
        log(f"  Training for {MAX_STEPS} steps...")
        train_result = trainer.train()

        # Step 6: Save model
        log(f"  Saving model to {save_dir}...")
        trainer.save_model(save_dir)
        barrier()
        trained_lora = snapshot_adapters(unwrap(trainer.model), expert_lora=False)

        # Step 7: Validate training
        log(f"\n  --- Training Validation ({mode_name}) ---")
        checks, step_losses = _validate_training(train_result, trainer, MAX_STEPS)
        log(f"  Per-step losses: {[f'{sl:.4f}' for sl in step_losses]}")

        # Step 8: Verify checkpoint files
        log(f"\n  --- Checkpoint Verification ({mode_name}) ---")
        ckpt_checks = adapter_save_checks(save_dir, rank)
        checks.update(ckpt_checks)

        # Step 9: Reload checkpoint and verify inference
        log(f"\n  --- Checkpoint Reload ({mode_name}) ---")
        # Must delete trainer/model before reloading to free GPU memory
        del trainer
        trainer = None
        del model
        model = None
        cleanup_memory()
        barrier()

        reload_checks = verify_adapter_reload(
            save_dir,
            trained_lora,
            model_name=MODEL_NAME,
            tokenizer=tokenizer,
            rank=rank,
            local_rank=local_rank,
            quantization_config=quantization_config,
        )
        checks.update(reload_checks)

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


def run(ctx) -> dict:
    rank, local_rank, base_output_dir = ctx.rank, ctx.local_rank, ctx.output_dir

    log(f"\n{'#' * 70}")
    log("  SFT LoRA/QLoRA Test on Qwen3-4B (FSDP + TP)")
    log(f"  World size: {ctx.world_size}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"  Max steps: {MAX_STEPS}, Batch size: {BATCH_SIZE}, Seq length: {MAX_SEQ_LENGTH}")
    log("  Packing: enabled")
    log(f"{'#' * 70}")

    # Ensure model is downloaded before all processes try to load
    ensure_model_downloaded(MODEL_NAME, rank)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1)
    log(f"  Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    results: dict[str, tuple[bool, str]] = {}

    # Shared SFTConfig kwargs
    shared_sft_kwargs = {
        "max_steps": MAX_STEPS,
        "per_device_train_batch_size": BATCH_SIZE,
        "bf16": True,
        "gradient_checkpointing": True,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "logging_nan_inf_filter": False,
        "max_length": MAX_SEQ_LENGTH,
        "packing": True,
        "dataloader_drop_last": True,
        "fsdp": "",
    }

    # Shared LoRA ModelConfig kwargs
    lora_model_kwargs = {
        "model_name_or_path": MODEL_NAME,
        "use_peft": True,
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "lora_target_modules": LORA_TARGET_MODULES,
        "lora_task_type": "CAUSAL_LM",
        "trust_remote_code": True,
        "attn_implementation": "flash_attention_2",
    }

    # ── Test 1: LoRA + FSDP ──────────────────────────────────────────────
    log(f"\n{'=' * 70}")
    log("  TEST 1: LoRA + FSDP (Qwen3-4B)")
    log(f"{'=' * 70}")

    lora_fsdp_model_config = ModelConfig(**lora_model_kwargs)
    lora_fsdp_sft_config = SFTConfig(
        output_dir=os.path.join(base_output_dir, "lora_fsdp_train"),
        learning_rate=LEARNING_RATE,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=True,
        **shared_sft_kwargs,
    )
    parallelism_fsdp = ParallelismConfig()

    success, detail = run_mode(
        "lora_fsdp",
        lora_fsdp_model_config,
        lora_fsdp_sft_config,
        parallelism_fsdp,
        tokenizer,
        train_dataset,
        eval_dataset,
        rank,
        local_rank,
        save_dir=os.path.join(base_output_dir, "lora_fsdp_save"),
    )
    results["lora_fsdp"] = (success, detail)

    # ── Test 2: QLoRA + FSDP ─────────────────────────────────────────────
    log(f"\n{'=' * 70}")
    log("  TEST 2: QLoRA (4-bit) + FSDP (Qwen3-4B)")
    log(f"{'=' * 70}")

    qlora_fsdp_model_config = ModelConfig(
        **lora_model_kwargs,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        use_bnb_nested_quant=True,
    )
    qlora_fsdp_sft_config = SFTConfig(
        output_dir=os.path.join(base_output_dir, "qlora_fsdp_train"),
        learning_rate=LEARNING_RATE,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=False,  # Liger incompatible with quantized models
        **shared_sft_kwargs,
    )

    success, detail = run_mode(
        "qlora_fsdp",
        qlora_fsdp_model_config,
        qlora_fsdp_sft_config,
        parallelism_fsdp,
        tokenizer,
        train_dataset,
        eval_dataset,
        rank,
        local_rank,
        save_dir=os.path.join(base_output_dir, "qlora_fsdp_save"),
    )
    results["qlora_fsdp"] = (success, detail)

    # ── Test 3: LoRA + TP=2 must be REJECTED ─────────────────────────────
    # TP shards attention as DTensors; PEFT adapters are plain tensors outside the TP graph, so
    # the adapter would train rank-inconsistent. The trainer must fail fast at construction.
    log(f"\n{'=' * 70}")
    log("  TEST 3: LoRA + TP=2 must be REJECTED (Qwen3-4B)")
    log(f"{'=' * 70}")

    parallelism_tp2 = ParallelismConfig(tp_size=2)

    lora_tp_model_config = ModelConfig(**lora_model_kwargs)
    lora_tp_sft_config = SFTConfig(
        output_dir=os.path.join(base_output_dir, "lora_tp2_train"),
        learning_rate=LEARNING_RATE,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=True,
        **shared_sft_kwargs,
    )

    success, detail = run_mode(
        "lora_tp2_rejected",
        lora_tp_model_config,
        lora_tp_sft_config,
        parallelism_tp2,
        tokenizer,
        train_dataset,
        eval_dataset,
        rank,
        local_rank,
        save_dir=os.path.join(base_output_dir, "lora_tp2_save"),
        expect_rejection=True,
    )
    results["lora_tp2_rejected"] = (success, detail)

    # ── Test 4: QLoRA + TP=2 ─────────────────────────────────────────────
    # QLoRA (4-bit quantization via BitsAndBytes) is incompatible with TP (DTensor).
    # DTensor requires sharding tensors across devices, but BnB Linear4bit layers
    # use custom integer storage that cannot be split by DTensor's ColwiseParallel
    # and RowwiseParallel. This is a known limitation.
    log(f"\n{'=' * 70}")
    log("  TEST 4: QLoRA + TP=2 -- SKIPPED (4-bit quantization incompatible with DTensor TP)")
    log(f"{'=' * 70}")

    for name, (passed, detail) in results.items():
        log(f"  {name:20s} {'PASSED' if passed else 'FAILED'} -- {detail}")

    return {"checks": {name: passed for name, (passed, _) in results.items()}}


main = gpu_test_main(min_world_size=2, prefix="test_sft_qwen3_4b_lora")(run)

if __name__ == "__main__":
    main()
