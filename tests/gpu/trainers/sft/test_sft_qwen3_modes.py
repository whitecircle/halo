#!/usr/bin/env python
"""SFT on Qwen3-0.6B through the sft.py pipeline: full fine-tune, LoRA, QLoRA, TP=2 and CP=2, with packing and save.

Under ``accelerate launch`` it skips TP, CP and QLoRA; on one GPU it skips TP and CP.
"""

import os
import traceback
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer
from trl import SFTConfig, get_quantization_config

from halo.configs.script_arguments import ModelArguments
from halo.data.collators.factory import select_data_collator
from halo.distributed.parallelism_config import ParallelismConfig
from halo.env import is_accelerate_fsdp_launch
from halo.model_loading.parallel.loader import load_distributed_model
from halo.model_loading.parallel.peft_setup import configure_adapter_training
from halo.runtime.collectives import barrier
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.adapters import adapter_save_checks
from tests.common.checkpoint_io import model_save_checks
from tests.common.datasets import create_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory, gpu_mem_gb
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log
from tests.common.training_checks import LM_TRAINING_LOSS_BAND, training_run_checks

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

# configure_adapter_training only needs the freeze/unfreeze patterns.
PEFT_ARGS = SimpleNamespace(
    unfreeze_layers_patterns=None,
    freeze_layers_patterns=None,
)


def run_mode(
    mode_name: str,
    model_config: ModelArguments,
    sft_config: SFTConfig,
    parallelism_config: ParallelismConfig,
    tokenizer,
    train_dataset,
    eval_dataset,
    rank: int,
    save_dir: str,
    data_collator=None,
) -> tuple[bool, str]:
    model = None
    trainer = None

    try:
        quantization_config = get_quantization_config(model_config)
        if quantization_config is not None:
            log(f"  Quantization: {quantization_config.quant_method}")

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

        # Already applied during model loading; keeps TRL from re-applying with different defaults.
        if sft_config.use_liger_kernel:
            sft_config.use_liger_kernel = False

        peft_config = configure_adapter_training(PEFT_ARGS, model, model_config, "CAUSAL_LM")
        if peft_config is not None:
            log(f"  PEFT config: r={peft_config.r}, alpha={peft_config.lora_alpha}")
        else:
            log("  PEFT: disabled (full fine-tune)")

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

        log(f"  Training for {MAX_STEPS} steps...")
        train_result = trainer.train()

        log(f"  Saving model to {save_dir}...")
        trainer.save_model(save_dir)
        barrier()

        log(f"\n  --- Validation ({mode_name}) ---")
        checks = training_run_checks(train_result, trainer, MAX_STEPS, loss_band=LM_TRAINING_LOSS_BAND)

        checks |= model_save_checks(save_dir, rank) if mode_name == "full" else adapter_save_checks(save_dir, rank)

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

    # Mirrors sft.py settings.
    shared_sft_kwargs = {
        "max_steps": MAX_STEPS,
        "per_device_train_batch_size": BATCH_SIZE,
        "bf16": True,
        "gradient_checkpointing": True,
        "logging_steps": 1,
        "save_strategy": "no",  # Manual save after training (avoids FSDP auto-save issues).
        "report_to": "none",
        "logging_nan_inf_filter": False,
        "max_length": MAX_SEQ_LENGTH,
        "packing": True,
        "dataloader_drop_last": True,
    }

    log(f"\n{'=' * 70}")
    log("  TEST 1: Full Fine-Tune")
    log(f"{'=' * 70}")

    full_model_config = ModelArguments(
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

    log(f"\n{'=' * 70}")
    log("  TEST 2: LoRA")
    log(f"{'=' * 70}")

    lora_model_config = ModelArguments(
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

    # Accelerate-managed FSDP can't flatten the integer tensors of 4-bit quantization.
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

        qlora_model_config = ModelArguments(
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
            use_liger_kernel=False,  # Liger is incompatible with quantized models.
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

    # TP and CP require torchrun and world_size >= 2.
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

        tp_model_config = ModelArguments(
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

        cp_model_config = ModelArguments(
            model_name_or_path=MODEL_NAME,
            use_peft=False,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        # Ulysses has no per-document boundaries, so CP rejects packing and needs padded sequences whose length divides
        # cp_size.
        cp_sft_config = SFTConfig(
            output_dir=os.path.join(base_output_dir, "cp2_train"),
            learning_rate=LEARNING_RATE,
            use_liger_kernel=True,
            **{**shared_sft_kwargs, "packing": False},
        )

        cp_collator = select_data_collator(
            train_on_completions_only=False,
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

    log(f"\n{'#' * 70}")
    log("  FINAL RESULTS")
    log(f"{'#' * 70}")
    # None means did not run: reporting a skipped mode as passed claims coverage the run never had.
    for name, (passed, detail) in results.items():
        status = "SKIPPED" if passed is None else ("PASSED" if passed else "FAILED")
        log(f"  {name:20s} {status} -- {detail}")
    log(f"{'#' * 70}")

    return {"checks": {name: passed for name, (passed, _) in results.items() if passed is not None}}


main = gpu_test_main(min_world_size=1, prefix="test_sft_modes")(run)

if __name__ == "__main__":
    main()
