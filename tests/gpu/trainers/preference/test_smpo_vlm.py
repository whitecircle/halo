#!/usr/bin/env python
"""Test: SmoothMarginPOTrainer on a VLM (Qwen3-VL-2B) — single-GPU vision-SMPO smoke.

A ProcessorMixin ``processing_class`` flips the trainer into VLM mode: rows are chat-templated with
the prefix-strip invariant, images are processed at collation (``DataCollatorForVLMSMPO``), and the
vision tensors ride the chosen|rejected concatenated forward duplicated row-major. Validates a real
model forward/backward on that path — finite loss over several steps, margins logged.

Run with 1 GPU:
    torchrun --nproc_per_node=1 tests/gpu/trainers/preference/test_smpo_vlm.py
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.configs.smpo_config import SmoothMarginPOConfig
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.preference.smpo import SmoothMarginPOTrainer
from tests.common.datasets import create_vlm_preference_dataset
from tests.common.harness import gpu_test_main
from tests.common.utils import log

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
NUM_TRAIN_SAMPLES = 16
NUM_TRAIN_STEPS = 4


def run(ctx):
    log("=" * 70)
    log("VLM SMPO Trainer Test (Qwen3-VL-2B)")
    log("=" * 70)

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = create_vlm_preference_dataset(NUM_TRAIN_SAMPLES)
    log(f"Train samples: {len(train_dataset)}")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa"
    )

    config = SmoothMarginPOConfig(
        output_dir=ctx.output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        beta=1.0,
        target_margin=0.4,
        bf16=True,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_length=1024,
        max_prompt_length=512,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        fsdp="",
    )

    trainer = SmoothMarginPOTrainer(
        model,
        args=config,
        train_dataset=train_dataset,
        processing_class=processor,
        parallelism_config=ParallelismConfig(),
    )

    assert trainer.is_vlm, "SMPO did not enter VLM mode for a processor processing_class"

    result = trainer.train()
    loss = result.training_loss
    log(f"Training loss: {loss:.6f}")

    logged = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    margins = [e for e in trainer.state.log_history if any(k.startswith("rewards/") for k in e)]
    checks = {
        "training_loss_finite": bool(torch.isfinite(torch.tensor(loss))),
        "enough_steps_logged": len(logged) >= 2,
        # No rewards/* metrics means the SMPO loss path did not run.
        "reward_metrics_logged": bool(margins),
    }

    return {"checks": checks, "metrics": {"training_loss": loss}}


main = gpu_test_main(min_world_size=1, prefix="test_smpo_vlm", partial_state=False)(run)

if __name__ == "__main__":
    main()
