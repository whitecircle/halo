#!/usr/bin/env python
"""Test: DistributedDPOTrainer on a VLM (Qwen3-VL-2B) — single-GPU vision-DPO smoke.

Validates that VLM DPO works end-to-end under TRL 1.6: passing a processor as processing_class
flips TRL into vision mode (_is_vlm) and an images-bearing preference dataset auto-selects TRL's
DataCollatorForVisionPreference, threading pixel_values to the model. Asserts training completes
with finite losses (so the vision path actually ran, not just the text arm).

Run:
    torchrun --nproc_per_node=1 tests/gpu/trainers/preference/test_dpo_vlm.py
"""

import torch
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import DPOConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.preference.dpo import DistributedDPOTrainer
from tests.common.datasets import create_vlm_preference_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_VL_2B
from tests.common.utils import log

MODEL_NAME = QWEN3_VL_2B
NUM_TRAIN_SAMPLES = 16
NUM_TRAIN_STEPS = 4


def run(ctx) -> dict:
    log("=" * 70)
    log("VLM DPO Trainer Test (Qwen3-VL-2B)")
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

    config = DPOConfig(
        output_dir=ctx.output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        beta=0.1,
        bf16=True,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_length=1024,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        fsdp="",
    )

    # PEFT reference so no second full model copy is needed for the smoke.
    trainer = DistributedDPOTrainer(
        model,
        args=config,
        ref_model=None,
        train_dataset=train_dataset,
        processing_class=processor,
        peft_config=LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
        parallelism_config=ParallelismConfig(),
    )

    assert getattr(trainer, "_is_vlm", False), "DPO did not enter VLM mode for a processor processing_class"

    result = trainer.train()
    loss = result.training_loss
    log(f"Training loss: {loss:.6f}")
    logged = [e["loss"] for e in trainer.state.log_history if "loss" in e]

    return {
        "checks": {
            "training_loss_finite": bool(torch.isfinite(torch.tensor(loss))),
            "enough_steps_logged": len(logged) >= 2,
        }
    }


main = gpu_test_main(min_world_size=1, prefix="test_dpo_vlm", partial_state=False)(run)

if __name__ == "__main__":
    main()
