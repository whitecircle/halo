#!/usr/bin/env python
"""SMPO TEXT run on a natively-multimodal checkpoint (Qwen3-VL-2B) — the run-verdict path.

The dispatch rule under test end-to-end: the model CLASS follows the checkpoint
(``AutoModelForImageTextToText``), the DATA path follows the run (``is_vlm=False`` for text pairs).
Forcing this shape onto the VLM branch instead would deny it CP, PP and ``padding_free`` and
template rows through the processor. Two things only a GPU run proves:

  * the text branch trains a real multimodal model on text preference pairs — finite loss, margins
    logged — with the row fn tokenizing through the processor's INNER tokenizer;
  * the export still ships ``processor_config.json``: the processor stays the saved processing
    class even though the run is text, so the checkpoint remains servable as what it is — a
    multimodal model.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 tests/gpu/trainers/preference/test_smpo_text_on_vlm.py
"""

import os
import shutil

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.configs.smpo_config import SmoothMarginPOConfig
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.preference.smpo import SmoothMarginPOTrainer
from tests.common.datasets import create_preference_dataset
from tests.common.distributed import shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.utils import log

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
NUM_TRAIN_SAMPLES = 32
NUM_TRAIN_STEPS = 4


def run(ctx) -> dict:
    log("=" * 70)
    log("SMPO text run on a multimodal checkpoint (Qwen3-VL-2B)")
    log("=" * 70)

    # Rank-shared: every rank saves to the same path and rank 0 inspects what landed there.
    export_root = shared_scratch_dir("smpo_text_on_vlm")
    save_dir = os.path.join(export_root, "export")
    if ctx.rank == 0:
        ctx.on_teardown(lambda: shutil.rmtree(export_root, ignore_errors=True))

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, attn_implementation="sdpa")

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
        max_length=512,
        max_prompt_length=256,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        fsdp="",
    )

    # The text branch consumes what the script's prepare_preference_datasets emits: rendered
    # STRING columns, tokenized by the row fn through the processor's inner tokenizer.
    train_dataset = create_preference_dataset(NUM_TRAIN_SAMPLES, tokenizer)
    trainer = SmoothMarginPOTrainer(
        model,
        args=config,
        train_dataset=train_dataset,
        processing_class=processor,
        parallelism_config=ParallelismConfig(),
        is_vlm=False,
    )
    assert not trainer.is_vlm, "the run verdict was ignored — a text run took the VLM branch"

    result = trainer.train()
    loss = result.training_loss
    log(f"Training loss: {loss:.6f}")

    checks = {"loss_finite": bool(torch.isfinite(torch.tensor(loss)))}
    if not checks["loss_finite"]:
        log(f"ISSUE: Training loss not finite: {loss}")
    margins = [e for e in trainer.state.log_history if any(k.startswith("rewards/") for k in e)]
    checks["rewards_logged"] = bool(margins)
    if not margins:
        log("ISSUE: No rewards/* metrics logged — the SMPO loss path did not run")

    trainer.save_model(save_dir)
    ctx.barrier()
    if ctx.rank == 0:
        exported = os.path.isfile(os.path.join(save_dir, "processor_config.json"))
        if not exported:
            log(
                f"ISSUE: export is unservable: processor_config.json missing from {save_dir} — the "
                f"processor must stay the saved processing class even on a text run"
            )
        checks["processor_config_exported"] = exported

    return {"checks": ctx.broadcast_checks(checks)}


main = gpu_test_main(min_world_size=2, prefix="test_smpo_text_on_vlm", partial_state=False)(run)

if __name__ == "__main__":
    main()
