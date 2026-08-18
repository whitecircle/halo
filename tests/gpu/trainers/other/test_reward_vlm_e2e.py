#!/usr/bin/env python
"""VLM Bradley-Terry reward e2e on Qwen3.5-2B: train → save → reload-and-score.

The full deliverable loop of the vision reward path, none of which a CPU test reaches with real
weights:

  * the trainer takes the vision branch off a processor ``processing_class``, and ``pixel_values``
    must actually reach the vision tower during training (hooked), with the Bradley-Terry loss
    finite under 2-rank FSDP2;
  * the gathered save must ship an artifact ``AutoModelForSequenceClassification`` +
    ``AutoProcessor`` load back and score an image pair with — including
    ``processor_config.json``, without which no engine can serve the reward model on images.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 tests/gpu/trainers/other/test_reward_vlm_e2e.py
"""

import os

import torch
from datasets import Dataset, Sequence
from datasets import Image as HFImage
from transformers import AutoModelForSequenceClassification, AutoProcessor
from trl import RewardConfig

from src.data.collators.vlm_preference import DataCollatorForVLMPreference
from src.data.pipeline.preferences import render_vlm_preference_row
from src.distributed.parallelism_config import ParallelismConfig
from src.models.loading.tokenizer_setup import sync_special_token_id
from src.trainers.reward.bradley_terry import DistributedRewardTrainer
from tests.common.datasets import digit_image
from tests.common.distributed import cleanup_dirs, shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.utils import cleanup_memory, log

MODEL_NAME = "Qwen/Qwen3.5-2B"
NUM_TRAIN_SAMPLES = 16
NUM_TRAIN_STEPS = 4
_VISION_TOWER_SUFFIXES = ("visual", "vision_tower", "vision_model")


def create_vlm_reward_dataset(n: int) -> Dataset:
    """Raw ``(prompt, chosen, rejected, images)`` rows — the shape the reward VLM map normalizes."""
    rows = []
    for i in range(n):
        d = i % 10
        wrong = (d + 1) % 10
        rows.append(
            {
                "images": [digit_image(d)],
                "prompt": [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What digit is shown?"}]}
                ],
                "chosen": [{"role": "assistant", "content": f"The digit is {d}."}],
                "rejected": [{"role": "assistant", "content": f"The digit is {wrong}."}],
            }
        )
    return Dataset.from_list(rows).cast_column("images", Sequence(HFImage()))


def _vision_tower(model) -> torch.nn.Module:
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] in _VISION_TOWER_SUFFIXES:
            return module
    raise AssertionError(f"no vision tower found on {type(model).__name__} — checkpoint not multimodal?")


def run(ctx):
    log("=" * 70)
    log("VLM Reward Trainer E2E (Qwen3.5-2B): train -> save -> reload-and-score")
    log("=" * 70)

    # Rank-shared: every rank saves into it, rank 0 reloads the artifact from it. Rank 0 clears it
    # first — a leftover export from an earlier run would satisfy the servability check on its own.
    output_dir = shared_scratch_dir("reward_vlm_e2e")
    if ctx.rank == 0:
        cleanup_dirs(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    ctx.on_teardown(lambda: cleanup_dirs(output_dir))
    ctx.barrier()
    save_dir = os.path.join(output_dir, "export")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    # The script path (apply_max_length → setup_model_and_tokenizer) records the pad id on
    # EVERY config level; transformers' pooling reads config.get_text_config().pad_token_id,
    # so a top-level write alone leaves a composite checkpoint refusing batches > 1.
    sync_special_token_id(model, "pad_token_id", tokenizer.pad_token_id)

    tower_calls = []
    _vision_tower(model).register_forward_pre_hook(lambda module, args: tower_calls.append(1))

    config = RewardConfig(
        output_dir=output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,
        bf16=True,
        gradient_checkpointing=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_length=1024,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_drop_last=True,
    )
    trainer = DistributedRewardTrainer(
        model=model,
        args=config,
        train_dataset=create_vlm_reward_dataset(NUM_TRAIN_SAMPLES),
        processing_class=processor,
        parallelism_config=ParallelismConfig(),
    )
    assert trainer._is_vlm, "reward trainer did not take the vision branch off a processor"

    result = trainer.train()
    loss = result.training_loss
    log(f"Training loss: {loss:.6f}")

    checks: dict[str, bool] = {}
    if ctx.rank == 0:
        checks["train_loss_finite"] = bool(torch.isfinite(torch.tensor(loss)))
        # No tower call means pixel_values never reached the vision tower — the run trained text-only.
        checks["pixel_values_reached_vision_tower"] = bool(tower_calls)

    trainer.save_model(save_dir)
    ctx.barrier()

    if ctx.rank == 0:
        missing = [
            f for f in ("processor_config.json", "config.json") if not os.path.isfile(os.path.join(save_dir, f))
        ]
        checks["export_is_servable"] = not missing
        if missing:
            log(f"ISSUE: export is unservable: {missing} missing from {save_dir}")

        # Serve half: free the training copy, reload the artifact, score one image pair
        # through the exact collation layout the trainer used.
        del trainer, model
        cleanup_memory()
        served = AutoModelForSequenceClassification.from_pretrained(
            save_dir, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).cuda()
        served_processor = AutoProcessor.from_pretrained(save_dir)
        row = render_vlm_preference_row(create_vlm_reward_dataset(1)[0], processing_class=served_processor)
        batch = DataCollatorForVLMPreference(processor=served_processor)([row])
        with torch.no_grad():
            logits = served(**{k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}).logits
        checks["served_pair_scored"] = logits.shape[0] == 2 and bool(torch.isfinite(logits).all())
        if not checks["served_pair_scored"]:
            log(
                f"ISSUE: served scoring failed: logits shape {tuple(logits.shape)}, "
                f"finite={torch.isfinite(logits).all()}"
            )
        log(f"Served pair scores: {logits.flatten().tolist()}")

    # Only rank 0 reloads and scores the export, so its verdict is what every rank reports.
    return {"checks": ctx.broadcast_checks(checks)}


main = gpu_test_main(min_world_size=2, prefix="test_reward_vlm_e2e", partial_state=False)(run)

if __name__ == "__main__":
    main()
