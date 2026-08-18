#!/usr/bin/env python
"""
Test: DistributedSelfDistillationTrainer (SDPG) with Qwen3-VL-2B (single-GPU smoke).

Validates the full SDPG self-distillation path end-to-end under TRL 1.6.0:
1. SelfDistillVLMDataCollator emits the student batch + teacher_* branch (privileged hint).
2. The trainer runs the student forward + privileged-teacher forward and computes
   L = L_sft + beta(k) * L_OPD on the shared response tokens.
3. ParallelismConfig() (standard mode).

Assertions: training completes, all logged losses finite, and the OPD term is actually
exercised (an opd_loss metric is recorded), so the privileged-teacher forward + full-vocab
reverse-KL ran — not just the SFT arm.

Run:
    torchrun --nproc_per_node=1 \
        tests/gpu/trainers/other/test_self_distillation_vlm.py
"""

import torch
from datasets import Dataset, Sequence
from datasets import Image as HFImage
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTConfig

from src.data.collators.vlm import SelfDistillVLMDataCollator
from src.distributed.parallelism_config import ParallelismConfig
from src.env import env_int
from src.trainers.distillation.self_distillation import DistributedSelfDistillationTrainer
from tests.common.datasets import digit_image
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_VL_2B
from tests.common.utils import log

MODEL_NAME = QWEN3_VL_2B
NUM_TRAIN_SAMPLES = 16
NUM_TRAIN_STEPS = 4
SEED = 42


def _parallelism_from_env() -> ParallelismConfig:
    """Standard by default; HALO_TEST_TP / HALO_TEST_EP select TP / EP (for parallelism sweeps)."""
    ep = env_int("HALO_TEST_EP", 1)
    tp = env_int("HALO_TEST_TP", 1)
    if ep > 1:
        return ParallelismConfig(ep_size=ep)
    if tp > 1:
        return ParallelismConfig(tp_size=tp)
    return ParallelismConfig()


def create_vlm_self_distill_dataset(n: int) -> Dataset:
    """Synthetic raw VLM conversations with a privileged ``answer`` field (read-a-digit task).

    Message content is uniformly list-of-parts (Arrow needs a consistent struct) and the
    ``images`` column is cast to the datasets Image feature so PIL images round-trip.
    """
    rows = []
    for i in range(n):
        d = i % 10
        rows.append(
            {
                "history": [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What digit is shown?"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": f"The digit is {d}."}]},
                ],
                "images": [digit_image(d)],
                "answer": str(d),
            }
        )
    return Dataset.from_list(rows).cast_column("images", Sequence(HFImage()))


def run(ctx) -> dict:
    log("=" * 70)
    log("Self-Distillation (SDPG) Trainer Test (Qwen3-VL-2B)")
    log("=" * 70)

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = create_vlm_self_distill_dataset(NUM_TRAIN_SAMPLES)
    log(f"Train samples: {len(train_dataset)}")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa"
    )

    collator = SelfDistillVLMDataCollator(
        processor,
        tokenizer,
        max_length=1024,
        hint_template="\n[Hint] The correct answer is: {answer}.\n",
        answer_field="answer",
        solution_field=None,
    )

    config = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        bf16=True,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_length=1024,
        dataloader_drop_last=True,
        dataset_num_proc=1,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        fsdp="",
    )

    trainer = DistributedSelfDistillationTrainer(
        model,
        args=config,
        train_dataset=train_dataset,
        processing_class=processor,
        data_collator=collator,
        parallelism_config=_parallelism_from_env(),
        sdpg_loss="reverse_kl",
        sdpg_beta_base=1.0,
        opd_exclude_eos=True,
    )
    ctx.on_teardown(lambda: trainer.cleanup_ep() if hasattr(trainer, "cleanup_ep") else None)

    ctx.barrier()
    train_result = trainer.train()

    checks: dict[str, bool] = {}
    training_loss = train_result.training_loss
    log(f"Training loss: {training_loss:.6f}")
    checks["training_loss_finite"] = bool(torch.isfinite(torch.tensor(training_loss)))

    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    checks["enough_steps_logged"] = len(losses) >= 2
    checks["step_losses_finite"] = all(bool(torch.isfinite(torch.tensor(lv))) for lv in losses)

    # The OPD term must have actually run (teacher forward + reverse-KL), not just SFT.
    checks["opd_loss_recorded"] = any("opd_loss" in e for e in trainer.state.log_history) or (
        hasattr(trainer, "_metrics") and any("opd_loss" in v for v in trainer._metrics.values())
    )

    log(f"  {len(losses)} steps logged, checks: {checks}")
    return {"checks": checks}


main = gpu_test_main(min_world_size=1, prefix="test_self_distillation_vlm")(run)

if __name__ == "__main__":
    main()
