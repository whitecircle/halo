#!/usr/bin/env python
"""Test: DistributedSelfDistillationTrainer (SDPG) on a text LLM — single-GPU smoke.

Validates the text SDPG self-distillation path end-to-end under TRL 1.6:
1. SelfDistillTextCollator emits the student batch + teacher_* branch (privileged hint) with
   byte-identical response tokens.
2. The trainer runs the student + privileged-teacher forward and computes
   L = L_sft + beta(k) * L_OPD on the shared response tokens (opd_loss metric recorded).

Model defaults to Qwen3-0.6B (CI); set HALO_TEST_MODEL to override (e.g. Qwen/Qwen3.5-4B).

Run:
    torchrun --nproc_per_node=1 tests/gpu/trainers/other/test_self_distillation_text.py
"""

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig

from src.data.collators.self_distill import SelfDistillTextCollator
from src.distributed.parallelism_config import ParallelismConfig
from src.env import env_str
from src.trainers.distillation.self_distillation import DistributedSelfDistillationTrainer
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.utils import log

MODEL_NAME = env_str("HALO_TEST_MODEL", QWEN3_0_6B)
NUM_TRAIN_SAMPLES = 16
NUM_TRAIN_STEPS = 4


def create_text_self_distill_dataset(n: int) -> Dataset:
    """Synthetic raw conversations with a privileged ``answer`` field (arithmetic task)."""
    rows = []
    for i in range(n):
        a, b = i % 7, (i * 3) % 5
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": f"What is {a} + {b}?"},
                    {"role": "assistant", "content": f"The answer is {a + b}."},
                ],
                "answer": str(a + b),
            }
        )
    return Dataset.from_list(rows)


def run(ctx) -> dict:
    log("=" * 70)
    log(f"Self-Distillation (SDPG) TEXT Trainer Test ({MODEL_NAME})")
    log("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = create_text_self_distill_dataset(NUM_TRAIN_SAMPLES)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, attn_implementation="sdpa")

    collator = SelfDistillTextCollator(
        tokenizer,
        max_length=512,
        conversation_field="messages",
        hint_template="\n[Hint] The correct answer is: {answer}.\n",
        answer_field="answer",
        solution_field=None,
        response_prompt_template="<|im_start|>assistant\n",
        train_on_completions_only=True,
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
        max_length=512,
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
        processing_class=tokenizer,
        data_collator=collator,
        parallelism_config=ParallelismConfig(),
        sdpg_loss="reverse_kl",
        sdpg_beta_base=1.0,
    )

    trainer.train()

    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    opd = [e["opd_loss"] for e in trainer.state.log_history if "opd_loss" in e]
    if opd:
        log(f"OPD loss recorded: {opd[0]:.4f}")

    return {
        "checks": {
            "enough_steps_logged": len(losses) >= 2,
            "losses_finite": all(torch.isfinite(torch.tensor(v)) for v in losses),
            # No opd_loss metric ⇒ the privileged-teacher forward never ran.
            "opd_loss_recorded": bool(opd),
        }
    }


main = gpu_test_main(min_world_size=1, prefix="test_self_distillation_text", partial_state=False)(run)

if __name__ == "__main__":
    main()
