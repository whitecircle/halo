#!/usr/bin/env python
"""LoRA / QLoRA / native-expert-LoRA on the off-policy teacher-distillation trainer (+ adapter ckpt).

``DistributedDistillationTrainer`` takes ``peft_config`` and wraps the student through
``prepare_peft_model`` (k-bit prep, the QLoRA bf16 cast and its autocast), as the script passes it.
The test asserts the adapter-only invariant, that the adapters move, and that an adapter checkpoint
round-trips. Liger is left on (toolkit default) so the LoRA path is exercised with fused kernels.

Modes (``--mode``): lora / qlora (dense Qwen3-0.6B, FSDP2) · lora_ep / expert_lora (GptOss-20B, EP=2).
The teacher is the same architecture as the student, loaded frozen + unparallelized (full per rank).

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_teacher_distill.py --mode lora
"""

import argparse
import math
import random

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling

from src.configs.distillation_config import DistillationConfig
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.distillation.teacher_distillation import DistributedDistillationTrainer
from tests.common.harness import gpu_test_main
from tests.common.peft_helpers import (
    assert_adapter_checkpoint,
    assert_adapters_moved,
    assert_only_adapters_trainable,
    is_expert_lora_active,
    load_peft_model,
    model_name_for,
    snapshot_adapters,
    unwrap,
)
from tests.common.utils import log

MAX_STEPS = 3
NUM_TRAIN_SAMPLES = 32
MAX_SEQ_LENGTH = 512
SEED = 42


def _distill_dataset(tokenizer, n: int) -> Dataset:
    rng = random.Random(SEED)
    rows = []
    for _ in range(n):
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": f"What is {a} + {b}?"},
                {"role": "assistant", "content": f"The answer is {a + b}."},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        enc = tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH)
        rows.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    return Dataset.from_list(rows)


def run(ctx) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="lora", choices=["lora", "qlora", "lora_ep", "expert_lora"])
    parser.add_argument("--ep", type=int, default=2)
    args, _ = parser.parse_known_args()
    mode, ep = args.mode, args.ep

    log(f"\n{'=' * 70}\n  Teacher-distill adapter test — mode={mode}, world={ctx.world_size}, ep={ep}\n{'=' * 70}")
    checks: dict[str, bool] = {}
    parallelism_config = ParallelismConfig(ep_size=ep) if mode in ("lora_ep", "expert_lora") else ParallelismConfig()
    attn = "flex_attention" if parallelism_config.is_ep_mode else "sdpa"
    # Student + adapter config via the production path; the trainer wraps it, as the script has it do.
    student, tokenizer, peft_config = load_peft_model(mode, parallelism_config, attn_implementation=attn)
    expert_lora = is_expert_lora_active(student)
    log(f"  expert_lora_active={expert_lora}, peft_config={'set' if peft_config else 'None'}")

    # Frozen teacher: same architecture, unparallelized (full copy per rank).
    teacher = AutoModelForCausalLM.from_pretrained(
        model_name_for(mode, parallelism_config),
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn,
        device_map={"": ctx.local_rank},
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    config = DistillationConfig(
        output_dir=ctx.output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        learning_rate=1e-3,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataset_num_proc=1,
        dataloader_num_workers=0,
        fsdp="",
        ddp_find_unused_parameters=parallelism_config.is_ep_mode,
        distill_loss="kl_divergence",
        distill_temperature=1.0,
        distill_alpha=0.5,
    )
    trainer = DistributedDistillationTrainer(
        student_model=student,
        teacher_model=teacher,
        teacher_tokenizer=tokenizer,
        args=config,
        train_dataset=_distill_dataset(tokenizer, NUM_TRAIN_SAMPLES),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer,
        peft_config=peft_config,
        parallelism_config=parallelism_config,
    )
    ctx.on_teardown(trainer.cleanup_ep)

    wrapped = unwrap(trainer.model)
    ok, detail = assert_only_adapters_trainable(wrapped)
    checks["only_adapters_trainable"] = ok
    log(f"  [trainable] {detail}")
    adapters_before = snapshot_adapters(wrapped, expert_lora=expert_lora)

    barrier()
    result = trainer.train()
    checks["loss_finite"] = math.isfinite(result.training_loss)
    checks["steps_done"] = result.global_step == MAX_STEPS
    log(f"  training_loss={result.training_loss:.5f}, steps={result.global_step}")

    adapters_after = snapshot_adapters(unwrap(trainer.model), expert_lora=expert_lora)
    ok, detail = assert_adapters_moved(adapters_before, adapters_after)
    checks["adapters_moved"] = ok
    log(f"  [moved] {detail}")

    ok, detail = assert_adapter_checkpoint(trainer, ctx.output_dir, ctx.rank, expert_lora=expert_lora)
    checks["adapter_checkpoint"] = ok
    log(f"  [checkpoint] {detail}")

    # Only rank 0 reads the checkpoint back; share its verdict so every rank exits alike.
    return {"checks": ctx.broadcast_checks(checks)}


main = gpu_test_main(min_world_size=2, prefix="lora_teacher_distill")(run)

if __name__ == "__main__":
    main()
