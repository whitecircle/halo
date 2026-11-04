#!/usr/bin/env python
"""Teacher distillation of GPT-OSS-20B into itself under every parallel layout of the student.

At step 1 teacher and student compute one function, so the KL is zero up to kernel noise, and a mis-combined expert or
mis-sharded projection shows far above it. Both keep pretrained sinks live: neutralized, the model is far off its
trained function, and the same noise reaches half a nat per token.
"""

import argparse
import math
import os
import random
from types import SimpleNamespace

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

import scripts.training.distillation.teacher_distill as teacher_distill
from halo.configs.distillation import TeacherDistillationConfig
from halo.distributed.parallelism_config import ParallelismConfig
from halo.hardware import is_blackwell_gpu
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.attention.gpt_oss_sinks import SinksPolicy, stamped_sinks_policy
from halo.models.attention.selection import effective_attn_implementation
from halo.trainers.distillation.teacher_distillation import DistributedTeacherDistillationTrainer
from tests.common.checkpoint_io import loading_problems
from tests.common.distributed import ensure_model_downloaded, shared_output_dir
from tests.common.dp_layout import dp_layout_checks, probe_forward
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.parallel_shape import parallel_shape_checks
from tests.common.rank_log import log
from tests.common.training_checks import step_losses

MODEL_NAME = GPT_OSS_20B
NUM_TRAIN_SAMPLES = 32
MAX_SEQ_LENGTH = 4096
NUM_TRAIN_STEPS = 5
BATCH_SIZE = 1
# Four updates move the student's distribution measurably off the teacher's.
LEARNING_RATE = 1e-4
SEED = 42
# Nats per token. With live sinks the bf16 noise of two layouts is ~0.02 on these rows; the defects guarded (two
# experts' down projections swapped, a rank holding the other's sink slice, a bias summed on both TP ranks) sit at 0.5
# and up.
STEP1_KL_ATOL = 0.1
# Past the live-sink noise, so a student whose updates never landed fails it.
KL_MOVED_MIN = 0.05
# Live and frozen, on both models.
RESET_SINKS = False
# Resolved per GPU at run time (:func:`sink_carrying_attention`).
FLASH = "flash"
MODES = {
    "fsdp": (2, {}, FLASH),
    "fsdp_unsharded": (2, {"fsdp_shard_ep1_experts": False}, FLASH),
    "ep": (2, {"ep_size": 2}, "flex_attention"),
    "tp": (2, {"tp_size": 2, "max_concurrent_loading": 1}, FLASH),
    "ep_tp": (2, {"ep_size": 2, "tp_size": 2}, "flex_attention"),
    "etp": (2, {"expert_tp_size": 2}, "flex_attention"),
    "ep_etp": (4, {"ep_size": 2, "expert_tp_size": 2}, "flex_attention"),
    "ep_dp2": (4, {"ep_size": 2}, "flex_attention"),
    "tp_dp2": (4, {"tp_size": 2, "max_concurrent_loading": 1}, FLASH),
}

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=sorted(MODES), required=True)
parser.add_argument("--gpus-per-node", type=int, default=0, help="simulated NVLink domain width (0: the launch's)")
ARGS = parser.parse_args()
WORLD_SIZE, MODE_FIELDS, ATTN_IMPL = MODES[ARGS.mode]
FIELDS = {**MODE_FIELDS, **({"gpus_per_node": ARGS.gpus_per_node} if ARGS.gpus_per_node else {})}


def create_distillation_dataset(num_samples: int, tokenizer, seed: int = SEED) -> Dataset:
    random.seed(seed)

    templates = [
        ("What is {a} + {b}?", lambda a, b: a + b),
        ("Calculate {a} * {b}.", lambda a, b: a * b),
        ("What is {a} - {b}?", lambda a, b: a - b),
    ]
    answers = [
        "The answer is {result}. I calculated this with {a} and {b}.",
        "That equals {result}.",
        "{result} is the answer.",
    ]

    data = []
    for _ in range(num_samples):
        tmpl, op = random.choice(templates)
        a, b = random.randint(1, 100), random.randint(1, 100)
        result = op(a, b)
        messages = [
            {"role": "user", "content": tmpl.format(a=a, b=b)},
            {"role": "assistant", "content": random.choice(answers).format(a=a, b=b, result=result)},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        data.append({"text": text})

    return Dataset.from_list(data)


def sink_carrying_attention(kind: str) -> str:
    """FA4 on Blackwell, eager elsewhere (FA2 and the stock FA3 build drop the sink column)."""
    if kind != FLASH:
        return kind
    return "flash_attention_4" if is_blackwell_gpu() else "eager"


def load_teacher(config: TeacherDistillationConfig, attn_implementation: str, local_rank: int):
    """The script's frozen-teacher load, on the student's request and sinks policy, unsharded."""
    return teacher_distill._load_distill_teacher(
        args=SimpleNamespace(teacher_model=MODEL_NAME, teacher_model_revision=None, reset_sinks=RESET_SINKS),
        model_config=SimpleNamespace(trust_remote_code=True),
        training_config=config,
        requested_attn=attn_implementation,
        is_vlm=False,
        local_rank=local_rank,
    )


@gpu_test_main(exact_world_size=WORLD_SIZE, prefix=f"distill_oss20b_{ARGS.mode}")
def run(ctx) -> dict:
    pc = ParallelismConfig(**FIELDS)
    attn_implementation = sink_carrying_attention(ATTN_IMPL)
    log(f"Distillation {ARGS.mode}: {FIELDS}, {attn_implementation}, {MODEL_NAME} as teacher and student")
    ensure_model_downloaded(MODEL_NAME, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = create_distillation_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED).map(
        lambda rows: tokenizer(rows["text"], truncation=True, max_length=MAX_SEQ_LENGTH),
        batched=True,
        remove_columns=["text"],
    )

    student_model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        use_liger_kernel=True,
        reset_sinks=RESET_SINKS,
    )
    config = TeacherDistillationConfig(
        output_dir=ctx.output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,  # Already applied by load_distributed_model.
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        dataset_num_proc=1,
        distill_loss="kl_divergence",
        distill_temperature=1.0,
        distill_alpha=0.5,
    )
    teacher_model = load_teacher(config, attn_implementation, ctx.local_rank)
    teacher_function = (stamped_sinks_policy(teacher_model), effective_attn_implementation(teacher_model.config))
    student_function = (stamped_sinks_policy(student_model), effective_attn_implementation(student_model.config))
    log(f"teacher (sinks, attention) {teacher_function}; student {student_function}")
    trainer = DistributedTeacherDistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        teacher_tokenizer=tokenizer,
        args=config,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer,
        parallelism_config=pc,
    )
    ctx.on_teardown(trainer.cleanup_ep)
    checks = parallel_shape_checks(trainer.model, pc)
    checks["teacher_frozen"] = not any(p.requires_grad for p in trainer.teacher_model.parameters())
    expected_function = (SinksPolicy.LIVE, attn_implementation)
    checks["teacher_runs_the_student_sinks_and_kernel"] = teacher_function == student_function == expected_function

    result = trainer.train()
    losses = step_losses(trainer)
    kl = [float(e["distillation_loss"]) for e in trainer.state.log_history if "distillation_loss" in e]
    log(f"step losses {losses}; distillation (KL) per step {kl}")
    checks["steps_completed"] = result.global_step == NUM_TRAIN_STEPS and len(losses) == NUM_TRAIN_STEPS
    checks["losses_finite"] = math.isfinite(result.training_loss) and all(map(math.isfinite, losses))
    checks["kl_logged_every_step"] = len(kl) == NUM_TRAIN_STEPS and all(map(math.isfinite, kl))
    checks["step1_kl_is_the_identical_teacher_zero"] = bool(kl) and abs(kl[0]) < STEP1_KL_ATOL
    checks["kl_moved_off_zero_as_the_student_trained"] = len(kl) > 1 and kl[-1] > max(kl[0], 0.0) + KL_MOVED_MIN
    checks["teacher_still_frozen"] = not any(p.requires_grad for p in trainer.teacher_model.parameters())
    checks |= dp_layout_checks(
        trainer.model,
        FIELDS,
        world_size=ctx.world_size,
        ep_config=trainer._ep_config,
        forward=lambda: probe_forward(trainer.model, ctx.device),
    )

    export = os.path.join(shared_output_dir(ctx), "student")
    trainer.save_model(export)
    ctx.barrier()
    if ctx.rank == 0:
        _, info = AutoModelForCausalLM.from_pretrained(
            export, dtype=torch.bfloat16, trust_remote_code=True, output_loading_info=True
        )
        problems = loading_problems(info)
        checks["saved_student_loads_every_weight"] = not problems
        log(f"saved student {export}: {problems or 'complete'}")
    return {"checks": ctx.broadcast_checks(checks), "metrics": {"step1_kl": kl[0] if kl else math.nan}}


if __name__ == "__main__":
    run()
