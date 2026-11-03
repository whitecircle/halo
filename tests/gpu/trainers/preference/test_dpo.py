#!/usr/bin/env python
"""DistributedDPOTrainer smoke test on Qwen3-0.6B. It compares no loss against a reference, so a wrong finite objective
passes."""

import random

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig

from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.preference.dpo import DistributedDPOTrainer
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log
from tests.common.training_checks import step_losses, training_run_checks

MODEL_NAME = QWEN3_0_6B
NUM_GPUS = 2
NUM_TRAIN_SAMPLES = 64
NUM_EVAL_SAMPLES = 16
MAX_SEQ_LENGTH = 4096
NUM_TRAIN_STEPS = 10
BATCH_SIZE = 1
LEARNING_RATE = 5e-6
SEED = 42


def create_word_problem_preference_dataset(
    num_samples: int,
    tokenizer,
    seed: int = SEED,
) -> Dataset:
    """Not ``tests.common.datasets.create_preference_dataset``: the wider templates and wording here are what this
    test's trajectory rests on."""
    random.seed(seed)

    question_templates = [
        ("What is {a} + {b}?", lambda a, b: a + b, "addition"),
        ("What is {a} * {b}?", lambda a, b: a * b, "multiplication"),
        ("What is {a} - {b}?", lambda a, b: a - b, "subtraction"),
        ("If you have {a} apples and get {b} more, how many do you have?", lambda a, b: a + b, "word_problem"),
    ]

    chosen_templates = {
        "addition": "The sum of {a} and {b} is {result}. I computed this by adding the two numbers together.",
        "multiplication": "The product of {a} and {b} is {result}. I multiplied {a} by {b} to get this answer.",
        "subtraction": "The difference is {result}. I subtracted {b} from {a} to get this.",
        "word_problem": "You would have {result} apples in total. Starting with {a} and adding {b} gives {result}.",
    }

    rejected_templates = {
        "addition": "The answer is {wrong}. Addition is straightforward.",
        "multiplication": "I think it is {wrong}. Let me reconsider... yes, {wrong}.",
        "subtraction": "That would be {wrong}.",
        "word_problem": "You would have {wrong} apples.",
    }

    data = []
    for i in range(num_samples):
        template_q, op_fn, op_name = random.choice(question_templates)
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        result = op_fn(a, b)
        wrong = result + random.choice([-3, -2, -1, 1, 2, 3])

        question = template_q.format(a=a, b=b)
        chosen_answer = chosen_templates[op_name].format(a=a, b=b, result=result)
        rejected_answer = rejected_templates[op_name].format(wrong=wrong)

        messages_prompt = [{"role": "user", "content": question}]
        messages_chosen = [{"role": "assistant", "content": chosen_answer}]
        messages_rejected = [{"role": "assistant", "content": rejected_answer}]

        if random.random() < 0.3:
            a2 = random.randint(1, 50)
            b2 = random.randint(1, 50)
            followup_q = f"Great, now what is {a2} + {b2}?"
            followup_correct = f"The answer is {a2 + b2}."
            followup_wrong = f"The answer is {a2 + b2 + random.randint(1, 5)}."

            messages_prompt.extend(
                [
                    {"role": "assistant", "content": chosen_answer},
                    {"role": "user", "content": followup_q},
                ]
            )
            messages_chosen = [{"role": "assistant", "content": followup_correct}]
            messages_rejected = [{"role": "assistant", "content": followup_wrong}]

        # TRL handles tokenization: prompt is user/assistant messages, chosen/rejected the completion messages only.
        data.append(
            {
                "prompt": messages_prompt,
                "chosen": messages_chosen,
                "rejected": messages_rejected,
            }
        )

    return Dataset.from_list(data)


def run(ctx) -> dict:
    log(f"{'=' * 70}")
    log("DPO Trainer Test")
    log(f"{'=' * 70}")
    log(f"Model: {MODEL_NAME}")
    log(f"World size: {ctx.world_size}")
    log(f"GPU: {torch.cuda.get_device_name(ctx.local_rank)}")

    if ctx.world_size != NUM_GPUS:
        log(f"WARNING: Expected {NUM_GPUS} GPUs, got {ctx.world_size}. Proceeding anyway.")

    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("Creating synthetic preference datasets...")
    train_dataset = create_word_problem_preference_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_word_problem_preference_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1)
    log(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    log("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    total_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {total_params / 1e6:.1f}M")

    log("Configuring DPO training...")
    config = DPOConfig(
        output_dir=ctx.output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        beta=0.1,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=MAX_SEQ_LENGTH,
        # TRL 1.14's DPOConfig has no max_prompt_length: prompt and completion truncate together to max_length.
        dataloader_drop_last=True,
        dataset_num_proc=1,
        dataloader_num_workers=0,
    )

    log("Creating DistributedDPOTrainer...")
    parallelism_config = ParallelismConfig()
    trainer = DistributedDPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    ctx.on_teardown(lambda: trainer.cleanup_ep() if hasattr(trainer, "cleanup_ep") else None)

    log("Starting DPO training...")
    ctx.barrier()
    train_result = trainer.train()

    checks = training_run_checks(train_result, trainer, NUM_TRAIN_STEPS)
    checks["enough_steps_logged"] = len(step_losses(trainer)) >= 2

    peak_mem_gb = torch.cuda.max_memory_allocated(ctx.local_rank) / 1e9
    log(f"Peak GPU memory: {peak_mem_gb:.2f} GB")

    return {"checks": checks}


main = gpu_test_main(min_world_size=1, prefix="test_dpo")(run)

if __name__ == "__main__":
    main()
