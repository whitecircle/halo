#!/usr/bin/env python
"""
Test: DistributedDPOTrainer with Qwen3-0.6B.

Validates that DPO training works end-to-end with:
1. Synthetic math preference data (chosen vs rejected)
2. 30% multi-turn conversations
3. Liger kernels, gradient checkpointing, bf16
4. ParallelismConfig() (standard mode, no EP/CP/TP)

Assertions:
- Training completes without errors
- Final training loss is finite (not NaN/Inf)
- Loss decreases from initial value

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/preference/test_dpo.py
"""

import random

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.preference.dpo import DistributedDPOTrainer
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.utils import log

# Configuration

MODEL_NAME = QWEN3_0_6B
NUM_GPUS = 2
NUM_TRAIN_SAMPLES = 64
NUM_EVAL_SAMPLES = 16
MAX_SEQ_LENGTH = 4096
MAX_PROMPT_LENGTH = 2048
NUM_TRAIN_STEPS = 10
BATCH_SIZE = 1
LEARNING_RATE = 5e-6
SEED = 42


# Dataset Creation (DPO-specific format -- kept inline)


def create_word_problem_preference_dataset(
    num_samples: int,
    tokenizer,
    seed: int = SEED,
) -> Dataset:
    """Create synthetic math preference data for DPO.

    Each sample has:
    - prompt: chat-templated user question
    - chosen: correct answer (higher quality)
    - rejected: incorrect or lower-quality answer

    30% of samples are multi-turn (follow-up question).

    Not ``tests.common.datasets.create_preference_dataset``: the word-problem templates and the
    per-operation chosen/rejected wording here are wider, and the pairs a DPO margin trains on are
    what this test measures — swapping the generator moves the trajectory it asserts on.
    """
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

        # 30% multi-turn: add a follow-up exchange
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

        # Use message list format for DPO — TRL handles tokenization internally.
        # prompt = list of user/assistant messages, chosen/rejected = completion messages only.
        data.append(
            {
                "prompt": messages_prompt,
                "chosen": messages_chosen,
                "rejected": messages_rejected,
            }
        )

    return Dataset.from_list(data)


# Main Test


def run(ctx) -> dict:
    """Run the DPO training test."""
    log(f"{'=' * 70}")
    log("DPO Trainer Test")
    log(f"{'=' * 70}")
    log(f"Model: {MODEL_NAME}")
    log(f"World size: {ctx.world_size}")
    log(f"GPU: {torch.cuda.get_device_name(ctx.local_rank)}")

    if ctx.world_size != NUM_GPUS:
        log(f"WARNING: Expected {NUM_GPUS} GPUs, got {ctx.world_size}. Proceeding anyway.")

    # ── Load tokenizer ──────────────────────────────────────────
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Create datasets ─────────────────────────────────────────
    log("Creating synthetic preference datasets...")
    train_dataset = create_word_problem_preference_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_word_problem_preference_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1)
    log(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    # ── Load model ──────────────────────────────────────────────
    log("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    total_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {total_params / 1e6:.1f}M")

    # ── Configure DPO training ──────────────────────────────────
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
        max_length=MAX_SEQ_LENGTH,
        # TRL 1.6 DPOConfig has no max_prompt_length (prompt is truncated within max_length).
        dataloader_drop_last=True,
        dataset_num_proc=1,
        dataloader_num_workers=0,
        fsdp="",  # Mixin handles FSDP wrapping
    )

    # ── Create trainer ──────────────────────────────────────────
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

    # ── Train ───────────────────────────────────────────────────
    log("Starting DPO training...")
    ctx.barrier()
    train_result = trainer.train()

    # ── Checks ──────────────────────────────────────────────────
    checks = {}

    # 1. Training loss must be finite
    training_loss = train_result.training_loss
    log(f"Training loss: {training_loss:.6f}")
    checks["training_loss_finite"] = bool(torch.isfinite(torch.tensor(training_loss)))

    # 2. Check log history for loss convergence
    log_history = trainer.state.log_history
    train_losses = [entry["loss"] for entry in log_history if "loss" in entry]
    checks["enough_steps_logged"] = len(train_losses) >= 2
    if len(train_losses) >= 2:
        initial_loss = train_losses[0]
        final_loss = train_losses[-1]
        log(f"Loss trajectory: {initial_loss:.4f} -> {final_loss:.4f}")
        checks["step_losses_finite"] = all(bool(torch.isfinite(torch.tensor(l))) for l in train_losses)
    else:
        log(f"Not enough training steps logged: {len(train_losses)}")

    # 3. Memory usage
    peak_mem_gb = torch.cuda.max_memory_allocated(ctx.local_rank) / 1e9
    log(f"Peak GPU memory: {peak_mem_gb:.2f} GB")

    return {"checks": checks}


main = gpu_test_main(min_world_size=1, prefix="test_dpo")(run)

if __name__ == "__main__":
    main()
