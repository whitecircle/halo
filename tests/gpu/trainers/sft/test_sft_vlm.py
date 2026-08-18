#!/usr/bin/env python
"""
VLM (Vision Language Model) SFT Training Test.

Validates that DistributedSFTTrainer works correctly with a VLM model
(AutoModelForImageTextToText) using text-only synthetic conversations.
This tests the VLM code path without requiring actual image data.

Test Setup:
- Model: Qwen/Qwen3-VL-2B-Instruct (VLM with text+vision capabilities)
- Parallelism: Standard DDP (no EP/CP/TP)
- Dataset: Synthetic math Q&A, text-only (no images)
- 30% multi-turn conversations

Test Phases:
1. Load VLM model via AutoModelForImageTextToText + AutoProcessor
2. Create synthetic text-only dataset with chat-templated conversations
3. Train for 5 steps with gradient checkpointing and Liger kernels
4. Validate: training completes, loss is finite

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_vlm.py
"""

import math
import random

import torch
from datasets import Dataset
from trl import ModelConfig, SFTConfig

from src.distributed.loading.vlm_setup import load_model_for_training
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_VL_2B
from tests.common.utils import log

MODEL_NAME = QWEN3_VL_2B
MAX_STEPS = 5
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 2e-5
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
SEED = 42
MULTI_TURN_RATIO = 0.3

SINGLE_TURN_TEMPLATES = [
    {
        "q": "What is {a} + {b}?",
        "a": "The sum of {a} and {b} is {r}.",
        "op": "+",
    },
    {
        "q": "Calculate {a} times {b}.",
        "a": "The product of {a} and {b} equals {r}.",
        "op": "*",
    },
    {
        "q": "What is {a} minus {b}?",
        "a": "The difference is {r}.",
        "op": "-",
    },
    {
        "q": "If you have {a} items and remove {b}, how many remain?",
        "a": "You would have {r} items remaining.",
        "op": "-",
    },
    {
        "q": "Compute the sum: {a} + {b}.",
        "a": "The result of {a} plus {b} is {r}.",
        "op": "+",
    },
]


def _compute_result(op: str, a: int, b: int) -> int:
    """Compute the result for a given operation."""
    if op == "+":
        return a + b
    elif op == "*":
        return a * b
    elif op == "-":
        return a - b
    return 0


def create_single_turn_sample(rng: random.Random) -> list:
    """Create a single-turn math Q&A conversation."""
    t = rng.choice(SINGLE_TURN_TEMPLATES)
    a = rng.randint(1, 100)
    b = rng.randint(1, 100)
    r = _compute_result(t["op"], a, b)
    messages = [
        {"role": "user", "content": t["q"].format(a=a, b=b, r=r)},
        {"role": "assistant", "content": t["a"].format(a=a, b=b, r=r)},
    ]
    return messages


def create_multi_turn_sample(rng: random.Random) -> list:
    """Create a multi-turn math conversation (2 exchanges)."""
    a = rng.randint(1, 50)
    b = rng.randint(1, 50)
    c = rng.randint(1, 30)

    first_op = rng.choice(["+", "*"])
    r1 = _compute_result(first_op, a, b)

    second_op = rng.choice(["+", "-"])
    r2 = _compute_result(second_op, r1, c)

    if first_op == "+":
        q1 = f"What is {a} + {b}?"
        a1 = f"The sum of {a} and {b} is {r1}."
    else:
        q1 = f"What is {a} times {b}?"
        a1 = f"The product of {a} and {b} is {r1}."

    if second_op == "+":
        q2 = f"Now add {c} to the result."
        a2 = f"Adding {c} to {r1} gives {r2}."
    else:
        q2 = f"Subtract {c} from that."
        a2 = f"Subtracting {c} from {r1} gives {r2}."

    messages = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": a1},
        {"role": "user", "content": q2},
        {"role": "assistant", "content": a2},
    ]
    return messages


def create_vlm_sft_dataset(
    tokenizer,
    num_samples: int,
    seed: int = SEED,
) -> Dataset:
    """Create a synthetic text-only VLM SFT dataset.

    Each sample is {"text": chat_templated_string} using the tokenizer's
    chat template. No images are included -- this tests the VLM model
    class (AutoModelForImageTextToText) with text-only data.

    30% of samples are multi-turn (2 user/assistant exchanges).
    """
    rng = random.Random(seed)
    data = []

    for _ in range(num_samples):
        messages = create_multi_turn_sample(rng) if rng.random() < MULTI_TURN_RATIO else create_single_turn_sample(rng)

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        data.append({"text": text})

    return Dataset.from_list(data)


def run(ctx):
    log(f"\n{'=' * 70}")
    log("  VLM SFT Training Test (Text-only, no EP/CP/TP)")
    log(f"  World size: {ctx.world_size}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  Max steps: {MAX_STEPS}, Batch size: {BATCH_SIZE}")
    log(f"  Seq length: {MAX_SEQ_LENGTH}")
    log(f"{'=' * 70}")

    # the loader reads dtype/liger settings off these
    config = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        fsdp="",  # Mixin handles FSDP wrapping
    )
    parallelism_config = ParallelismConfig()

    # production entry point, so VLM detection and processor load are under test too
    log("\n[1/5] Loading VLM via load_model_for_training (production path)...")
    model_config = ModelConfig(
        model_name_or_path=MODEL_NAME, attn_implementation="flash_attention_2", trust_remote_code=True
    )
    model, processing_class, tokenizer, is_vlm = load_model_for_training(model_config, config, parallelism_config)
    assert is_vlm, "Qwen3-VL must be detected as a VLM by load_model_for_training"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"  Model: {type(model).__name__}, processor: {type(processing_class).__name__}")

    log("\n[2/5] Creating synthetic text-only datasets...")
    train_dataset = create_vlm_sft_dataset(tokenizer, NUM_TRAIN_SAMPLES, seed=SEED)
    eval_dataset = create_vlm_sft_dataset(tokenizer, NUM_EVAL_SAMPLES, seed=SEED + 1)
    log(f"  Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")
    if ctx.rank == 0:
        log(f"  Sample (truncated): {train_dataset[0]['text'][:200]}...")

    log("\n[3/5] Configuring trainer...")
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        parallelism_config=parallelism_config,
    )
    log(f"  Trainer created: {type(trainer).__name__}")
    log(f"  Parallelism: {parallelism_config.mode_string or 'Standard DDP'}")

    log("\n[5/5] Training...")
    train_result = trainer.train()

    training_loss = train_result.training_loss
    log_history = trainer.state.log_history
    step_losses = [entry["loss"] for entry in log_history if "loss" in entry and "eval_loss" not in entry]

    log("\n  --- Training Results ---")
    log(f"  Final training loss: {training_loss:.6f}")
    log(f"  Per-step losses: {[f'{l:.4f}' for l in step_losses]}")

    log("\n  --- Assertions ---")
    checks = {}

    loss_finite = math.isfinite(training_loss)
    checks["loss_finite"] = loss_finite
    log(f"  Loss is finite: {'PASS' if loss_finite else 'FAIL'} (loss={training_loss:.6f})")

    all_finite = all(math.isfinite(l) for l in step_losses)
    checks["all_steps_finite"] = all_finite
    log(f"  All step losses finite: {'PASS' if all_finite else 'FAIL'}")

    no_nan_inf = not any(math.isnan(l) or math.isinf(l) for l in step_losses)
    checks["no_nan_inf"] = no_nan_inf
    log(f"  No NaN/Inf in step losses: {'PASS' if no_nan_inf else 'FAIL'}")

    loss_reasonable = training_loss < 100.0
    checks["loss_reasonable"] = loss_reasonable
    log(f"  Loss is reasonable (<100): {'PASS' if loss_reasonable else 'FAIL'} (loss={training_loss:.6f})")

    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="test_sft_vlm")(run)

if __name__ == "__main__":
    main()
