#!/usr/bin/env python
"""
Offline GRPO Trainer test with batch_size=4 and mixed group sizes.

Validates that OfflineGRPOTrainer works correctly with per_device_train_batch_size > 1,
where a single batch contains examples from multiple different prompt groups with
different group sizes (4 and 8 completions per prompt).

Tests all 3 loss types: grpo, bnpo, dr_grpo.

Key differences from test_offline_grpo.py:
- batch_size=4 (vs 1) to stress multi-example batching
- Mixed group sizes: 50% prompts with 4 completions, 50% with 8 completions
- Tests all 3 loss types sequentially
- Verifies loss consistency across loss types

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/grpo/test_offline_grpo_bs4.py
"""

import math
import os
import random

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.grpo.offline import OfflineGRPOTrainer
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.utils import cleanup_memory, log

# Configuration

MODEL_NAME = QWEN3_0_6B
MAX_STEPS = 6
BATCH_SIZE = 4
LEARNING_RATE = 5e-6
MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 256
NUM_TRAIN_SAMPLES = 48
NUM_EVAL_SAMPLES = 12
SEED = 123
LOSS_TYPES = ["grpo", "bnpo", "dr_grpo"]

# Dataset Creation — mixed group sizes

QUESTION_TEMPLATES = [
    {"q": "What is {a} + {b}?", "op": "+"},
    {"q": "Calculate {a} times {b}.", "op": "*"},
    {"q": "What is {a} minus {b}?", "op": "-"},
    {"q": "Compute the sum of {a} and {b}.", "op": "+"},
    {"q": "Find the product of {a} and {b}.", "op": "*"},
]

ANSWER_TEMPLATES = [
    "The answer is {val}.",
    "The result equals {val}.",
    "That gives us {val}.",
    "I calculate the answer as {val}.",
    "It computes to {val}.",
    "After calculation, I get {val}.",
    "The final result is {val}.",
    "Working it out, the answer is {val}.",
]


def _compute(op: str, a: int, b: int) -> int:
    if op == "+":
        return a + b
    elif op == "*":
        return a * b
    elif op == "-":
        return a - b
    return 0


def _generate_graded_completions(
    correct: int,
    rng: random.Random,
    num_completions: int,
) -> tuple[list[str], list[float]]:
    """Generate completions with graded rewards for a given group size."""
    templates = [rng.choice(ANSWER_TEMPLATES) for _ in range(num_completions)]

    completions = []
    rewards = []

    # Create a gradient of quality from best (1.0) to worst (0.0)
    for i in range(num_completions):
        reward = max(0.0, 1.0 - i * (1.0 / (num_completions - 1)))
        offset = i * rng.randint(1, 5) * rng.choice([1, -1]) if i > 0 else 0
        val = correct + offset
        completions.append(templates[i].format(val=val))
        rewards.append(round(reward, 2))

    # Shuffle completions and rewards together
    combined = list(zip(completions, rewards, strict=False))
    rng.shuffle(combined)
    completions, rewards = zip(*combined, strict=False)
    return list(completions), list(rewards)


def create_offline_grpo_dataset(
    tokenizer,
    num_samples: int,
    seed: int = SEED,
) -> Dataset:
    """Create synthetic offline GRPO dataset with mixed group sizes.

    50% of prompts get 4 completions, 50% get 8 completions.
    This tests the group_weights = 1/group_size normalization.
    """
    rng = random.Random(seed)
    data = []

    for i in range(num_samples):
        t = rng.choice(QUESTION_TEMPLATES)
        a = rng.randint(1, 100)
        b = rng.randint(1, 100)
        correct = _compute(t["op"], a, b)

        messages = [
            {"role": "user", "content": t["q"].format(a=a, b=b)},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Alternate between 4 and 8 completions
        num_completions = 4 if i % 2 == 0 else 8
        completions, rewards = _generate_graded_completions(correct, rng, num_completions)

        data.append(
            {
                "prompt": prompt,
                "completions": completions,
                "rewards": rewards,
            }
        )

    return Dataset.from_list(data)


# Single loss-type training run


def run_single_loss_type(
    loss_type: str,
    model_name: str,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_dir: str,
) -> dict:
    """Run training with a specific loss type and return results."""
    # Load fresh model for each loss type
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    config = OfflineGRPOConfig(
        output_dir=output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        dataloader_drop_last=True,
        loss_type=loss_type,
        fsdp="",
    )

    trainer = OfflineGRPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=ParallelismConfig(),
    )

    train_result = trainer.train()
    training_loss = train_result.training_loss
    log_history = trainer.state.log_history
    step_losses = [entry["loss"] for entry in log_history if "loss" in entry and "eval_loss" not in entry]

    # Collect per-split metrics
    pos_logps = [e.get("positive/logps_mean") for e in log_history if "positive/logps_mean" in e]
    neg_logps = [e.get("negative/logps_mean") for e in log_history if "negative/logps_mean" in e]

    result = {
        "loss": training_loss,
        "step_losses": step_losses,
        "pos_logps": pos_logps,
        "neg_logps": neg_logps,
    }

    del model, trainer
    cleanup_memory()
    return result


# Main Test


def run(ctx):
    log(f"\n{'=' * 70}")
    log("  Offline GRPO batch_size=4 Test (mixed group sizes, all loss types)")
    log(f"  World size: {ctx.world_size}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  Batch size: {BATCH_SIZE}, Loss types: {LOSS_TYPES}")
    log("  Group sizes: mixed 4 and 8 completions per prompt")
    log(f"{'=' * 70}")

    # ── Load tokenizer ──
    log("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Create datasets ──
    log("\n[2/4] Creating synthetic datasets with mixed group sizes...")
    train_dataset = create_offline_grpo_dataset(tokenizer, NUM_TRAIN_SAMPLES, seed=SEED)
    eval_dataset = create_offline_grpo_dataset(tokenizer, NUM_EVAL_SAMPLES, seed=SEED + 1)

    # Log dataset composition
    group_sizes = [len(s["completions"]) for s in train_dataset]
    size_4_count = sum(1 for s in group_sizes if s == 4)
    size_8_count = sum(1 for s in group_sizes if s == 8)
    total_examples = sum(group_sizes)
    log(f"  Train: {len(train_dataset)} prompts → {total_examples} expanded examples")
    log(f"  Group composition: {size_4_count} groups of 4, {size_8_count} groups of 8")
    log(f"  Eval: {len(eval_dataset)} prompts")

    # ── Run each loss type ──
    log("\n[3/4] Training with each loss type...")
    results = {}
    checks = {}
    metrics = {}

    for loss_type in LOSS_TYPES:
        log(f"\n  --- Training with loss_type='{loss_type}' ---")
        # A fresh output dir per loss type, inside the harness-owned scratch it reclaims.
        output_dir = os.path.join(ctx.output_dir, loss_type)
        result = run_single_loss_type(loss_type, MODEL_NAME, tokenizer, train_dataset, eval_dataset, output_dir)
        results[loss_type] = result

        loss = result["loss"]
        step_losses = result["step_losses"]

        log(f"    Final loss: {loss:.6f}")
        log(f"    Step losses: {[f'{l:.4f}' for l in step_losses]}")
        if result["pos_logps"]:
            log(f"    Positive logps (first/last): {result['pos_logps'][0]:.4f} / {result['pos_logps'][-1]:.4f}")
        if result["neg_logps"]:
            log(f"    Negative logps (first/last): {result['neg_logps'][0]:.4f} / {result['neg_logps'][-1]:.4f}")

        # Checks per loss type
        finite = math.isfinite(loss)
        checks[f"{loss_type}_loss_finite"] = finite
        log(f"    Loss finite: {'PASS' if finite else 'FAIL'}")

        all_finite = all(math.isfinite(l) for l in step_losses)
        checks[f"{loss_type}_all_steps_finite"] = all_finite
        log(f"    All steps finite: {'PASS' if all_finite else 'FAIL'}")

        reasonable = loss < 100.0
        checks[f"{loss_type}_loss_reasonable"] = reasonable
        log(f"    Loss reasonable (<100): {'PASS' if reasonable else 'FAIL'}")

        metrics[f"{loss_type}_final_loss"] = loss

    # ── Cross-loss-type validation ──
    log("\n[4/4] Cross-validation...")

    # All loss types should produce finite, reasonable losses
    all_losses = {lt: results[lt]["loss"] for lt in LOSS_TYPES}
    log(f"  Losses: {', '.join(f'{lt}={v:.6f}' for lt, v in all_losses.items())}")

    # Verify no loss type diverged vs the others (within 100x is a sanity check)
    loss_values = list(all_losses.values())
    if all(v > 0 for v in loss_values):
        max_ratio = max(loss_values) / min(loss_values)
        ratio_ok = max_ratio < 100.0
        checks["loss_ratio_reasonable"] = ratio_ok
        metrics["max_loss_ratio"] = max_ratio
        log(f"  Max loss ratio: {max_ratio:.2f} {'PASS' if ratio_ok else 'FAIL'}")

    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(min_world_size=1, prefix="ogrpo_bs4")(run)

if __name__ == "__main__":
    main()
