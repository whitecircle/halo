#!/usr/bin/env python
"""DistributedRewardTrainer end to end, with the logged loss pinned to TRL 1.14's exact BT objective.

Chunk order, pooling position and sign all fail silently: each keeps the loss finite and the step count intact. A
reversed chunk trains the model to prefer the rejected answer. The test pools the logits itself, since the reference's
own ``chunk`` shares the trainer's convention, and anchors chunk order against the tokenized dataset's columns.
"""

import argparse
import random

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig

from halo.configs.script_arguments import CommonScriptArguments
from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.tensor_parallel.hf_plan import is_input_embedding_tp_sharded
from halo.model_loading.tokenizer_setup import configure_run_tokenizer
from halo.trainers.reward.bradley_terry import DistributedRewardTrainer
from tests.common.distributed import (
    ensure_model_downloaded,
    snapshot_full_weights,
    world_any,
    world_mean,
)
from tests.common.dp_layout import dp_layout_checks, hsdp_layout, probe_forward
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log
from tests.common.tolerances import TOL
from tests.common.training_checks import step_losses

MODEL_NAME = QWEN3_0_6B
NUM_TRAIN_SAMPLES = 64
NUM_EVAL_SAMPLES = 16
MAX_SEQ_LENGTH = 1024
NUM_TRAIN_STEPS = 10
# >1 pair per device so the batch is ragged; the `pooling_at_pad` control needs some row to carry padding.
BATCH_SIZE = 4
# Far below a bf16 weight's spacing, yet AdamWBF16's stochastic rounding (auto-enabled under bf16) applies it in
# expectation, which grows the margin by ~0.07 a step: by PINNED_STEP it sits inside RESPONSIVE_MARGIN instead of
# saturating -logsigmoid to 0, where the pin compares two zeros.
LEARNING_RATE = 1e-7
SEED = 42
# Late, so the pinned weights are ones the trainer reached and the snapshot path is under test.
PINNED_STEP = NUM_TRAIN_STEPS

LOSS_REL_TOL = TOL.exact_objective_rel
# A control shows the pin would fail, so its threshold is the pin's tolerance.
CONTROL_MIN_GAP = LOSS_REL_TOL
# Set by what the pin must catch: a swapped assignment negates the DP-mean margin, which
# control_halves_swapped_breaks_the_margin_pin holds further than this from the logged one. The trainer's bf16 forward
# and the reference's measured up to 0.036 apart on it and 0.026 on the mean reward, at 1, 2 and 4 ranks.
REWARD_ABS_TOL = 0.15
# An exact rate over the trainer's own pair comparisons, averaged in fp32: no dtype slack.
ACCURACY_ABS_TOL = 1e-6
# Below this band every control collapses onto log(2); above it -logsigmoid saturates and the sign-drop control can't
# show.
RESPONSIVE_MARGIN = (0.3, 5.0)


def create_reward_dataset(num_samples: int, tokenizer, seed: int = SEED) -> Dataset:
    """30% of rows are multi-turn, which makes the batch ragged enough for the pooling rule to matter."""
    random.seed(seed)

    question_templates = [
        ("What is {a} + {b}?", lambda a, b: a + b, "addition"),
        ("Calculate {a} times {b}.", lambda a, b: a * b, "multiplication"),
        ("What is {a} minus {b}?", lambda a, b: a - b, "subtraction"),
        ("If you divide {a} by {b}, what is the integer quotient?", lambda a, b: a // b, "division"),
    ]
    chosen_answers = {
        "addition": "The sum of {a} and {b} is {result}. This is computed by straightforward addition.",
        "multiplication": "{a} multiplied by {b} gives {result}. I computed this step by step.",
        "subtraction": "{a} minus {b} equals {result}. Subtraction gives us this difference.",
        "division": "The integer quotient of {a} divided by {b} is {result}. This is floor division.",
    }
    rejected_answers = {
        "addition": "I believe the answer is {wrong}.",
        "multiplication": "The result should be {wrong}.",
        "subtraction": "That gives {wrong}.",
        "division": "The quotient is {wrong}.",
    }

    data = []
    for _ in range(num_samples):
        template_q, op_fn, op_name = random.choice(question_templates)
        a = random.randint(2, 100)
        b = random.randint(1, min(a, 50))  # b <= a keeps subtraction/division clean.
        result = op_fn(a, b)
        wrong = result + random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
        question = template_q.format(a=a, b=b)

        chosen_messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": chosen_answers[op_name].format(a=a, b=b, result=result)},
        ]
        rejected_messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": rejected_answers[op_name].format(wrong=wrong)},
        ]

        if random.random() < 0.3:
            a2, b2 = random.randint(1, 50), random.randint(1, 50)
            followup_q = f"And what is {a2} + {b2}?"
            chosen_messages += [
                {"role": "user", "content": followup_q},
                {
                    "role": "assistant",
                    "content": f"That equals {a2 + b2}. Adding {a2} and {b2} together gives this result.",
                },
            ]
            rejected_messages += [
                {"role": "user", "content": followup_q},
                {"role": "assistant", "content": f"That would be {a2 + b2 + random.randint(1, 5)}."},
            ]

        data.append(
            {
                "chosen": tokenizer.apply_chat_template(chosen_messages, tokenize=False, add_generation_prompt=False),
                "rejected": tokenizer.apply_chat_template(
                    rejected_messages, tokenize=False, add_generation_prompt=False
                ),
            }
        )
    return Dataset.from_list(data)


def build_model(tokenizer):
    # The scalar `score` head is not in the checkpoint; seed first, or the margin the controls depend on differs per
    # run.
    torch.manual_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    # Through the script's own seam: the pooling rule reads this id back, so setting it by hand would survive a sync
    # regression.
    configure_run_tokenizer(
        CommonScriptArguments(), model, tokenizer, MAX_SEQ_LENGTH, embeddings_sharded=is_input_embedding_tp_sharded
    )
    return model


def token_rewards(model, input_ids, attention_mask) -> torch.Tensor:
    """Per-token reward logits in fp32. ``model(...)`` returns already-pooled logits, which would hide a wrong
    position."""
    with torch.no_grad():
        hidden = model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        return model.score(hidden).float().squeeze(-1)


def pool_rewards(token_logits, input_ids, pad_token_id, *, at_pad: bool = False) -> torch.Tensor:
    """Scalar reward at the last non-pad token, transformers' position rule reimplemented. ``at_pad`` is that bug's
    control."""
    rows, seq_len = input_ids.shape
    if at_pad:
        positions = torch.full((rows,), seq_len - 1, device=input_ids.device, dtype=torch.long)
    else:
        non_pad = (input_ids != pad_token_id).to(torch.int32)
        token_indices = torch.arange(seq_len, device=input_ids.device, dtype=torch.int32)
        positions = (token_indices * non_pad).argmax(-1)
    return token_logits[torch.arange(rows, device=input_ids.device), positions]


def bt_objective(rewards: torch.Tensor, *, swap: bool = False, drop_sign: bool = False) -> float:
    """``-logsigmoid(chosen - rejected)`` over TRL's ``[all chosen ; all rejected]``. ``swap`` and ``drop_sign`` are
    controls."""
    chosen, rejected = torch.chunk(rewards, chunks=2)
    if swap:
        chosen, rejected = rejected, chosen
    difference = chosen - rejected
    return float(difference.sigmoid().log().mean() if drop_sign else -F.logsigmoid(difference).mean())


parser = argparse.ArgumentParser()
parser.add_argument("--hsdp", action="store_true", help="HSDP over two simulated NVLink domains (4+ ranks)")
parser.add_argument("--world", type=int, default=None, help="the exact rank count (1: one process, no FSDP wrap)")
ARGS = parser.parse_args()


@gpu_test_main(min_world_size=4 if ARGS.hsdp else 1, exact_world_size=ARGS.world, prefix="test_reward")
def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    layout = hsdp_layout(ctx.world_size) if ARGS.hsdp else {}

    ensure_model_downloaded(MODEL_NAME, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = RewardConfig(
        output_dir=ctx.output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
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
    )

    trainer = DistributedRewardTrainer(
        model=build_model(tokenizer),
        args=config,
        train_dataset=create_reward_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED),
        eval_dataset=create_reward_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1),
        processing_class=tokenizer,
        parallelism_config=ParallelismConfig(**layout),
    )
    ctx.on_teardown(trainer.cleanup_ep)

    captured: dict = {}
    weights: dict = {}
    trainer_rewards: list[torch.Tensor] = []
    real_compute_loss = trainer.compute_loss

    def spying_compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
        # global_step is still N-1 while step N's loss is being computed.
        if trainer.state.global_step != PINNED_STEP - 1 or captured:
            return real_compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        captured.update({k: v.clone() for k, v in inputs.items() if torch.is_tensor(v)})
        weights.update(snapshot_full_weights(model))
        loss, outputs = real_compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        # The bf16 rewards TRL's accuracy compares; a second bf16 forward can order a near-tied pair the other way.
        trainer_rewards.append(outputs.logits.detach().float().squeeze(-1))
        return (loss, outputs) if return_outputs else loss

    trainer.compute_loss = spying_compute_loss
    trainer.train()

    losses = step_losses(trainer)
    checks["ran_all_steps"] = trainer.state.global_step == NUM_TRAIN_STEPS and len(losses) == NUM_TRAIN_STEPS
    checks["losses_finite"] = bool(torch.isfinite(torch.tensor(losses)).all())
    checks |= dp_layout_checks(
        trainer.model,
        layout,
        world_size=ctx.world_size,
        ep_config=None,
        forward=lambda: probe_forward(trainer.model, ctx.device),
    )
    # Early return: without the capture every line below raises KeyError, and the harness reports `error` over FAIL.
    checks["pinned_step_captured"] = bool(captured) and bool(weights) and bool(trainer_rewards)
    if not checks["pinned_step_captured"]:
        return {"checks": checks, "metrics": metrics}

    rows = captured["input_ids"].shape[0]
    checks["batch_is_chosen_then_rejected_pairs"] = rows == 2 * BATCH_SIZE

    # Anchored on the dataset: the reference's `chunk` shares the trainer's convention, so a collator emitting
    # [rejected ; chosen] would make both sides agree and the swap control still pass.
    tokenized = trainer.train_dataset
    columns = tokenized.column_names
    chosen_column = next(c for c in ("chosen_input_ids", "chosen_ids") if c in columns)
    rejected_column = next(c for c in ("rejected_input_ids", "rejected_ids") if c in columns)
    chosen_rows = {tuple(row[chosen_column]) for row in tokenized}
    rejected_rows = {tuple(row[rejected_column]) for row in tokenized}
    # Anti-vacuity: if a sequence could be both, membership proves nothing about the ordering.
    checks["chosen_and_rejected_are_distinguishable"] = chosen_rows.isdisjoint(rejected_rows)
    captured_rows = [
        tuple(captured["input_ids"][i][: int(captured["attention_mask"][i].sum())].tolist()) for i in range(rows)
    ]
    checks["first_half_is_the_chosen_column"] = all(row in chosen_rows for row in captured_rows[:BATCH_SIZE]) and all(
        row in rejected_rows for row in captured_rows[BATCH_SIZE:]
    )
    # World-wide: the compared loss is a DP mean, and a shard whose rows tokenize alike would flake a per-rank form.
    lengths = captured["attention_mask"].sum(-1)
    checks["batch_is_ragged"] = world_any(int(lengths.min()) < int(lengths.max()), ctx.device)

    scorer = build_model(tokenizer)
    scorer.load_state_dict(weights)
    scorer = scorer.to(ctx.device).eval()
    per_token = token_rewards(scorer, captured["input_ids"], captured["attention_mask"])
    rewards = pool_rewards(per_token, captured["input_ids"], tokenizer.pad_token_id)
    expected = world_mean(bt_objective(rewards), ctx.device)
    logged = losses[PINNED_STEP - 1]
    metrics["pinned_loss"], metrics["pinned_reference"] = logged, expected
    checks["loss_matches_reference"] = abs(logged - expected) < LOSS_REL_TOL * max(1.0, abs(expected))

    chosen, rejected = torch.chunk(rewards, chunks=2)
    margin = float((chosen - rejected).mean())
    metrics["reward_margin"] = margin
    # Anti-vacuity: outside this band every control below collapses onto the real objective.
    checks["margin_in_responsive_region"] = RESPONSIVE_MARGIN[0] < abs(margin) < RESPONSIVE_MARGIN[1]

    # Pinned independently of the loss, which separates the chosen/rejected assignment. Accuracy reads the trainer's
    # own rewards, whose chunk order first_half_is_the_chosen_column anchors.
    entry = [e for e in trainer.state.log_history if "loss" in e][PINNED_STEP - 1]
    trainer_chosen, trainer_rejected = torch.chunk(trainer_rewards[0], chunks=2)
    metric_pins = {
        "accuracy": (
            world_mean(float((trainer_chosen > trainer_rejected).float().mean()), ctx.device),
            ACCURACY_ABS_TOL,
        ),
        "margin": (world_mean(margin, ctx.device), REWARD_ABS_TOL),
        "mean_reward": (world_mean(float(rewards.mean()), ctx.device), REWARD_ABS_TOL),
    }
    for key, (want, tol) in metric_pins.items():
        got = entry.get(key)
        metrics[f"logged_{key}"], metrics[f"expected_{key}"] = (got if got is not None else float("nan")), want
        checks[f"metric_matches_reference[{key}]"] = got is not None and abs(got - want) < tol
    # Anti-vacuity that can fail by data: a batch whose pairs split evenly logs the same accuracy either way round.
    swapped_accuracy = world_mean(float((trainer_rejected > trainer_chosen).float().mean()), ctx.device)
    metrics["control_halves_swapped_accuracy"] = swapped_accuracy
    logged_accuracy = entry.get("accuracy")
    checks["control_halves_swapped_breaks_the_accuracy_pin"] = (
        logged_accuracy is not None and abs(logged_accuracy - swapped_accuracy) > ACCURACY_ABS_TOL
    )
    # Anti-vacuity that can fail by data: ranks whose margins cancel log nearly the same margin either way round.
    logged_margin = entry.get("margin")
    checks["control_halves_swapped_breaks_the_margin_pin"] = (
        logged_margin is not None and abs(logged_margin + metric_pins["margin"][0]) > REWARD_ABS_TOL
    )

    controls = {
        # Chunk-order inversion: trains the model to prefer the rejected answer.
        "halves_swapped": world_mean(bt_objective(rewards, swap=True), ctx.device),
        # Ascent on the preference instead of descent.
        "sign_dropped": world_mean(bt_objective(rewards, drop_sign=True), ctx.device),
        "pooling_at_pad": world_mean(
            bt_objective(pool_rewards(per_token, captured["input_ids"], tokenizer.pad_token_id, at_pad=True)),
            ctx.device,
        ),
    }
    for name, value in controls.items():
        gap = abs(logged - value)
        metrics[f"control_{name}_gap"] = gap
        log(f"  control {name}: reference={value:.6f} gap-from-logged={gap:.3e}")
        checks[f"control_{name}_breaks_the_pin"] = gap > CONTROL_MIN_GAP

    log(
        f"  logged margin per step: {[round(e.get('margin', float('nan')), 3) for e in trainer.state.log_history if 'loss' in e]}"
    )
    log(f"  pinned step {PINNED_STEP}: loss={logged:.6f} reference={expected:.6f} margin={margin:.4f}")
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
