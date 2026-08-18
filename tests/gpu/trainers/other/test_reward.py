#!/usr/bin/env python
"""DistributedRewardTrainer end-to-end, with the logged loss pinned to TRL 1.6's exact BT objective.

A reward model's loss is ``-logsigmoid(r_chosen - r_rejected)`` over a batch that TRL collates as
``[all chosen ; all rejected]`` and scores at each row's last non-pad token. Three things there fail
silently rather than loudly: the chunk order (``torch.chunk`` reading the halves the wrong way round
inverts the preference — the model learns to prefer the rejected answer), the pooling position (a
fixed index instead of the last non-pad token scores padding), and the sign. All keep the loss finite
and the step count intact, so finiteness cannot see any of them.

So the reward head is scored independently here — this test pools the logits itself, at the position
rule transformers documents, from an independently constructed copy of the model holding the weight
snapshot the trainer had at the pinned step — and:

  * the logged loss at ``PINNED_STEP`` must equal that reference objective on the batch the trainer
    actually consumed;
  * the captured rows must map back to the tokenized dataset's own ``chosen``/``rejected`` columns,
    which is what pins the chunk ORDER — the reference's own ``chunk`` shares the trainer's
    convention, so nothing else here would notice a collator emitting the halves reversed;
  * the logged ``accuracy`` / ``mean_reward`` / ``margin`` metrics must equal the same independent
    quantities, which pins the chosen/rejected assignment separately from the loss;
  * three NEGATIVE CONTROLS must each miss the logged loss by far more than the pin's tolerance —
    swapping the two halves (the chunk-order inversion), dropping the sign, and pooling at the last
    *padded* position instead of the last real token.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/other/test_reward.py
"""

import random

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig

from src.args.common_script_args import CommonScriptArguments
from src.distributed.parallelism_config import ParallelismConfig
from src.models.loading.tokenizer_setup import setup_model_and_tokenizer
from src.trainers.reward.bradley_terry import DistributedRewardTrainer
from tests.common.distributed import (
    ensure_model_downloaded,
    snapshot_full_weights,
    world_any,
    world_mean,
    world_min,
)
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.tolerances import TOL
from tests.common.utils import log

MODEL_NAME = QWEN3_0_6B
NUM_TRAIN_SAMPLES = 64
NUM_EVAL_SAMPLES = 16
MAX_SEQ_LENGTH = 1024
NUM_TRAIN_STEPS = 10
# >1 pair per device so the batch is ragged; the `pooling_at_pad` control is only distinguishable
# once some row actually carries padding.
BATCH_SIZE = 4
# Tiny on purpose: the O(1) margin the controls need comes from the randomly initialised `score`
# head, not from training, and this rate keeps it inside RESPONSIVE_MARGIN instead of saturating
# -logsigmoid to 0 — where the pin degenerates into comparing two zeros.
LEARNING_RATE = 1e-7
SEED = 42
# Late, so the pinned weights are ones the trainer reached — that puts the snapshot path under test.
PINNED_STEP = NUM_TRAIN_STEPS

LOSS_REL_TOL = TOL.exact_objective_rel
# A control must show the pin would FAIL, so its threshold IS the pin's tolerance.
CONTROL_MIN_GAP = LOSS_REL_TOL
# bf16 pooled rewards of magnitude ~3 differenced against an fp32 reference (measured 0.023). A
# swapped chosen/rejected assignment flips the margin's sign (~1.2), so this still catches it.
REWARD_ABS_TOL = 0.05
# Below this band chosen and rejected score alike and every control collapses onto log(2); above it
# -logsigmoid saturates to ~0 and the sign-drop control becomes undetectable.
RESPONSIVE_MARGIN = (0.3, 5.0)
# The trainer compares bf16 rewards, this test fp32: a pair inside bf16 noise (~0.03 at magnitude ~3)
# can flip and move `accuracy` by 1/BATCH_SIZE, so the exact rate is only pinnable clear of it.
MIN_PAIR_GAP = 0.1


def create_reward_dataset(num_samples: int, tokenizer, seed: int = SEED) -> Dataset:
    """Synthetic math preference data as chat-templated conversation strings.

    30% of rows are multi-turn, which is what makes the batch ragged enough for the pooling rule to
    matter. TRL's RewardTrainer takes ``{"chosen": str, "rejected": str}`` complete conversations.
    """
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
        b = random.randint(1, min(a, 50))  # b <= a keeps subtraction/division clean
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
    # The scalar `score` head is not in the checkpoint; seed before construction (the Trainer seeds
    # later) or the margin the controls depend on differs run to run.
    torch.manual_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    # Through the script's own seam, never a hand-written config.pad_token_id: the pooling rule
    # pinned here reads that id back, so setting it by hand would survive a regression in the sync.
    setup_model_and_tokenizer(CommonScriptArguments(), model, tokenizer, MAX_SEQ_LENGTH)
    return model


def token_rewards(model, input_ids, attention_mask) -> torch.Tensor:
    """Per-TOKEN reward logits ``[B, T]`` in fp32: the backbone plus the scalar head, run by hand.

    ``model(...)`` returns logits that are already pooled ``[B, 1]``, so scoring the head directly is
    what lets this test own the pooling rule and therefore detect a wrong pooling position.
    """
    with torch.no_grad():
        hidden = model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        return model.score(hidden).float().squeeze(-1)


def pool_rewards(token_logits, input_ids, pad_token_id, *, at_pad: bool = False) -> torch.Tensor:
    """Per-row scalar reward at the LAST NON-PAD token — transformers' own sequence-classification
    position rule, reimplemented here. ``at_pad`` pools at the final column instead, which is the
    control for exactly that bug.
    """
    rows, seq_len = input_ids.shape
    if at_pad:
        positions = torch.full((rows,), seq_len - 1, device=input_ids.device, dtype=torch.long)
    else:
        non_pad = (input_ids != pad_token_id).to(torch.int32)
        token_indices = torch.arange(seq_len, device=input_ids.device, dtype=torch.int32)
        positions = (token_indices * non_pad).argmax(-1)
    return token_logits[torch.arange(rows, device=input_ids.device), positions]


def bt_objective(rewards: torch.Tensor, *, swap: bool = False, drop_sign: bool = False) -> float:
    """TRL's exact Bradley-Terry loss: ``-logsigmoid(chosen - rejected)`` meaned over pairs.

    The row order is TRL's ``[all chosen ; all rejected]``, recovered with a single ``chunk``.
    ``swap`` reads the halves the wrong way round; ``drop_sign`` omits the negation.
    """
    chosen, rejected = torch.chunk(rewards, chunks=2)
    if swap:
        chosen, rejected = rejected, chosen
    difference = chosen - rejected
    return float(difference.sigmoid().log().mean() if drop_sign else -F.logsigmoid(difference).mean())


@gpu_test_main(min_world_size=2, prefix="test_reward")
def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}

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
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataset_num_proc=1,
        dataloader_num_workers=0,
        fsdp="",  # the mixin owns FSDP wrapping
    )

    trainer = DistributedRewardTrainer(
        model=build_model(tokenizer),
        args=config,
        train_dataset=create_reward_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED),
        eval_dataset=create_reward_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1),
        processing_class=tokenizer,
        parallelism_config=ParallelismConfig(),
    )
    ctx.on_teardown(trainer.cleanup_ep)

    # The reference must score exactly the pinned step's rows with exactly its weights.
    captured: dict = {}
    weights: dict = {}
    real_compute_loss = trainer.compute_loss

    def spying_compute_loss(model, inputs, *args, **kwargs):
        # global_step is still N-1 while step N's loss is being computed.
        if trainer.state.global_step == PINNED_STEP - 1 and not captured:
            captured.update({k: v.clone() for k, v in inputs.items() if torch.is_tensor(v)})
            weights.update(snapshot_full_weights(model))
        return real_compute_loss(model, inputs, *args, **kwargs)

    trainer.compute_loss = spying_compute_loss
    trainer.train()

    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    checks["ran_all_steps"] = trainer.state.global_step == NUM_TRAIN_STEPS and len(losses) == NUM_TRAIN_STEPS
    checks["losses_finite"] = bool(torch.isfinite(torch.tensor(losses)).all())
    # Early return, not a bare check: without the capture every line below raises KeyError and the
    # harness reports `error` instead of this check failing.
    checks["pinned_step_captured"] = bool(captured) and bool(weights)
    if not checks["pinned_step_captured"]:
        return {"checks": checks, "metrics": metrics}

    rows = captured["input_ids"].shape[0]
    checks["batch_is_chosen_then_rejected_pairs"] = rows == 2 * BATCH_SIZE

    # Anchor WHICH half is which against the trainer's own tokenized dataset: the reference's
    # `chunk` shares the trainer's convention, so a collator emitting [rejected ; chosen] would make
    # both sides agree and the swap control would still pass.
    tokenized = trainer.train_dataset
    columns = tokenized.column_names
    chosen_column = next(c for c in ("chosen_input_ids", "chosen_ids") if c in columns)
    rejected_column = next(c for c in ("rejected_input_ids", "rejected_ids") if c in columns)
    chosen_rows = {tuple(row[chosen_column]) for row in tokenized}
    rejected_rows = {tuple(row[rejected_column]) for row in tokenized}
    # ANTI-VACUITY: if a sequence could be both, membership proves nothing about the ordering.
    checks["chosen_and_rejected_are_distinguishable"] = chosen_rows.isdisjoint(rejected_rows)
    captured_rows = [
        tuple(captured["input_ids"][i][: int(captured["attention_mask"][i].sum())].tolist()) for i in range(rows)
    ]
    checks["first_half_is_the_chosen_column"] = all(row in chosen_rows for row in captured_rows[:BATCH_SIZE]) and all(
        row in rejected_rows for row in captured_rows[BATCH_SIZE:]
    )
    # World-wide, not per-shard: the compared loss is a DP mean, so padding need only exist
    # somewhere, and a shard whose rows tokenize alike would flake a per-rank form.
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
    # ANTI-VACUITY: outside this band every control below collapses onto the real objective.
    checks["margin_in_responsive_region"] = RESPONSIVE_MARGIN[0] < abs(margin) < RESPONSIVE_MARGIN[1]
    # Precondition for the exact `accuracy` pin: no pair may sit inside bf16 reward noise, where the
    # trainer's bf16 comparison and this test's fp32 one can disagree on its sign.
    min_pair_gap = float((chosen - rejected).abs().min())
    metrics["min_pair_gap"] = min_pair_gap
    checks["pair_gaps_clear_of_bf16_noise"] = world_min(min_pair_gap, ctx.device) > MIN_PAIR_GAP

    # Pinned independently of the loss, which is what separates the chosen/rejected assignment.
    entry = [e for e in trainer.state.log_history if "loss" in e][PINNED_STEP - 1]
    metric_pins = {
        # An exact rate over pair comparisons — no accumulation, so no dtype slack.
        "accuracy": (world_mean(float((chosen > rejected).float().mean()), ctx.device), 1e-6),
        "margin": (world_mean(margin, ctx.device), REWARD_ABS_TOL),
        "mean_reward": (world_mean(float(rewards.mean()), ctx.device), REWARD_ABS_TOL),
    }
    for key, (want, tol) in metric_pins.items():
        got = entry.get(key)
        metrics[f"logged_{key}"], metrics[f"expected_{key}"] = (got if got is not None else float("nan")), want
        checks[f"metric_matches_reference[{key}]"] = got is not None and abs(got - want) < tol

    # NEGATIVE CONTROLS: each degenerate objective must MISS the logged loss.
    controls = {
        # Chunk-order inversion: the model is trained to prefer the rejected answer.
        "halves_swapped": world_mean(bt_objective(rewards, swap=True), ctx.device),
        # Negation dropped: ascent on the preference instead of descent.
        "sign_dropped": world_mean(bt_objective(rewards, drop_sign=True), ctx.device),
        # Pooled at the final column, so padded rows score padding.
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
