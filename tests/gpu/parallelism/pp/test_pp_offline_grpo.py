#!/usr/bin/env python
"""End-to-end PP through the real OfflineGRPOTrainer at kl_beta == 0, step-1 loss pinned against an unsplit copy.

Each loss type's whole-batch denominator folds into the runner's normalizer while the microbatch loss has the bare
numerator, and every seam yields a silently wrong number. Inert row-padding uses group_size filler 1, since 0 would NaN
via inf x 0.
"""

import sys

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from halo.configs.grpo_offline import OfflineGRPOConfig
from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.grpo.offline.trainer import OfflineGRPOTrainer
from tests.common.distributed import hf_eval_loss_reference, world_mean
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.pp_equivalence import build_dense_model
from tests.common.pp_trainers import (
    build_pp_args,
    build_pp_trainer,
    expect_construction_error,
    losses_identical_across_ranks,
)

PP_SIZE = 2
MICROBATCHES = 2
BATCH = 4
N_STEPS = 10
# 16 completions split 8/8 across dp=2: exactly 2 full batches per DP shard (the eval BatchSampler drops partial tails)
# and a zero gather-trim remainder, so HF's eval_loss is the plain mean over the 4 per-batch losses.
EVAL_GROUP_SIZES = [4, 4, 2, 3, 3]
EVAL_STEPS = 5
MAX_PROMPT = 16
MAX_COMPLETION = 16
MIN_LOG_PROB = -3.0
REL_TOL = 1e-4

# The real Qwen tokenizer feeds tokenize_row, so the tiny model must span its full id range.
VOCAB_SIZE = 151936


def build_dataset(sizes: list[int] | None = None, prompt_offset: int = 0) -> Dataset:
    """Distinct rewards give z-norm advantages of both signs for the clamp; varied sizes make denominators differ per
    split."""
    sizes = sizes if sizes is not None else [2, 3, 4, 3, 2, 4, 3, 3]
    rows = {"prompt": [], "completions": [], "rewards": []}
    for g, size in enumerate(sizes):
        p = g + prompt_offset
        rows["prompt"].append(f"Question {p}: name object number {p} and describe it briefly.")
        rows["completions"].append([f"Answer {p}.{j}: " + "word " * (j + 2) for j in range(size)])
        rows["rewards"].append([float(j) + 0.25 * ((p + j) % 3) for j in range(size)])
    return Dataset.from_dict(rows)


def offline_grpo_reference(
    model,
    input_ids,
    attention_mask,
    labels,
    advantage,
    group_size,
    *,
    loss_type,
    policy_gradient_formulation,
    min_log_prob,
) -> float:
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    logps = torch.log_softmax(logits[:, :-1, :], dim=-1)
    logps = logps.gather(-1, shift_labels.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
    if min_log_prob is not None:
        logps = torch.where((advantage < 0).unsqueeze(1), logps.clamp(min=min_log_prob), logps)
    weights = logps.exp() if policy_gradient_formulation == "prob_weighted" else logps
    per_token_loss = -(weights * advantage.float().unsqueeze(1)) * mask
    group_weights = 1.0 / group_size.float()
    weighted = per_token_loss * group_weights.unsqueeze(1)
    if loss_type == "grpo":
        return float((weighted.sum(1) / mask.sum(1).clamp(min=1)).sum() / group_weights.sum())
    if loss_type == "bnpo":
        return float(weighted.sum() / (mask.float() * group_weights.unsqueeze(1)).sum().clamp(min=1.0))
    return float(weighted.sum() / (group_weights.sum() * MAX_COMPLETION))  # dr_grpo


def build_args(output_dir: str, **overrides) -> OfflineGRPOConfig:
    return build_pp_args(
        OfflineGRPOConfig,
        output_dir,
        **{
            "max_steps": N_STEPS,
            "per_device_train_batch_size": BATCH,
            "per_device_eval_batch_size": BATCH,  # PP freezes eval batches to the train shape.
            "max_prompt_length": MAX_PROMPT,
            "max_completion_length": MAX_COMPLETION,
            "kl_beta": 0.0,
            "advantage_method": "z_norm",
            "loss_type": "grpo",
            "policy_gradient_formulation": "reinforce",
            "min_log_prob": MIN_LOG_PROB,
            **overrides,
        },
    )


def run_trainer(config, args, dataset, tokenizer, eval_dataset=None):
    """With ``eval_dataset``, before training (so an unsplit copy can reproduce the weights): an init-weights
    ``evaluate()``, then a partial-batch ``prediction_step`` (2 rows < the frozen 4) through the inert row padding, its
    loss returned."""
    trainer, steps, evals = build_pp_trainer(
        OfflineGRPOTrainer,
        model=build_dense_model(tokenizer, vocab_size=VOCAB_SIZE),
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=config,
    )
    init_eval_metrics = partial_loss = None
    if eval_dataset is not None:
        init_eval_metrics = trainer.evaluate()
        # COLLECTIVE: every rank calls it with the same rows.
        partial_batch = trainer.data_collator([trainer.eval_dataset[i] for i in range(2)])
        loss, _, _ = trainer.prediction_step(trainer.model, partial_batch, prediction_loss_only=True)
        partial_loss = float(loss)
    # In-loop evals keep appending; trained-weights batches must stay out of the init-weights reference.
    init_eval_calls = list(evals)
    trainer.train()
    return trainer, steps[0], init_eval_calls, init_eval_metrics, partial_loss


@gpu_test_main(exact_world_size=4, prefix="pp_offline_grpo")
def run(ctx):
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    dataset = build_dataset()
    checks, metrics = {}, {}

    # Raises before any collective or model load. This prediction_step returns (loss, None, None) in every mode, so
    # compute_metrics under PP would broadcast the raw logits plane and feed metrics a tensor no non-PP run produces.
    expect_construction_error(
        OfflineGRPOTrainer,
        "compute_metrics_rejected",
        checks,
        "NO predictions",
        model=build_dense_model(tokenizer, vocab_size=VOCAB_SIZE),
        args=build_args(ctx.output_dir + "/metrics"),
        train_dataset=dataset,
        processing_class=tokenizer,
        parallelism_config=config,
        compute_metrics=lambda eval_pred: {"noop": 0.0},
    )

    args = build_args(ctx.output_dir, eval_strategy="steps", eval_steps=EVAL_STEPS)
    eval_dataset = build_dataset(sizes=EVAL_GROUP_SIZES, prompt_offset=100)
    trainer, captured, eval_calls, init_eval_metrics, partial_loss = run_trainer(
        config, args, dataset, tokenizer, eval_dataset=eval_dataset
    )

    checks["accumulation_became_microbatches"] = (
        args.gradient_accumulation_steps == 1 and trainer._pp_n_microbatches == MICROBATCHES
    )
    checks["max_length_derived_from_caps"] = args.max_length == MAX_PROMPT + MAX_COMPLETION
    checks["fixed_shape_batch"] = captured["input_ids"].shape == (BATCH, MAX_PROMPT + MAX_COMPLETION)

    # Captured rows map back to the raw dataset, transform-independently: the equivalence reference consumes
    # batch_transform's output, so a transform bug corrupting both sides identically would pass it.
    completion_index = {}
    for g, completions in enumerate(dataset["completions"]):
        for j, completion in enumerate(completions):
            ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            completion_index[tuple((ids + [tokenizer.eos_token_id])[:MAX_COMPLETION])] = (g, j)
    group_sizes = [len(c) for c in dataset["completions"]]
    rows_pinned = True
    for r in range(BATCH):
        labels_row = captured["labels"][r]
        supervised = (labels_row != -100).nonzero().flatten().tolist()
        key = tuple(labels_row[supervised].tolist())
        if key not in completion_index:
            rows_pinned = False
            continue
        g, _ = completion_index[key]
        real = (captured["attention_mask"][r] == 1).nonzero().flatten().tolist()
        prompt_positions = [p for p in real if p < supervised[0]]
        expected_prompt = tokenizer(dataset["prompt"][g], add_special_tokens=False)["input_ids"][-MAX_PROMPT:]
        if captured["input_ids"][r][prompt_positions].tolist() != expected_prompt:
            rows_pinned = False
        if int(captured["group_size"][r]) != group_sizes[g]:
            rows_pinned = False
    checks["captured_rows_pin_to_raw_dataset"] = rows_pinned

    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    checks["ran_all_steps"] = trainer.state.global_step == N_STEPS and len(losses) == N_STEPS
    checks["losses_finite"] = bool(torch.isfinite(torch.tensor(losses)).all())

    # The runner's per-step divisor must be this loss type's whole-batch denominator.
    expected_normalizer = float((1.0 / captured["group_size"].float()).sum())
    checks["normalizer_is_group_weight_sum"] = abs(captured["num_items"] - expected_normalizer) < 1e-6

    # Step-1 logged loss is the exact objective on the unsplit model over the captured batches.
    reference_model = build_dense_model(tokenizer, vocab_size=VOCAB_SIZE).to(ctx.device).eval()
    reference = offline_grpo_reference(
        reference_model,
        captured["input_ids"],
        captured["attention_mask"],
        captured["labels"],
        captured["advantage"],
        captured["group_size"],
        loss_type="grpo",
        policy_gradient_formulation="reinforce",
        min_log_prob=MIN_LOG_PROB,
    )
    expected = world_mean(reference)
    metrics["step1_loss"], metrics["step1_reference"] = losses[0], expected
    checks["step1_loss_matches_reference"] = abs(losses[0] - expected) < REL_TOL * max(1.0, abs(expected))

    # An unclamped reference must not match (random-init logps ~ -11.9 sit far below the -3.0 floor).
    unclamped = world_mean(
        offline_grpo_reference(
            reference_model,
            captured["input_ids"],
            captured["attention_mask"],
            captured["labels"],
            captured["advantage"],
            captured["group_size"],
            loss_type="grpo",
            policy_gradient_formulation="reinforce",
            min_log_prob=None,
        )
    )
    checks["min_log_prob_clamp_wired"] = abs(losses[0] - unclamped) > 1e-2

    # The captures are the two evaluate() batches plus the direct partial-batch call. Each reference runs over its real
    # rows only, which for the partial capture proves the inert rows contribute 0.
    checks["pad_spec_declares_group_size_filler"] = dict(trainer._pp_adapter.pad_spec) == {"group_size": 1}
    checks["eval_captured_two_batches_plus_partial"] = len(eval_calls) == 3

    def eval_reference(call) -> float:
        real = (call["labels"] != -100).any(dim=-1)
        return offline_grpo_reference(
            reference_model,
            call["input_ids"][real],
            call["attention_mask"][real],
            call["labels"][real],
            call["advantage"][real],
            call["group_size"][real],
            loss_type="grpo",
            policy_gradient_formulation="reinforce",
            min_log_prob=MIN_LOG_PROB,
        )

    step_losses = [eval_reference(call) for call in eval_calls[:2]]
    expected_eval = hf_eval_loss_reference(
        step_losses,
        BATCH,
        remainder=0,
        dp_rank=trainer.parallelism_config.get_data_parallel_rank(),
        dp_size=trainer.parallelism_config.data_parallel_size,
    )
    eval_loss = init_eval_metrics["eval_loss"]
    metrics["init_eval_loss"], metrics["init_eval_reference"] = eval_loss, expected_eval
    checks["eval_loss_matches_reference"] = abs(eval_loss - expected_eval) < REL_TOL * max(1.0, abs(expected_eval))

    checks["eval_normalizer_is_group_weight_sum"] = all(
        abs(call["num_items"] - float((1.0 / call["group_size"][(call["labels"] != -100).any(dim=-1)].float()).sum()))
        < 1e-6
        for call in eval_calls
    )

    # 2 real rows row-padded to the frozen 4 with group_size filler 1: finite and exactly the 2-row objective.
    partial_call = eval_calls[2]
    partial_real = (partial_call["labels"] != -100).any(dim=-1)
    checks["partial_batch_row_padded"] = partial_call["input_ids"].size(0) == BATCH and int(partial_real.sum()) == 2
    partial_reference = eval_reference(partial_call)
    metrics["partial_loss"], metrics["partial_reference"] = partial_loss, partial_reference
    checks["partial_batch_loss_matches"] = abs(partial_loss - partial_reference) < REL_TOL * max(
        1.0, abs(partial_reference)
    )

    checks["eval_loss_identical_across_ranks"] = losses_identical_across_ranks([eval_loss])

    inloop_eval = [e["eval_loss"] for e in trainer.state.log_history if "eval_loss" in e]
    checks["inloop_evals_ran_finite"] = len(inloop_eval) == 2 and bool(torch.isfinite(torch.tensor(inloop_eval)).all())

    checks["loss_decreased"] = losses[-1] < losses[0] - 0.01
    metrics["first_loss"], metrics["last_loss"] = losses[0], losses[-1]

    checks["losses_identical_across_ranks"] = losses_identical_across_ranks(losses)

    grad_norms = [e["grad_norm"] for e in trainer.state.log_history if "grad_norm" in e]
    checks["grad_norm_logged_finite"] = bool(grad_norms) and bool(torch.isfinite(torch.tensor(grad_norms)).all())

    dr_args = build_args(
        ctx.output_dir + "/dr",
        max_steps=2,
        loss_type="dr_grpo",
        policy_gradient_formulation="prob_weighted",
        min_log_prob=None,
    )
    dr_trainer, dr_captured, _, _, _ = run_trainer(config, dr_args, dataset, tokenizer)
    dr_losses = [e["loss"] for e in dr_trainer.state.log_history if "loss" in e]
    dr_expected = world_mean(
        offline_grpo_reference(
            reference_model,
            dr_captured["input_ids"],
            dr_captured["attention_mask"],
            dr_captured["labels"],
            dr_captured["advantage"],
            dr_captured["group_size"],
            loss_type="dr_grpo",
            policy_gradient_formulation="prob_weighted",
            min_log_prob=None,
        )
    )
    metrics["dr_grpo_step1_loss"], metrics["dr_grpo_step1_reference"] = dr_losses[0], dr_expected
    checks["dr_grpo_step1_loss_matches_reference"] = abs(dr_losses[0] - dr_expected) < REL_TOL * max(
        1.0, abs(dr_expected)
    )
    dr_expected_normalizer = float((1.0 / dr_captured["group_size"].float()).sum()) * MAX_COMPLETION
    checks["dr_grpo_normalizer"] = abs(dr_captured["num_items"] - dr_expected_normalizer) < 1e-5

    bnpo_args = build_args(ctx.output_dir + "/bnpo", max_steps=2, loss_type="bnpo", min_log_prob=None)
    bnpo_trainer, bnpo_captured, _, _, _ = run_trainer(config, bnpo_args, dataset, tokenizer)
    bnpo_losses = [e["loss"] for e in bnpo_trainer.state.log_history if "loss" in e]
    bnpo_expected = world_mean(
        offline_grpo_reference(
            reference_model,
            bnpo_captured["input_ids"],
            bnpo_captured["attention_mask"],
            bnpo_captured["labels"],
            bnpo_captured["advantage"],
            bnpo_captured["group_size"],
            loss_type="bnpo",
            policy_gradient_formulation="reinforce",
            min_log_prob=None,
        )
    )
    metrics["bnpo_step1_loss"], metrics["bnpo_step1_reference"] = bnpo_losses[0], bnpo_expected
    checks["bnpo_step1_loss_matches_reference"] = abs(bnpo_losses[0] - bnpo_expected) < REL_TOL * max(
        1.0, abs(bnpo_expected)
    )

    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
