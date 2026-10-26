#!/usr/bin/env python
"""Offline GRPO with a KL term (kl_beta > 0) through the real OfflineGRPOTrainer under pipeline parallelism.

No pipeline rank holds a full model, so the KL reference is scored once at construction by a
forward-only sweep through the schedule and shipped per completion token as a dataset column. Every
seam yields a silently wrong objective rather than a crash, so each is pinned against an unsplit
fp32 copy of the init weights:

  * the sweep itself — every row's column, on the training and evaluation datasets, equals the
    unsplit model's per-token log-probs of that completion (DP-sharded sweep, gathered over the
    stage group, with a partial final batch per shard that the sweep row-pads and trims);
  * the placement — the reference extra target the step receives carries each row's column values
    at exactly the shifted positions the loss mask scores, zeros elsewhere;
  * the objective — with the column deliberately perturbed after the sweep (so the KL is nonzero at
    step 1, where policy == reference would otherwise zero it), the step-1 logged loss and the
    init-weights eval_loss equal the exact kl_beta objective, including the negative-advantage
    min_log_prob clamp on policy and reference and the k3 cap; and differ from the kl_beta=0 one;
  * construction accepts kl_beta > 0 and builds no resident reference model.

Topology: 4 GPUs as 2 simulated nodes of 2, pp=2 x dp=2 (stage FSDP over 2 ranks).

Run: torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_offline_grpo_kl.py
"""

import math
import sys

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from src.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.grpo.objective.logratio import KL_LOGRATIO_CLAMP
from src.trainers.grpo.offline.trainer import OfflineGRPOTrainer
from tests.common.distributed import hf_eval_loss_reference, shared_output_dir, world_mean
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.pp_equivalence import build_dense_model
from tests.gpu.parallelism.pp.test_pp_offline_grpo import (
    BATCH,
    EVAL_GROUP_SIZES,
    MAX_COMPLETION,
    MIN_LOG_PROB,
    PP_SIZE,
    REL_TOL,
    VOCAB_SIZE,
    build_dataset,
    make_args,
)

KL_BETA = 0.2
N_STEPS = 6
# 26 rows: each dp shard of 13 is three full batches of 4 plus a partial final batch of one row.
TRAIN_GROUP_SIZES = [2, 3, 4, 3, 2, 4, 3, 3, 2]
# Per-token log-prob tolerance for the pipelined sweep vs the unsplit model: both fp32, the chain
# reassociates the residual stream across the stage boundary and pads differently.
SWEEP_ABS_TOL = 2e-3
PERTURB_SEED = 7


def offline_grpo_kl_reference(
    model,
    input_ids,
    attention_mask,
    labels,
    advantage,
    group_size,
    ref_logps,
    *,
    loss_type,
    policy_gradient_formulation,
    min_log_prob,
    beta,
) -> float:
    """The exact offline-GRPO objective with the k3 KL term on an unsplit model, fp32, independent of src/.

    ``ref_logps`` is on the shifted ``[B, S-1]`` grid (the extra target the step receives).
    """
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    logps = torch.log_softmax(logits[:, :-1, :], dim=-1)
    logps = logps.gather(-1, shift_labels.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
    ref = ref_logps.float()
    negative = (advantage < 0).unsqueeze(1)
    if min_log_prob is not None:
        logps = torch.where(negative, logps.clamp(min=min_log_prob), logps)
        ref = torch.where(negative, ref.clamp(min=min_log_prob), ref)
    weights = logps.exp() if policy_gradient_formulation == "prob_weighted" else logps
    per_token_loss = -(weights * advantage.float().unsqueeze(1))
    if beta != 0.0:
        ref = torch.minimum(ref, logps + KL_LOGRATIO_CLAMP)
        per_token_loss = per_token_loss + beta * (torch.exp(ref - logps) - (ref - logps) - 1)
    group_weights = 1.0 / group_size.float()
    weighted = per_token_loss * mask * group_weights.unsqueeze(1)
    if loss_type == "grpo":
        return float((weighted.sum(1) / mask.sum(1).clamp(min=1)).sum() / group_weights.sum())
    if loss_type == "bnpo":
        return float(weighted.sum() / (mask.float() * group_weights.unsqueeze(1)).sum().clamp(min=1.0))
    return float(weighted.sum() / (group_weights.sum() * MAX_COMPLETION))  # dr_grpo


def unsplit_completion_logps(model, device, prompt_ids: list[int], completion_ids: list[int]) -> torch.Tensor:
    """Per-completion-token log-probs of one row under the unsplit model, unpadded, fp32."""
    ids = torch.tensor([prompt_ids + completion_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits.float()[0]
    logps = torch.log_softmax(logits, dim=-1)
    prompt_width = len(prompt_ids)
    return torch.stack(
        [logps[prompt_width - 1 + j, token] for j, token in enumerate(completion_ids)]
        if completion_ids
        else [torch.zeros((), device=device)][:0]
    ).cpu()


def sweep_matches_unsplit(dataset, reference_model, device) -> tuple[bool, float]:
    """Every row's column == the unsplit model's completion log-probs (length and values)."""
    worst = 0.0
    for row in dataset:
        column = torch.tensor(row[REF_PER_TOKEN_LOGPS_COLUMN], dtype=torch.float32)
        if len(column) != len(row["completion_input_ids"]):
            return False, float("inf")
        expected = unsplit_completion_logps(
            reference_model, device, row["prompt_input_ids"], row["completion_input_ids"]
        )
        if len(column):
            diff = float((column - expected).abs().max())
            worst = max(worst, diff) if math.isfinite(diff) else math.inf
    return worst < SWEEP_ABS_TOL, worst


def perturb_column(dataset, seed: int):
    """Replace the swept column with column + U(-1, 1) noise, rank-identically (seeded)."""
    generator = torch.Generator().manual_seed(seed)
    perturbed = [
        (torch.tensor(values, dtype=torch.float32) + torch.rand(len(values), generator=generator) * 2 - 1).tolist()
        for values in dataset[REF_PER_TOKEN_LOGPS_COLUMN]
    ]
    return dataset.remove_columns([REF_PER_TOKEN_LOGPS_COLUMN]).add_column(REF_PER_TOKEN_LOGPS_COLUMN, perturbed)


def run_trainer(config, args, dataset, tokenizer, eval_dataset):
    """Build the PP trainer (the sweep runs inside construction), perturb both columns, spy, eval, train."""
    trainer = OfflineGRPOTrainer(
        model=build_dense_model(tokenizer, vocab_size=VOCAB_SIZE),
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=config,
    )
    swept = {"train": trainer.train_dataset, "eval": trainer.eval_dataset}
    trainer.train_dataset = perturb_column(trainer.train_dataset, PERTURB_SEED)
    trainer.eval_dataset = perturb_column(trainer.eval_dataset, PERTURB_SEED + 1)

    captured: dict = {}
    real_step = trainer._pp_runner.step

    def spying_step(input_ids, labels, **kw):
        if "input_ids" not in captured:
            captured["input_ids"] = input_ids.clone()
            captured["labels"] = labels.clone()
            captured["attention_mask"] = kw["attention_mask"].clone()
            captured.update({k: v.clone() for k, v in kw["extra_targets"].items()})
        return real_step(input_ids, labels, **kw)

    trainer._pp_runner.step = spying_step

    eval_calls: list[dict] = []
    real_eval = trainer._pp_runner.eval_loss

    def spying_eval(input_ids, labels, **kw):
        eval_calls.append(
            {
                "input_ids": input_ids.clone(),
                "labels": labels.clone(),
                "attention_mask": kw["attention_mask"].clone(),
                **{k: v.clone() for k, v in kw["extra_targets"].items()},
            }
        )
        return real_eval(input_ids, labels, **kw)

    trainer._pp_runner.eval_loss = spying_eval
    init_eval_metrics = trainer.evaluate()
    init_eval_calls = list(eval_calls)
    trainer.train()
    return trainer, swept, captured, init_eval_calls, init_eval_metrics


@gpu_test_main(exact_world_size=4, prefix="pp_offline_grpo_kl")
def run(ctx):
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = build_dataset(sizes=TRAIN_GROUP_SIZES)
    eval_dataset = build_dataset(sizes=EVAL_GROUP_SIZES, prompt_offset=100)
    checks, metrics = {}, {}
    dp_size = config.data_parallel_size
    checks["fixture_has_a_partial_sweep_batch"] = (sum(TRAIN_GROUP_SIZES) // dp_size) % BATCH != 0

    # The run-start reference cache lives under output_dir: one rank writes it, every rank maps it.
    output_dir = shared_output_dir(ctx)
    args = make_args(output_dir, max_steps=N_STEPS, kl_beta=KL_BETA, eval_strategy="steps", eval_steps=N_STEPS)
    trainer, swept, captured, eval_calls, init_eval_metrics = run_trainer(
        config, args, dataset, tokenizer, eval_dataset
    )
    checks["constructs_with_kl_beta_and_no_resident_reference"] = trainer.beta == KL_BETA and trainer.ref_model is None
    checks["adapter_ships_reference_extra_target"] = (
        REF_PER_TOKEN_LOGPS_COLUMN in trainer._pp_adapter.extra_target_keys
    )
    checks["sweep_wrote_both_columns"] = all(
        REF_PER_TOKEN_LOGPS_COLUMN in ds.column_names and len(ds) == n
        for ds, n in ((swept["train"], sum(TRAIN_GROUP_SIZES)), (swept["eval"], sum(EVAL_GROUP_SIZES)))
    )

    # ── The sweep: every row of both datasets, against the unsplit init model ────────────────────
    reference_model = build_dense_model(tokenizer, vocab_size=VOCAB_SIZE).to(ctx.device).eval()
    ok, worst = sweep_matches_unsplit(swept["train"], reference_model, ctx.device)
    checks["train_sweep_matches_unsplit_model"], metrics["train_sweep_worst_abs"] = ok, worst
    ok, worst = sweep_matches_unsplit(swept["eval"], reference_model, ctx.device)
    checks["eval_sweep_matches_unsplit_model"], metrics["eval_sweep_worst_abs"] = ok, worst

    # ── Placement: the step's reference grid carries each row's (perturbed) column at the scored
    # positions and nothing elsewhere. Rows are pinned to the dataset through their label span.
    by_completion = {tuple(ids): i for i, ids in enumerate(trainer.train_dataset["completion_input_ids"])}
    grid = captured[REF_PER_TOKEN_LOGPS_COLUMN]
    placed = grid.shape == (BATCH, args.max_length - 1)
    for r in range(BATCH):
        positions = (captured["labels"][r] != -100).nonzero().flatten().tolist()
        row = by_completion.get(tuple(captured["labels"][r][positions].tolist()))
        if row is None or not positions:
            placed = False
            continue
        expected = torch.tensor(trainer.train_dataset[row][REF_PER_TOKEN_LOGPS_COLUMN], device=grid.device)
        span = grid[r, positions[0] - 1 : positions[-1]]
        outside = grid[r].clone()
        outside[positions[0] - 1 : positions[-1]] = 0
        placed &= bool(torch.allclose(span, expected, atol=1e-6)) and bool(outside.eq(0).all())
    checks["reference_grid_placed_at_scored_positions"] = placed

    # ── The objective at step 1 (policy == init; reference perturbed, so the KL is live) ─────────
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    checks["ran_all_steps"] = trainer.state.global_step == N_STEPS and len(losses) == N_STEPS
    checks["losses_finite"] = bool(torch.isfinite(torch.tensor(losses)).all())

    def reference(call, beta, **overrides) -> float:
        real = (call["labels"] != -100).any(dim=-1)
        params = {"loss_type": "grpo", "policy_gradient_formulation": "reinforce", "min_log_prob": MIN_LOG_PROB}
        params.update(overrides)
        return offline_grpo_kl_reference(
            reference_model,
            call["input_ids"][real],
            call["attention_mask"][real],
            call["labels"][real],
            call["advantage"][real],
            call["group_size"][real],
            call[REF_PER_TOKEN_LOGPS_COLUMN][real],
            beta=beta,
            **params,
        )

    expected = world_mean(reference(captured, KL_BETA))
    metrics["step1_loss"], metrics["step1_reference"] = losses[0], expected
    checks["step1_loss_matches_kl_reference"] = abs(losses[0] - expected) < REL_TOL * max(1.0, abs(expected))
    without_kl = world_mean(reference(captured, 0.0))
    metrics["step1_reference_without_kl"] = without_kl
    checks["kl_term_enters_the_loss"] = abs(losses[0] - without_kl) > 1e-2

    # ── Eval leg on init weights: HF's mean over the per-batch objectives (zero remainder) ───────
    step_losses = [reference(call, KL_BETA) for call in eval_calls]
    expected_eval = hf_eval_loss_reference(
        step_losses,
        BATCH,
        remainder=0,
        dp_rank=trainer.parallelism_config.get_data_parallel_rank(),
        dp_size=trainer.parallelism_config.data_parallel_size,
    )
    eval_loss = init_eval_metrics["eval_loss"]
    metrics["init_eval_loss"], metrics["init_eval_reference"] = eval_loss, expected_eval
    checks["eval_loss_matches_kl_reference"] = abs(eval_loss - expected_eval) < REL_TOL * max(1.0, abs(expected_eval))
    checks["eval_captured_full_batches"] = len(eval_calls) == 2 and all(
        bool((call["labels"] != -100).any(dim=-1).all()) for call in eval_calls
    )

    local = torch.tensor(losses + [eval_loss], device=ctx.device)
    gathered = [torch.zeros_like(local) for _ in range(ctx.world_size)]
    dist.all_gather(gathered, local)
    checks["losses_identical_across_ranks"] = all(torch.allclose(local, g, atol=1e-6) for g in gathered)
    checks["loss_decreased"] = losses[-1] < losses[0] - 0.01
    metrics["first_loss"], metrics["last_loss"] = losses[0], losses[-1]
    grad_norms = [e["grad_norm"] for e in trainer.state.log_history if "grad_norm" in e]
    checks["grad_norm_logged_finite"] = bool(grad_norms) and bool(torch.isfinite(torch.tensor(grad_norms)).all())

    # ── bnpo / prob_weighted / no clamp: the KL rides every loss type's numerator ─────────────────
    bnpo_args = make_args(
        output_dir + "/bnpo",
        max_steps=2,
        kl_beta=KL_BETA,
        loss_type="bnpo",
        policy_gradient_formulation="prob_weighted",
        min_log_prob=None,
        eval_strategy="steps",
        eval_steps=2,
    )
    bnpo_trainer, _, bnpo_captured, _, _ = run_trainer(config, bnpo_args, dataset, tokenizer, eval_dataset)
    bnpo_losses = [e["loss"] for e in bnpo_trainer.state.log_history if "loss" in e]
    bnpo_expected = world_mean(
        reference(
            bnpo_captured, KL_BETA, loss_type="bnpo", policy_gradient_formulation="prob_weighted", min_log_prob=None
        )
    )
    metrics["bnpo_step1_loss"], metrics["bnpo_step1_reference"] = bnpo_losses[0], bnpo_expected
    checks["bnpo_step1_loss_matches_kl_reference"] = abs(bnpo_losses[0] - bnpo_expected) < REL_TOL * max(
        1.0, abs(bnpo_expected)
    )

    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
