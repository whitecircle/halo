#!/usr/bin/env python
"""End-to-end pipeline parallelism through the real DistributedKTOTrainer (apo_zero_unpaired).

KTO under PP is restricted to the KL-free ``apo_zero_unpaired`` loss with reference log-probs
shipped as the ``ref_logps`` dataset column — no reference model exists anywhere, and rows are
independent (no pair layout). Every seam produces a silently wrong number, not a crash, so the
step-1 logged loss is pinned against TRL's exact objective computed on an unsplit copy of the
model over the captured batches:

  * labels derived from completion_mask (the positions TRL's shifted completion_mask scores)
  * ref_logps and the per-row desirability label riding as row-aligned extra targets
  * the row-count normalizer (TRL's loss is a per-replica nanmean over rows)
  * desirable_weight / undesirable_weight on their respective sides
  * the single-sided branch: a run whose every microbatch is all-desirable (empty rejected side)
  * loss-only PP evaluation: init-weights eval_loss pinned against the exact per-batch objective
    with HF's gather/trim averaging reproduced, a partial final eval batch proving the inert
    row-padding contributes exactly 0, and two in-loop evals staying finite and rank-identical
  * construction-time rejections (default 'kto' loss, precompute off, missing ref_logps column
    in the train and eval datasets, live ref model, compute_metrics)

Topology: 4 GPUs as 2 simulated nodes of 2, pp=2 x dp=2 (stage FSDP over 2 ranks).

Run: torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_kto_apo.py
"""

import sys

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import AutoTokenizer
from trl.experimental.kto import KTOConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.preference.kto import DistributedKTOTrainer
from tests.common.distributed import eval_loss_over_examples, hf_eval_loss_reference, world_mean
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.pp_equivalence import build_dense_model
from tests.common.utils import log

PP_SIZE = 2
MICROBATCHES = 2
ROWS_PER_RANK = 8  # per_device_train_batch_size; unpaired — one row per example
SEQ_LEN = 48
N_STEPS = 10
# 28 rows over dp=2 at eval batch 8 (even_batches off) = batches of 8/8/8/4 rows: one DP shard
# ends on a partial batch, so the eval leg exercises the inert row-padding for real.
EVAL_ROWS = 28
EVAL_STEPS = 5
# Small beta keeps beta*logratio out of sigmoid saturation for a random-init policy (per-sequence
# logps ~ -12 * completion length against synthetic refs), so gradients stay alive.
BETA = 0.05
# Non-default side weights so the equivalence check pins the weighting, not just the sigmoids.
DESIRABLE_WEIGHT = 1.3
UNDESIRABLE_WEIGHT = 0.7
REL_TOL = 1e-4

# The real Qwen tokenizer tokenizes the text rows, so the tiny model must span its full id range.
VOCAB_SIZE = 151936


def build_dataset(n_rows: int = 32, seed: int = 7, with_ref_column: bool = True, all_desirable: bool = False):
    """Unpaired text rows with ragged completion lengths plus a synthetic precomputed ref column.

    Mixed labels by default (2 desirable : 1 undesirable — also exercises unequal side counts);
    ``all_desirable`` makes every microbatch single-sided (the empty-rejected branch). The ref
    values are synthetic scalars — the reference computation consumes the same captured values, so
    only their transport (column → collator → per-row extra target) is under test.
    """
    g = torch.Generator().manual_seed(seed)
    rows = {"prompt": [], "completion": [], "label": []}
    if with_ref_column:
        rows["ref_logps"] = []
    for i in range(n_rows):
        rows["prompt"].append(f"Q{i}: what is {i} plus {i}?")
        rows["completion"].append(" The answer is " + " ".join(str(2 * i + j) for j in range(1 + i % 4)) + ".")
        rows["label"].append(True if all_desirable else i % 3 != 0)
        if with_ref_column:
            # Same magnitude as a random-init policy's seq logps (~ -12/token over the full vocab,
            # 5-10 completion tokens), so beta*logratio stays out of the sigmoid tails — gradients
            # stay alive and zeroing the refs visibly moves the loss (the transport check).
            rows["ref_logps"].append(-float(70 + 40 * torch.rand(1, generator=g)))
    return Dataset.from_dict(rows)


def kto_reference(model, input_ids, attention_mask, labels, ref_logps, row_labels) -> float:
    """TRL's exact apo_zero_unpaired objective with precomputed refs, fp32, independent of src/.

    Per-row weighted losses then a mean over all rows — identical to TRL's
    ``cat(w_d·chosen, w_u·rejected).nanmean()`` since every row lands in exactly one side.
    """
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    token_logps = torch.log_softmax(logits[:, :-1, :], dim=-1)
    token_logps = token_logps.gather(-1, shift_labels.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1) * mask
    logratios = token_logps.sum(-1) - ref_logps
    desirable = row_labels.bool()
    per_row = torch.where(
        desirable,
        DESIRABLE_WEIGHT * (1 - torch.sigmoid(BETA * logratios)),
        UNDESIRABLE_WEIGHT * torch.sigmoid(BETA * logratios),
    )
    return float(per_row.mean())


def make_args(output_dir: str, **overrides):
    base = {
        "output_dir": output_dir,
        "max_steps": N_STEPS,
        "per_device_train_batch_size": ROWS_PER_RANK,
        "gradient_accumulation_steps": MICROBATCHES,  # becomes the pipeline's microbatch count
        "learning_rate": 5e-3,
        "lr_scheduler_type": "constant",
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
        "logging_nan_inf_filter": False,
        "seed": 42,
        "bf16": False,
        "gradient_checkpointing": False,
        "max_length": SEQ_LEN,
        "beta": BETA,
        "desirable_weight": DESIRABLE_WEIGHT,
        "undesirable_weight": UNDESIRABLE_WEIGHT,
        "loss_type": "apo_zero_unpaired",
        "precompute_ref_log_probs": True,
        "per_device_eval_batch_size": ROWS_PER_RANK,  # PP freezes eval batches to the train shape
        # even_batches would complete the tail eval batch by wrapping around to the dataset start;
        # off, the final batch stays partial and must be row-padded with inert rows.
        "accelerator_config": {"even_batches": False},
    }
    base.update(overrides)
    return KTOConfig(**base)


def expect_construction_error(name: str, checks: dict, match: str, **trainer_kwargs) -> None:
    """Construct on every rank expecting a uniform pre-collective ValueError."""
    try:
        DistributedKTOTrainer(**trainer_kwargs)
        checks[name] = False
    except ValueError as exc:
        checks[name] = match in str(exc)
        if not checks[name]:
            log(f"[{name}] raised without the expected mechanism: {exc}")


def run_trainer(config, args, dataset, tokenizer, eval_dataset=None):
    """Build a PP KTO trainer with spies on the train step and (when evaluating) the eval step.

    With ``eval_dataset``, an init-weights ``evaluate()`` runs before training (its per-call eval
    batches are captured for the exact eval reference — trained-weights evals cannot be reproduced
    on an unsplit copy) and its metrics are returned alongside the captures.
    """
    trainer = DistributedKTOTrainer(
        model=build_dense_model(tokenizer, vocab_size=VOCAB_SIZE),
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=config,
    )

    captured: dict = {}
    normalizers: list[float] = []
    real_step = trainer._pp_runtime.step

    def spying_step(input_ids, labels, **kw):
        normalizers.append(float(kw["num_items_in_batch"]))
        if "input_ids" not in captured:
            captured["input_ids"] = input_ids.clone()
            captured["labels"] = labels.clone()
            captured["attention_mask"] = kw["attention_mask"].clone()
            captured["ref_logps"] = kw["extra_targets"]["ref_logps"].clone()
            captured["label"] = kw["extra_targets"]["label"].clone()
        return real_step(input_ids, labels, **kw)

    trainer._pp_runtime.step = spying_step

    eval_calls: list[dict] = []
    init_eval_metrics = None
    if eval_dataset is not None:
        real_eval = trainer._pp_runtime.eval_loss

        def spying_eval(input_ids, labels, **kw):
            eval_calls.append(
                {
                    "input_ids": input_ids.clone(),
                    "labels": labels.clone(),
                    "attention_mask": kw["attention_mask"].clone(),
                    "ref_logps": kw["extra_targets"]["ref_logps"].clone(),
                    "label": kw["extra_targets"]["label"].clone(),
                    "num_items": float(kw["num_items_in_batch"]),
                }
            )
            return real_eval(input_ids, labels, **kw)

        trainer._pp_runtime.eval_loss = spying_eval
        init_eval_metrics = trainer.evaluate()

    # Snapshot the init-weights captures under a new name: in-loop evals keep appending to
    # eval_calls (the closure's cell), and trained-weights batches must not leak into the
    # init-weights reference.
    init_eval_calls = list(eval_calls)
    trainer.train()
    return trainer, captured, normalizers, init_eval_calls, init_eval_metrics


@gpu_test_main(exact_world_size=4, prefix="pp_kto_apo")
def run(ctx):
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = build_dataset()
    checks, metrics = {}, {}
    # Rejections raise before any model use, so one instance serves every construction below.
    model = build_dense_model(tokenizer, vocab_size=VOCAB_SIZE)

    # ── Construction-time rejections (all raise before any collective or model split) ────────────
    common = {"train_dataset": dataset, "processing_class": tokenizer, "parallelism_config": config}
    expect_construction_error(
        "default_kto_loss_rejected",
        checks,
        "world-global detached KL baseline",
        model=model,
        args=make_args(ctx.output_dir + "/kto", loss_type="kto"),
        **common,
    )
    expect_construction_error(
        "live_ref_model_rejected",
        checks,
        "ref_model",
        model=model,
        ref_model=build_dense_model(tokenizer, vocab_size=VOCAB_SIZE),
        args=make_args(ctx.output_dir + "/ref"),
        **common,
    )
    expect_construction_error(
        "precompute_off_rejected",
        checks,
        "precompute-only",
        model=model,
        args=make_args(ctx.output_dir + "/off", precompute_ref_log_probs=False),
        **common,
    )
    expect_construction_error(
        "missing_column_rejected",
        checks,
        "ALREADY",
        model=model,
        args=make_args(ctx.output_dir + "/nocol"),
        train_dataset=build_dataset(with_ref_column=False),
        processing_class=tokenizer,
        parallelism_config=config,
    )
    expect_construction_error(
        "eval_missing_ref_column_rejected",
        checks,
        "eval dataset",
        model=model,
        args=make_args(ctx.output_dir + "/evalcol"),
        eval_dataset=build_dataset(n_rows=8, seed=23, with_ref_column=False),
        **common,
    )
    # TRL's KTO prediction_step predicts the raw logits plane; serving that under PP means
    # broadcasting [rows, max_length, vocab] from the last stage to every rank of the chain, and any
    # reduction would be a different prediction than the non-PP path's. PP rejects it instead.
    expect_construction_error(
        "compute_metrics_rejected",
        checks,
        "RAW logits plane",
        model=model,
        args=make_args(ctx.output_dir + "/metrics"),
        compute_metrics=lambda eval_pred: {"noop": 0.0},
        **common,
    )

    # ── Main run: mixed desirable/undesirable rows, with the loss-only PP eval leg ───────────────
    args = make_args(ctx.output_dir, eval_strategy="steps", eval_steps=EVAL_STEPS)
    eval_dataset = build_dataset(n_rows=EVAL_ROWS, seed=23)
    trainer, captured, normalizers, eval_calls, init_eval_metrics = run_trainer(
        config, args, dataset, tokenizer, eval_dataset=eval_dataset
    )

    checks["accumulation_became_microbatches"] = (
        args.gradient_accumulation_steps == 1 and trainer._pp_n_microbatches == MICROBATCHES
    )
    checks["batch_padded_to_fixed_shape"] = captured["input_ids"].shape == (ROWS_PER_RANK, SEQ_LEN)
    checks["unpaired_rows_declared"] = trainer._pp_runtime.paired_examples is False
    checks["no_reference_model"] = trainer.ref_model is None
    checks["normalizer_is_row_count"] = len(normalizers) == N_STEPS and all(
        abs(value - ROWS_PER_RANK) < 1e-6 for value in normalizers
    )
    row_labels = captured["label"].bool()
    checks["captured_batch_has_both_sides"] = bool(row_labels.any()) and bool((~row_labels).any())

    # ── Row pinning: captured rows must map back to TRL's own tokenized dataset. The equivalence
    # reference consumes batch_transform's output, so a transform bug corrupting both sides
    # identically (labels off the wrong mask, extras shuffled) would pass it; this check is
    # transform-independent — each row must be prompt⧺completion of some example, the supervised
    # label span exactly the completion tokens, and the ref/label extras that example's own values.
    tokenized = trainer.train_dataset
    by_prompt = {
        tuple(row["prompt_ids"]): (list(row["completion_ids"]), bool(row["label"]), float(row["ref_logps"]))
        for row in tokenized
    }
    rows_pinned = True
    for k in range(ROWS_PER_RANK):
        length = int(captured["attention_mask"][k].sum())
        seq = captured["input_ids"][k][:length].tolist()
        matches = [p for p in by_prompt if seq[: len(p)] == list(p)]
        if not matches:
            rows_pinned = False
            continue
        prompt = max(matches, key=len)
        completion, label, ref_logp = by_prompt[prompt]
        if seq != list(prompt) + completion:
            rows_pinned = False
        labels_row = captured["labels"][k][:length]
        supervised = (labels_row != -100).nonzero().flatten().tolist()
        expected_span = list(range(len(prompt), len(prompt) + len(completion)))
        if supervised != expected_span or labels_row[supervised].tolist() != completion:
            rows_pinned = False
        if bool(captured["label"][k]) != label or abs(float(captured["ref_logps"][k]) - ref_logp) > 1e-6:
            rows_pinned = False
    checks["captured_rows_pin_to_raw_tokenized_examples"] = rows_pinned

    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    checks["ran_all_steps"] = trainer.state.global_step == N_STEPS and len(losses) == N_STEPS
    checks["losses_finite"] = bool(torch.isfinite(torch.tensor(losses)).all())

    reference_model = build_dense_model(tokenizer, vocab_size=VOCAB_SIZE).to(ctx.device).eval()
    reference = kto_reference(
        reference_model,
        captured["input_ids"],
        captured["attention_mask"],
        captured["labels"],
        captured["ref_logps"],
        captured["label"],
    )
    expected = world_mean(reference)
    metrics["step1_loss"], metrics["step1_reference"] = losses[0], expected
    checks["step1_loss_matches_reference"] = abs(losses[0] - expected) < REL_TOL * max(1.0, abs(expected))

    # The ref column genuinely enters the loss: zeroed refs must not match.
    zero_ref = world_mean(
        kto_reference(
            reference_model,
            captured["input_ids"],
            captured["attention_mask"],
            captured["labels"],
            torch.zeros_like(captured["ref_logps"]),
            captured["label"],
        )
    )
    checks["ref_logps_shift_the_loss"] = abs(losses[0] - zero_ref) > 1e-2

    # ── Eval leg: the init-weights eval_loss must equal HF's averaging of the exact per-batch KTO
    # objective over the real rows of each captured eval batch (inert padding rows excluded — a
    # broken inert-row mask shifts every partial batch's loss by w_u·σ(0)), reproducing the gather
    # order and accelerate's final-step remainder trim.
    real_row_flags = [(call["labels"] != -100).any(dim=-1) for call in eval_calls]
    step_losses = [
        kto_reference(
            reference_model,
            call["input_ids"][real],
            call["attention_mask"][real],
            call["labels"][real],
            call["ref_logps"][real],
            call["label"][real],
        )
        for call, real in zip(eval_calls, real_row_flags, strict=True)
    ]
    dp_size, dp_rank = trainer.get_data_parallel_size(), trainer.get_data_parallel_rank()
    remainder = EVAL_ROWS % (ROWS_PER_RANK * dp_size)
    expected_eval = hf_eval_loss_reference(step_losses, ROWS_PER_RANK, remainder, dp_rank=dp_rank, dp_size=dp_size)
    eval_loss = init_eval_metrics["eval_loss"]
    metrics["init_eval_loss"], metrics["init_eval_reference"] = eval_loss, expected_eval
    checks["eval_loss_matches_reference"] = abs(eval_loss - expected_eval) < REL_TOL * max(1.0, abs(expected_eval))
    # Second, independent oracle: every eval row scored once, walked example by example with no
    # model of the gather at all. On this partial final batch a per-stage-duplicating gather keeps a
    # prefix of the duplicates — a different set of rows, which the two derivations disagree about.
    by_example = eval_loss_over_examples(step_losses, ROWS_PER_RANK, EVAL_ROWS, dp_rank=dp_rank, dp_size=dp_size)
    metrics["init_eval_by_example"] = by_example
    checks["eval_loss_matches_independent_example_walk"] = abs(eval_loss - by_example) < REL_TOL * max(
        1.0, abs(by_example)
    )

    # The eval normalizer must be each batch's real row count (partial batch: 4, not 8).
    checks["eval_normalizer_is_real_row_count"] = all(
        abs(call["num_items"] - float(real.sum())) < 1e-6
        for call, real in zip(eval_calls, real_row_flags, strict=True)
    )
    # The partial batch genuinely occurred somewhere in the world (inert rows were padded in).
    had_partial = torch.tensor(float(any(not bool(real.all()) for real in real_row_flags)), device=ctx.device)
    dist.all_reduce(had_partial, op=dist.ReduceOp.MAX)
    checks["partial_eval_batch_exercised"] = bool(had_partial)

    local_eval = torch.tensor([eval_loss], device=ctx.device)
    gathered_eval = [torch.zeros_like(local_eval) for _ in range(ctx.world_size)]
    dist.all_gather(gathered_eval, local_eval)
    checks["eval_loss_identical_across_ranks"] = all(torch.allclose(local_eval, g) for g in gathered_eval)

    # Two in-loop evals (steps 5 and 10) ran on trained weights: finite and rank-identical.
    inloop_eval = [e["eval_loss"] for e in trainer.state.log_history if "eval_loss" in e]
    checks["inloop_evals_ran_finite"] = len(inloop_eval) == 2 and bool(torch.isfinite(torch.tensor(inloop_eval)).all())

    checks["loss_decreased"] = losses[-1] < losses[0] - 1e-3
    metrics["first_loss"], metrics["last_loss"] = losses[0], losses[-1]

    local = torch.tensor(losses, device=ctx.device)
    gathered = [torch.zeros_like(local) for _ in range(ctx.world_size)]
    dist.all_gather(gathered, local)
    checks["losses_identical_across_ranks"] = all(torch.allclose(local, g, atol=1e-6) for g in gathered)

    grad_norms = [e["grad_norm"] for e in trainer.state.log_history if "grad_norm" in e]
    checks["grad_norm_logged_finite"] = bool(grad_norms) and bool(torch.isfinite(torch.tensor(grad_norms)).all())

    # ── All-desirable run: every microbatch single-sided (empty rejected side) ───────────────────
    des_args = make_args(ctx.output_dir + "/des", max_steps=2)
    des_trainer, des_captured, _, _, _ = run_trainer(
        config, des_args, build_dataset(seed=11, all_desirable=True), tokenizer
    )
    des_losses = [e["loss"] for e in des_trainer.state.log_history if "loss" in e]
    checks["all_desirable_batch_is_single_sided"] = bool(des_captured["label"].bool().all())
    des_expected = world_mean(
        kto_reference(
            reference_model,
            des_captured["input_ids"],
            des_captured["attention_mask"],
            des_captured["labels"],
            des_captured["ref_logps"],
            des_captured["label"],
        )
    )
    metrics["all_desirable_step1_loss"], metrics["all_desirable_reference"] = des_losses[0], des_expected
    checks["all_desirable_step1_matches_reference"] = abs(des_losses[0] - des_expected) < REL_TOL * max(
        1.0, abs(des_expected)
    )

    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
