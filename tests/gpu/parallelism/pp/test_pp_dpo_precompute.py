#!/usr/bin/env python
"""End-to-end pipeline parallelism through the real DistributedDPOTrainer (precompute-only).

DPO under PP consumes reference log-probs shipped as dataset columns — no reference model exists
anywhere — and re-lays TRL's [chosen ⧺ rejected] concat out as interleaved pairs. Every seam
produces a silently wrong number, not a crash, so the step-1 logged loss is pinned against TRL's
exact objective computed on an unsplit copy of the model over the captured batches:

  * the interleave (a chunk-style microbatch split would compare unrelated examples)
  * labels derived from completion_mask (the positions TRL's shift_completion_mask scores)
  * the ref columns riding as an interleaved per-example extra target
  * the pair-count normalizer (TRL's loss is a per-replica mean over pairs)
  * the loss-type dispatch: sigmoid and ipo (ipo's per-sequence token-count normalization)
  * loss-only PP evaluation: init-weights eval_loss pinned against the exact per-batch objective
    with HF's gather/trim averaging reproduced, a partial final eval batch proving the inert
    row-padding contributes exactly 0, and two in-loop evals staying finite and rank-identical
  * construction-time rejections (live ref model, precompute off, missing train/eval columns,
    aot's batch-global sort, an unaudited per-pair type, compute_metrics)

Topology: 4 GPUs as 2 simulated nodes of 2, pp=2 x dp=2 (stage FSDP over 2 ranks).

Run: torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_dpo_precompute.py
"""

import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer
from trl import DPOConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.preference.dpo import DistributedDPOTrainer
from tests.common.distributed import eval_loss_over_examples, hf_eval_loss_reference, world_mean
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.pp_equivalence import build_dense_model
from tests.common.utils import log

PP_SIZE = 2
MICROBATCHES = 2
PAIRS_PER_RANK = 4  # per_device_train_batch_size counts pairs; the collator doubles the rows
SEQ_LEN = 48
N_STEPS = 10
# 14 pairs over dp=2 at eval batch 4 (even_batches off) = batches of 4/4/4/2 pairs: one DP shard
# ends on a partial batch, so the eval leg exercises the inert row-padding for real.
EVAL_PAIRS = 14
EVAL_STEPS = 5
# Small beta keeps beta*delta out of logsigmoid saturation for a random-init policy (per-sequence
# logps ~ -12 * completion length against synthetic refs), so gradients stay alive.
BETA = 0.05
REL_TOL = 1e-4

# The real Qwen tokenizer tokenizes the text rows, so the tiny model must span its full id range.
VOCAB_SIZE = 151936


def build_dataset(n_rows: int = 24, seed: int = 7, with_ref_columns: bool = True) -> Dataset:
    """Text preference rows with ragged completion lengths plus synthetic precomputed ref columns.

    Ragged lengths make ipo's per-sequence token-count normalization observable; the ref values are
    synthetic scalars — the reference computation consumes the same captured values, so only their
    transport (column → collator → interleaved extra) is under test, not their provenance.
    """
    g = torch.Generator().manual_seed(seed)
    rows = {"prompt": [], "chosen": [], "rejected": []}
    if with_ref_columns:
        rows["ref_chosen_logps"] = []
        rows["ref_rejected_logps"] = []
    for i in range(n_rows):
        rows["prompt"].append(f"Q{i}: what is {i} plus {i}?")
        rows["chosen"].append(" The answer is " + " ".join(str(2 * i + j) for j in range(1 + i % 4)) + ".")
        rows["rejected"].append(
            " It could be " + " ".join(str(2 * i + 1 + j) for j in range(1 + (i + 2) % 4)) + " maybe."
        )
        if with_ref_columns:
            rows["ref_chosen_logps"].append(-float(10 + 20 * torch.rand(1, generator=g)))
            rows["ref_rejected_logps"].append(-float(10 + 20 * torch.rand(1, generator=g)))
    return Dataset.from_dict(rows)


def dpo_reference(model, input_ids, attention_mask, labels, ref_logps, *, loss_type) -> float:
    """TRL's exact DPO objective on an interleaved batch with precomputed refs, fp32, independent of src/."""
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    token_logps = torch.log_softmax(logits[:, :-1, :], dim=-1)
    token_logps = token_logps.gather(-1, shift_labels.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1) * mask
    seq_logps = token_logps.sum(-1)
    chosen_scores = seq_logps[0::2] - ref_logps[0::2]
    rejected_scores = seq_logps[1::2] - ref_logps[1::2]
    delta = chosen_scores - rejected_scores
    if loss_type == "sigmoid":
        per_pair = -F.logsigmoid(BETA * delta)
    elif loss_type == "ipo":
        counts = (labels != -100).sum(-1).clamp(min=1).float()  # TRL divides by the unshifted count
        ipo_delta = chosen_scores / counts[0::2] - rejected_scores / counts[1::2]
        per_pair = (ipo_delta - 1 / (2 * BETA)) ** 2
    else:
        raise ValueError(loss_type)
    return float(per_pair.mean())


def make_args(output_dir: str, **overrides) -> DPOConfig:
    base = {
        "output_dir": output_dir,
        "max_steps": N_STEPS,
        "per_device_train_batch_size": PAIRS_PER_RANK,
        "gradient_accumulation_steps": MICROBATCHES,  # becomes the pipeline's microbatch count
        "learning_rate": 5e-3,
        "lr_scheduler_type": "constant",
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
        "logging_nan_inf_filter": False,
        "seed": 42,
        "bf16": False,
        "max_length": SEQ_LEN,
        "beta": BETA,
        "loss_type": ["sigmoid"],
        "precompute_ref_log_probs": True,
        "per_device_eval_batch_size": PAIRS_PER_RANK,  # PP freezes eval batches to the train shape
        # even_batches would complete the tail eval batch by wrapping around to the dataset start;
        # off, the final batch stays partial and must be row-padded with inert rows.
        "accelerator_config": {"even_batches": False},
    }
    base.update(overrides)
    return DPOConfig(**base)


def expect_construction_error(name: str, checks: dict, match: str, **trainer_kwargs) -> None:
    """Construct on every rank expecting a uniform pre-collective ValueError."""
    try:
        DistributedDPOTrainer(**trainer_kwargs)
        checks[name] = False
    except ValueError as exc:
        checks[name] = match in str(exc)
        if not checks[name]:
            log(f"[{name}] raised without the expected mechanism: {exc}")


def run_trainer(config, args, dataset, tokenizer, eval_dataset=None):
    """Build a PP DPO trainer with spies on the train step and (when evaluating) the eval step.

    With ``eval_dataset``, an init-weights ``evaluate()`` runs before training (its per-call eval
    batches are captured for the exact eval reference — trained-weights evals cannot be reproduced
    on an unsplit copy) and its metrics are returned alongside the captures.
    """
    trainer = DistributedDPOTrainer(
        model=build_dense_model(tokenizer, vocab_size=VOCAB_SIZE),
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=config,
    )

    captured: dict = {}
    normalizers: list[float] = []
    real_step = trainer._pp_runner.step

    def spying_step(input_ids, labels, **kw):
        normalizers.append(float(kw["num_items_in_batch"]))
        if "input_ids" not in captured:
            captured["input_ids"] = input_ids.clone()
            captured["labels"] = labels.clone()
            captured["attention_mask"] = kw["attention_mask"].clone()
            captured["ref_logps"] = kw["extra_targets"]["ref_logps"].clone()
        return real_step(input_ids, labels, **kw)

    trainer._pp_runner.step = spying_step

    eval_calls: list[dict] = []
    init_eval_metrics = None
    if eval_dataset is not None:
        real_eval = trainer._pp_runner.eval_loss

        def spying_eval(input_ids, labels, **kw):
            eval_calls.append(
                {
                    "input_ids": input_ids.clone(),
                    "labels": labels.clone(),
                    "attention_mask": kw["attention_mask"].clone(),
                    "ref_logps": kw["extra_targets"]["ref_logps"].clone(),
                    "num_items": float(kw["num_items_in_batch"]),
                }
            )
            return real_eval(input_ids, labels, **kw)

        trainer._pp_runner.eval_loss = spying_eval
        init_eval_metrics = trainer.evaluate()

    # Snapshot the init-weights captures under a new name: in-loop evals keep appending to
    # eval_calls (the closure's cell), and trained-weights batches must not leak into the
    # init-weights reference.
    init_eval_calls = list(eval_calls)
    trainer.train()
    return trainer, captured, normalizers, init_eval_calls, init_eval_metrics


@gpu_test_main(exact_world_size=4, prefix="pp_dpo_precompute")
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
        "missing_columns_rejected",
        checks,
        "ALREADY",
        model=model,
        args=make_args(ctx.output_dir + "/nocols"),
        train_dataset=build_dataset(with_ref_columns=False),
        processing_class=tokenizer,
        parallelism_config=config,
    )
    expect_construction_error(
        "aot_rejected",
        checks,
        "sorts",
        model=model,
        args=make_args(ctx.output_dir + "/aot", loss_type=["aot"]),
        **common,
    )
    expect_construction_error(
        "unaudited_type_rejected",
        checks,
        "not re-implemented",
        model=model,
        args=make_args(ctx.output_dir + "/exo", loss_type=["exo_pair"], label_smoothing=1e-3),
        **common,
    )
    expect_construction_error(
        "eval_missing_ref_columns_rejected",
        checks,
        "eval dataset",
        model=model,
        args=make_args(ctx.output_dir + "/evalcols"),
        eval_dataset=build_dataset(n_rows=8, seed=23, with_ref_columns=False),
        **common,
    )
    # TRL's DPO prediction_step predicts the raw logits plane over the [chosen ⧺ rejected] concat,
    # a convention the pipeline's interleaved rows cannot express and whose plane would cross the
    # chain in full. Reducing it to anything else would be a different prediction, so PP rejects it.
    expect_construction_error(
        "compute_metrics_rejected",
        checks,
        "RAW logits plane",
        model=model,
        args=make_args(ctx.output_dir + "/metrics"),
        compute_metrics=lambda eval_pred: {"noop": 0.0},
        **common,
    )

    # ── Main run: sigmoid, with the loss-only PP eval leg (init-weights evaluate + 2 in-loop) ────
    args = make_args(ctx.output_dir, eval_strategy="steps", eval_steps=EVAL_STEPS)
    eval_dataset = build_dataset(n_rows=EVAL_PAIRS, seed=23)
    trainer, captured, normalizers, eval_calls, init_eval_metrics = run_trainer(
        config, args, dataset, tokenizer, eval_dataset=eval_dataset
    )

    checks["accumulation_became_microbatches"] = (
        args.gradient_accumulation_steps == 1 and trainer._pp_n_microbatches == MICROBATCHES
    )
    checks["pairs_interleaved_2x_batch"] = captured["input_ids"].shape == (2 * PAIRS_PER_RANK, SEQ_LEN)
    checks["paired_examples_declared"] = trainer._pp_runner.rows_per_example == 2
    checks["no_reference_model"] = trainer.ref_model is None
    checks["normalizer_is_pair_count"] = len(normalizers) == N_STEPS and all(
        abs(value - PAIRS_PER_RANK) < 1e-6 for value in normalizers
    )

    # ── Row pinning: captured interleaved rows must map back to TRL's own tokenized dataset. The
    # equivalence reference consumes batch_transform's output, so a transform bug corrupting both
    # sides identically (broken interleave, labels off the wrong mask) would pass it; this check is
    # transform-independent — row 2k must be prompt⧺chosen of some example, row 2k+1 that example's
    # prompt⧺rejected, and the supervised label span exactly the chosen tokens.
    tokenized = trainer.train_dataset
    columns = tokenized.column_names
    p_col = next(c for c in ("prompt_input_ids", "prompt_ids") if c in columns)
    c_col = next(c for c in ("chosen_input_ids", "chosen_ids") if c in columns)
    r_col = next(c for c in ("rejected_input_ids", "rejected_ids") if c in columns)
    pair_by_prompt = {tuple(row[p_col]): (list(row[c_col]), list(row[r_col])) for row in tokenized}
    rows_pinned = True
    for k in range(PAIRS_PER_RANK):
        len_c = int(captured["attention_mask"][2 * k].sum())
        len_r = int(captured["attention_mask"][2 * k + 1].sum())
        seq_c = captured["input_ids"][2 * k][:len_c].tolist()
        seq_r = captured["input_ids"][2 * k + 1][:len_r].tolist()
        matches = [p for p in pair_by_prompt if seq_c[: len(p)] == list(p)]
        if not matches:
            rows_pinned = False
            continue
        prompt = max(matches, key=len)
        chosen, rejected = pair_by_prompt[prompt]
        if seq_c != list(prompt) + chosen or seq_r != list(prompt) + rejected:
            rows_pinned = False
        labels_row = captured["labels"][2 * k][:len_c]
        supervised = (labels_row != -100).nonzero().flatten().tolist()
        if (
            supervised != list(range(len(prompt), len(prompt) + len(chosen)))
            or labels_row[supervised].tolist() != chosen
        ):
            rows_pinned = False
    checks["captured_rows_pin_to_raw_tokenized_pairs"] = rows_pinned

    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    checks["ran_all_steps"] = trainer.state.global_step == N_STEPS and len(losses) == N_STEPS
    checks["losses_finite"] = bool(torch.isfinite(torch.tensor(losses)).all())

    reference_model = build_dense_model(tokenizer, vocab_size=VOCAB_SIZE).to(ctx.device).eval()
    reference = dpo_reference(
        reference_model,
        captured["input_ids"],
        captured["attention_mask"],
        captured["labels"],
        captured["ref_logps"],
        loss_type="sigmoid",
    )
    expected = world_mean(reference)
    metrics["step1_loss"], metrics["step1_reference"] = losses[0], expected
    checks["step1_loss_matches_reference"] = abs(losses[0] - expected) < REL_TOL * max(1.0, abs(expected))

    # The ref columns genuinely enter the loss: zeroed refs must not match.
    zero_ref = world_mean(
        dpo_reference(
            reference_model,
            captured["input_ids"],
            captured["attention_mask"],
            captured["labels"],
            torch.zeros_like(captured["ref_logps"]),
            loss_type="sigmoid",
        )
    )
    checks["ref_logps_shift_the_loss"] = abs(losses[0] - zero_ref) > 1e-2

    # ── Eval leg: the init-weights eval_loss must equal HF's averaging of the exact per-batch DPO
    # objective over the real pairs of each captured eval batch (inert padding rows excluded — a
    # broken inert-pair mask shifts every partial batch's loss by -logsigmoid(0)), reproducing the
    # gather order and accelerate's final-step remainder trim.
    real_row_flags = [(call["labels"] != -100).any(dim=-1) for call in eval_calls]
    step_losses = [
        dpo_reference(
            reference_model,
            call["input_ids"][real],
            call["attention_mask"][real],
            call["labels"][real],
            call["ref_logps"][real],
            loss_type="sigmoid",
        )
        for call, real in zip(eval_calls, real_row_flags, strict=True)
    ]
    dp_size, dp_rank = trainer.get_data_parallel_size(), trainer.get_data_parallel_rank()
    remainder = EVAL_PAIRS % (PAIRS_PER_RANK * dp_size)
    expected_eval = hf_eval_loss_reference(step_losses, PAIRS_PER_RANK, remainder, dp_rank=dp_rank, dp_size=dp_size)
    eval_loss = init_eval_metrics["eval_loss"]
    metrics["init_eval_loss"], metrics["init_eval_reference"] = eval_loss, expected_eval
    checks["eval_loss_matches_reference"] = abs(eval_loss - expected_eval) < REL_TOL * max(1.0, abs(expected_eval))
    # Second, independent oracle: every eval pair scored once, walked example by example with no
    # model of the gather at all. A gather that repeated each replica per stage would keep a prefix
    # of the duplicates on this partial final batch — a different set of pairs, which the two
    # derivations disagree about even though a mean over full batches would hide it.
    by_example = eval_loss_over_examples(step_losses, PAIRS_PER_RANK, EVAL_PAIRS, dp_rank=dp_rank, dp_size=dp_size)
    metrics["init_eval_by_example"] = by_example
    checks["eval_loss_matches_independent_example_walk"] = abs(eval_loss - by_example) < REL_TOL * max(
        1.0, abs(by_example)
    )

    # The eval normalizer must be each batch's real pair count (partial batch: 2, not 4).
    checks["eval_normalizer_is_real_pair_count"] = all(
        abs(call["num_items"] - float(real.sum()) / 2) < 1e-6
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

    # ── IPO run: per-sequence token-count normalization through the same closure ─────────────────
    ipo_args = make_args(ctx.output_dir + "/ipo", max_steps=2, loss_type=["ipo"])
    ipo_trainer, ipo_captured, _, _, _ = run_trainer(config, ipo_args, build_dataset(seed=11), tokenizer)
    ipo_losses = [e["loss"] for e in ipo_trainer.state.log_history if "loss" in e]
    ipo_expected = world_mean(
        dpo_reference(
            reference_model,
            ipo_captured["input_ids"],
            ipo_captured["attention_mask"],
            ipo_captured["labels"],
            ipo_captured["ref_logps"],
            loss_type="ipo",
        )
    )
    metrics["ipo_step1_loss"], metrics["ipo_step1_reference"] = ipo_losses[0], ipo_expected
    checks["ipo_step1_loss_matches_reference"] = abs(ipo_losses[0] - ipo_expected) < REL_TOL * max(
        1.0, abs(ipo_expected)
    )

    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
