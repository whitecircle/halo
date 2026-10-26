#!/usr/bin/env python
"""End-to-end PP through the real DPO or KTO trainer on precomputed reference log-probs (no reference model).

DPO re-lays TRL's [chosen ⧺ rejected] concat out as interleaved pairs, KTO runs only the KL-free apo_zero_unpaired
loss on independent rows; each seam is a silently wrong number, so step-1 and init-weights eval losses are pinned
against TRL's exact objective on an unsplit copy. Run: torchrun --nproc_per_node=4 <this file> --trainer {dpo,kto}
"""

import argparse
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer
from trl import DPOConfig, KTOConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.preference.dpo import DistributedDPOTrainer
from src.trainers.preference.kto import DistributedKTOTrainer
from tests.common.distributed import eval_loss_over_examples, hf_eval_loss_reference, world_any, world_mean
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.pp_equivalence import build_dense_model
from tests.common.pp_trainers import (
    build_pp_args,
    build_pp_trainer,
    expect_construction_error,
    losses_identical_across_ranks,
)
from tests.common.utils import step_losses

PP_SIZE = 2
MICROBATCHES = 2
SEQ_LEN = 48
N_STEPS = 10
EVAL_STEPS = 5
# Small beta keeps beta * logratio out of the sigmoid's saturation for a random-init policy (per-sequence
# logps ~ -12 per completion token against synthetic refs), so gradients stay alive.
BETA = 0.05
# KTO's non-default side weights, so the objective pin covers the weighting, not just the sigmoids.
DESIRABLE_WEIGHT = 1.3
UNDESIRABLE_WEIGHT = 0.7
REL_TOL = 1e-4
# The real Qwen tokenizer tokenizes the text rows, so the tiny model must span its full id range.
VOCAB_SIZE = 151936
SPEC = None  # SPECS[--trainer], set before the harness runs


def build_rows(
    spec: "PreferenceSpec", n_rows: int, seed: int = 7, with_refs: bool = True, all_desirable: bool = False
) -> Dataset:
    """Text rows for the trainer under test with ragged completion lengths, plus synthetic precomputed ref
    columns (only their transport is under test).

    DPO: chosen/rejected pairs (ragged lengths make ipo's per-sequence token-count normalization observable).
    KTO: 2 desirable : 1 undesirable rows (unequal side counts), every row desirable with ``all_desirable``;
    its refs sit at a random-init policy's magnitude (~ -12/token over 5-10 tokens), so zeroed refs show.
    """
    g = torch.Generator().manual_seed(seed)
    rows = defaultdict(list)
    for i in range(n_rows):
        rows["prompt"].append(f"Q{i}: what is {i} plus {i}?")
        answer = " The answer is " + " ".join(str(2 * i + j) for j in range(1 + i % 4)) + "."
        if spec.rows_per_example == 2:
            rows["chosen"].append(answer)
            rows["rejected"].append(
                " It could be " + " ".join(str(2 * i + 1 + j) for j in range(1 + (i + 2) % 4)) + " maybe."
            )
            if with_refs:
                rows["ref_chosen_logps"].append(-float(10 + 20 * torch.rand(1, generator=g)))
                rows["ref_rejected_logps"].append(-float(10 + 20 * torch.rand(1, generator=g)))
        else:
            rows["completion"].append(answer)
            rows["label"].append(True if all_desirable else i % 3 != 0)
            if with_refs:
                rows["ref_logps"].append(-float(70 + 40 * torch.rand(1, generator=g)))
    return Dataset.from_dict(dict(rows))


def completion_logps(model, batch: dict) -> torch.Tensor:
    """Each row's summed log-prob of its supervised tokens under ``model``, fp32, independent of src/."""
    with torch.no_grad():
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
    shift_labels = batch["labels"][:, 1:]
    mask = shift_labels != -100
    token_logps = torch.log_softmax(logits[:, :-1, :], dim=-1)
    return (token_logps.gather(-1, shift_labels.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1) * mask).sum(-1)


def dpo_objective(model, batch: dict, loss_type: str = "sigmoid") -> float:
    """TRL's exact DPO objective on an interleaved [c0, r0, ...] batch with precomputed refs."""
    scores = completion_logps(model, batch) - batch["ref_logps"]
    chosen, rejected = scores[0::2], scores[1::2]
    if loss_type == "ipo":
        counts = (batch["labels"] != -100).sum(-1).clamp(min=1).float()  # TRL divides by the unshifted count
        per_pair = (chosen / counts[0::2] - rejected / counts[1::2] - 1 / (2 * BETA)) ** 2
    else:
        per_pair = -F.logsigmoid(BETA * (chosen - rejected))
    return float(per_pair.mean())


def kto_objective(model, batch: dict) -> float:
    """TRL's exact apo_zero_unpaired objective with precomputed refs: per-row weighted losses then a mean over
    all rows, which is TRL's ``cat(w_d·chosen, w_u·rejected).nanmean()`` since every row is on one side."""
    logratios = completion_logps(model, batch) - batch["ref_logps"]
    per_row = torch.where(
        batch["label"].bool(),
        DESIRABLE_WEIGHT * (1 - torch.sigmoid(BETA * logratios)),
        UNDESIRABLE_WEIGHT * torch.sigmoid(BETA * logratios),
    )
    return float(per_row.mean())


def _matching_prompt(seq: list[int], prompts) -> tuple | None:
    matches = [p for p in prompts if seq[: len(p)] == list(p)]
    return max(matches, key=len) if matches else None


def _supervises_exactly(labels_row: torch.Tensor, prompt, completion: list[int]) -> bool:
    supervised = (labels_row != -100).nonzero().flatten().tolist()
    return supervised == list(range(len(prompt), len(prompt) + len(completion))) and (
        labels_row[supervised].tolist() == completion
    )


def dpo_rows_pinned(captured: dict, tokenized: Dataset) -> bool:
    """Row 2k is prompt⧺chosen of some example of TRL's own tokenized dataset and row 2k+1 that example's
    prompt⧺rejected, the supervised span exactly the chosen tokens."""
    columns = tokenized.column_names
    p_col, c_col, r_col = (
        next(c for c in (f"{side}_input_ids", f"{side}_ids") if c in columns)
        for side in ("prompt", "chosen", "rejected")
    )
    pair_by_prompt = {tuple(row[p_col]): (list(row[c_col]), list(row[r_col])) for row in tokenized}
    pinned = True
    for k in range(captured["input_ids"].size(0) // 2):
        len_c = int(captured["attention_mask"][2 * k].sum())
        len_r = int(captured["attention_mask"][2 * k + 1].sum())
        seq_c = captured["input_ids"][2 * k][:len_c].tolist()
        seq_r = captured["input_ids"][2 * k + 1][:len_r].tolist()
        prompt = _matching_prompt(seq_c, pair_by_prompt)
        if prompt is None:
            pinned = False
            continue
        chosen, rejected = pair_by_prompt[prompt]
        pinned &= seq_c == list(prompt) + chosen and seq_r == list(prompt) + rejected
        pinned &= _supervises_exactly(captured["labels"][2 * k][:len_c], prompt, chosen)
    return pinned


def kto_rows_pinned(captured: dict, tokenized: Dataset) -> bool:
    """Each row is prompt⧺completion of some example of TRL's own tokenized dataset, the supervised span exactly
    the completion tokens, and the ref/label extras that example's own values."""
    by_prompt = {
        tuple(row["prompt_ids"]): (list(row["completion_ids"]), bool(row["label"]), float(row["ref_logps"]))
        for row in tokenized
    }
    pinned = True
    for k in range(captured["input_ids"].size(0)):
        length = int(captured["attention_mask"][k].sum())
        seq = captured["input_ids"][k][:length].tolist()
        prompt = _matching_prompt(seq, by_prompt)
        if prompt is None:
            pinned = False
            continue
        completion, label, ref_logp = by_prompt[prompt]
        pinned &= seq == list(prompt) + completion
        pinned &= _supervises_exactly(captured["labels"][k][:length], prompt, completion)
        pinned &= bool(captured["label"][k]) == label and abs(float(captured["ref_logps"][k]) - ref_logp) <= 1e-6
    return pinned


@dataclass(frozen=True)
class SecondRun:
    """Two steps down the trainer's other loss path through the same closure, the step-1 loss pinned."""

    prefix: str  # of its check and metric names
    config: dict
    row_kwargs: dict
    objective_kwargs: dict
    batch_checks: Callable[[dict], dict] = lambda batch: {}  # on its first captured batch


@dataclass(frozen=True)
class PreferenceSpec:
    """What differs between the two trainers' pipeline contracts."""

    trainer_cls: type
    config_cls: type
    rows_per_example: int  # DPO interleaves a (chosen, rejected) pair; KTO rows are independent
    examples_per_rank: int  # per_device_train_batch_size
    train_examples: int
    # Over dp=2 at the train batch (even_batches off) one DP shard ends on a partial eval batch (4/4/4/2
    # pairs, 8/8/8/4 rows), so the eval leg exercises the inert row padding for real.
    eval_examples: int
    objective: Callable[..., float]
    rows_pinned: Callable[[dict, Dataset], bool]
    config: dict
    refusals: tuple  # (check name, error fragment, config overrides) beyond the shared refusals
    second_run: SecondRun
    batch_checks: Callable[[dict], dict] = lambda batch: {}  # on the main run's first captured batch


SPECS = {
    "dpo": PreferenceSpec(
        trainer_cls=DistributedDPOTrainer,
        config_cls=DPOConfig,
        rows_per_example=2,
        examples_per_rank=4,
        train_examples=24,
        eval_examples=14,
        objective=dpo_objective,
        rows_pinned=dpo_rows_pinned,
        config={"loss_type": ["sigmoid"]},
        refusals=(
            ("aot_rejected", "sorts", {"loss_type": ["aot"]}),
            ("unaudited_type_rejected", "not re-implemented", {"loss_type": ["exo_pair"], "label_smoothing": 1e-3}),
        ),
        # ipo's per-sequence token-count normalization.
        second_run=SecondRun(
            prefix="ipo", config={"loss_type": ["ipo"]}, row_kwargs={}, objective_kwargs={"loss_type": "ipo"}
        ),
    ),
    "kto": PreferenceSpec(
        trainer_cls=DistributedKTOTrainer,
        config_cls=KTOConfig,
        rows_per_example=1,
        examples_per_rank=8,
        train_examples=32,
        eval_examples=28,
        objective=kto_objective,
        rows_pinned=kto_rows_pinned,
        config={
            "gradient_checkpointing": False,
            "desirable_weight": DESIRABLE_WEIGHT,
            "undesirable_weight": UNDESIRABLE_WEIGHT,
            "loss_type": "apo_zero_unpaired",
        },
        refusals=(("default_kto_loss_rejected", "world-global detached KL baseline", {"loss_type": "kto"}),),
        # Every microbatch single-sided: the empty-rejected branch.
        second_run=SecondRun(
            prefix="all_desirable",
            config={},
            row_kwargs={"all_desirable": True},
            objective_kwargs={},
            batch_checks=lambda batch: {"all_desirable_batch_is_single_sided": bool(batch["label"].bool().all())},
        ),
        batch_checks=lambda batch: {
            "captured_batch_has_both_sides": bool(batch["label"].bool().any()) and bool((~batch["label"].bool()).any())
        },
    ),
}


def build_args(spec: PreferenceSpec, output_dir: str, **overrides):
    return build_pp_args(
        spec.config_cls,
        output_dir,
        **{
            "max_steps": N_STEPS,
            "per_device_train_batch_size": spec.examples_per_rank,
            "gradient_accumulation_steps": MICROBATCHES,
            "per_device_eval_batch_size": spec.examples_per_rank,  # PP freezes eval batches to the train shape
            # even_batches would complete the tail eval batch by wrapping around to the dataset start; off, the
            # final batch stays partial and must be row-padded with inert rows.
            "accelerator_config": {"even_batches": False},
            "max_length": SEQ_LEN,
            "beta": BETA,
            "precompute_ref_log_probs": True,
            **spec.config,
            **overrides,
        },
    )


def run_trainer(spec: PreferenceSpec, config, args, dataset, tokenizer, eval_dataset=None):
    """The PP trainer with its runner spied, trained. With ``eval_dataset`` an init-weights ``evaluate()`` runs
    first: its batches feed the exact eval reference (trained-weights evals cannot be reproduced on an unsplit
    copy). Returns the trainer, every step's batch, and the init-weights eval calls and metrics."""
    trainer, steps, evals = build_pp_trainer(
        spec.trainer_cls,
        model=build_dense_model(tokenizer, vocab_size=VOCAB_SIZE),
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=config,
    )
    init_eval_metrics = trainer.evaluate() if eval_dataset is not None else None
    # In-loop evals keep appending to evals; trained-weights batches must stay out of the init reference.
    init_eval_calls = list(evals)
    trainer.train()
    return trainer, steps, init_eval_calls, init_eval_metrics


def refusal_checks(spec: PreferenceSpec, config, tokenizer, dataset, output_dir: str) -> dict:
    """Construction-time refusals: all raise before any collective or model split, so one model serves."""
    checks = {}
    common = {
        "model": build_dense_model(tokenizer, vocab_size=VOCAB_SIZE),
        "train_dataset": dataset,
        "processing_class": tokenizer,
        "parallelism_config": config,
    }
    refusals = [
        (
            "live_ref_model_rejected",
            "ref_model",
            {},
            {"ref_model": build_dense_model(tokenizer, vocab_size=VOCAB_SIZE)},
        ),
        ("precompute_off_rejected", "precompute-only", {"precompute_ref_log_probs": False}, {}),
        (
            "missing_ref_columns_rejected",
            "ALREADY",
            {},
            {"train_dataset": build_rows(spec, spec.train_examples, with_refs=False)},
        ),
        (
            "eval_missing_ref_columns_rejected",
            "eval dataset",
            {},
            {"eval_dataset": build_rows(spec, 8, seed=23, with_refs=False)},
        ),
        # TRL's prediction_step predicts the raw logits plane; serving it under PP means broadcasting
        # [rows, max_length, vocab] down the chain, and any reduction would be a different prediction.
        ("compute_metrics_rejected", "RAW logits plane", {}, {"compute_metrics": lambda eval_pred: {"noop": 0.0}}),
        *((name, match, overrides, {}) for name, match, overrides in spec.refusals),
    ]
    for name, match, overrides, kwargs in refusals:
        args = build_args(spec, f"{output_dir}/{name}", **overrides)
        expect_construction_error(spec.trainer_cls, name, checks, match, **{**common, "args": args, **kwargs})
    return checks


def eval_leg_checks(spec: PreferenceSpec, trainer, eval_calls, init_eval_metrics, reference_model):
    """COLLECTIVE (eval-loss gathers, world reductions). The init-weights eval_loss must equal HF's averaging
    of the exact per-batch objective over the real rows of each captured eval batch (inert padding rows
    excluded — a broken inert-row mask shifts every partial batch's loss), reproducing the gather order and
    accelerate's final-step remainder trim. Returns ``(checks, metrics)``."""
    checks, metrics = {}, {}
    real_row_flags = [(call["labels"] != -100).any(dim=-1) for call in eval_calls]
    per_batch = [
        spec.objective(reference_model, {key: value[real] for key, value in call.items() if torch.is_tensor(value)})
        for call, real in zip(eval_calls, real_row_flags, strict=True)
    ]
    dp_size = trainer.parallelism_config.data_parallel_size
    dp_rank = trainer.parallelism_config.get_data_parallel_rank()
    remainder = spec.eval_examples % (spec.examples_per_rank * dp_size)
    expected = hf_eval_loss_reference(per_batch, spec.examples_per_rank, remainder, dp_rank=dp_rank, dp_size=dp_size)
    eval_loss = init_eval_metrics["eval_loss"]
    metrics["init_eval_loss"], metrics["init_eval_reference"] = eval_loss, expected
    checks["eval_loss_matches_reference"] = abs(eval_loss - expected) < REL_TOL * max(1.0, abs(expected))
    # Second, independent oracle: every eval example scored once, walked example by example with no model of
    # the gather at all. On the partial final batch a per-stage-duplicating gather keeps a prefix of the
    # duplicates — a different set of examples, which the two derivations disagree about.
    by_example = eval_loss_over_examples(
        per_batch, spec.examples_per_rank, spec.eval_examples, dp_rank=dp_rank, dp_size=dp_size
    )
    metrics["init_eval_by_example"] = by_example
    checks["eval_loss_matches_independent_example_walk"] = abs(eval_loss - by_example) < REL_TOL * max(
        1.0, abs(by_example)
    )
    # The eval normalizer must be each batch's real example count (the partial batch's, not the frozen one's).
    checks["eval_normalizer_is_real_example_count"] = all(
        abs(call["num_items"] - float(real.sum()) / spec.rows_per_example) < 1e-6
        for call, real in zip(eval_calls, real_row_flags, strict=True)
    )
    # The partial batch genuinely occurred somewhere in the world (inert rows were padded in).
    checks["partial_eval_batch_exercised"] = world_any(any(not bool(real.all()) for real in real_row_flags))
    checks["eval_loss_identical_across_ranks"] = losses_identical_across_ranks([eval_loss])
    # Two in-loop evals (steps 5 and 10) ran on trained weights: finite and rank-identical.
    inloop_eval = [e["eval_loss"] for e in trainer.state.log_history if "eval_loss" in e]
    checks["inloop_evals_ran_finite"] = len(inloop_eval) == 2 and bool(torch.isfinite(torch.tensor(inloop_eval)).all())
    return checks, metrics


def second_run_checks(spec: PreferenceSpec, config, tokenizer, reference_model, output_dir: str):
    """COLLECTIVE (a whole training run). Two steps of the trainer's other loss path through the same closure,
    the step-1 loss pinned like the main run's. Returns ``(checks, metrics)``."""
    second = spec.second_run
    args = build_args(spec, f"{output_dir}/{second.prefix}", max_steps=2, **second.config)
    rows = build_rows(spec, spec.train_examples, seed=11, **second.row_kwargs)
    trainer, steps, _, _ = run_trainer(spec, config, args, rows, tokenizer)
    checks = second.batch_checks(steps[0])
    loss = step_losses(trainer)[0]
    expected = world_mean(spec.objective(reference_model, steps[0], **second.objective_kwargs))
    checks[f"{second.prefix}_step1_loss_matches_reference"] = abs(loss - expected) < REL_TOL * max(1.0, abs(expected))
    return checks, {f"{second.prefix}_step1_loss": loss, f"{second.prefix}_step1_reference": expected}


@gpu_test_main(exact_world_size=4, prefix="pp_preference_precompute")
def run(ctx):
    # 4 GPUs as 2 simulated nodes of 2, pp=2 x dp=2 (stage FSDP over 2 ranks).
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    dataset = build_rows(SPEC, SPEC.train_examples)
    checks, metrics = refusal_checks(SPEC, config, tokenizer, dataset, ctx.output_dir), {}

    # ── Main run, with the loss-only PP eval leg (init-weights evaluate + 2 in-loop) ─────────────────────
    args = build_args(SPEC, ctx.output_dir, eval_strategy="steps", eval_steps=EVAL_STEPS)
    eval_dataset = build_rows(SPEC, SPEC.eval_examples, seed=23)
    trainer, steps, eval_calls, init_eval_metrics = run_trainer(SPEC, config, args, dataset, tokenizer, eval_dataset)
    captured = steps[0]
    checks["accumulation_became_microbatches"] = (
        args.gradient_accumulation_steps == 1 and trainer._pp_n_microbatches == MICROBATCHES
    )
    # The collated batch reaches the runner padded to the fixed shape: every key the batch transform reads,
    # completion_mask included (the adapter's pad_spec), or the transform sees two widths.
    rows_per_rank = SPEC.rows_per_example * SPEC.examples_per_rank
    checks["batch_padded_to_fixed_shape"] = captured["input_ids"].shape == (rows_per_rank, SEQ_LEN)
    checks["rows_per_example_declared"] = trainer._pp_runner.rows_per_example == SPEC.rows_per_example
    checks["no_reference_model"] = trainer.ref_model is None
    checks["normalizer_is_example_count"] = len(steps) == N_STEPS and all(
        abs(step["num_items"] - SPEC.examples_per_rank) < 1e-6 for step in steps
    )
    checks.update(SPEC.batch_checks(captured))
    # The equivalence reference consumes batch_transform's output, so a transform bug corrupting both sides
    # identically (labels off the wrong mask, a broken interleave, shuffled extras) would pass it; pinning the
    # captured rows to TRL's own tokenized dataset is transform-independent.
    checks["captured_rows_pin_to_raw_tokenized_examples"] = SPEC.rows_pinned(captured, trainer.train_dataset)

    losses = step_losses(trainer)
    checks["ran_all_steps"] = trainer.state.global_step == N_STEPS and len(losses) == N_STEPS
    checks["losses_finite"] = bool(torch.isfinite(torch.tensor(losses)).all())
    reference_model = build_dense_model(tokenizer, vocab_size=VOCAB_SIZE).to(ctx.device).eval()
    expected = world_mean(SPEC.objective(reference_model, captured))
    metrics["step1_loss"], metrics["step1_reference"] = losses[0], expected
    checks["step1_loss_matches_reference"] = abs(losses[0] - expected) < REL_TOL * max(1.0, abs(expected))
    # The ref columns genuinely enter the loss: zeroed refs must not match.
    zero_refs = {**captured, "ref_logps": torch.zeros_like(captured["ref_logps"])}
    checks["ref_logps_shift_the_loss"] = abs(losses[0] - world_mean(SPEC.objective(reference_model, zero_refs))) > 1e-2

    eval_checks, eval_metrics = eval_leg_checks(SPEC, trainer, eval_calls, init_eval_metrics, reference_model)
    checks.update(eval_checks)
    metrics.update(eval_metrics)
    checks["loss_decreased"] = losses[-1] < losses[0] - 1e-3
    metrics["first_loss"], metrics["last_loss"] = losses[0], losses[-1]
    checks["losses_identical_across_ranks"] = losses_identical_across_ranks(losses)
    grad_norms = [e["grad_norm"] for e in trainer.state.log_history if "grad_norm" in e]
    checks["grad_norm_logged_finite"] = bool(grad_norms) and bool(torch.isfinite(torch.tensor(grad_norms)).all())

    second_checks, second_metrics = second_run_checks(SPEC, config, tokenizer, reference_model, ctx.output_dir)
    checks.update(second_checks)
    metrics.update(second_metrics)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True, choices=sorted(SPECS))
    SPEC = SPECS[parser.parse_args().trainer]
    sys.exit(run())
