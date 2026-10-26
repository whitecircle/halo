#!/usr/bin/env python
"""End-to-end pipeline parallelism through the real OfflineGRPOTrainer (kl_beta == 0; the KL term
has its own gate, test_pp_offline_grpo_kl.py).

Offline GRPO under PP folds each loss type's whole-batch denominator into the runner's per-step
normalizer while the microbatch loss produces the bare numerator — every seam yields a silently
wrong number, not a crash, so the step-1 logged loss is pinned against the exact offline-GRPO
objective computed on an unsplit copy of the model over the captured batches:

  * the prompt|completion concat + labels derivation (split collator keys → runner contract)
  * the per-loss-type normalizer (grpo: Σ group weights; bnpo: group-weighted token count;
    dr_grpo: Σ group weights × max_completion_length) with heterogeneous group sizes, so the
    denominators genuinely differ across the microbatch split
  * the negative-advantage min_log_prob clamp (live off the model attr)
  * both pg formulations (reinforce in the main/bnpo runs, prob_weighted in the dr_grpo run)
  * loss-only PP evaluation: init-weights eval_loss pinned against the exact per-batch objective
    (the eval dataloader's BatchSampler drops partial tails, so full batches only), two in-loop
    evals staying finite and rank-identical, and a direct partial-batch prediction_step proving
    the inert row-padding contributes exactly 0 (pad_spec group_size=1 — a 0 filler would NaN
    via inf x 0)
  * construction-time rejections (undersized max_length, compute_metrics)

Topology: 4 GPUs as 2 simulated nodes of 2, pp=2 x dp=2 (stage FSDP over 2 ranks).

Run: torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_offline_grpo.py
"""

import sys

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import AutoTokenizer

from src.configs.grpo_offline import OfflineGRPOConfig
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.grpo.offline.trainer import OfflineGRPOTrainer
from tests.common.distributed import hf_eval_loss_reference, world_mean
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.pp_equivalence import build_dense_model
from tests.common.utils import log

PP_SIZE = 2
MICROBATCHES = 2
BATCH = 4
N_STEPS = 10
# Eval group sizes: 16 completions split 8/8 across dp=2 by the contiguous MultiGroupSampler
# split — exactly 2 full batches per DP shard (the eval BatchSampler drops partial tails) and a
# zero gather-trim remainder, so HF's eval_loss is the plain mean over the 4 per-batch losses.
EVAL_GROUP_SIZES = [4, 4, 2, 3, 3]
EVAL_STEPS = 5
MAX_PROMPT = 16
MAX_COMPLETION = 16
MIN_LOG_PROB = -3.0
REL_TOL = 1e-4

# The real Qwen tokenizer feeds tokenize_row, so the tiny model must span its full id range.
VOCAB_SIZE = 151936


def build_dataset(sizes: list[int] | None = None, prompt_offset: int = 0) -> Dataset:
    """(prompt, completions, rewards) groups with heterogeneous sizes and distinct rewards.

    Distinct rewards give nonzero z-norm advantages of both signs (exercising the clamp's
    negative-advantage gating); varied group sizes make the group-weight denominators differ per
    microbatch split. ``prompt_offset`` keeps a second (eval) dataset's prompts distinct.
    """
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
    """The exact offline-GRPO objective (kl_beta=0) on an unsplit model, fp32, independent of src/."""
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


def make_args(output_dir: str, **overrides) -> OfflineGRPOConfig:
    base = {
        "output_dir": output_dir,
        "max_steps": N_STEPS,
        "per_device_train_batch_size": BATCH,
        "per_device_eval_batch_size": BATCH,  # PP freezes eval batches to the train shape
        "gradient_accumulation_steps": MICROBATCHES,  # becomes the pipeline's microbatch count
        "learning_rate": 5e-3,
        "lr_scheduler_type": "constant",
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
        "logging_nan_inf_filter": False,
        "seed": 42,
        "bf16": False,
        "max_prompt_length": MAX_PROMPT,
        "max_completion_length": MAX_COMPLETION,
        "kl_beta": 0.0,
        "advantage_method": "z_norm",
        "loss_type": "grpo",
        "policy_gradient_formulation": "reinforce",
        "min_log_prob": MIN_LOG_PROB,
    }
    base.update(overrides)
    return OfflineGRPOConfig(**base)


def expect_construction_error(name: str, checks: dict, match: str, **trainer_kwargs) -> None:
    """Construct on every rank expecting a uniform pre-collective ValueError."""
    try:
        OfflineGRPOTrainer(**trainer_kwargs)
        checks[name] = False
    except ValueError as exc:
        checks[name] = match in str(exc)
        if not checks[name]:
            log(f"[{name}] raised without the expected mechanism: {exc}")


def run_trainer(config, args, dataset, tokenizer, eval_dataset=None):
    """Build a PP offline-GRPO trainer with spies on the train step and (when evaluating) the eval step.

    With ``eval_dataset``, before training (weights still init, so an unsplit copy can reproduce
    them): an init-weights ``evaluate()`` runs with its per-call eval batches captured, then a
    direct partial-batch ``prediction_step`` (2 rows < the frozen 4) exercises the inert row
    padding — its post-pad batch lands in the capture list too and its loss is returned.
    """
    trainer = OfflineGRPOTrainer(
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
            captured.update({k: v.clone() for k, v in kw["extra_targets"].items()})
        return real_step(input_ids, labels, **kw)

    trainer._pp_runner.step = spying_step

    eval_calls: list[dict] = []
    init_eval_metrics = None
    partial_loss = None
    if eval_dataset is not None:
        real_eval = trainer._pp_runner.eval_loss

        def spying_eval(input_ids, labels, **kw):
            eval_calls.append(
                {
                    "input_ids": input_ids.clone(),
                    "labels": labels.clone(),
                    "attention_mask": kw["attention_mask"].clone(),
                    "num_items": float(kw["num_items_in_batch"]),
                    **{k: v.clone() for k, v in kw["extra_targets"].items()},
                }
            )
            return real_eval(input_ids, labels, **kw)

        trainer._pp_runner.eval_loss = spying_eval
        init_eval_metrics = trainer.evaluate()
        # Direct partial-batch prediction step: 2 collated rows < the frozen 4. Collective — every
        # rank calls it with the same rows (eval_dataset is rank-identical).
        partial_batch = trainer.data_collator([trainer.eval_dataset[i] for i in range(2)])
        loss, _, _ = trainer.prediction_step(trainer.model, partial_batch, prediction_loss_only=True)
        partial_loss = float(loss)

    # Snapshot the init-weights captures under a new name: in-loop evals keep appending to
    # eval_calls (the closure's cell), and trained-weights batches must not leak into the
    # init-weights reference.
    init_eval_calls = list(eval_calls)
    trainer.train()
    return trainer, captured, normalizers, init_eval_calls, init_eval_metrics, partial_loss


@gpu_test_main(exact_world_size=4, prefix="pp_offline_grpo")
def run(ctx):
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = build_dataset()
    checks, metrics = {}, {}

    # ── Construction-time rejections (all raise before any collective or model load) ─────────────
    expect_construction_error(
        "undersized_max_length_rejected",
        checks,
        "below max_prompt_length",
        model=build_dense_model(tokenizer, vocab_size=VOCAB_SIZE),
        args=make_args(ctx.output_dir + "/short", max_length=8),
        train_dataset=dataset,
        processing_class=tokenizer,
        parallelism_config=config,
    )
    # This trainer's own prediction_step returns (loss, None, None) in every mode, so there is no
    # prediction convention for the pipeline to reproduce — serving compute_metrics under PP would
    # broadcast the raw logits plane and feed metrics a tensor no non-PP run produces.
    expect_construction_error(
        "compute_metrics_rejected",
        checks,
        "NO predictions",
        model=build_dense_model(tokenizer, vocab_size=VOCAB_SIZE),
        args=make_args(ctx.output_dir + "/metrics"),
        train_dataset=dataset,
        processing_class=tokenizer,
        parallelism_config=config,
        compute_metrics=lambda eval_pred: {"noop": 0.0},
    )

    # ── Main run: loss_type=grpo, reinforce, live min_log_prob clamp, loss-only PP eval leg ──────
    args = make_args(ctx.output_dir, eval_strategy="steps", eval_steps=EVAL_STEPS)
    eval_dataset = build_dataset(sizes=EVAL_GROUP_SIZES, prompt_offset=100)
    trainer, captured, normalizers, eval_calls, init_eval_metrics, partial_loss = run_trainer(
        config, args, dataset, tokenizer, eval_dataset=eval_dataset
    )

    checks["accumulation_became_microbatches"] = (
        args.gradient_accumulation_steps == 1 and trainer._pp_n_microbatches == MICROBATCHES
    )
    checks["max_length_derived_from_caps"] = args.max_length == MAX_PROMPT + MAX_COMPLETION
    checks["fixed_shape_batch"] = captured["input_ids"].shape == (BATCH, MAX_PROMPT + MAX_COMPLETION)

    # ── Row pinning: captured rows must map back to the raw dataset, transform-independently. The
    # equivalence reference consumes batch_transform's output, so a transform bug corrupting both
    # sides identically (labels off the wrong mask, prompt/completion concat misaligned) would pass
    # it; here the supervised label span must be a raw completion's tokenization (trainer contract:
    # add_special_tokens=False + EOS, truncated at the cap), the preceding real tokens that prompt's
    # left-truncated tokenization, and group_size the raw group's size.
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
    checks["normalizer_is_group_weight_sum"] = abs(normalizers[0] - expected_normalizer) < 1e-6

    # Step-1 logged loss == the exact objective on the unsplit model over the captured batches.
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

    # The clamp genuinely enters the loss: an unclamped reference must not match (random-init logps
    # ~ -11.9 sit far below the -3.0 floor on every negative-advantage row).
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

    # ── Eval leg: captures are the two evaluate() batches plus the direct partial-batch call. The
    # init-weights eval_loss must equal HF's averaging (zero remainder here — plain mean over the
    # per-batch losses) of the exact per-batch objective; each batch's reference runs over its
    # real rows only, which for the partial capture is what proves the inert rows contribute 0.
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

    # The eval normalizer must be the same per-loss-type denominator as training, over real rows.
    checks["eval_normalizer_is_group_weight_sum"] = all(
        abs(call["num_items"] - float((1.0 / call["group_size"][(call["labels"] != -100).any(dim=-1)].float()).sum()))
        < 1e-6
        for call in eval_calls
    )

    # Direct partial-batch prediction step (init weights): 2 real rows row-padded to the frozen 4
    # with group_size filler 1 — finite (a 0 filler would NaN via inf x 0) and exactly the 2-row
    # objective. Its capture must really carry inert rows.
    partial_call = eval_calls[2]
    partial_real = (partial_call["labels"] != -100).any(dim=-1)
    checks["partial_batch_row_padded"] = partial_call["input_ids"].size(0) == BATCH and int(partial_real.sum()) == 2
    partial_reference = eval_reference(partial_call)
    metrics["partial_loss"], metrics["partial_reference"] = partial_loss, partial_reference
    checks["partial_batch_loss_matches"] = abs(partial_loss - partial_reference) < REL_TOL * max(
        1.0, abs(partial_reference)
    )

    local_eval = torch.tensor([eval_loss], device=ctx.device)
    gathered_eval = [torch.zeros_like(local_eval) for _ in range(ctx.world_size)]
    dist.all_gather(gathered_eval, local_eval)
    checks["eval_loss_identical_across_ranks"] = all(torch.allclose(local_eval, g) for g in gathered_eval)

    # Two in-loop evals (steps 5 and 10) ran on trained weights: finite and rank-identical.
    inloop_eval = [e["eval_loss"] for e in trainer.state.log_history if "eval_loss" in e]
    checks["inloop_evals_ran_finite"] = len(inloop_eval) == 2 and bool(torch.isfinite(torch.tensor(inloop_eval)).all())

    checks["loss_decreased"] = losses[-1] < losses[0] - 0.01
    metrics["first_loss"], metrics["last_loss"] = losses[0], losses[-1]

    local = torch.tensor(losses, device=ctx.device)
    gathered = [torch.zeros_like(local) for _ in range(ctx.world_size)]
    dist.all_gather(gathered, local)
    checks["losses_identical_across_ranks"] = all(torch.allclose(local, g, atol=1e-6) for g in gathered)

    grad_norms = [e["grad_norm"] for e in trainer.state.log_history if "grad_norm" in e]
    checks["grad_norm_logged_finite"] = bool(grad_norms) and bool(torch.isfinite(torch.tensor(grad_norms)).all())

    # ── dr_grpo run: constant-length denominator, prob_weighted formulation, no clamp ────────────
    dr_args = make_args(
        ctx.output_dir + "/dr",
        max_steps=2,
        loss_type="dr_grpo",
        policy_gradient_formulation="prob_weighted",
        min_log_prob=None,
    )
    dr_trainer, dr_captured, dr_normalizers, _, _, _ = run_trainer(config, dr_args, dataset, tokenizer)
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
    checks["dr_grpo_normalizer"] = abs(dr_normalizers[0] - dr_expected_normalizer) < 1e-5

    # ── bnpo run: group-weighted token-count denominator ─────────────────────────────────────────
    bnpo_args = make_args(ctx.output_dir + "/bnpo", max_steps=2, loss_type="bnpo", min_log_prob=None)
    bnpo_trainer, bnpo_captured, _, _, _, _ = run_trainer(config, bnpo_args, dataset, tokenizer)
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
