#!/usr/bin/env python
"""End-to-end pipeline parallelism through the real SmoothMarginPOTrainer.

SMPO under PP re-lays the [chosen ⧺ rejected] concat out as interleaved pairs and folds its three
normalizers (pair count, chosen/rejected token counts) into the microbatch loss as per-step
constants — every one of those seams produces a silently wrong number, not a crash, so the test
pins the step-1 logged loss against the exact SMPO objective computed on an unsplit copy of the
model over the same batches:

  * pair interleaving (a chunk-style split would compare unrelated examples)
  * the three-normalizer folding (a wrong denominator shifts the loss, not the trend)
  * the live scheduled target_margin (VariableSchedulerCallback -> stage attr -> step state)
  * the per-token min_log_prob clamp on rejected tokens
  * loss-only PP evaluation: init-weights eval_loss pinned against the exact per-batch objective
    with HF's gather/trim averaging reproduced, a partial final eval batch proving the inert
    row-padding contributes exactly 0 (the doubled-row frozen shape via rows_per_example=2), and
    two in-loop evals staying finite and rank-identical
  * construction-time rejections (percentile clip, pair-splitting microbatch shape)
  * compute_metrics under PP: the predictions the last stage broadcasts must be the per-pair
    (chosen, rejected) rewards prediction_step reports, not the raw [rows, S, vocab] plane

Topology: 4 GPUs as 2 simulated nodes of 2, pp=2 x dp=2 (stage FSDP over 2 ranks).

Run: torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_smpo.py
"""

import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer

from src.configs.preference import SmoothMarginPOConfig
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.preference.smpo import SmoothMarginPOTrainer
from tests.common.distributed import hf_eval_loss_reference
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_CONFIG
from tests.common.pp_equivalence import build_dense_model
from tests.common.utils import log

PP_SIZE = 2
SEQ_LEN = 48  # pipeline fixed shape; batch-max concat lengths stay below it on purpose
N_STEPS = 12
REL_TOL = 1e-4
PAIRS_PER_RANK = 4  # per_device_train_batch_size counts pairs; the pair transform doubles the rows
# 14 pairs over dp=2 at eval batch 4 (even_batches off) = batches of 4/4/4/2 pairs: one DP shard
# ends on a partial batch, so the eval leg exercises the inert row-padding for real.
EVAL_PAIRS = 14
EVAL_STEPS = 6

# Main-run objective knobs, shared with the reference computation.
BETA = 1.0
SFT_RATIO = 0.8
MIN_LOG_PROB = -3.5
INITIAL_MARGIN = 0.2
TARGET_MARGIN = 0.8
IPO_MARGIN = 0.4


def build_pref_dataset(vocab_size: int, n_rows: int = 16, seed: int = 7) -> Dataset:
    """Pre-tokenized preference rows with ragged prompt/completion lengths.

    Ragged lengths make the chosen/rejected token counts unequal per rank, which is what exercises
    the per-step normalizer constants — equal counts would let a wrong denominator cancel out.
    """
    g = torch.Generator().manual_seed(seed)

    def rand_ids(lo: int, hi: int) -> list[int]:
        length = int(torch.randint(lo, hi, (1,), generator=g))
        return torch.randint(4, vocab_size, (length,), generator=g).tolist()

    rows = {"prompt_input_ids": [], "chosen_input_ids": [], "rejected_input_ids": []}
    for _ in range(n_rows):
        rows["prompt_input_ids"].append(rand_ids(3, 10))
        rows["chosen_input_ids"].append(rand_ids(4, 14) + [1])  # terminal EOS like _tokenize_row
        rows["rejected_input_ids"].append(rand_ids(4, 14) + [1])
    return Dataset.from_dict(rows)


def smpo_reference(model, input_ids, attention_mask, labels, *, margin, min_log_prob, loss_type) -> float:
    """The exact SMPO objective on an interleaved [c0, r0, ...] batch, fp32, independent of src/."""
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    logps = torch.log_softmax(logits[:, :-1, :], dim=-1)
    logps = logps.gather(-1, shift_labels.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1) * mask

    chosen, rejected = logps[0::2], logps[1::2]
    cmask, rmask = mask[0::2], mask[1::2]
    nll_chosen, nll_rejected = -chosen.sum(), -rejected.sum()  # SFT anchor is pre-clip
    if min_log_prob is not None:
        rejected = torch.where(rmask & (rejected < min_log_prob), torch.tensor(min_log_prob), rejected)

    z = chosen.sum(-1) / cmask.sum(-1).clamp(min=1) - rejected.sum(-1) / rmask.sum(-1).clamp(min=1) - margin
    if loss_type == "sigmoid":
        pair_losses = -F.logsigmoid(BETA * z)
    elif loss_type == "ipo":
        pair_losses = (BETA * z).pow(2)
    else:
        raise ValueError(loss_type)
    sft = SFT_RATIO * nll_chosen / cmask.sum().clamp(min=1) + (1 - SFT_RATIO) * nll_rejected / rmask.sum().clamp(min=1)
    return float(pair_losses.mean() + sft)


def make_args(output_dir: str, **overrides) -> SmoothMarginPOConfig:
    base = {
        "output_dir": output_dir,
        "max_steps": N_STEPS,
        "per_device_train_batch_size": PAIRS_PER_RANK,
        "per_device_eval_batch_size": PAIRS_PER_RANK,  # PP freezes eval batches to the train shape
        # even_batches would complete the tail eval batch by wrapping around to the dataset start;
        # off, the final batch stays partial and must be row-padded with inert rows.
        "accelerator_config": {"even_batches": False},
        "gradient_accumulation_steps": 2,  # becomes the pipeline's microbatch count
        "learning_rate": 5e-3,
        "lr_scheduler_type": "constant",
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
        "logging_nan_inf_filter": False,
        "seed": 42,
        "bf16": False,
        "max_length": SEQ_LEN,
        "max_prompt_length": 16,
        "max_completion_length": 16,
        "beta": BETA,
        "chosen_sft_ratio": SFT_RATIO,
        "loss_type": "sigmoid",
        "target_margin": TARGET_MARGIN,
        "initial_margin": INITIAL_MARGIN,
        "use_margin_schedule": True,
        "min_log_prob": MIN_LOG_PROB,
        # The config defaults enable the percentile clip; PP rejects it (batch-global quantiles).
        "lower_clip_percentile": None,
        "upper_clip_percentile": None,
    }
    base.update(overrides)
    return SmoothMarginPOConfig(**base)


def gathered_mean(value: float, world_size: int) -> float:
    """Mean of a rank-local scalar across all ranks — HF logs the world-gathered mean loss."""
    local = torch.tensor([value], device=torch.cuda.current_device())
    peers = [torch.zeros_like(local) for _ in range(world_size)]
    dist.all_gather(peers, local)
    return float(torch.stack(peers).mean())


def expect_construction_error(name: str, checks: dict, match: str, **trainer_kwargs) -> None:
    """Construct on every rank expecting a uniform pre-collective ValueError."""
    try:
        SmoothMarginPOTrainer(**trainer_kwargs)
        checks[name] = False
    except ValueError as exc:
        checks[name] = match in str(exc)
        if not checks[name]:
            log(f"[{name}] raised without the expected mechanism: {exc}")


def stage_shards(trainer) -> dict[str, torch.Tensor]:
    """This rank's sharded parameter data — the tensors the optimizer must be updating."""
    return {n: p.detach().to_local().clone() for n, p in trainer._pp_runner.stage.submod.named_parameters()}


def run_trainer(config, args, dataset, tokenizer, eval_dataset=None):
    """Build a PP SMPO trainer with spies on the train step and (when evaluating) the eval step.

    With ``eval_dataset``, an init-weights ``evaluate()`` runs before training (its per-call eval
    batches and the live margin the step state carried are captured for the exact eval reference —
    trained-weights evals cannot be reproduced on an unsplit copy) and its metrics are returned
    alongside the captures, together with the init-weights shards taken before that eval.
    """
    trainer = SmoothMarginPOTrainer(
        model=build_dense_model(),
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=config,
    )
    init_shards = stage_shards(trainer)

    captured: dict = {}
    margins: list[float] = []
    real_step = trainer._pp_runner.step

    def spying_step(input_ids, labels, **kw):
        if "batch" not in captured:
            captured["batch"] = (
                input_ids.detach().cpu(),
                kw["attention_mask"].detach().cpu(),
                labels.detach().cpu(),
            )
        margins.append(float(trainer._pp_loss.constants.target_margin))
        return real_step(input_ids, labels, **kw)

    trainer._pp_runner.step = spying_step

    eval_calls: list[dict] = []
    init_eval_metrics = None
    if eval_dataset is not None:
        real_eval = trainer._pp_runner.eval_loss

        def spying_eval(input_ids, labels, **kw):
            eval_calls.append(
                {
                    "input_ids": input_ids.detach().cpu(),
                    "labels": labels.detach().cpu(),
                    "attention_mask": kw["attention_mask"].detach().cpu(),
                    "margin": float(trainer._pp_loss.constants.target_margin),
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
    return trainer, captured["batch"], margins, init_eval_calls, init_eval_metrics, init_shards


@gpu_test_main(exact_world_size=4, prefix="pp_smpo")
def run(ctx):
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    # The full tokenizer's pad id (151643) is outside the tiny 1024-token vocab and would send the
    # stage-0 embedding gather out of bounds; repoint padding at id 0 (the model config's pad id).
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(0)
    dataset = build_pref_dataset(TINY_QWEN3_CONFIG["vocab_size"])
    checks, metrics = {}, {}

    # ── Construction-time rejections (all raise before any collective) ───────────────────────────
    # Percentile clip: the config default (lower_clip_percentile=0.02) must be rejected under PP.
    expect_construction_error(
        "percentile_clip_rejected",
        checks,
        "quantile",
        model=build_dense_model(),
        args=make_args(ctx.output_dir + "/clip", lower_clip_percentile=0.02),
        train_dataset=dataset,
        processing_class=tokenizer,
        parallelism_config=config,
    )
    # Pair alignment: bs=3 with 2 microbatches would split a chosen/rejected pair.
    expect_construction_error(
        "pair_splitting_batch_rejected",
        checks,
        "divisible",
        model=build_dense_model(),
        args=make_args(ctx.output_dir + "/odd", per_device_train_batch_size=3),
        train_dataset=dataset,
        processing_class=tokenizer,
        parallelism_config=config,
    )
    # ── Main run: sigmoid + margin schedule + min_log_prob clamp + loss-only PP eval leg ─────────
    args = make_args(ctx.output_dir, eval_strategy="steps", eval_steps=EVAL_STEPS)
    eval_dataset = build_pref_dataset(TINY_QWEN3_CONFIG["vocab_size"], n_rows=EVAL_PAIRS, seed=23)
    trainer, (ids, mask, labels), margins, eval_calls, init_eval_metrics, init_shards = run_trainer(
        config, args, dataset, tokenizer, eval_dataset=eval_dataset
    )
    # The pre-train evaluate() leaves the FSDP2 stage gathered; the optimizer HF then builds must still
    # hold the shards, or every step updates orphan tensors and the weights never move (a frozen run
    # shows finite grad norms, an exact step-1 loss, and a loss that only follows the margin schedule).
    trained_shards = stage_shards(trainer)
    checks["shards_updated_after_pretrain_eval"] = (
        max(float((trained_shards[name] - init).abs().max()) for name, init in init_shards.items()) > 0.0
    )

    checks["accumulation_became_microbatches"] = (
        args.gradient_accumulation_steps == 1 and trainer._pp_n_microbatches == 2
    )
    checks["pairs_interleaved_2x_batch"] = ids.size(0) == 8 and ids.size(1) == SEQ_LEN

    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    log(f"[rank {ctx.rank}] losses={[f'{v:.4f}' for v in losses]} margins={[f'{m:.3f}' for m in margins[:3]]}...")
    checks["ran_all_steps"] = trainer.state.global_step == N_STEPS and len(losses) == N_STEPS
    checks["losses_finite"] = all(torch.isfinite(torch.tensor(losses)))

    # Step-1 logged loss == the exact SMPO objective on the unsplit model over the same batches.
    # Each chain's loss is rank-local-normalized; HF logs the world-gathered mean, which equals the
    # mean of the per-rank references (chain ranks hold identical batches).
    ref_model = build_dense_model()
    reference = smpo_reference(
        ref_model, ids, mask, labels, margin=INITIAL_MARGIN, min_log_prob=MIN_LOG_PROB, loss_type="sigmoid"
    )
    expected = gathered_mean(reference, ctx.world_size)
    metrics["step1_loss"], metrics["step1_reference"] = losses[0], expected
    checks["step1_loss_matches_reference"] = abs(losses[0] - expected) / max(1.0, abs(expected)) < REL_TOL

    # The margin genuinely enters the loss: a zero-margin reference must not match.
    zero_margin = gathered_mean(
        smpo_reference(ref_model, ids, mask, labels, margin=0.0, min_log_prob=MIN_LOG_PROB, loss_type="sigmoid"),
        ctx.world_size,
    )
    checks["margin_shifts_the_loss"] = abs(losses[0] - zero_margin) > 0.01
    # And so does the clamp: an unclamped reference must not match (random init logps ~ -6.9).
    unclamped = gathered_mean(
        smpo_reference(ref_model, ids, mask, labels, margin=INITIAL_MARGIN, min_log_prob=None, loss_type="sigmoid"),
        ctx.world_size,
    )
    checks["min_log_prob_clamp_wired"] = abs(losses[0] - unclamped) > 0.01

    # ── Eval leg: the init-weights eval_loss must equal HF's averaging of the exact per-batch SMPO
    # objective over the real pairs of each captured eval batch (inert padding rows excluded — a
    # broken inert-pair mask shifts every partial batch's loss by the margin loss at zero
    # log-probs), reproducing the gather order and accelerate's final-step remainder trim. The
    # margin is the one the step state carried at each call — the stage attribute seeded by the
    # PP adapter, i.e. INITIAL_MARGIN before any training step.
    checks["init_eval_margin_is_initial"] = all(abs(c["margin"] - INITIAL_MARGIN) < 1e-6 for c in eval_calls)
    real_row_flags = [(call["labels"] != -100).any(dim=-1) for call in eval_calls]
    step_losses = [
        smpo_reference(
            ref_model,
            call["input_ids"][real],
            call["attention_mask"][real],
            call["labels"][real],
            margin=call["margin"],
            min_log_prob=MIN_LOG_PROB,
            loss_type="sigmoid",
        )
        for call, real in zip(eval_calls, real_row_flags, strict=True)
    ]
    dp_size, dp_rank = (
        trainer.parallelism_config.data_parallel_size,
        trainer.parallelism_config.get_data_parallel_rank(),
    )
    remainder = EVAL_PAIRS % (PAIRS_PER_RANK * dp_size)
    expected_eval = hf_eval_loss_reference(step_losses, PAIRS_PER_RANK, remainder, dp_rank=dp_rank, dp_size=dp_size)
    eval_loss = init_eval_metrics["eval_loss"]
    metrics["init_eval_loss"], metrics["init_eval_reference"] = eval_loss, expected_eval
    checks["eval_loss_matches_reference"] = abs(eval_loss - expected_eval) / max(1.0, abs(expected_eval)) < REL_TOL

    # SMPO normalizes inside the loss closure — the runner's eval divisor must stay 1.
    checks["eval_normalizer_is_one"] = all(abs(call["num_items"] - 1.0) < 1e-9 for call in eval_calls)
    # The partial batch genuinely occurred somewhere in the world (inert rows were padded in).
    had_partial = torch.tensor(
        float(any(not bool(real.all()) for real in real_row_flags)), device=torch.cuda.current_device()
    )
    dist.all_reduce(had_partial, op=dist.ReduceOp.MAX)
    checks["partial_eval_batch_exercised"] = bool(had_partial)

    local_eval = torch.tensor([eval_loss], device=torch.cuda.current_device())
    gathered_eval = [torch.zeros_like(local_eval) for _ in range(ctx.world_size)]
    dist.all_gather(gathered_eval, local_eval)
    checks["eval_loss_identical_across_ranks"] = all(torch.allclose(local_eval, g) for g in gathered_eval)

    # Two in-loop evals (steps 6 and 12) ran on trained weights: finite and rank-identical.
    inloop_eval = [e["eval_loss"] for e in trainer.state.log_history if "eval_loss" in e]
    checks["inloop_evals_ran_finite"] = len(inloop_eval) == 2 and bool(torch.isfinite(torch.tensor(inloop_eval)).all())

    # Live schedule: step 1 reads initial_margin; by step 12 the linear schedule has moved it.
    checks["margin_schedule_live"] = (
        len(margins) == N_STEPS and abs(margins[0] - INITIAL_MARGIN) < 1e-6 and margins[-1] > INITIAL_MARGIN + 0.3
    )

    checks["loss_decreased"] = losses[-1] < losses[0] - 0.05
    metrics["first_loss"], metrics["last_loss"] = losses[0], losses[-1]

    local = torch.tensor(losses, device=torch.cuda.current_device())
    gathered = [torch.zeros_like(local) for _ in range(ctx.world_size)]
    dist.all_gather(gathered, local)
    checks["losses_identical_across_ranks"] = all(torch.allclose(local, g, atol=1e-6) for g in gathered)

    grad_norms = [e["grad_norm"] for e in trainer.state.log_history if "grad_norm" in e]
    checks["grad_norm_logged_finite"] = bool(grad_norms) and all(torch.isfinite(torch.tensor(grad_norms)))

    # ── IPO run: second loss type through the same closure, fixed nonzero margin, no clamp ───────
    ipo_args = make_args(
        ctx.output_dir + "/ipo",
        max_steps=2,
        loss_type="ipo",
        use_margin_schedule=False,
        target_margin=IPO_MARGIN,
        min_log_prob=None,
    )
    ipo_trainer, (ipo_ids, ipo_mask, ipo_labels), _, _, _, _ = run_trainer(
        config, ipo_args, build_pref_dataset(TINY_QWEN3_CONFIG["vocab_size"], seed=11), tokenizer
    )
    ipo_losses = [e["loss"] for e in ipo_trainer.state.log_history if "loss" in e]
    ipo_expected = gathered_mean(
        smpo_reference(
            ref_model, ipo_ids, ipo_mask, ipo_labels, margin=IPO_MARGIN, min_log_prob=None, loss_type="ipo"
        ),
        ctx.world_size,
    )
    metrics["ipo_step1_loss"], metrics["ipo_step1_reference"] = ipo_losses[0], ipo_expected
    checks["ipo_step1_loss_matches_reference"] = (
        abs(ipo_losses[0] - ipo_expected) / max(1.0, abs(ipo_expected)) < REL_TOL
    )

    # ── compute_metrics leg: SMPO's prediction_step predicts each pair's (chosen, rejected) rewards,
    # so the PP path must reduce the last stage's output to those [pairs, 2] before the chain
    # broadcast. Without the adapter's predictions_fn the raw [rows, S, vocab] plane would arrive at
    # compute_metrics instead — the received shape is what catches that regression.
    seen_shapes: list[tuple[int, ...]] = []

    def record_prediction_shape(eval_pred) -> dict:
        predictions = torch.as_tensor(eval_pred.predictions)
        seen_shapes.append(tuple(predictions.shape))
        return {"reward_gap": float(predictions[:, 0].mean() - predictions[:, 1].mean())}

    metrics_trainer = SmoothMarginPOTrainer(
        model=build_dense_model(),
        args=make_args(ctx.output_dir + "/metrics", max_steps=1, use_margin_schedule=False),
        train_dataset=dataset,
        # PAIRS_PER_RANK pairs per DP shard: exactly one full eval batch each, no partial tail.
        eval_dataset=build_pref_dataset(TINY_QWEN3_CONFIG["vocab_size"], n_rows=2 * PAIRS_PER_RANK, seed=31),
        processing_class=tokenizer,
        parallelism_config=config,
        compute_metrics=record_prediction_shape,
    )
    metrics_eval = metrics_trainer.evaluate()
    log(f"[rank {ctx.rank}] compute_metrics saw shapes={seen_shapes} metrics={metrics_eval}")
    # One (chosen, rejected) row per pair over the data-parallel replicas — the chain's stages all
    # return the same rows, and the metric gather keeps one copy. 2-D [pairs, 2], never the logits
    # plane (which would arrive 3-D as [rows, S, vocab] no matter how HF trims the gather).
    checks["metrics_receive_per_pair_rewards"] = seen_shapes == [(PAIRS_PER_RANK * ctx.world_size // PP_SIZE, 2)]
    checks["reward_gap_reported_finite"] = "eval_reward_gap" in metrics_eval and bool(
        torch.isfinite(torch.tensor(metrics_eval.get("eval_reward_gap", float("nan"))))
    )
    metrics["metrics_prediction_rows"] = float(seen_shapes[0][0]) if seen_shapes else float("nan")

    # Direct pin on the reduction itself: pair 0's rewards are beta times each side's mean shifted
    # log-prob over its supervised positions (the rejected side clamped at min_log_prob), chosen row
    # even and rejected odd; the inert all-ignore padding pair predicts zero.
    probe_logits = torch.randn(4, 6, 5, device=ctx.device)
    probe_labels = torch.full((4, 6), -100, dtype=torch.long, device=ctx.device)
    probe_labels[:2, 3:5] = 3  # rows 2 and 3 stay all-ignore: the inert row padding
    rewards = metrics_trainer._pp_loss.predictions(probe_logits, {"labels": probe_labels})
    probe_mask = probe_labels[:, 1:] != -100
    logps = F.log_softmax(probe_logits[:, :-1].float(), dim=-1)
    logps = logps.gather(-1, probe_labels[:, 1:].clamp(min=0).unsqueeze(-1)).squeeze(-1)
    rejected = logps[1].clamp(min=MIN_LOG_PROB)
    expected = BETA * torch.stack([logps[0][probe_mask[0]].mean(), rejected[probe_mask[1]].mean()])
    checks["predictions_are_per_pair_rewards"] = rewards.shape == (2, 2) and torch.allclose(
        rewards[0], expected, atol=1e-5
    )
    checks["inert_pair_predicts_zero"] = bool((rewards[1] == 0).all())

    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
