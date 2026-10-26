#!/usr/bin/env python
"""Pipeline parallelism through the real DistributedRewardTrainer (Bradley-Terry reward model).

The reward PP contract has three seams the generic SFT e2e test cannot cover, and this test fails
when any of them breaks:

  * the interleave: TRL collates ``[chosen ⧺ rejected]``; a microbatch split of that layout compares
    unrelated examples. The step-1 loss is checked against the exact TRL objective (margin +
    centering) computed on the captured interleaved batch by transformers' own pooled forward —
    wrong pairing or wrong pooling both show as a loss mismatch, not a crash.
  * the normalizer: the BT loss is a per-pair mean, so the runner must divide by the chain-local
    pair count (spied per step), not the SFT token count.
  * the pair-alignment guard: a batch that would split a pair across microbatches must raise.

Also covers PP evaluation on the doubled-row preference batches (the trainer rescales the base
row bound) against a reference computed with the init weights: a full eval, whose reported accuracy
and prediction row count are both pinned (a mean is blind to duplicated rows, a row count is blind to
which pairs they are), plus a partial batch of one pair driven straight through the prediction step —
the only shape in which the inert row padding and the pair-unit trim actually execute.

Topology: 4 GPUs as 2 simulated nodes of 2 (NVLink domain = 2), pp2 x dp2.

Run: torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_reward_bt.py
"""

import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForSequenceClassification
from trl import RewardConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.reward.bradley_terry import DistributedRewardTrainer
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_CONFIG
from tests.common.utils import log

PP_SIZE = 2
SEQ_LEN = 48
N_STEPS = 10
PAIRS_PER_RANK = 4  # per_device_train_batch_size counts pairs; the collator doubles the rows
MICROBATCHES = 2
PAD_ID = 0
CENTER_COEF = 0.05
LOSS_RTOL = 2e-4


def build_model():
    torch.manual_seed(1234)
    config = Qwen3Config(**TINY_QWEN3_CONFIG, num_labels=1, pad_token_id=PAD_ID, eos_token_id=1)
    return Qwen3ForSequenceClassification(config)


def build_dataset(vocab_size: int, n_rows: int, seed: int) -> Dataset:
    """Pre-tokenized preference rows with ragged lengths (exercises the rightmost-non-pad pooling)
    and a nonzero margin column (exercises the margin extra)."""
    g = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(n_rows):
        chosen_len = int(torch.randint(6, SEQ_LEN, (1,), generator=g))
        rejected_len = int(torch.randint(6, SEQ_LEN, (1,), generator=g))
        rows.append(
            {
                "chosen_ids": torch.randint(4, vocab_size, (chosen_len,), generator=g).tolist(),
                "rejected_ids": torch.randint(4, vocab_size, (rejected_len,), generator=g).tolist(),
                "margin": round(float(torch.rand(1, generator=g)) * 0.5, 3),
            }
        )
    return Dataset.from_list(rows)


def reference_bt_loss(model, input_ids, attention_mask, margin_interleaved) -> float:
    """TRL's exact objective on an interleaved batch, pooled by transformers' own seq-cls forward."""
    with torch.no_grad():
        pooled = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1).float()
    chosen, rejected = pooled[0::2], pooled[1::2]
    loss = -F.logsigmoid(chosen - rejected - margin_interleaved.float()[0::2]).mean()
    loss = loss + CENTER_COEF * ((chosen + rejected) ** 2).mean()
    return float(loss)


def interleave_pairs(rows) -> list[list[int]]:
    """``[c0, r0, c1, r1, ...]`` — the row order ``_pp_reward_batch_transform`` produces."""
    return [ids for pair in zip(rows["chosen_ids"], rows["rejected_ids"], strict=False) for ids in pair]


def reference_pair_scores(model, rows, device) -> torch.Tensor:
    """``[pairs, 2]`` pooled (chosen, rejected) scores — what the BT adapter's ``predictions_fn`` must produce."""
    input_ids, attention_mask = pad_to(interleave_pairs(rows), SEQ_LEN)
    with torch.no_grad():
        pooled = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
    scores = pooled.logits.squeeze(-1).float()
    return torch.stack((scores[0::2], scores[1::2]), dim=-1)


def reference_pairwise_accuracy(model, rows, device) -> float:
    """``pairwise_accuracy`` recomputed from the dataset, over every pair exactly once.

    The metric the trainer reports is a mean, so it survives the per-stage duplication of the
    prediction rows unchanged — but not a dropped pair, and not an inert pad pair, which scores as a
    tie and lands on ``argmax == 0``, i.e. counts as correct.
    """
    scores = reference_pair_scores(model, rows, device)
    return float((scores[:, 0] >= scores[:, 1]).float().mean())


def pad_to(ids: list[list[int]], length: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch = torch.full((len(ids), length), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(ids), length), dtype=torch.long)
    for i, row in enumerate(ids):
        batch[i, : len(row)] = torch.tensor(row)
        mask[i, : len(row)] = 1
    return batch, mask


@gpu_test_main(exact_world_size=4, prefix="pp_reward_bt")
def run(ctx):
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    # The tiny model's vocab is 1024; Qwen's real pad id would overflow the embedding, so pad with
    # token 0 (data rows are drawn from [4, vocab) — pad never collides with a real token).
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(PAD_ID)

    args = RewardConfig(
        output_dir=ctx.output_dir,
        max_steps=N_STEPS,
        per_device_train_batch_size=PAIRS_PER_RANK,
        gradient_accumulation_steps=MICROBATCHES,  # becomes the pipeline's microbatch count
        learning_rate=8e-3,
        lr_scheduler_type="constant",
        max_length=SEQ_LEN,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        logging_nan_inf_filter=False,
        seed=42,
        bf16=False,
        gradient_checkpointing=False,
        center_rewards_coefficient=CENTER_COEF,
        per_device_eval_batch_size=PAIRS_PER_RANK,
    )

    eval_dataset = build_dataset(TINY_QWEN3_CONFIG["vocab_size"], n_rows=2 * PAIRS_PER_RANK, seed=23)

    # A preference metric consumes (chosen, rejected) scores. Without the adapter's predictions_fn
    # the raw [rows, S, 1] per-token plane would arrive here instead — shaped nothing like a
    # prediction — so recording the shape is what catches that regression.
    seen_shapes: list[tuple[int, ...]] = []

    def pairwise_accuracy(eval_pred) -> dict:
        predictions = torch.as_tensor(eval_pred.predictions)
        seen_shapes.append(tuple(predictions.shape))
        return {"accuracy": float((predictions.argmax(dim=-1) == 0).float().mean())}

    trainer = DistributedRewardTrainer(
        model=build_model(),
        args=args,
        train_dataset=build_dataset(TINY_QWEN3_CONFIG["vocab_size"], n_rows=32, seed=7),
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=config,
        compute_metrics=pairwise_accuracy,
    )

    checks, metrics = {}, {}
    checks["accumulation_became_microbatches"] = (
        args.gradient_accumulation_steps == 1 and trainer._pp_n_microbatches == MICROBATCHES
    )
    checks["margin_shipped_as_extra_target"] = "margin" in trainer._pp_adapter.extra_target_keys
    checks["paired_examples_declared"] = trainer._pp_runner.rows_per_example == 2

    reference = build_model().to(ctx.device).float().eval()

    # ── Eval with the init weights (before any training step): the logged eval_loss must equal the
    # mean over the two DP shards of TRL's per-batch BT loss, computed with transformers' own
    # pooled forward. Covers the doubled-row frozen shape, the eval normalizer (= pair count via
    # the labels markers), margins in eval, and the loss broadcast.
    eval_metrics = trainer.evaluate()
    shard_losses = []
    for lo in (0, PAIRS_PER_RANK):
        shard = eval_dataset[lo : lo + PAIRS_PER_RANK]
        input_ids, attention_mask = pad_to(interleave_pairs(shard), SEQ_LEN)
        margin = torch.tensor(shard["margin"]).repeat_interleave(2)
        shard_losses.append(
            reference_bt_loss(
                reference, input_ids.to(ctx.device), attention_mask.to(ctx.device), margin.to(ctx.device)
            )
        )
    reference_eval = sum(shard_losses) / len(shard_losses)
    log(f"[rank {ctx.rank}] eval_loss={eval_metrics['eval_loss']:.6f} reference={reference_eval:.6f}")
    checks["eval_loss_matches_reference"] = (
        abs(eval_metrics["eval_loss"] - reference_eval) / abs(reference_eval) < LOSS_RTOL
    )
    metrics["eval_loss"], metrics["reference_eval"] = eval_metrics["eval_loss"], reference_eval

    # compute_metrics must receive pooled pairwise scores, not the raw per-token plane. The shape is
    # the assertion: [pairs, 2] is a prediction, [rows, S, 1] is not.
    checks["metrics_receive_pairwise_scores"] = bool(seen_shapes) and all(
        len(shape) == 2 and shape[1] == 2 for shape in seen_shapes
    )
    checks["pairwise_accuracy_reported"] = "eval_accuracy" in eval_metrics
    # The value, not a range: ``0 <= accuracy <= 1`` holds for any input of any shape, so it passes
    # on a predictions_fn that scores the wrong pairs, half of them, or inert padding.
    accuracy = eval_metrics.get("eval_accuracy")
    reference_accuracy = reference_pairwise_accuracy(reference, eval_dataset[:], ctx.device)
    checks["pairwise_accuracy_matches_reference"] = accuracy is not None and abs(accuracy - reference_accuracy) < 1e-6
    # Every rank of a chain returns that chain's predictions, so a world gather would carry the
    # dataset pp_size times; the metric gather is scoped to the data-parallel replicas, which is
    # what makes this the dataset exactly once. Pinning the count is what makes a dropped or a
    # duplicated batch visible: the mean above is blind to duplication, and blind to a tail replaced
    # by pad pairs that score 1.0.
    checks["metrics_see_every_pair_exactly_once"] = seen_shapes == [(len(eval_dataset), 2)]
    log(
        f"[rank {ctx.rank}] compute_metrics saw shapes={seen_shapes} eval_accuracy={accuracy} "
        f"reference={reference_accuracy}"
    )
    metrics["eval_accuracy"] = float(accuracy) if accuracy is not None else float("nan")

    # ── A partial eval batch, driven straight through the prediction step. The exact-multiple set
    # above arrives already at the frozen row count, so neither the inert row padding nor the pair-unit
    # trim runs. One pair against the frozen 8 makes both: 3 inert pairs are appended and must be
    # trimmed back off, or compute_metrics gets 4 pairs where 1 belongs, three of them ties scoring as
    # correct. prediction_step keeps accelerate's gather out of it. Collective: every rank calls it.
    one_pair = eval_dataset[:1]
    chosen_ids, chosen_mask = pad_to(one_pair["chosen_ids"], SEQ_LEN)
    rejected_ids, rejected_mask = pad_to(one_pair["rejected_ids"], SEQ_LEN)
    partial_batch = {
        # The [chosen ⧺ rejected] concat TRL's collator emits, which the batch transform interleaves.
        "input_ids": torch.cat([chosen_ids, rejected_ids]).to(ctx.device),
        "attention_mask": torch.cat([chosen_mask, rejected_mask]).to(ctx.device),
        "margin": torch.tensor(one_pair["margin"], device=ctx.device),
    }
    _, partial_predictions, partial_labels = trainer.prediction_step(trainer.model, partial_batch, False)
    partial_reference = reference_pair_scores(reference, one_pair, ctx.device)
    log(f"[rank {ctx.rank}] partial-batch predictions={tuple(partial_predictions.shape)} vs 1 real pair")
    checks["partial_eval_batch_trims_to_the_real_pair"] = tuple(partial_predictions.shape) == (1, 2)
    checks["partial_eval_batch_scores_match_reference"] = torch.allclose(
        partial_predictions.float().cpu(), partial_reference.cpu(), atol=1e-4
    )
    # Predictions and labels trim together or compute_metrics gets mismatched lengths.
    checks["partial_eval_batch_labels_trim_with_predictions"] = partial_labels.shape[0] == 1

    # ── Spy on the runner step: capture step 1's interleaved batch + extras + normalizer.
    captured: dict = {}
    normalizers: list[float] = []
    real_step = trainer._pp_runner.step

    def spying_step(input_ids, labels, **kwargs):
        normalizers.append(float(kwargs.get("num_items_in_batch")))
        if not captured:
            captured["input_ids"] = input_ids.clone()
            captured["labels"] = labels.clone()
            captured["attention_mask"] = kwargs["attention_mask"].clone()
            captured["extras"] = {k: v.clone() for k, v in (kwargs.get("extra_targets") or {}).items()}
        return real_step(input_ids, labels, **kwargs)

    trainer._pp_runner.step = spying_step
    trainer.train()

    checks["normalizer_is_chain_pair_count"] = len(normalizers) == N_STEPS and all(
        abs(value - PAIRS_PER_RANK) < 1e-6 for value in normalizers
    )
    # The plane carries the pooling: exactly one marker per row, at that row's last non-pad token
    # (rows are right-padded with PAD_ID and data tokens start at 4, so length-1 is that position).
    # Marking only the chosen rows would leave the rejected halves unpooled and unvalidated.
    marked = captured["labels"] != -100
    checks["plane_marks_every_row_at_pooled_position"] = bool(
        (marked.sum(dim=-1) == 1).all()
        and torch.equal(marked.int().argmax(dim=-1), captured["attention_mask"].sum(dim=-1) - 1)
    )
    checks["no_input_ids_extra_target"] = "input_ids" not in trainer._pp_adapter.extra_target_keys

    # ── Step-1 loss against the exact TRL objective on the captured batch (init weights). The
    # logged loss is the world mean of the per-chain losses; each chain contributes pp_size times
    # with identical values, so the mean over ranks equals the mean over chains.
    local_reference = torch.tensor(
        reference_bt_loss(reference, captured["input_ids"], captured["attention_mask"], captured["extras"]["margin"]),
        device=ctx.device,
    )
    dist.all_reduce(local_reference, op=dist.ReduceOp.AVG)
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    log(f"[rank {ctx.rank}] step1={losses[0]:.6f} reference={float(local_reference):.6f}")
    checks["step1_loss_matches_reference"] = (
        abs(losses[0] - float(local_reference)) / abs(float(local_reference)) < LOSS_RTOL
    )
    metrics["step1_loss"], metrics["step1_reference"] = losses[0], float(local_reference)

    checks["ran_all_steps"] = trainer.state.global_step == N_STEPS
    checks["losses_finite"] = len(losses) == N_STEPS and bool(torch.isfinite(torch.tensor(losses)).all())
    checks["loss_decreased"] = losses[-1] < losses[0] - 0.05
    metrics["first_loss"], metrics["last_loss"] = losses[0], losses[-1]

    local = torch.tensor(losses, device=ctx.device)
    gathered = [torch.zeros_like(local) for _ in range(ctx.world_size)]
    dist.all_gather(gathered, local)
    checks["losses_identical_across_ranks"] = all(torch.allclose(local, peer, atol=1e-6) for peer in gathered)

    # ── Negative check: a batch whose rows cannot be microbatched pair-aligned must raise. 6 rows
    # (3 pairs) across 2 microbatches would split pair 1 — the runner validates before any
    # collective, so every rank raises and catches locally with no desync.
    bad_ids = torch.full((6, SEQ_LEN), 5, dtype=torch.long, device=ctx.device)
    raised = False
    try:
        trainer._pp_runner.step(
            bad_ids,
            torch.full_like(bad_ids, -100),
            attention_mask=torch.ones_like(bad_ids),
            num_items_in_batch=3.0,
            extra_targets={"margin": torch.zeros(6, device=ctx.device)},
        )
    except ValueError as exc:
        raised = "pair" in str(exc)
    checks["odd_pair_split_raises"] = raised

    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
