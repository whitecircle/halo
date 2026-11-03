#!/usr/bin/env python
"""PP through the real DistributedRewardTrainer: interleaved pairs, the pair-count normalizer, the pair-alignment
guard.

A microbatch split of TRL's ``[chosen ⧺ rejected]`` compares unrelated examples, which shows as a loss mismatch, not a
crash. Eval pins accuracy and row count: a mean is blind to duplicated rows, a count to which pairs.
"""

import sys

import torch
import torch.nn.functional as F
from datasets import Dataset
from trl import RewardConfig

from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.reward.bradley_terry import DistributedRewardTrainer
from tests.common.distributed import world_mean
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.pp_trainers import (
    build_pp_args,
    build_pp_trainer,
    build_seq_cls_model,
    losses_identical_across_ranks,
    right_pad,
    tiny_vocab_tokenizer,
)
from tests.common.rank_log import log

PP_SIZE = 2
SEQ_LEN = 48
N_STEPS = 10
PAIRS_PER_RANK = 4  # per_device_train_batch_size counts pairs; the collator doubles the rows.
MICROBATCHES = 2
CENTER_COEF = 0.05
LOSS_RTOL = 2e-4


def build_dataset(vocab_size: int, n_rows: int, seed: int) -> Dataset:
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


def build_args(output_dir: str, **overrides) -> RewardConfig:
    return build_pp_args(
        RewardConfig,
        output_dir,
        **{
            "max_steps": N_STEPS,
            "per_device_train_batch_size": PAIRS_PER_RANK,
            "gradient_accumulation_steps": MICROBATCHES,  # Becomes the pipeline's microbatch count.
            "learning_rate": 8e-3,
            "max_length": SEQ_LEN,
            "gradient_checkpointing": False,
            "center_rewards_coefficient": CENTER_COEF,
            "per_device_eval_batch_size": PAIRS_PER_RANK,
            **overrides,
        },
    )


def reference_bt_loss(model, input_ids, attention_mask, margin_interleaved) -> float:
    with torch.no_grad():
        pooled = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1).float()
    chosen, rejected = pooled[0::2], pooled[1::2]
    loss = -F.logsigmoid(chosen - rejected - margin_interleaved.float()[0::2]).mean()
    loss = loss + CENTER_COEF * ((chosen + rejected) ** 2).mean()
    return float(loss)


def interleave_pairs(rows) -> list[list[int]]:
    return [ids for pair in zip(rows["chosen_ids"], rows["rejected_ids"], strict=False) for ids in pair]


def reference_pair_scores(model, rows, device) -> torch.Tensor:
    """``[pairs, 2]`` pooled scores, what the BT adapter's ``predictions_fn`` must produce."""
    with torch.no_grad():
        pooled = model(**right_pad(interleave_pairs(rows), SEQ_LEN, device))
    scores = pooled.logits.squeeze(-1).float()
    return torch.stack((scores[0::2], scores[1::2]), dim=-1)


@gpu_test_main(exact_world_size=4, prefix="pp_reward_bt")
def run(ctx):
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)
    tokenizer = tiny_vocab_tokenizer()
    args = build_args(ctx.output_dir)

    eval_dataset = build_dataset(TINY_QWEN3_CONFIG["vocab_size"], n_rows=2 * PAIRS_PER_RANK, seed=23)

    # Without the adapter's predictions_fn the raw [rows, S, 1] plane arrives here, so the recorded shape catches that.
    seen_shapes: list[tuple[int, ...]] = []

    def pairwise_accuracy(eval_pred) -> dict:
        predictions = torch.as_tensor(eval_pred.predictions)
        seen_shapes.append(tuple(predictions.shape))
        return {"accuracy": float((predictions.argmax(dim=-1) == 0).float().mean())}

    trainer, steps, _ = build_pp_trainer(
        DistributedRewardTrainer,
        model=build_seq_cls_model(num_labels=1),
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

    reference = build_seq_cls_model(num_labels=1).to(ctx.device).float().eval()

    # Init-weights eval: the logged eval_loss equals the mean over the two DP shards of TRL's per-batch BT loss.
    eval_metrics = trainer.evaluate()
    shard_losses = []
    for lo in (0, PAIRS_PER_RANK):
        shard = eval_dataset[lo : lo + PAIRS_PER_RANK]
        padded = right_pad(interleave_pairs(shard), SEQ_LEN, ctx.device)
        margin = torch.tensor(shard["margin"], device=ctx.device).repeat_interleave(2)
        shard_losses.append(reference_bt_loss(reference, padded["input_ids"], padded["attention_mask"], margin))
    reference_eval = sum(shard_losses) / len(shard_losses)
    log(f"[rank {ctx.rank}] eval_loss={eval_metrics['eval_loss']:.6f} reference={reference_eval:.6f}")
    checks["eval_loss_matches_reference"] = (
        abs(eval_metrics["eval_loss"] - reference_eval) / abs(reference_eval) < LOSS_RTOL
    )
    metrics["eval_loss"], metrics["reference_eval"] = eval_metrics["eval_loss"], reference_eval

    checks["metrics_receive_pairwise_scores"] = bool(seen_shapes) and all(
        len(shape) == 2 and shape[1] == 2 for shape in seen_shapes
    )
    checks["pairwise_accuracy_reported"] = "eval_accuracy" in eval_metrics
    # The value: ``0 <= accuracy <= 1`` passes on a predictions_fn that scores the wrong pairs, half of them, or
    # padding.
    accuracy = eval_metrics.get("eval_accuracy")
    # Over every pair once: a mean survives per-stage duplication, but not a dropped pair or an inert pad pair (a tie:
    # argmax 0, counted correct).
    reference_scores = reference_pair_scores(reference, eval_dataset[:], ctx.device)
    reference_accuracy = float((reference_scores[:, 0] >= reference_scores[:, 1]).float().mean())
    checks["pairwise_accuracy_matches_reference"] = accuracy is not None and abs(accuracy - reference_accuracy) < 1e-6
    # Every rank of a chain returns its predictions, so a world gather carries the dataset pp_size times; the metric
    # gather is scoped to the DP replicas. The count makes a dropped or duplicated batch visible.
    checks["metrics_see_every_pair_exactly_once"] = seen_shapes == [(len(eval_dataset), 2)]
    log(
        f"[rank {ctx.rank}] compute_metrics saw shapes={seen_shapes} eval_accuracy={accuracy} "
        f"reference={reference_accuracy}"
    )
    metrics["eval_accuracy"] = float(accuracy) if accuracy is not None else float("nan")

    # The exact-multiple set above arrives at the frozen row count, so neither inert padding nor the pair-unit trim
    # runs. One pair against the frozen 8 makes both: 3 inert pairs must be trimmed back off. COLLECTIVE: every rank
    # calls it.
    one_pair = eval_dataset[:1]
    # TRL's concat, which the batch transform interleaves.
    partial_batch = right_pad(one_pair["chosen_ids"] + one_pair["rejected_ids"], SEQ_LEN, ctx.device)
    partial_batch["margin"] = torch.tensor(one_pair["margin"], device=ctx.device)
    _, partial_predictions, partial_labels = trainer.prediction_step(trainer.model, partial_batch, False)
    partial_reference = reference_pair_scores(reference, one_pair, ctx.device)
    log(f"[rank {ctx.rank}] partial-batch predictions={tuple(partial_predictions.shape)} vs 1 real pair")
    checks["partial_eval_batch_trims_to_the_real_pair"] = tuple(partial_predictions.shape) == (1, 2)
    checks["partial_eval_batch_scores_match_reference"] = torch.allclose(
        partial_predictions.float().cpu(), partial_reference.cpu(), atol=1e-4
    )
    # Predictions and labels trim together or compute_metrics gets mismatched lengths.
    checks["partial_eval_batch_labels_trim_with_predictions"] = partial_labels.shape[0] == 1

    trainer.train()
    captured = steps[0]
    normalizers = [step["num_items"] for step in steps]

    checks["normalizer_is_chain_pair_count"] = len(normalizers) == N_STEPS and all(
        abs(value - PAIRS_PER_RANK) < 1e-6 for value in normalizers
    )
    # One marker per row, at its last non-pad token (right-padded with id 0, data tokens from 4). Marking only the
    # chosen rows would leave the rejected halves unpooled and unvalidated.
    marked = captured["labels"] != -100
    checks["plane_marks_every_row_at_pooled_position"] = bool(
        (marked.sum(dim=-1) == 1).all()
        and torch.equal(marked.int().argmax(dim=-1), captured["attention_mask"].sum(dim=-1) - 1)
    )
    checks["no_input_ids_extra_target"] = "input_ids" not in trainer._pp_adapter.extra_target_keys

    # The logged loss is the world mean; each chain contributes pp_size times with identical values.
    expected = world_mean(
        reference_bt_loss(reference, captured["input_ids"], captured["attention_mask"], captured["margin"])
    )
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    log(f"[rank {ctx.rank}] step1={losses[0]:.6f} reference={expected:.6f}")
    checks["step1_loss_matches_reference"] = abs(losses[0] - expected) / abs(expected) < LOSS_RTOL
    metrics["step1_loss"], metrics["step1_reference"] = losses[0], expected

    checks["ran_all_steps"] = trainer.state.global_step == N_STEPS
    checks["losses_finite"] = len(losses) == N_STEPS and bool(torch.isfinite(torch.tensor(losses)).all())
    checks["loss_decreased"] = losses[-1] < losses[0] - 0.05
    metrics["first_loss"], metrics["last_loss"] = losses[0], losses[-1]

    checks["losses_identical_across_ranks"] = losses_identical_across_ranks(losses)

    # 6 rows (3 pairs) across 2 microbatches would split pair 1. The runner validates before any collective, so every
    # rank raises and catches locally with no desync.
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
