#!/usr/bin/env python
"""Pipeline parallelism through the real ClassificationTrainer (pooled sequence classification).

Two scenarios cover the classification PP contract end to end:

  * weighted cross-entropy (single-label): the class id rides inside the synthesized labels plane,
    and the normalizer must be the sum of the target classes' weights — a plain count silently
    rescales the objective. Step-1 loss is checked against the exact non-PP objective computed on
    the captured batch by transformers' own pooled forward; evaluation (which under PP pools
    [B, num_labels] logits via forward_only and broadcasts them down the chain) is checked for
    both eval_loss and sklearn metrics against a full recomputation with the init weights.
  * weighted BCE (multi-label): multi-hot float targets cannot ride the long labels plane, so they
    ship as the ``class_targets`` extra; the normalizer is the element count B*C. Also drives a
    direct partial-batch prediction_step (rows < frozen shape) to exercise the inert row padding.

Topology: 4 GPUs as 2 simulated nodes of 2 (NVLink domain = 2), pp2 x dp2.

Run: torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_classification.py
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import accuracy_score

from src.configs.classification import ClassificationConfig
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.reward.classification import ClassificationTrainer
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
from tests.common.utils import log

PP_SIZE = 2
SEQ_LEN = 48
N_STEPS = 10
BATCH = 4  # rows per rank; gradient_accumulation_steps=2 becomes 2 microbatches of 2 rows
MICROBATCHES = 2
NUM_LABELS = 4
CLASS_WEIGHTS = [0.5, 1.5, 1.0, 2.0]
LOSS_RTOL = 2e-4


def random_rows(n_rows: int, vocab_size: int, g: torch.Generator) -> tuple[list, list]:
    """Ragged pre-tokenized rows (variable lengths exercise the rightmost-non-pad pooling)."""
    ids, masks = [], []
    for _ in range(n_rows):
        length = int(torch.randint(6, SEQ_LEN, (1,), generator=g))
        ids.append(torch.randint(4, vocab_size, (length,), generator=g).tolist())
        masks.append([1] * length)
    return ids, masks


def build_single_label_dataset(n_rows: int, seed: int) -> Dataset:
    g = torch.Generator().manual_seed(seed)
    ids, masks = random_rows(n_rows, TINY_QWEN3_CONFIG["vocab_size"], g)
    labels = torch.randint(0, NUM_LABELS, (n_rows,), generator=g).tolist()
    return Dataset.from_dict({"input_ids": ids, "attention_mask": masks, "labels": labels})


def build_multi_label_dataset(n_rows: int, seed: int) -> Dataset:
    g = torch.Generator().manual_seed(seed)
    ids, masks = random_rows(n_rows, TINY_QWEN3_CONFIG["vocab_size"], g)
    labels = (torch.rand(n_rows, NUM_LABELS, generator=g) > 0.5).float().tolist()
    return Dataset.from_dict({"input_ids": ids, "attention_mask": masks, "labels": labels})


def pooled_reference_logits(model, batch: dict) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()


def base_args(output_dir: str, **overrides) -> ClassificationConfig:
    return build_pp_args(
        ClassificationConfig,
        output_dir,
        max_steps=N_STEPS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        gradient_accumulation_steps=MICROBATCHES,  # becomes the pipeline's microbatch count
        learning_rate=8e-3,
        max_length=SEQ_LEN,
        **overrides,
    )


def decode_plane(plane: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(positions, values) at the synthesized labels plane's single marker per row."""
    mask = plane != -100
    positions = mask.int().argmax(dim=-1)
    values = plane[torch.arange(plane.size(0), device=plane.device), positions]
    return positions, values


def check_training(trainer, step1_reference: float, decrease: float, prefix: str) -> tuple[dict, dict]:
    """COLLECTIVE (the rank-identical loss gather). The finished run's shape and loss curve, and its step-1
    loss against the world-averaged exact objective ``step1_reference``; metrics named under ``prefix``."""
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    log(f"[{prefix}] step1={losses[0]:.6f} reference={step1_reference:.6f}")
    checks = {
        "step1_loss_matches_reference": abs(losses[0] - step1_reference) / abs(step1_reference) < LOSS_RTOL,
        "accumulation_became_microbatches": trainer._pp_n_microbatches == MICROBATCHES,
        "ran_all_steps": trainer.state.global_step == N_STEPS,
        "losses_finite": len(losses) == N_STEPS and bool(torch.isfinite(torch.tensor(losses)).all()),
        "loss_decreased": losses[-1] < losses[0] - decrease,
        "losses_identical_across_ranks": losses_identical_across_ranks(losses),
    }
    metrics = {
        f"{prefix}_step1": losses[0],
        f"{prefix}_step1_reference": step1_reference,
        f"{prefix}_first_loss": losses[0],
        f"{prefix}_last_loss": losses[-1],
    }
    return checks, metrics


def scenario_weighted_ce(ctx, tokenizer, output_dir) -> tuple[dict, dict]:
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)
    eval_dataset = build_single_label_dataset(2 * BATCH, seed=23)
    trainer, steps, _ = build_pp_trainer(
        ClassificationTrainer,
        model=build_seq_cls_model(NUM_LABELS),
        # eval_strategy exercises the in-loop PP eval too (the construction-time PP eval gates run
        # only when it is set); the exactness checks below use a manual evaluate() at init weights.
        args=base_args(
            output_dir, loss_type="cross_entropy", class_weights=CLASS_WEIGHTS, eval_strategy="steps", eval_steps=5
        ),
        train_dataset=build_single_label_dataset(24, seed=7),
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        is_binary=False,
        is_multi_label=False,
        parallelism_config=config,
    )
    reference = build_seq_cls_model(NUM_LABELS).to(ctx.device).float().eval()
    weight = torch.tensor(CLASS_WEIGHTS, device=ctx.device)
    checks, metrics = {}, {}
    checks["no_extra_targets_single_label"] = trainer._pp_adapter.extra_target_keys == ()

    # ── Eval with init weights: eval_loss (mean over the two DP shards of the weighted-CE batch
    # mean) and sklearn metrics from the pooled logits the PP prediction step broadcasts.
    eval_metrics = trainer.evaluate()
    all_labels = torch.tensor(eval_dataset["labels"], device=ctx.device)
    pooled_full = pooled_reference_logits(reference, right_pad(eval_dataset["input_ids"], SEQ_LEN, ctx.device))
    shard_losses = [
        float(F.cross_entropy(pooled_full[lo : lo + BATCH], all_labels[lo : lo + BATCH], weight=weight))
        for lo in (0, BATCH)
    ]
    reference_eval = sum(shard_losses) / len(shard_losses)
    reference_accuracy = accuracy_score(np.array(eval_dataset["labels"]), pooled_full.argmax(dim=-1).cpu().numpy())
    log(
        f"[rank {ctx.rank}] weighted-CE eval_loss={eval_metrics['eval_loss']:.6f} ref={reference_eval:.6f} "
        f"eval_accuracy={eval_metrics.get('eval_accuracy')} ref={reference_accuracy}"
    )
    checks["eval_loss_matches_reference"] = (
        abs(eval_metrics["eval_loss"] - reference_eval) / abs(reference_eval) < LOSS_RTOL
    )
    checks["eval_accuracy_matches_sklearn_recompute"] = abs(eval_metrics["eval_accuracy"] - reference_accuracy) < 1e-6
    metrics["ce_eval_loss"], metrics["ce_reference_eval"] = eval_metrics["eval_loss"], reference_eval

    trainer.train()
    captured = steps[0]
    normalizers = [step["num_items"] for step in steps]

    positions, targets = decode_plane(captured["labels"])
    lengths = captured["attention_mask"].sum(dim=-1)
    checks["plane_marks_last_non_pad_position"] = torch.equal(positions, lengths - 1)
    expected_normalizer = float(weight[targets].sum())
    checks["normalizer_is_target_weight_sum"] = (
        len(normalizers) == N_STEPS
        and all(abs(value - expected_normalizer) < 1e-4 for value in normalizers[:1])
        and all(value > 0 for value in normalizers)
    )

    pooled = pooled_reference_logits(
        reference, {"input_ids": captured["input_ids"], "attention_mask": captured["attention_mask"]}
    )
    expected = world_mean(float(F.cross_entropy(pooled, targets, weight=weight)))

    # eval_strategy="steps" ran two in-loop evals (steps 5 and 10) through the PP prediction step
    # with compute_metrics active on trained weights — they must have produced finite losses.
    eval_losses = [entry["eval_loss"] for entry in trainer.state.log_history if "eval_loss" in entry]
    checks["midtrain_eval_ran_finite"] = len(eval_losses) == 2 and bool(
        torch.isfinite(torch.tensor(eval_losses)).all()
    )

    c, m = check_training(trainer, expected, decrease=0.05, prefix="ce")
    checks.update(c)
    metrics.update(m)
    del trainer
    torch.cuda.empty_cache()
    return {f"ce_{name}": ok for name, ok in checks.items()}, metrics


def scenario_multi_label_bce(ctx, tokenizer, output_dir) -> tuple[dict, dict]:
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)
    train_dataset = build_multi_label_dataset(24, seed=11)
    trainer, steps, _ = build_pp_trainer(
        ClassificationTrainer,
        model=build_seq_cls_model(NUM_LABELS, seed=4321),
        args=base_args(output_dir, loss_type="cross_entropy", class_weights=CLASS_WEIGHTS),
        train_dataset=train_dataset,
        processing_class=tokenizer,
        is_binary=False,
        is_multi_label=True,
        parallelism_config=config,
    )
    reference = build_seq_cls_model(NUM_LABELS, seed=4321).to(ctx.device).float().eval()
    pos_weight = torch.tensor(CLASS_WEIGHTS, device=ctx.device)
    checks, metrics = {}, {}
    checks["class_targets_declared"] = trainer._pp_adapter.extra_target_keys == ("class_targets",)

    # ── Direct partial-batch prediction step (2 rows < frozen 4): exercises inert row padding,
    # trimming, and the pooled-logits broadcast against transformers' own pooled forward.
    small = train_dataset[:2]
    small_batch = right_pad(small["input_ids"], SEQ_LEN, ctx.device)
    small_batch["labels"] = torch.tensor(small["labels"], dtype=torch.float32, device=ctx.device)
    loss, logits, returned_labels = trainer.prediction_step(trainer.model, dict(small_batch), False)
    reference_pooled = pooled_reference_logits(reference, small_batch)
    reference_loss = F.binary_cross_entropy_with_logits(reference_pooled, small_batch["labels"], pos_weight=pos_weight)
    log(f"[rank {ctx.rank}] BCE partial-batch loss={float(loss):.6f} reference={float(reference_loss):.6f}")
    checks["partial_batch_pooled_logits_match"] = torch.allclose(logits, reference_pooled, atol=1e-4)
    checks["partial_batch_loss_matches"] = abs(float(loss) - float(reference_loss)) < 1e-4
    checks["partial_batch_labels_roundtrip"] = torch.equal(returned_labels, small_batch["labels"])

    trainer.train()
    captured = steps[0]
    normalizers = [step["num_items"] for step in steps]

    checks["normalizer_is_element_count"] = len(normalizers) == N_STEPS and all(
        abs(value - BATCH * NUM_LABELS) < 1e-6 for value in normalizers
    )
    targets = captured["class_targets"]
    checks["class_targets_are_multi_hot"] = targets.shape == (BATCH, NUM_LABELS) and bool(
        ((targets == 0) | (targets == 1)).all()
    )

    pooled = pooled_reference_logits(
        reference, {"input_ids": captured["input_ids"], "attention_mask": captured["attention_mask"]}
    )
    expected = world_mean(float(F.binary_cross_entropy_with_logits(pooled, targets, pos_weight=pos_weight)))
    c, m = check_training(trainer, expected, decrease=0.03, prefix="bce")
    checks.update(c)
    metrics.update(m)
    del trainer
    torch.cuda.empty_cache()
    return {f"bce_{name}": ok for name, ok in checks.items()}, metrics


@gpu_test_main(exact_world_size=4, prefix="pp_classification")
def run(ctx):
    tokenizer = tiny_vocab_tokenizer()

    checks, metrics = {}, {}
    ce_checks, ce_metrics = scenario_weighted_ce(ctx, tokenizer, ctx.output_dir)
    checks.update(ce_checks)
    metrics.update(ce_metrics)
    ctx.barrier()
    bce_checks, bce_metrics = scenario_multi_label_bce(ctx, tokenizer, ctx.output_dir)
    checks.update(bce_checks)
    metrics.update(bce_metrics)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
