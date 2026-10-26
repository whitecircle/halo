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
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForSequenceClassification

from src.configs.classification_config import ClassificationConfig
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.reward.classification import ClassificationTrainer
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_CONFIG
from tests.common.utils import log

PP_SIZE = 2
SEQ_LEN = 48
N_STEPS = 10
BATCH = 4  # rows per rank; gradient_accumulation_steps=2 becomes 2 microbatches of 2 rows
MICROBATCHES = 2
NUM_LABELS = 4
PAD_ID = 0
CLASS_WEIGHTS = [0.5, 1.5, 1.0, 2.0]
LOSS_RTOL = 2e-4


def build_model(seed: int):
    torch.manual_seed(seed)
    config = Qwen3Config(**TINY_QWEN3_CONFIG, num_labels=NUM_LABELS, pad_token_id=PAD_ID, eos_token_id=1)
    return Qwen3ForSequenceClassification(config)


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


def pad_batch(ids: list[list[int]], device) -> dict:
    input_ids = torch.full((len(ids), SEQ_LEN), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(ids), SEQ_LEN), dtype=torch.long)
    for i, row in enumerate(ids):
        input_ids[i, : len(row)] = torch.tensor(row)
        attention_mask[i, : len(row)] = 1
    return {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}


def pooled_reference_logits(model, batch: dict) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()


def base_args(output_dir: str, **overrides) -> ClassificationConfig:
    return ClassificationConfig(
        output_dir=output_dir,
        max_steps=N_STEPS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
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
        **overrides,
    )


def make_spy(trainer, captured: dict, normalizers: list):
    real_step = trainer._pp_runtime.step

    def spying_step(input_ids, labels, **kwargs):
        normalizers.append(float(kwargs.get("num_items_in_batch")))
        if not captured:
            captured["input_ids"] = input_ids.clone()
            captured["attention_mask"] = kwargs["attention_mask"].clone()
            captured["labels"] = labels.clone()
            captured["extras"] = {k: v.clone() for k, v in (kwargs.get("extra_targets") or {}).items()}
        return real_step(input_ids, labels, **kwargs)

    trainer._pp_runtime.step = spying_step


def decode_plane(plane: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(positions, values) at the synthesized labels plane's single marker per row."""
    mask = plane != -100
    positions = mask.int().argmax(dim=-1)
    values = plane[torch.arange(plane.size(0), device=plane.device), positions]
    return positions, values


def check_common(trainer, losses: list, decrease: float) -> dict:
    checks = {
        "accumulation_became_microbatches": trainer._pp_n_microbatches == MICROBATCHES,
        "ran_all_steps": trainer.state.global_step == N_STEPS,
        "losses_finite": len(losses) == N_STEPS and bool(torch.isfinite(torch.tensor(losses)).all()),
        "loss_decreased": losses[-1] < losses[0] - decrease,
    }
    device = torch.cuda.current_device()
    local = torch.tensor(losses, device=device)
    gathered = [torch.zeros_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    checks["losses_identical_across_ranks"] = all(torch.allclose(local, peer, atol=1e-6) for peer in gathered)
    return checks


def scenario_weighted_ce(ctx, tokenizer, output_dir) -> tuple[dict, dict]:
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)
    eval_dataset = build_single_label_dataset(2 * BATCH, seed=23)
    trainer = ClassificationTrainer(
        model=build_model(1234),
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
    reference = build_model(1234).to(ctx.device).float().eval()
    weight = torch.tensor(CLASS_WEIGHTS, device=ctx.device)
    checks, metrics = {}, {}
    checks["no_extra_targets_single_label"] = trainer._pp_adapter.extra_target_keys == ()

    # ── Eval with init weights: eval_loss (mean over the two DP shards of the weighted-CE batch
    # mean) and sklearn metrics from the pooled logits the PP prediction step broadcasts.
    eval_metrics = trainer.evaluate()
    all_labels = torch.tensor(eval_dataset["labels"], device=ctx.device)
    pooled_full = pooled_reference_logits(reference, pad_batch(eval_dataset["input_ids"], ctx.device))
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

    captured, normalizers = {}, []
    make_spy(trainer, captured, normalizers)
    trainer.train()

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
    local_reference = F.cross_entropy(pooled, targets, weight=weight)
    dist.all_reduce(local_reference, op=dist.ReduceOp.AVG)
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    log(f"[rank {ctx.rank}] weighted-CE step1={losses[0]:.6f} reference={float(local_reference):.6f}")
    checks["step1_loss_matches_reference"] = (
        abs(losses[0] - float(local_reference)) / abs(float(local_reference)) < LOSS_RTOL
    )
    metrics["ce_step1"], metrics["ce_step1_reference"] = losses[0], float(local_reference)
    metrics["ce_first_loss"], metrics["ce_last_loss"] = losses[0], losses[-1]

    # eval_strategy="steps" ran two in-loop evals (steps 5 and 10) through the PP prediction step
    # with compute_metrics active on trained weights — they must have produced finite losses.
    eval_losses = [entry["eval_loss"] for entry in trainer.state.log_history if "eval_loss" in entry]
    checks["midtrain_eval_ran_finite"] = len(eval_losses) == 2 and bool(
        torch.isfinite(torch.tensor(eval_losses)).all()
    )

    checks.update(check_common(trainer, losses, decrease=0.05))
    del trainer
    torch.cuda.empty_cache()
    return {f"ce_{name}": ok for name, ok in checks.items()}, metrics


def scenario_multi_label_bce(ctx, tokenizer, output_dir) -> tuple[dict, dict]:
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)
    train_dataset = build_multi_label_dataset(24, seed=11)
    trainer = ClassificationTrainer(
        model=build_model(4321),
        args=base_args(output_dir, loss_type="cross_entropy", class_weights=CLASS_WEIGHTS),
        train_dataset=train_dataset,
        processing_class=tokenizer,
        is_binary=False,
        is_multi_label=True,
        parallelism_config=config,
    )
    reference = build_model(4321).to(ctx.device).float().eval()
    pos_weight = torch.tensor(CLASS_WEIGHTS, device=ctx.device)
    checks, metrics = {}, {}
    checks["class_targets_declared"] = trainer._pp_adapter.extra_target_keys == ("class_targets",)

    # ── Direct partial-batch prediction step (2 rows < frozen 4): exercises inert row padding,
    # trimming, and the pooled-logits broadcast against transformers' own pooled forward.
    small = train_dataset[:2]
    small_batch = pad_batch(small["input_ids"], ctx.device)
    small_batch["labels"] = torch.tensor(small["labels"], dtype=torch.float32, device=ctx.device)
    loss, logits, returned_labels = trainer.prediction_step(trainer.model, dict(small_batch), False)
    reference_pooled = pooled_reference_logits(reference, small_batch)
    reference_loss = F.binary_cross_entropy_with_logits(reference_pooled, small_batch["labels"], pos_weight=pos_weight)
    log(f"[rank {ctx.rank}] BCE partial-batch loss={float(loss):.6f} reference={float(reference_loss):.6f}")
    checks["partial_batch_pooled_logits_match"] = torch.allclose(logits, reference_pooled, atol=1e-4)
    checks["partial_batch_loss_matches"] = abs(float(loss) - float(reference_loss)) < 1e-4
    checks["partial_batch_labels_roundtrip"] = torch.equal(returned_labels, small_batch["labels"])

    captured, normalizers = {}, []
    make_spy(trainer, captured, normalizers)
    trainer.train()

    checks["normalizer_is_element_count"] = len(normalizers) == N_STEPS and all(
        abs(value - BATCH * NUM_LABELS) < 1e-6 for value in normalizers
    )
    targets = captured["extras"]["class_targets"]
    checks["class_targets_are_multi_hot"] = targets.shape == (BATCH, NUM_LABELS) and bool(
        ((targets == 0) | (targets == 1)).all()
    )

    pooled = pooled_reference_logits(
        reference, {"input_ids": captured["input_ids"], "attention_mask": captured["attention_mask"]}
    )
    local_reference = F.binary_cross_entropy_with_logits(pooled, targets, pos_weight=pos_weight)
    dist.all_reduce(local_reference, op=dist.ReduceOp.AVG)
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    log(f"[rank {ctx.rank}] BCE step1={losses[0]:.6f} reference={float(local_reference):.6f}")
    checks["step1_loss_matches_reference"] = (
        abs(losses[0] - float(local_reference)) / abs(float(local_reference)) < LOSS_RTOL
    )
    metrics["bce_step1"], metrics["bce_step1_reference"] = losses[0], float(local_reference)
    metrics["bce_first_loss"], metrics["bce_last_loss"] = losses[0], losses[-1]

    checks.update(check_common(trainer, losses, decrease=0.03))
    del trainer
    torch.cuda.empty_cache()
    return {f"bce_{name}": ok for name, ok in checks.items()}, metrics


@gpu_test_main(exact_world_size=4, prefix="pp_classification")
def run(ctx):
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    # The tiny model's vocab is 1024; Qwen's real pad id would overflow the embedding, so pad with
    # token 0 (data rows are drawn from [4, vocab) — pad never collides with a real token).
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(PAD_ID)

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
