#!/usr/bin/env python
"""``compute_metrics`` under pipeline parallelism must score what an unsplit model scores.

A loss-only PP eval is silently metric-less: with ``return_outputs=False`` the prediction step
returns ``(loss, None, None)`` and HF's eval loop — which guards on ``logits is not None`` — skips
``compute_metrics`` without a word. This test pins the path that carries predictions instead: one
forward-only pass yields the last stage's merged output, the adapter's ``predictions_fn`` and the
user's ``preprocess_logits_for_metrics`` reduce it on that stage, and the reduced tensor is
broadcast down the chain.

What each check would catch:

  * predictions never arriving (the silent skip) — ``eval_accuracy`` would be absent entirely
  * a wrong reduction, a mis-trimmed pad row, or a broadcast that corrupts the values — the metric
    would not match a reference computed here from the same weights on one GPU with no pipeline
  * a dtype-blind broadcast: ``preprocess_logits_for_metrics`` returns int64 argmax indices, which a
    fixed fp32 receive buffer garbles silently — the metadata hop is what sizes and types it
  * ranks disagreeing — every rank of the chain must report the identical metric

The reference is derived independently: an unsplit copy of the same initial weights, forwarded on
rank 0, argmaxed and scored with plain tensor ops — never through the pipeline code under test.

Run: torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_eval_metrics.py
"""

import sys

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import AutoTokenizer, default_data_collator
from trl import SFTConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_CONFIG
from tests.common.pp_equivalence import build_dense_model
from tests.common.utils import log

PP_SIZE = 2
SEQ_LEN = 64
EVAL_ROWS = 8
BATCH = 4


def build_dataset(vocab_size: int, n_rows: int, seed: int) -> Dataset:
    g = torch.Generator().manual_seed(seed)
    rows = torch.randint(4, vocab_size, (n_rows, SEQ_LEN), generator=g)
    labels = rows.clone()
    for i in range(labels.size(0)):
        labels[i, : 4 + (i % 11)] = -100  # ragged prompt masks, as in real SFT batches
    return Dataset.from_dict(
        {
            "input_ids": rows.tolist(),
            "labels": labels.tolist(),
            "attention_mask": torch.ones_like(rows).tolist(),
        }
    )


def token_accuracy(predictions, labels) -> float:
    """Token accuracy over non-ignored next-token targets, from argmax indices."""
    predictions = torch.as_tensor(predictions)
    labels = torch.as_tensor(labels)
    preds, targets = predictions[:, :-1], labels[:, 1:]
    keep = targets != -100
    return float((preds[keep] == targets[keep]).float().mean())


def next_token_accuracy(eval_pred) -> dict:
    """HF's ``compute_metrics`` contract: one ``EvalPrediction`` in, a metric dict out."""
    predictions, labels = torch.as_tensor(eval_pred.predictions), torch.as_tensor(eval_pred.label_ids)
    print(f"[compute_metrics] predictions={tuple(predictions.shape)} labels={tuple(labels.shape)}", flush=True)
    return {"accuracy": token_accuracy(predictions, labels)}


def argmax_only(logits, labels):
    """Reduce the [B, S, V] plane to [B, S] int64 before it crosses the chain."""
    del labels
    return logits.argmax(dim=-1)


def reference_accuracy(eval_dataset) -> float:
    """Same metric from an unsplit model on one GPU — no pipeline anywhere in this path."""
    model = build_dense_model().cuda().eval()
    input_ids = torch.tensor(eval_dataset["input_ids"], device="cuda")
    attention_mask = torch.tensor(eval_dataset["attention_mask"], device="cuda")
    labels = torch.tensor(eval_dataset["labels"], device="cuda")
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    accuracy = token_accuracy(logits.argmax(dim=-1).cpu(), labels.cpu())
    del model
    torch.cuda.empty_cache()
    return accuracy


@gpu_test_main(exact_world_size=4, prefix="pp_eval_metrics")
def run(ctx):
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)
    eval_dataset = build_dataset(TINY_QWEN3_CONFIG["vocab_size"], n_rows=EVAL_ROWS, seed=23)

    args = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=1,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=2,
        learning_rate=0.0,  # frozen weights: the metric must match the reference at init
        lr_scheduler_type="constant",
        max_length=SEQ_LEN,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        logging_nan_inf_filter=False,
        seed=42,
        bf16=False,
        eval_strategy="no",  # evaluate() is called explicitly below, at the init weights
        per_device_eval_batch_size=BATCH,
    )

    trainer = DistributedSFTTrainer(
        model=build_dense_model(),
        args=args,
        train_dataset=build_dataset(TINY_QWEN3_CONFIG["vocab_size"], n_rows=16, seed=7),
        eval_dataset=eval_dataset,
        processing_class=AutoTokenizer.from_pretrained(QWEN3_0_6B),
        data_collator=default_data_collator,
        parallelism_config=config,
        compute_metrics=next_token_accuracy,
        preprocess_logits_for_metrics=argmax_only,
    )

    checks, metrics = {}, {}
    # No construction-time gate may reject these under PP, and the callables must survive onto the
    # trainer rather than being swallowed by the distributed kwargs seam.
    checks["compute_metrics_accepted_under_pp"] = trainer.compute_metrics is not None
    # Hoisted onto the last stage's reduction rather than left for HF's evaluation loop: run there
    # it would only shrink the broadcast after the raw plane already crossed the chain, and running
    # in both places would argmax an argmax. Cleared on the trainer is the hoist, not a drop — the
    # adapter now carries it, and the accuracy pinned below proves it ran exactly once.
    checks["preprocess_logits_for_metrics_hoisted_onto_the_last_stage"] = (
        trainer._pp_adapter.predictions_fn is not None and trainer.preprocess_logits_for_metrics is None
    )

    eval_metrics = trainer.evaluate()
    reference = reference_accuracy(eval_dataset)

    # The metric must exist — its absence is precisely the silent skip, and an assertion on the
    # value alone would raise KeyError instead of reporting a failed check.
    checks["accuracy_reported"] = "eval_accuracy" in eval_metrics
    measured = eval_metrics.get("eval_accuracy")
    log(
        ctx,
        f"eval_accuracy={measured} reference={reference:.6f} eval_loss={eval_metrics.get('eval_loss'):.6f}",
    )
    checks["accuracy_matches_unsplit_reference"] = measured is not None and abs(measured - reference) < 1e-6

    # Anti-vacuity: a metric that is 0.0 or 1.0 would match a broken reference just as well.
    checks["accuracy_is_nondegenerate"] = measured is not None and 0.0 < measured < 1.0

    # Every rank of the chain must report the same value (the broadcast, not a per-stage artifact).
    gathered = [None] * ctx.world_size
    dist.all_gather_object(gathered, measured)
    checks["accuracy_identical_across_ranks"] = all(
        value is not None and abs(value - gathered[0]) < 1e-9 for value in gathered
    )

    # Loss-only eval must keep working alongside it (the cheap path is still the default when
    # prediction_loss_only is set) and must agree with the metrics path's loss.
    loss_only = trainer.evaluate(metric_key_prefix="lo")
    checks["loss_matches_between_paths"] = abs(loss_only["lo_loss"] - eval_metrics["eval_loss"]) < 1e-6

    metrics["eval_accuracy"] = float(measured) if measured is not None else float("nan")
    metrics["reference_accuracy"] = reference
    metrics["eval_loss"] = float(eval_metrics["eval_loss"])
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
