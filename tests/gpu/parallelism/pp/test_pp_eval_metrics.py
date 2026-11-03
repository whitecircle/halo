#!/usr/bin/env python
"""``compute_metrics`` under PP scores what an unsplit model scores.

A loss-only PP eval returns ``(loss, None, None)`` and HF's loop skips ``compute_metrics`` without a word.
``preprocess_logits_for_metrics`` returns int64 argmax indices, which a fixed fp32 receive buffer garbles silently, so
the metadata hop sizes and types them.
"""

import sys

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, default_data_collator
from trl import SFTConfig

from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_CONFIG
from tests.common.pp_equivalence import build_dense_model
from tests.common.pp_trainers import build_pp_args, build_sft_rows
from tests.common.rank_log import log

PP_SIZE = 2
SEQ_LEN = 64
EVAL_ROWS = 8
BATCH = 4


def token_accuracy(predictions, labels) -> float:
    predictions = torch.as_tensor(predictions)
    labels = torch.as_tensor(labels)
    preds, targets = predictions[:, :-1], labels[:, 1:]
    keep = targets != -100
    return float((preds[keep] == targets[keep]).float().mean())


def next_token_accuracy(eval_pred) -> dict:
    predictions, labels = torch.as_tensor(eval_pred.predictions), torch.as_tensor(eval_pred.label_ids)
    print(f"[compute_metrics] predictions={tuple(predictions.shape)} labels={tuple(labels.shape)}", flush=True)
    return {"accuracy": token_accuracy(predictions, labels)}


def argmax_only(logits, labels):
    del labels
    return logits.argmax(dim=-1)


def reference_accuracy(eval_dataset) -> float:
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
    eval_dataset = build_sft_rows(TINY_QWEN3_CONFIG["vocab_size"], n_rows=EVAL_ROWS, seed=23)

    args = build_pp_args(
        SFTConfig,
        ctx.output_dir,
        max_steps=1,
        per_device_train_batch_size=BATCH,
        learning_rate=0.0,  # Frozen weights: the metric must match the reference at init.
        max_length=SEQ_LEN,
        eval_strategy="no",  # Called explicitly below, at the init weights.
        per_device_eval_batch_size=BATCH,
    )

    trainer = DistributedSFTTrainer(
        model=build_dense_model(),
        args=args,
        train_dataset=build_sft_rows(TINY_QWEN3_CONFIG["vocab_size"]),
        eval_dataset=eval_dataset,
        processing_class=AutoTokenizer.from_pretrained(QWEN3_0_6B),
        data_collator=default_data_collator,
        parallelism_config=config,
        compute_metrics=next_token_accuracy,
        preprocess_logits_for_metrics=argmax_only,
    )

    checks, metrics = {}, {}
    # The callables must survive onto the trainer rather than being swallowed by the distributed kwargs seam.
    checks["compute_metrics_accepted_under_pp"] = trainer.compute_metrics is not None
    # Hoisted onto the last stage: run in HF's loop it would only shrink the broadcast after the raw plane crossed the
    # chain, and in both places would argmax an argmax. The accuracy pinned below proves it ran exactly once.
    checks["preprocess_logits_for_metrics_hoisted_onto_the_last_stage"] = (
        trainer._pp_adapter.predictions_fn is not None and trainer.preprocess_logits_for_metrics is None
    )

    eval_metrics = trainer.evaluate()
    reference = reference_accuracy(eval_dataset)

    # Its absence is the silent skip, and an assertion on the value alone would raise KeyError over a failed check.
    checks["accuracy_reported"] = "eval_accuracy" in eval_metrics
    measured = eval_metrics.get("eval_accuracy")
    log(
        ctx,
        f"eval_accuracy={measured} reference={reference:.6f} eval_loss={eval_metrics.get('eval_loss'):.6f}",
    )
    checks["accuracy_matches_unsplit_reference"] = measured is not None and abs(measured - reference) < 1e-6

    # Anti-vacuity: 0.0 or 1.0 would match a broken reference just as well.
    checks["accuracy_is_nondegenerate"] = measured is not None and 0.0 < measured < 1.0

    # Every rank of the chain must report the same value.
    gathered = [None] * ctx.world_size
    dist.all_gather_object(gathered, measured)
    checks["accuracy_identical_across_ranks"] = all(
        value is not None and abs(value - gathered[0]) < 1e-9 for value in gathered
    )

    # Loss-only eval is still the default under prediction_loss_only and must agree with the metrics path's loss.
    loss_only = trainer.evaluate(metric_key_prefix="lo")
    checks["loss_matches_between_paths"] = abs(loss_only["lo_loss"] - eval_metrics["eval_loss"]) < 1e-6

    metrics["eval_accuracy"] = float(measured) if measured is not None else float("nan")
    metrics["reference_accuracy"] = reference
    metrics["eval_loss"] = float(eval_metrics["eval_loss"])
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
