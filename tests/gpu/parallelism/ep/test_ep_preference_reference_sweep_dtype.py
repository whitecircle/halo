#!/usr/bin/env python
"""DPO ``precompute_ref_log_probs`` under EP2 on a tiny Qwen3-MoE, with and without ``fp32_non_ep_params``.

The sweep runs after the FSDP2 wrap, at the training forward's compute dtype: swept in the constructor, fp32 masters
reach DeepEP's combine as fp32 and raise. The swept columns must equal the wrapped model's own reference pass, and
with ``precompute_ref_log_probs`` off no sweep runs.
"""

import argparse
import sys

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.preference_precompute import TRAINERS, column
from tests.common.preference_precompute_e2e import build_text_rows
from tests.common.tiny_models import TINY_MOE_FAMILIES, shared_tiny_family_checkpoint

ARMS = {
    "fp32_non_ep": {"precompute": True, "fp32_non_ep_params": True},
    "bf16": {"precompute": True, "fp32_non_ep_params": False},
    "no_precompute": {"precompute": False, "fp32_non_ep_params": False},
}
N_ROWS = 8
BATCH_SIZE = 2
MAX_LENGTH = 96
SEED = 42
# Same weights, same rows, same dtype: only kernel nondeterminism separates the two passes. A sweep at another dtype
# moves a sequence sum by well over this.
SWEEP_ATOL = 1e-3


@gpu_test_main(exact_world_size=2, prefix="ep_reference_sweep_dtype")
def run(ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    arm = ARMS[parser.parse_args().arm]
    trainer_cls, config_cls = TRAINERS["dpo"]
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}

    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    checkpoint = shared_tiny_family_checkpoint(
        ctx, TINY_MOE_FAMILIES["qwen3_moe"], "ep_reference_sweep_dtype_tiny_model", tokenizer, SEED
    )
    pc = ParallelismConfig(
        ep_size=2, bf16_optimizer=not arm["fp32_non_ep_params"], fp32_non_ep_params=arm["fp32_non_ep_params"]
    )
    model, tokenizer = load_distributed_model(
        model_name_or_path=checkpoint,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
        use_liger_kernel=False,
    )
    args = config_cls(
        output_dir=ctx.output_dir,
        max_steps=1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        save_strategy="no",
        report_to=[],
        seed=SEED,
        bf16=True,
        gradient_checkpointing=False,
        precompute_ref_log_probs=arm["precompute"],
        dataloader_num_workers=0,
        max_length=MAX_LENGTH,
    )
    rows = build_text_rows("dpo", N_ROWS)
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=Dataset.from_dict(rows),
        eval_dataset={"alpha": Dataset.from_dict(build_text_rows("dpo", N_ROWS, N_ROWS))},
        processing_class=tokenizer,
        parallelism_config=pc,
    )
    ctx.on_teardown(trainer.cleanup_ep)

    reference_columns = trainer.REFERENCE_COLUMNS
    splits = {"train": trainer.train_dataset, "alpha": trainer.eval_dataset["alpha"]}
    if not arm["precompute"]:
        checks["no_reference_columns_attached"] = not any(
            name in dataset.column_names for dataset in splits.values() for name in reference_columns
        )
        return {"checks": checks, "metrics": metrics}

    for split, dataset in splits.items():
        checks[f"{split}_columns_attached"] = all(name in dataset.column_names for name in reference_columns)
        batch = trainer.data_collator(list(dataset.select(range(BATCH_SIZE))))
        batch = {key: value.to(ctx.device) for key, value in batch.items()}
        with torch.no_grad(), trainer.autocast_smart_context_manager():
            chosen, rejected = trainer.compute_ref_log_probs(trainer.model, batch)
        for name, wrapped in zip(reference_columns, (chosen, rejected), strict=True):
            swept = column(dataset.select(range(BATCH_SIZE)), name).to(ctx.device)
            delta = float((wrapped.float() - swept).abs().max())
            metrics[f"{split}_{name}_max_abs_delta"] = delta
            checks[f"{split}_{name}_equals_the_wrapped_reference_pass"] = delta < SWEEP_ATOL
            checks[f"{split}_{name}_nondegenerate"] = bool(torch.isfinite(swept).all() and (swept < -1.0).all())
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
