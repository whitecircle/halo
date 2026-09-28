#!/usr/bin/env python
"""
EP+TP replicated-gradient sync correctness.

Under TP, truly-replicated params (LayerNorms, router; shared expert on families that have one)
must be kept identical across the TP axis or the replicas drift apart over training → a
rank-inconsistent model and per-rank-divergent grad clipping. TP-only does this inside
``tp_clip_grad_norm_``; EP+TP installs ``ep_clip_grad_norm_`` instead, so that path owns the sync:
``ep_clip_grad_norm_`` (and the max_grad_norm=0 fallback hook) TP-axis-average the replicated grads
— EXCLUDING the EP-distributed experts (different experts per TP rank; averaging them would corrupt
them) and harmlessly including the already-world-synced router. A path that skips the average lets
the replicas drift.

This test runs EP+TP with DP=1 (ep=2, tp=2 on 2 GPUs), where the replicated params are PLAIN
tensors (no FSDP), so the gap is maximally visible, and asserts after training:
  1. Every replicated non-expert, non-DTensor param is BIT-IDENTICAL across the TP group.
  2. The EP-distributed expert weights still DIFFER across the TP group (NOT corrupted by an
     over-broad all-reduce).
  3. Training is healthy (loss finite).

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/combined/test_ep_tp_replicated_grad_sync.py
"""

import math

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.expert_parallel.base_layer import has_grouped_mm
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded, group_max_abs_diff
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.utils import log

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 2
TP_SIZE = 2
NUM_TRAIN_STEPS = 8
SEED = 42


# DP=1 EP+TP keeps the replicated params as PLAIN tensors (no FSDP), which is where the missing
# TP-axis sync is maximally visible. EP and TP are coextensive on the same ranks, so DP=1 means
# world_size == tp_size (== ep_group_size) — exactly 2 GPUs here (ep=tp=2).
@gpu_test_main(exact_world_size=TP_SIZE, prefix="ep_tp_replicated_grad_sync")
def run(ctx):
    log(f"\n{'#' * 70}\n  EP+TP replicated-grad sync (ep={EP_SIZE}, tp={TP_SIZE}, DP=1)\n{'#' * 70}")

    ensure_model_downloaded(MODEL_NAME, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = create_sft_dataset(32, tokenizer, seed=SEED)

    parallelism_config = ParallelismConfig(ep_size=EP_SIZE, tp_size=TP_SIZE, use_grouped_gemm=has_grouped_mm())
    log(f"Config: {parallelism_config.summary()}")
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flex_attention",
        use_liger_kernel=True,
    )

    sft_config = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,  # a bit higher so any drift would be visible
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=2048,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        max_grad_norm=1.0,  # exercises the ep_clip_grad_norm_ sync path
        ddp_find_unused_parameters=True,
        fsdp="",
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    ctx.on_teardown(trainer.cleanup_ep)
    assert trainer.parallelism_config.is_ep_tp_mode, "expected EP+TP mode"

    train_result = trainer.train()
    log(f"Final loss: {train_result.training_loss:.6f}")

    tp_group = trainer._get_tp_process_group()
    assert tp_group is not None and dist.get_world_size(tp_group) == TP_SIZE, "TP group not reachable"
    expert_ids = trainer._get_sharded_expert_param_ids()

    # 1) Replicated non-expert plain params must be bit-identical across the TP axis. A NaN diff
    # counts as drifted (NaN != 0.0).
    replicated_diffs = [
        group_max_abs_diff(p.data, tp_group)
        for p in model.parameters()
        if not isinstance(p.data, DTensor) and id(p) not in expert_ids
    ]
    drifted = sum(1 for d in replicated_diffs if d != 0.0)
    replicated_ok = bool(replicated_diffs) and drifted == 0
    log(
        f"  Replicated params identical across TP: {'PASS' if replicated_ok else 'FAIL'} "
        f"({len(replicated_diffs)} params, {drifted} not bit-identical)"
    )

    # 2) EP-distributed experts must STILL differ across TP (not averaged/corrupted).
    expert_diffs = [
        group_max_abs_diff(p.data, tp_group)
        for p in model.parameters()
        if id(p) in expert_ids and not isinstance(p.data, DTensor)
    ]
    experts_distinct = any(d > 0.0 for d in expert_diffs) and all(math.isfinite(d) for d in expert_diffs)
    log(
        f"  EP experts still distinct across TP: {'PASS' if experts_distinct else 'FAIL'} "
        f"({sum(1 for d in expert_diffs if d > 0.0)} of {len(expert_diffs)} differ)"
    )

    checks = {
        "loss_finite": math.isfinite(train_result.training_loss),
        "replicated_synced": replicated_ok,
        "experts_not_corrupted": experts_distinct,
    }
    return {"checks": checks}


if __name__ == "__main__":
    run()
