#!/usr/bin/env python
"""PP + packing through the real trainer: the padded tail must be inert and cheap.

Pipeline parallelism keeps the packed rows (one per microbatch) and pads every batch to
``max_length`` — the one supported configuration where pad tails reach the kernels instead of being
dropped by the flatten. Those tails carry chunked-ramp position_ids (``PAD_TAIL_SEGMENT_CHUNK``):
per-pad position-0 segments would each cost the FA4 backward a fixed per-segment overhead, while a
ramp is a handful of no-op segments.

The assertable contract is inertness: the same packed rows trained at two different ``max_length``
values differ only in tail size, so every per-step loss must match to float tolerance. A tail that
attends into real documents, carries labels, or shifts the loss normalizer fails this immediately —
and it holds regardless of how the tail's positions are encoded, so it pins the contract the ramp
must keep, not the ramp's implementation.

Run: torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_packing_pad_tail.py
"""

import sys

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTConfig

from src.data.collators.packing import PAD_TAIL_SEGMENT_CHUNK, DataCollatorWithPacking
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_CONFIG
from tests.common.pp_equivalence import build_dense_model
from tests.common.utils import log

PP_SIZE = 2
N_STEPS = 6
REAL_LEN = 96  # per-row real tokens; max_length above this is pure pad tail


def build_packed_dataset(vocab_size: int, n_rows: int = 16, seed: int = 7) -> Dataset:
    """Packed rows of REAL_LEN real tokens each: 3 documents per row, seq_lengths carried."""
    g = torch.Generator().manual_seed(seed)
    rows = torch.randint(4, vocab_size, (n_rows, REAL_LEN), generator=g).tolist()
    seq_lengths = [REAL_LEN // 3, REAL_LEN // 3, REAL_LEN - 2 * (REAL_LEN // 3)]
    return Dataset.from_dict(
        {
            "input_ids": rows,
            "labels": [list(r) for r in rows],
            "seq_lengths": [list(seq_lengths) for _ in rows],
        }
    )


def _losses_at(ctx, max_length: int) -> list[float]:
    # Same topology as the PP e2e test: 4 GPUs as 2 simulated nodes of 2, stage per node.
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)
    args = SFTConfig(
        output_dir=f"{ctx.output_dir}/len{max_length}",
        max_steps=N_STEPS,
        # PP packing requires one packed row per microbatch: batch == microbatch count.
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-3,
        lr_scheduler_type="constant",
        max_length=max_length,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        logging_nan_inf_filter=False,
        seed=42,
        bf16=False,  # fp32: the two runs must match to float tolerance, not bf16 noise
        gradient_checkpointing=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    # The tiny model's vocab cannot embed the real tokenizer's pad id (151643): FixedShapeCollator
    # pads input_ids with tokenizer.pad_token_id, and an out-of-vocab id is an embedding-gather
    # CUDA assert. Pin it to the model config's pad id instead.
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(0)
    trainer = DistributedSFTTrainer(
        model=build_dense_model(),
        args=args,
        train_dataset=build_packed_dataset(TINY_QWEN3_CONFIG["vocab_size"]),
        processing_class=tokenizer,
        # The production preprocessed-packed path: rows carry seq_lengths, the toolkit's collator is
        # passed explicitly (select_data_collator does this in the scripts), and the PP mixin keeps
        # the rows by flipping flatten_to_single_row before wrapping in FixedShapeCollator.
        data_collator=DataCollatorWithPacking(tokenizer=tokenizer),
        parallelism_config=config,
    )
    trainer.train()
    return [h["loss"] for h in trainer.state.log_history if "loss" in h]


@gpu_test_main(exact_world_size=4, prefix="pp_packing_pad_tail")
def run(ctx):
    # Tail sizes straddle the ramp chunk so both the sub-chunk and multi-chunk tail paths run.
    short_tail = REAL_LEN + PAD_TAIL_SEGMENT_CHUNK // 4  # 64-token tail: one partial chunk
    long_tail = REAL_LEN + 2 * PAD_TAIL_SEGMENT_CHUNK  # 512-token tail: several chunks

    losses_short = _losses_at(ctx, short_tail)
    losses_long = _losses_at(ctx, long_tail)

    checks = {}
    checks["both_runs_completed"] = len(losses_short) == N_STEPS and len(losses_long) == N_STEPS
    checks["losses_finite"] = all(map(torch.isfinite, map(torch.tensor, losses_short + losses_long)))

    # Same real tokens, different tail size: an attending/trained/normalizer-shifting tail diverges here.
    drift = (
        max(abs(a - b) for a, b in zip(losses_short, losses_long, strict=True))
        if checks["both_runs_completed"]
        else float("inf")
    )
    checks["pad_tail_is_inert"] = drift < 1e-4
    log(f"per-step |loss(len={short_tail}) - loss(len={long_tail})| max = {drift:.2e}", rank=ctx.rank)
    if not checks["pad_tail_is_inert"]:
        log(
            f"FAIL: growing only the pad tail moved the loss by {drift:.2e} — the tail attends into "
            f"real documents, is being trained, or shifts the loss normalizer",
            rank=ctx.rank,
        )
    return {"checks": checks, "metrics": {"max_loss_drift": drift}}


if __name__ == "__main__":
    sys.exit(run())
