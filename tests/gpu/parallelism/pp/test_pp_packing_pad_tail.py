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
from trl import SFTConfig

from src.data.collators.packing import PAD_TAIL_SEGMENT_CHUNK, DataCollatorWithPacking
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.pp_equivalence import build_dense_model
from tests.common.pp_trainers import build_pp_args, tiny_vocab_tokenizer
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
    # fp32 (the default): the two runs must match to float tolerance, not bf16 noise.
    args = build_pp_args(
        SFTConfig,
        f"{ctx.output_dir}/len{max_length}",
        max_steps=N_STEPS,
        # PP packing requires one packed row per microbatch: batch == microbatch count.
        per_device_train_batch_size=2,
        max_length=max_length,
        gradient_checkpointing=True,
    )
    # The FixedShapeCollator pads input_ids with the tokenizer's pad id, which must embed in the tiny vocab.
    tokenizer = tiny_vocab_tokenizer()
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
