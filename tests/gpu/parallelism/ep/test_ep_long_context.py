#!/usr/bin/env python
"""Elastic EP runs long context at ep8 — there is no DeepEP "symmetric-window token ceiling".

The apparent ceiling at `num_max_tokens_per_rank × ep_size × hidden ≈ 2³⁰` (~46.6k tokens/rank at
ep8) is not a DeepEP limit: it is the `fused_gptoss_glu` int32 program offset overflowing on the
skewed grouped activation (>2³¹ elements), which async-reports on peer ranks as DeepEP's
`symmetric.hpp 719`. The int64 offset (`src/kernels/fused_glu.py`) lets elastic stream arbitrary
sequence length.

Sweeps one no_grad forward per seq ∈ {45056 (below the 2³⁰ ep8 arena bound), 47104 (just above), 65536 (well
above — past the GLU int32-overflow point: a freshly-loaded skewed router piles ~759k tokens × 2880 >
2³¹ onto one rank)} on elastic ep8 (gpt-oss-20b, hidden 2880), and asserts a finite loss at every
length — no guard, no CUDA fault. With an int32 offset seq=65536 illegal-accesses in `fused_glu` (the
grouped activation overflows). The int64 offset's *backward* equivalence is covered by
`tests/gpu/kernels/test_fused_glu.py` (the >2³¹-numel case).

Forward-only: the fault is a forward-dispatch fault, so a no_grad forward proves the lift
without the memory (and DeepEP-barrier sensitivity) of a 65536-token backward.

Run with 8 GPUs (single node):
    torchrun --nproc_per_node=8 \
        tests/gpu/parallelism/ep/test_ep_long_context.py

Requirements: 8x GPUs (>=80GB), DeepEP; Model: unsloth/gpt-oss-20b-BF16 (auto-downloaded).
"""

import math
import sys
import traceback

import torch
from accelerate import PartialState

from src.distributed.expert_parallel.base_layer import has_grouped_mm
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from tests.common.distributed import (
    ensure_model_downloaded,
    init_distributed,
    teardown_distributed,
)
from tests.common.models import GPT_OSS_20B
from tests.common.utils import cleanup_memory, log, log_all

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 8
SEED = 7
# below the 2³⁰ ep8 arena bound (45056×8×2880 < 2³⁰), just above (47104), and well above — 65536 is past
# the GLU int32-overflow point (~759k tokens × 2880 > 2³¹ onto one rank), the decisive regression point.
SWEEP = (45056, 47104, 65536)


def main():
    rank, world_size, local_rank = init_distributed()
    PartialState()
    if world_size != EP_SIZE:
        log(f"  Needs exactly {EP_SIZE} GPUs (got {world_size}). SKIPPING.")
        teardown_distributed()
        return 0
    device = f"cuda:{local_rank}"
    try:
        ensure_model_downloaded(MODEL_NAME, rank)
        cfg = ParallelismConfig(ep_size=EP_SIZE, use_grouped_gemm=has_grouped_mm(), ep_buffer_backend="elastic")
        model = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=cfg,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            use_liger_kernel=True,
        )[0]
        model.eval()

        losses = {}
        for seq in SWEEP:
            torch.manual_seed(SEED)
            input_ids = torch.randint(0, 1000, (1, seq), device=device, dtype=torch.long)
            with torch.no_grad():
                losses[seq] = float(model(input_ids=input_ids, labels=input_ids).loss.item())
            log_all(f"  [elastic ep{EP_SIZE}] seq={seq:>6}: loss={losses[seq]:.5f}")
            cleanup_memory()
        barrier()

        def is_finite(v):
            return isinstance(v, float) and math.isfinite(v)

        # elastic has no symmetric-window ceiling: finite loss at every swept length (the int64-offset
        # regression guard — int32 illegal-accesses on the >2³¹-element grouped activation at seq=65536).
        checks = {f"finite_loss_seq{s}": is_finite(losses[s]) for s in SWEEP}

        if rank == 0:
            log(
                f"\n{'=' * 70}\nEP LONG-CONTEXT (ep{EP_SIZE}, gpt-oss hidden 2880; old ceiling ≈ 46.6k tok/rank)\n{'=' * 70}"
            )
            log(f"  {'seq/rank':>10} {'extent=seq×ep×hidden':>22} {'vs 2³⁰':>10} {'loss':>12}")
            for s in SWEEP:
                extent = s * EP_SIZE * 2880
                log(f"  {s:>10} {extent:>21,} {'≥' if extent >= 2**30 else '<':>10} {losses[s]:>12.5f}")
            for k, v in checks.items():
                log(f"  [{'PASS' if v else 'FAIL'}] {k}")
            all_passed = all(checks.values())
            log(f"\n{'#' * 70}\n  EP LONG-CONTEXT TEST {'PASSED' if all_passed else 'FAILED'}")
            if not all_passed:
                log(f"  Failed: {[k for k, v in checks.items() if not v]}")
            log(f"{'#' * 70}\n")
        else:
            all_passed = all(checks.values())

        del model
        cleanup_memory()
        barrier()
        teardown_distributed()
        return 0 if all_passed else 1
    except Exception:
        log_all(f"EXCEPTION on rank {rank}:\n{traceback.format_exc()}")
        teardown_distributed()
        return 1


if __name__ == "__main__":
    sys.exit(main())
