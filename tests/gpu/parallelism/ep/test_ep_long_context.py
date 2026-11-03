#!/usr/bin/env python
"""Elastic EP runs long context at ep8: no DeepEP "symmetric-window token ceiling" exists.

The apparent ceiling at ~46.6k tokens/rank is the GptOss GLU kernel's int32 offset overflowing on the skewed activation
(>2³¹ elements), which async-reports on peers as DeepEP's `symmetric.hpp 719`. Forward-only: it is a forward-dispatch
fault, and a 65536-token backward costs memory and DeepEP-barrier sensitivity.
"""

import math

import torch

from halo.distributed.parallelism_config import ParallelismConfig
from halo.hardware import supports_grouped_mm
from halo.model_loading.parallel.loader import load_distributed_model
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import GPT_OSS_20B
from tests.common.rank_log import log, log_all

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 8
SEED = 7
# Below the 2³⁰ ep8 arena bound (45056×8×2880 < 2³⁰), just above, and past the GLU int32-overflow point (~759k tokens ×
# 2880 > 2³¹ onto one rank).
SWEEP = (45056, 47104, 65536)


@gpu_test_main(exact_world_size=EP_SIZE, prefix="ep_long_context")
def run(ctx):
    ensure_model_downloaded(MODEL_NAME, ctx.rank)
    cfg = ParallelismConfig(ep_size=EP_SIZE, use_grouped_gemm=supports_grouped_mm(), ep_buffer_backend="elastic")
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
        input_ids = torch.randint(0, 1000, (1, seq), device=ctx.device, dtype=torch.long)
        with torch.no_grad():
            losses[seq] = float(model(input_ids=input_ids, labels=input_ids).loss.item())
        log_all(f"  [elastic ep{EP_SIZE}] seq={seq:>6}: loss={losses[seq]:.5f}")
        cleanup_memory()

    # Int32 illegal-accesses on the >2³¹-element grouped activation at seq=65536.
    checks = {f"finite_loss_seq{s}": math.isfinite(losses[s]) for s in SWEEP}

    if ctx.rank == 0:
        hidden = model.config.hidden_size
        arena_bound = 2**30 // (EP_SIZE * hidden)
        log(
            f"\n{'=' * 70}\nEP LONG-CONTEXT (ep{EP_SIZE}, hidden {hidden}; 2³⁰ arena bound at {arena_bound:,} tok/rank)"
            f"\n{'=' * 70}"
        )
        log(f"  {'seq/rank':>10} {'extent=seq×ep×hidden':>22} {'vs 2³⁰':>10} {'loss':>12}")
        for s in SWEEP:
            extent = s * EP_SIZE * hidden
            log(f"  {s:>10} {extent:>21,} {'≥' if extent >= 2**30 else '<':>10} {losses[s]:>12.5f}")
    return {"checks": checks}


if __name__ == "__main__":
    run()
