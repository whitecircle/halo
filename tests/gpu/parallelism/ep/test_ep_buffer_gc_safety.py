#!/usr/bin/env python
"""A garbage-collected MoE layer must not free its DeepEP buffer.

Freeing a DeepEP buffer is a **collective**: ``buffer.destroy()`` spins on the device-side NVLink
barrier for ``HALO_DEEPEP_GPU_TIMEOUT_SECONDS`` (100 s). A Python finalizer, by contrast, runs whenever
this rank's garbage collector happens to fire, and no other rank agrees on that moment — so a
finalizer that frees puts one rank alone in a barrier its peers have not reached. After a gathered
save the writer rank can be far more than 100 s behind, and the job dies with ``DeepEP NVLink
barrier timeout`` followed by ``cudaErrorLaunchFailure``, which reads as a DeepEP fault rather than
as the teardown race it is.

So a dropped dispatcher only relinquishes its claim, and the buffer is freed by
:func:`free_unclaimed_ep_buffers` at an explicit, barriered teardown. This pins both halves:

  1. Dropping the last reference to a dispatching layer and forcing a full GC leaves the buffer
     ALIVE and the claim at zero — the finalizer ran and freed nothing.
  2. The explicit sweep then frees exactly that buffer, so nothing leaks for the rest of the process.

Half 1 alone would pass if the buffer were simply never freed; half 2 alone would pass if the
finalizer had freed it first. Restoring a collective free to ``__del__`` fails half 1; dropping the
sweep fails half 2.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_ep_buffer_gc_safety.py

Requirements:
    - 2x GPUs, single node, DeepEP installed. Builds no model: a bare dispatcher is enough.
"""

import gc

from src.distributed.expert_parallel import dispatcher as dispatcher_mod
from src.distributed.expert_parallel.dispatcher import DeepEPDispatcher, free_unclaimed_ep_buffers
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from tests.common.harness import gpu_test_main
from tests.common.utils import log

HIDDEN = 1024
NUM_TOPK = 4
TOKENS_PER_RANK = 128
EXPERTS_PER_RANK = 8
# Two layers on one EP group, so "the buffer survived" cannot be confused with "this layer kept its
# own": both share one arena, and dropping the first must leave the second's transport intact.
NUM_LAYERS = 2


def live_arenas() -> list:
    return list(dispatcher_mod._SharedArena._REGISTRY.values())


def build_dispatchers(ep_config, num_experts: int) -> list:
    """``NUM_LAYERS`` sized dispatchers on one EP group (sizing the transport is collective).

    A function rather than a loop in ``main`` so the loop variable dies with this frame: a stray
    reference to the last dispatcher would keep one claim alive and quietly weaken the GC check.
    """
    built = []
    for _ in range(NUM_LAYERS):
        dispatcher = DeepEPDispatcher(ep_config, num_experts, HIDDEN)
        dispatcher.create_buffer()
        dispatcher._ensure_buffer(TOKENS_PER_RANK, NUM_TOPK)
        built.append(dispatcher)
    return built


def run(ctx) -> dict:
    ep_config = ParallelismConfig(ep_size=ctx.world_size).create_ep_config()
    dispatchers = build_dispatchers(ep_config, EXPERTS_PER_RANK * ctx.world_size)

    arenas = live_arenas()
    assert len(arenas) == 1, f"expected the {NUM_LAYERS} layers to share ONE arena, found {len(arenas)}"
    arena = arenas[0]
    assert arena.buffer is not None, "arena reports no buffer after a dispatch was sized"
    assert arena._users == NUM_LAYERS, f"expected {NUM_LAYERS} claims, got {arena._users}"

    checks = {}

    # ---- 1. the finalizer must relinquish, never free ---------------------
    # Ranks are deliberately NOT aligned around this: that is the whole point — a collective here
    # would be entered by one rank at a GC-chosen moment.
    dispatchers.clear()
    gc.collect()

    checks["buffer_survives_gc"] = arena.buffer is not None
    checks["arena_still_registered"] = arena in live_arenas()
    checks["claims_dropped_by_gc"] = arena._users == 0
    log(
        f"after dropping {NUM_LAYERS} dispatchers + gc.collect(): buffer alive="
        f"{arena.buffer is not None}, claims={arena._users}, registered arenas={len(live_arenas())}"
    )

    # ---- 2. the explicit, barriered sweep must free it --------------------
    barrier()
    freed = free_unclaimed_ep_buffers()
    checks["sweep_freed_the_buffer"] = freed == 1
    checks["buffer_released"] = arena.buffer is None
    checks["registry_empty"] = live_arenas() == []
    log(f"free_unclaimed_ep_buffers() freed {freed}; registered arenas now {len(live_arenas())}")

    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="ep_buffer_gc_safety")(run)

if __name__ == "__main__":
    main()
