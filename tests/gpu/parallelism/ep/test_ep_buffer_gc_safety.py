#!/usr/bin/env python
"""A garbage-collected MoE layer must not free its DeepEP buffer, since ``buffer.destroy()`` is collective.

A finalizer that freed it would fire on one rank's GC and spin on the NVLink barrier for 100 s. After a gathered save
the writer can lag past that, and the job dies with ``DeepEP NVLink barrier timeout``, which reads as a DeepEP fault.
:func:`free_unclaimed_ep_buffers` frees at a barriered teardown instead.
"""

import gc

from halo.distributed.expert_parallel.deepep import backends as backends_mod
from halo.distributed.expert_parallel.deepep.backends import free_unclaimed_ep_buffers
from halo.distributed.expert_parallel.deepep.dispatcher import DeepEPDispatcher
from halo.distributed.parallelism_config import ParallelismConfig
from halo.runtime.collectives import barrier
from tests.common.harness import gpu_test_main
from tests.common.rank_log import log

HIDDEN = 1024
NUM_TOPK = 4
TOKENS_PER_RANK = 128
EXPERTS_PER_RANK = 8
# Two layers sharing one arena: dropping the first must leave the second's transport intact.
NUM_LAYERS = 2


def live_arenas() -> list:
    return list(backends_mod._Arena._registry.values())


def build_dispatchers(ep_config, num_experts: int) -> list:
    """COLLECTIVE (sizing the transport). A function so the loop variable dies with this frame: a stray reference would
    keep a claim alive and weaken the GC check."""
    built = []
    for _ in range(NUM_LAYERS):
        dispatcher = DeepEPDispatcher(ep_config, num_experts, HIDDEN)
        dispatcher._ensure_buffer(TOKENS_PER_RANK, NUM_TOPK)
        built.append(dispatcher)
    return built


def run(ctx) -> dict:
    ep_config = ParallelismConfig(ep_size=ctx.world_size).ep_config()
    dispatchers = build_dispatchers(ep_config, EXPERTS_PER_RANK * ctx.world_size)

    arenas = live_arenas()
    assert len(arenas) == 1, f"expected the {NUM_LAYERS} layers to share ONE arena, found {len(arenas)}"
    arena = arenas[0]
    assert arena.buffer is not None, "arena reports no buffer after a dispatch was sized"
    assert arena._users == NUM_LAYERS, f"expected {NUM_LAYERS} claims, got {arena._users}"

    checks = {}

    # Ranks are deliberately unaligned here: a collective would be entered by one rank at a GC-chosen moment.
    dispatchers.clear()
    gc.collect()

    checks["buffer_survives_gc"] = arena.buffer is not None
    checks["arena_still_registered"] = arena in live_arenas()
    checks["claims_dropped_by_gc"] = arena._users == 0
    log(
        f"after dropping {NUM_LAYERS} dispatchers + gc.collect(): buffer alive="
        f"{arena.buffer is not None}, claims={arena._users}, registered arenas={len(live_arenas())}"
    )

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
