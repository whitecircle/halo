#!/usr/bin/env python
"""
Routing-replay mask broadcast under TP/ETP: NCCL-safe int16 transport.

The routing-replay mask is int16 (`[rows, seq, layers, top_k]`, `-1` = natural routing) and rides
`_broadcast_tensors_from_tp_leader` when a TP/ETP group must train on identical batches. NCCL has
no 16-bit-integer type, so `nccl_safe_broadcast` routes such tensors through a uint8 bit view.

Validates on 2 GPUs (the TP-group shape):

1. int16 broadcast: per-rank-divergent replay-style masks (including `-1` sentinels) become
   bit-identical to the leader's after `nccl_safe_broadcast` — a plain `dist.broadcast` raises
   here, so this test fails if the uint8-view routing is dropped.
2. Odd trailing dim: `top_k` odd (the case an int32 view could not handle) round-trips exactly.
3. Mixed batch dict: a GRPO-shaped dict (int64 ids, float32 logps, bool masks, int16 routing mask,
   one non-tensor) driven through the PRODUCTION `_broadcast_tensors_from_tp_leader` — not a local
   copy of its loop, which would keep passing after the method stopped routing int16 safely.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/tp/test_replay_mask_tp_broadcast.py
"""

import torch
import torch.distributed as dist

from src.distributed.runtime import nccl_safe_broadcast
from src.trainers.mixins.base import DistributedTrainerMixin
from tests.common.harness import gpu_test_main

ROWS, SEQ, LAYERS = 2, 16, 3
SEED = 42


class _TPLeaderCaller:
    """The production ``_broadcast_tensors_from_tp_leader``, over this run's world as the TP group.

    Bound to the two collaborators the method consults so the test drives the REAL loop — its tensor
    filter and its ``nccl_safe_broadcast`` routing. A local re-implementation of that loop would keep
    passing after the method stopped routing int16 through the uint8 view.
    """

    _broadcast_tensors_from_tp_leader = DistributedTrainerMixin._broadcast_tensors_from_tp_leader

    def _get_tp_or_etp_process_group(self):
        return dist.group.WORLD

    def _get_tp_group_src_rank(self) -> int:
        return 0


def _replay_style_mask(rank: int, top_k: int, device: torch.device) -> torch.Tensor:
    """Per-rank-divergent int16 mask shaped like the routing-replay batch tensor."""
    gen = torch.Generator().manual_seed(SEED + rank)
    mask = torch.randint(0, 128, (ROWS, SEQ, LAYERS, top_k), generator=gen, dtype=torch.int16)
    mask[:, : 2 + rank] = -1  # natural-routing sentinel spans, rank-divergent on purpose
    return mask.to(device)


def check_int16_broadcast_matches_leader(rank: int, device: torch.device) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for top_k in (4, 5):  # even and odd trailing dim (odd breaks a naive int32 view)
        mask = _replay_style_mask(rank, top_k, device)
        leader_copy = _replay_style_mask(0, top_k, device)
        if rank != 0:
            # Premise: without a divergent start the sync is not observable.
            checks[f"ranks_start_divergent_top_k{top_k}"] = not torch.equal(mask, leader_copy)
        nccl_safe_broadcast(mask, src=0)
        checks[f"dtype_unchanged_in_flight_top_k{top_k}"] = mask.dtype == torch.int16
        checks[f"mask_matches_leader_top_k{top_k}"] = torch.equal(mask, leader_copy)
        # The -1 sentinel must survive the uint8 bit view.
        checks[f"sentinel_survives_top_k{top_k}"] = int((mask == -1).sum()) > 0
    return checks


def _grpo_shaped_batch(rank: int, device: torch.device) -> dict:
    """Per-rank-divergent GRPO-shaped batch; the same recipe at rank 0 is the leader's expected state."""
    gen = torch.Generator().manual_seed(SEED + rank)
    return {
        "prompt_ids": torch.randint(0, 1000, (ROWS, SEQ), generator=gen, dtype=torch.int64).to(device),
        "logps": torch.randn(ROWS, SEQ, generator=gen, dtype=torch.float32).to(device),
        "mask": (torch.rand(ROWS, SEQ, generator=gen) > 0.5).to(device),
        "routing_masks": _replay_style_mask(rank, 4, device),
    }


def check_mixed_batch_dict_broadcast(rank: int, device: torch.device) -> dict[str, bool]:
    batch = _grpo_shaped_batch(rank, device)
    batch["not_a_tensor"] = rank  # the loop's isinstance filter must leave this alone
    expected = _grpo_shaped_batch(0, device)
    returned = _TPLeaderCaller()._broadcast_tensors_from_tp_leader(batch)
    checks: dict[str, bool] = {"batch_synced_in_place": returned is batch}
    for key, want in expected.items():
        got = batch[key]
        checks[f"{key}_is_tensor"] = isinstance(got, torch.Tensor)
        checks[f"{key}_dtype_preserved"] = got.dtype == want.dtype
        checks[f"{key}_matches_leader"] = torch.equal(got, want)
    # A non-tensor entry the filter mishandled would be broadcast (or crash); it must survive as the
    # rank's own value, which is what makes rank 1's copy differ from the leader's here.
    checks["non_tensor_untouched"] = batch["not_a_tensor"] == rank
    return checks


def run(ctx):
    checks = check_int16_broadcast_matches_leader(ctx.rank, ctx.device)
    checks.update(check_mixed_batch_dict_broadcast(ctx.rank, ctx.device))
    ctx.barrier()
    return {"checks": checks}


main = gpu_test_main(exact_world_size=2, prefix="replay_mask_tp_broadcast", partial_state=False)(run)

if __name__ == "__main__":
    main()
