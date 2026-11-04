#!/usr/bin/env python
"""Routing-replay mask broadcast under TP/ETP: NCCL has no 16-bit-integer type, so `nccl_safe_broadcast` routes int16
through a uint8 view."""

import torch
import torch.distributed as dist

from halo.runtime.collectives import nccl_safe_broadcast
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.harness import gpu_test_main

ROWS, SEQ, LAYERS = 2, 16, 3
SEED = 42


class _TPLeaderCaller:
    """Drives the real loop: a local copy would keep passing after the method stopped routing int16 through the uint8
    view."""

    _broadcast_tensors_from_tp_leader = DistributedTrainerMixin._broadcast_tensors_from_tp_leader

    def _get_tp_or_etp_process_group(self):
        return dist.group.WORLD

    def _get_tp_group_src_rank(self) -> int:
        return 0


def _replay_style_mask(rank: int, top_k: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator().manual_seed(SEED + rank)
    mask = torch.randint(0, 128, (ROWS, SEQ, LAYERS, top_k), generator=gen, dtype=torch.int16)
    mask[:, : 2 + rank] = -1  # Natural-routing sentinel spans, rank-divergent on purpose.
    return mask.to(device)


def check_int16_broadcast_matches_leader(rank: int, device: torch.device) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for top_k in (4, 5):  # Even and odd trailing dim (odd breaks a naive int32 view).
        mask = _replay_style_mask(rank, top_k, device)
        leader_copy = _replay_style_mask(0, top_k, device)
        if rank != 0:
            # Premise: without a divergent start the sync is not observable.
            checks[f"ranks_start_divergent_top_k{top_k}"] = not torch.equal(mask, leader_copy)
        nccl_safe_broadcast(mask, src=0)
        checks[f"dtype_unchanged_in_flight_top_k{top_k}"] = mask.dtype == torch.int16
        checks[f"mask_matches_leader_top_k{top_k}"] = torch.equal(mask, leader_copy)
        checks[f"sentinel_survives_top_k{top_k}"] = int((mask == -1).sum()) > 0
    return checks


def _grpo_shaped_batch(rank: int, device: torch.device) -> dict:
    gen = torch.Generator().manual_seed(SEED + rank)
    return {
        "prompt_ids": torch.randint(0, 1000, (ROWS, SEQ), generator=gen, dtype=torch.int64).to(device),
        "logps": torch.randn(ROWS, SEQ, generator=gen, dtype=torch.float32).to(device),
        "mask": (torch.rand(ROWS, SEQ, generator=gen) > 0.5).to(device),
        "routing_masks": _replay_style_mask(rank, 4, device),
    }


def check_mixed_batch_dict_broadcast(rank: int, device: torch.device) -> dict[str, bool]:
    batch = _grpo_shaped_batch(rank, device)
    batch["not_a_tensor"] = rank  # The loop's isinstance filter must leave this alone.
    expected = _grpo_shaped_batch(0, device)
    returned = _TPLeaderCaller()._broadcast_tensors_from_tp_leader(batch)
    checks: dict[str, bool] = {"batch_synced_in_place": returned is batch}
    for key, want in expected.items():
        got = batch[key]
        checks[f"{key}_is_tensor"] = isinstance(got, torch.Tensor)
        checks[f"{key}_dtype_preserved"] = got.dtype == want.dtype
        checks[f"{key}_matches_leader"] = torch.equal(got, want)
    # A mishandled non-tensor would be broadcast or crash; it must survive as the rank's own value.
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
