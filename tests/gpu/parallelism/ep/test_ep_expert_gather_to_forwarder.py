#!/usr/bin/env python
"""The weight sync's expert gather on real NCCL groups: the forwarding rank receives the all-gather's bytes (negative
zeros and NaN payloads included) with a layer issued ahead on the side stream and strictly in turn, over one EP group of
four and over two EP groups of two, whose second group joins no collective."""

import os

import torch
import torch.distributed as dist

from halo.models.moe.expert_layout import ExpertLayout, ExpertLayoutKind
from halo.models.moe.family_specs import MoEFamilySpec
from halo.runtime.collectives import build_groups
from halo.runtime.consensus import DeferredRankFailure
from halo.trainers.grpo.rollout.weight_push import _send_ep_expert_weights
from tests.common.ep_stubs import GatherStubLayer, RecordingForwarder, signed_zero_nan
from tests.common.harness import gpu_test_main

LAYERS = 4
_SPEC = MoEFamilySpec(model_types=("gather_to_forwarder",), expert_layout=ExpertLayout(ExpertLayoutKind.FUSED))


def _fill(rank: int, layer: int, shape, device) -> torch.Tensor:
    values = torch.randn(shape, generator=torch.Generator().manual_seed(1000 * layer + rank)).to(device)
    return signed_zero_nan(values)


def _layer(rank, index, group, ep_size, device) -> GatherStubLayer:
    return GatherStubLayer(
        _SPEC,
        _fill(rank, index, (4, 64, 96), device),
        _fill(rank, index, (4, 96, 64), device),
        process_group=group,
        dispatch_ep_group=group,
        ep_size=ep_size,
    )


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.view(torch.int16).equal(b.view(torch.int16))


def _row(ctx, group_lists, ahead: int) -> bool:
    os.environ["HALO_WEIGHT_SYNC_GATHER_AHEAD"] = str(ahead)
    mine = next(i for i, ranks in enumerate(group_lists) if ctx.rank in ranks)
    group = build_groups(group_lists, mine=mine, what="gather test group")
    ep_size = len(group_lists[mine])
    layers = {f"layers.{i}": _layer(ctx.rank, i, group, ep_size, ctx.device) for i in range(LAYERS)}
    forwarder = RecordingForwarder() if ctx.rank == 0 else None
    guard = DeferredRankFailure("gather test")
    _send_ep_expert_weights(layers, forwarder, guard)
    guard.reject()
    torch.cuda.synchronize()
    # The old path over this rank's group, every rank of a group entering it.
    reference = {
        f"layers.{i}.{name}": tensor
        for i, layer in enumerate(layers.values())
        for name, tensor in layer.gather_expert_state_dict(device=ctx.device).items()
    }
    if ctx.rank != 0:
        return not forwarder
    return set(forwarder.sent) == set(reference) and all(
        _bitwise_equal(forwarder.sent[k], v) for k, v in reference.items()
    )


def run(ctx):
    checks = {}
    for ahead in (0, 1):
        checks[f"one_group_of_four_ahead_{ahead}"] = _row(ctx, [[0, 1, 2, 3]], ahead)
        checks[f"two_groups_of_two_ahead_{ahead}"] = _row(ctx, [[0, 1], [2, 3]], ahead)
    local = torch.tensor([all(checks.values())], device=ctx.device, dtype=torch.int32)
    dist.all_reduce(local, op=dist.ReduceOp.MIN)
    checks["every_rank_agrees"] = bool(local.item())
    return {"checks": checks}


main = gpu_test_main(exact_world_size=4, prefix="ep_gather_forwarder")(run)

if __name__ == "__main__":
    main()
