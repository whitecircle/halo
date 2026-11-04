#!/usr/bin/env python3
"""``get_data_parallel_rank`` partitions every accepted shape into exactly ``data_parallel_size`` batches. They come
from different code, and a mismatch reads another replica's shard at a finite loss."""

import itertools
from collections import Counter

import pytest

from halo.distributed.group_layout import NodeLocalLayout
from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.parallelism_rules import SUPPORTED_AXIS_SETS
from tests.common.parallelism import make_parallelism_config

# (world_size, gpus_per_node); the 2-GPU-node boxes make PP and multi-domain layouts reachable.
_TOPOLOGIES = ((8, 8), (16, 8), (32, 8), (8, 2), (16, 2))
_SIZES = (1, 2, 4, 8)


def _config(rank: int, world: int, gpus_per_node: int, **kwargs) -> ParallelismConfig:
    return make_parallelism_config(world_size=world, gpus_per_node=gpus_per_node, rank=rank, **kwargs)


def _accepted_shapes() -> list[tuple[int, int, dict]]:
    shapes = []
    for world, gpn in _TOPOLOGIES:
        for pp, ep, etp, tp, cp in itertools.product(_SIZES, repeat=5):
            kwargs = {
                "pp_size": pp,
                "ep_size": ep,
                "expert_tp_size": etp,
                "tp_size": tp,
                "cp_size": cp,
            }
            for scope in ("node", "global"):
                try:
                    _config(0, world, gpn, ep_scope=scope, **kwargs)
                except ValueError:
                    continue
                shapes.append((world, gpn, {**kwargs, "ep_scope": scope}))
    return shapes


_SHAPES = _accepted_shapes()


def test_the_sweep_reaches_every_supported_axis_set():
    """A grid that stopped building configs would make the sweep vacuous."""
    assert len(_SHAPES) > 100, len(_SHAPES)
    reached = set()
    for world, gpn, kwargs in _SHAPES:
        reached.add(_config(0, world, gpn, **kwargs).active_axes)
    assert reached == set(SUPPORTED_AXIS_SETS), sorted(map(sorted, SUPPORTED_AXIS_SETS - reached))


@pytest.mark.parametrize("world,gpn,kwargs", _SHAPES, ids=lambda v: str(v))
def test_dp_ranks_partition_the_world_into_data_parallel_size_batches(world, gpn, kwargs):
    configs = [_config(rank, world, gpn, **kwargs) for rank in range(world)]
    dp_size = configs[0].data_parallel_size
    claims = Counter(c.get_data_parallel_rank() for c in configs)

    assert set(claims) == set(range(dp_size)), (
        f"{kwargs} on world={world}/domain={gpn}: DP ranks {sorted(claims)} are not "
        f"range({dp_size}) — the sampler indexes {dp_size} shards by this value."
    )
    # pp_size chains x the batch-sharing axis width; EP does not count.
    expected_per_replica = world // dp_size
    assert set(claims.values()) == {expected_per_replica}, f"{kwargs} on world={world}: {claims}"


@pytest.mark.parametrize("world,gpn,kwargs", _SHAPES, ids=lambda v: str(v))
def test_ranks_that_must_share_a_batch_do(world, gpn, kwargs):
    """CP chunks one sequence, TP shards one matmul, expert-TP sums in token space, and a chain passes one
    microbatch down the stages."""
    configs = [_config(rank, world, gpn, **kwargs) for rank in range(world)]
    dp_of = [c.get_data_parallel_rank() for c in configs]
    cfg = configs[0]

    for rank, c in enumerate(configs):
        if c.cp_size > 1:
            cp_layout = NodeLocalLayout(
                world_size=c.stage_world_size, group_size=c.cp_size, domain_size=c.nvlink_domain_size
            )
            partners = [
                c.stage_base_rank + r for r in cp_layout.group_ranks(cp_layout.rank_and_group(c.stage_local_rank)[1])
            ]
            assert {dp_of[r] for r in partners} == {dp_of[rank]}, ("cp", rank, partners, dp_of)
        if c.tp_size > 1:
            # The (dp, tp) mesh is row-major.
            base = c.stage_base_rank + (c.stage_local_rank // c.tp_size) * c.tp_size
            partners = range(base, base + c.tp_size)
            assert {dp_of[r] for r in partners} == {dp_of[rank]}, ("tp", rank, list(partners), dp_of)
        if c.expert_tp_size > 1:
            # Derived from the layout, since DP equality would self-prove.
            layout = c.ep_layout
            group = [c.stage_base_rank + r for r in layout.group_ranks(layout.rank_and_group(c.stage_local_rank)[1])]
            dispatch_of = {
                r: layout.etp_partition(layout.rank_and_group(r - c.stage_base_rank)[0], c.ep_size, c.expert_tp_size)[
                    0
                ]
                for r in group
            }
            partners = [r for r in group if dispatch_of[r] == dispatch_of[rank]]
            assert len(partners) == c.expert_tp_size, ("etp-layout", rank, dispatch_of)
            assert {dp_of[r] for r in partners} == {dp_of[rank]}, ("etp", rank, partners, dp_of)
            # Or the ETP all-reduce sums unrelated tokens.
            others = [r for r in group if dispatch_of[r] != dispatch_of[rank]]
            assert dp_of[rank] not in {dp_of[r] for r in others}, ("etp-distinct", rank, others, dp_of)
        if c.pp_size > 1:
            chain = c.pp_chain_ranks()
            assert {dp_of[r] for r in chain} == {dp_of[rank]}, ("pp", rank, chain, dp_of)

    # EP must not reduce the batch count (an EP group may still share one).
    if cfg.ep_size > 1:
        no_ep = _config(0, world, gpn, **{**kwargs, "ep_size": 1, "ep_scope": "node"})
        assert cfg.data_parallel_size == no_ep.data_parallel_size, (cfg.data_parallel_size, no_ep.data_parallel_size)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
