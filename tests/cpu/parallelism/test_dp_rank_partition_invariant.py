#!/usr/bin/env python3
"""``get_data_parallel_rank`` must partition every ACCEPTED shape into exactly
``data_parallel_size`` batches.

The two numbers come from different code: ``data_parallel_size`` is arithmetic on the axis sizes
(``stage_world_size // max(tp, cp, expert_tp)``), while ``get_data_parallel_rank`` walks a
per-axis branch (the expert-TP dispatch formula, the ``// max(tp, cp)`` floor, or the CP group
index). Nothing forces them to agree, and a disagreement is silent: the sampler shards the dataset
``data_parallel_size`` ways and indexes it by the rank, so a rank whose DP index falls outside that
range reads another replica's shard while some shard is never read — a data bug with a finite loss.
That failure is half of the ``{etp, cp}`` rejection's stated mechanism, and it is exactly what an
allowlist addition would reintroduce.

So this asserts the invariant over the whole ACCEPTED space rather than a case list: for every
config a grid sweep can build, the DP ranks of all its world ranks must be ``range(dp_size)``,
equally claimed, with TP / CP / expert-TP partners and whole pipeline chains landing on the same
batch. Only shapes ``ParallelismConfig`` accepts are checked — rejected ones are another test's job.

Usage:
    python tests/cpu/parallelism/test_dp_rank_partition_invariant.py
"""

import itertools
import sys
from collections import Counter

import pytest

from src.distributed.group_layout import etp_dispatch_coords
from src.distributed.parallelism_config import SUPPORTED_AXIS_SETS, ParallelismConfig
from tests.common.parallelism import make_parallelism_config

# (world_size, gpus_per_node); the 2-GPU-node boxes make PP and multi-domain layouts reachable.
_TOPOLOGIES = ((8, 8), (16, 8), (32, 8), (8, 2), (16, 2))
_SIZES = (1, 2, 4, 8)


def _config(rank: int, world: int, gpus_per_node: int, **kwargs) -> ParallelismConfig:
    return make_parallelism_config(world_size=world, gpus_per_node=gpus_per_node, rank=rank, **kwargs)


def _accepted_shapes() -> list[tuple[int, int, dict]]:
    """Every (world, domain, kwargs) the validators accept across the grid, rank 0 as the probe."""
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
    """Guard the guard: a grid that silently stopped building configs would make the sweep vacuous."""
    assert len(_SHAPES) > 100, len(_SHAPES)
    reached = set()
    for world, gpn, kwargs in _SHAPES:
        reached.add(_config(0, world, gpn, **kwargs).active_axes)
    assert reached == set(SUPPORTED_AXIS_SETS), sorted(map(sorted, SUPPORTED_AXIS_SETS - reached))


@pytest.mark.parametrize("world,gpn,kwargs", _SHAPES, ids=lambda v: str(v))
def test_dp_ranks_partition_the_world_into_data_parallel_size_batches(world, gpn, kwargs):
    """The DP ranks of a shape's world must be ``range(dp_size)``, each claimed equally often."""
    configs = [_config(rank, world, gpn, **kwargs) for rank in range(world)]
    dp_size = configs[0].data_parallel_size
    claims = Counter(c.get_data_parallel_rank() for c in configs)

    assert set(claims) == set(range(dp_size)), (
        f"{kwargs} on world={world}/domain={gpn}: DP ranks {sorted(claims)} are not "
        f"range({dp_size}) — the sampler indexes {dp_size} shards by this value."
    )
    # Each replica is claimed by pp_size chains x the batch-sharing axis width; EP does not count.
    expected_per_replica = world // dp_size
    assert set(claims.values()) == {expected_per_replica}, f"{kwargs} on world={world}: {claims}"


@pytest.mark.parametrize("world,gpn,kwargs", _SHAPES, ids=lambda v: str(v))
def test_ranks_that_must_share_a_batch_do(world, gpn, kwargs):
    """CP / TP / expert-TP partners and whole pipeline chains must land on one DP rank.

    Each for a different reason, all silent if broken: CP partners hold chunks of the SAME sequence,
    TP partners hold shards of one matmul over the SAME tokens, expert-TP partners' outputs are
    summed element-wise in token space by ``ReduceFromExpertTP``, and a pipeline chain passes one
    microbatch's activations down the stages (stage 0 reads ``input_ids``, the last reads ``labels``).
    """
    configs = [_config(rank, world, gpn, **kwargs) for rank in range(world)]
    dp_of = [c.get_data_parallel_rank() for c in configs]
    cfg = configs[0]

    for rank, c in enumerate(configs):
        if c.cp_size > 1:
            partners = c.get_cp_group_ranks()
            assert {dp_of[r] for r in partners} == {dp_of[rank]}, ("cp", rank, partners, dp_of)
        if c.tp_size > 1:
            # TP groups are contiguous rank blocks inside a stage (the (dp, tp) mesh is row-major).
            base = c.stage_base_rank + (c.stage_local_rank // c.tp_size) * c.tp_size
            partners = range(base, base + c.tp_size)
            assert {dp_of[r] for r in partners} == {dp_of[rank]}, ("tp", rank, list(partners), dp_of)
        if c.expert_tp_size > 1:
            # Partners derived from the layout leaf, never from DP equality (which would self-prove).
            group = c.get_ep_group_ranks()
            dispatch_of = {
                r: etp_dispatch_coords(configs[r].get_ep_rank(), c.ep_size, c.expert_tp_size, c.is_node_local_ep)[0]
                for r in group
            }
            partners = [r for r in group if dispatch_of[r] == dispatch_of[rank]]
            assert len(partners) == c.expert_tp_size, ("etp-layout", rank, dispatch_of)
            assert {dp_of[r] for r in partners} == {dp_of[rank]}, ("etp", rank, partners, dp_of)
            # A different dispatch position must be a different batch, or the ETP all-reduce sums unrelated tokens.
            others = [r for r in group if dispatch_of[r] != dispatch_of[rank]]
            assert dp_of[rank] not in {dp_of[r] for r in others}, ("etp-distinct", rank, others, dp_of)
        if c.pp_size > 1:
            chain = c.get_pp_group_ranks()
            assert {dp_of[r] for r in chain} == {dp_of[rank]}, ("pp", rank, chain, dp_of)

    # EP is orthogonal to DP: it must not reduce the batch count (an EP group may still share one).
    if cfg.ep_size > 1:
        no_ep = _config(0, world, gpn, **{**kwargs, "ep_size": 1, "ep_scope": "node"})
        assert cfg.data_parallel_size == no_ep.data_parallel_size, (cfg.data_parallel_size, no_ep.data_parallel_size)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
