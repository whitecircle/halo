#!/usr/bin/env python
"""Ulysses' all-to-all moves real data: pin what it moves, on real gloo ranks.

The reference is the GLOBAL tensor the shards come from, not a second copy of the exchange's packing: through
``sequence_to_heads`` every rank must come out holding its own heads over the whole sequence, through
``heads_to_sequence`` its own sequence chunk of every head, and each backward must hand every rank the gradient of
the layout it sent. A transposed axis, a rotated rank order or a backward that does not invert the forward keeps
plausible shapes and norms, and the CP GPU suites see it only as one contribution to an end-to-end loss.

Q, K and V travel in one buffer, so the cases differ where the families do: GQA's K/V carry fewer heads than Q,
MLA's V a narrower head dim, and the inputs come either contiguous or as the views the families hand over (head
slices of a fused QKV projection, a channel slice for MLA's V). Each runs at ``cp_size`` 2 and 4 (at 2, scatter and
gather chunk indices coincide often enough that a rank-ordering error can survive), at batch 1 and 2 (the exchange
views and copies on different paths), on the world group and on two subgroups of a 4-rank world: CP's own
contiguous blocks and a strided split, where group rank and global rank differ on three of the four ranks. An
exchange that took the world group, its size or the global rank order fails on the subgroups. Each direction issues
exactly one ``all_to_all_single`` forward and one backward.

Run: pytest tests/cpu/parallelism/test_cp_autograd_collectives.py
"""

import contextlib
import datetime
import inspect
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from halo.distributed.context_parallel.all_to_all import heads_to_sequence, sequence_to_heads
from halo.distributed.group_layout import NodeLocalLayout
from tests.common.gloo import new_groups, run_gloo_ranks

SEQ = 8
# (heads, head_dim) of Q, K and V, every head count divisible by 4.
SHAPES = {
    "gqa": ((8, 3), (4, 3), (4, 3)),
    "mla": ((4, 5), (4, 5), (4, 2)),
}
BATCHES = (1, 2)
# The dim each of Q, K and V is a slice of a wider buffer along, or None for a contiguous tensor.
LAYOUTS = {
    "contiguous": (None, None, None),
    "views": (2, 2, 3),
}
SUBGROUP_WORLD = 4
CP_BLOCKS = [
    NodeLocalLayout(world_size=SUBGROUP_WORLD, group_size=2, domain_size=SUBGROUP_WORLD).group_ranks(group)
    for group in range(2)
]
STRIDED = [[0, 2], [1, 3]]
TIMEOUT = datetime.timedelta(seconds=60)
_ALL_TO_ALL_SINGLE = inspect.signature(dist.all_to_all_single)


def _seq_shard(tensor: torch.Tensor, rank: int, cp_size: int) -> torch.Tensor:
    """Rank's slice of ``[batch, seq, ...]`` along the sequence: CP's layout outside attention."""
    chunk = tensor.shape[1] // cp_size
    return tensor[:, rank * chunk : (rank + 1) * chunk].contiguous()


def _head_shard(tensor: torch.Tensor, rank: int, cp_size: int) -> torch.Tensor:
    """Rank's slice of ``[batch, seq, heads, dim]`` along the heads: Ulysses' layout inside attention."""
    chunk = tensor.shape[2] // cp_size
    return tensor[:, :, rank * chunk : (rank + 1) * chunk].contiguous()


def _leaf(values: torch.Tensor, sliced_dim: int | None) -> torch.Tensor:
    """``values`` as a leaf that requires grad: contiguous, or a view one slot in from either end of ``sliced_dim``
    of a wider buffer (not contiguous, offset into its storage)."""
    if sliced_dim is None:
        return values.clone().requires_grad_()
    shape = list(values.shape)
    shape[sliced_dim] += 2
    view = values.new_zeros(shape).narrow(sliced_dim, 1, values.shape[sliced_dim])
    view.copy_(values)
    assert not view.is_contiguous()
    return view.requires_grad_()


@contextlib.contextmanager
def _exchanges():
    """The ``all_to_all_single`` calls issued inside, each still run for real; a list-form ``all_to_all`` raises."""
    with (
        patch.object(dist, "all_to_all_single", wraps=dist.all_to_all_single) as single,
        patch.object(dist, "all_to_all", side_effect=AssertionError("the exchange issued a list-form all_to_all")),
    ):
        yield single


def _groups(exchanges) -> list:
    return [
        _ALL_TO_ALL_SINGLE.bind(*call.args, **call.kwargs).arguments.get("group") for call in exchanges.call_args_list
    ]


def _check_sequence_to_heads(group, shapes, batch: int, layout) -> None:
    cp_rank, cp_size = dist.get_rank(group), dist.get_world_size(group)
    fulls = [torch.randn(batch, SEQ, heads, head_dim) for heads, head_dim in shapes]
    upstreams = [torch.randn_like(full) for full in fulls]
    inputs = [_leaf(_seq_shard(full, cp_rank, cp_size), dim) for full, dim in zip(fulls, layout, strict=True)]

    with _exchanges() as exchanges:
        outputs = sequence_to_heads(*inputs, group)
        assert _groups(exchanges) == [group]
        torch.autograd.backward(outputs, [_head_shard(upstream, cp_rank, cp_size) for upstream in upstreams])
        assert _groups(exchanges) == [group, group]

    for name, output, full in zip("qkv", outputs, fulls, strict=True):
        assert torch.equal(output, _head_shard(full, cp_rank, cp_size)), (name, cp_rank)
    for name, local, upstream in zip("qkv", inputs, upstreams, strict=True):
        assert torch.equal(local.grad, _seq_shard(upstream, cp_rank, cp_size)), (name, cp_rank)


def _check_heads_to_sequence(group, shape, batch: int, sliced_dim: int | None) -> None:
    cp_rank, cp_size = dist.get_rank(group), dist.get_world_size(group)
    full, upstream = torch.randn(batch, SEQ, *shape), torch.randn(batch, SEQ, *shape)
    local = _leaf(_head_shard(full, cp_rank, cp_size), sliced_dim)

    with _exchanges() as exchanges:
        output = heads_to_sequence(local, group)
        assert _groups(exchanges) == [group]
        output.backward(_seq_shard(upstream, cp_rank, cp_size))
        assert _groups(exchanges) == [group, group]

    assert torch.equal(output, _seq_shard(full, cp_rank, cp_size)), cp_rank
    assert torch.equal(local.grad, _head_shard(upstream, cp_rank, cp_size)), cp_rank


def _exchange_worker(rank: int, subgroups: list[list[int]] | None) -> None:
    group = dist.group.WORLD if subgroups is None else new_groups(rank, subgroups, TIMEOUT)
    # The same global tensors on every rank, so each rank's expected shard is a slice of what its peers hold.
    torch.manual_seed(0)
    for shapes in SHAPES.values():
        for batch in BATCHES:
            for layout in LAYOUTS.values():
                _check_sequence_to_heads(group, shapes, batch, layout)
                _check_heads_to_sequence(group, shapes[0], batch, layout[0])


@pytest.mark.parametrize("cp_size", (2, 4))
def test_each_rank_gets_its_heads_and_its_sequence_chunk_back(cp_size):
    run_gloo_ranks(_exchange_worker, cp_size, None, pg_timeout=TIMEOUT)


@pytest.mark.parametrize("subgroups", (CP_BLOCKS, STRIDED), ids=("cp-blocks", "strided"))
def test_a_cp_subgroup_exchanges_in_its_own_rank_order(subgroups):
    run_gloo_ranks(_exchange_worker, SUBGROUP_WORLD, subgroups, pg_timeout=TIMEOUT)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
