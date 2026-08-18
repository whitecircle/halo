#!/usr/bin/env python
"""``EPMoELayerBase._etp_narrow`` — the one home for the expert-TP shard offset.

Three expert layouts slice the intermediate dimension for this ETP rank: separately-stored GLU
(Qwen3, Bailing), fused contiguous halves (every other family), and GptOss's interleaved pair. They
differ in WHICH axis carries that dimension — it moves with the transpose each layout applies — but
the ``rank * shard`` offset is the same arithmetic eight times over, so it lives once.

This pins the primitive against explicit slicing at every axis and rank, which is what makes the
call-site rewrites literal substitutions. The layouts themselves are GPU-covered
(``tests/gpu/parallelism/combined/test_ep_etp_*``); nothing on CPU builds a real ETP layer.

Run: python tests/cpu/parallelism/test_etp_narrow.py
"""

from __future__ import annotations

import pytest
import torch

from src.distributed.expert_parallel.base_layer import EPMoELayerBase


class _Probe:
    """The two attributes ``_etp_narrow`` reads, without building a real EP layer."""

    _etp_shard_size = EPMoELayerBase._etp_shard_size
    _etp_narrow = EPMoELayerBase._etp_narrow

    def __init__(self, expert_tp_size: int, expert_tp_rank: int):
        self.expert_tp_size = expert_tp_size
        self.expert_tp_rank = expert_tp_rank


def test_narrow_matches_explicit_slicing_on_every_axis_and_rank():
    """Exact equality with the hand-written ``[s : s + shard]`` on each of the three axes the layouts
    shard, and the shards tile the axis exactly once (no overlap, no gap)."""
    tensor = torch.arange(2 * 8 * 12, dtype=torch.float32).reshape(2, 8, 12)
    for dim, length in ((1, 8), (2, 12)):
        for tp_size in (1, 2, 4):
            if length % tp_size:
                continue
            shard = length // tp_size
            rebuilt = []
            for rank in range(tp_size):
                got = _Probe(tp_size, rank)._etp_narrow(tensor, dim)
                start = rank * shard
                # The explicit ``[..., s : s + shard, ...]`` the helper replaces at the call sites.
                expected = tensor[(slice(None),) * dim + (slice(start, start + shard),)]
                assert torch.equal(got, expected), (dim, tp_size, rank)
                assert got.shape[dim] == shard
                rebuilt.append(got)
            assert torch.equal(torch.cat(rebuilt, dim=dim), tensor), (dim, tp_size)


def test_narrow_is_a_view_of_the_source():
    """Callers materialize with ``.contiguous()`` / ``.clone()``, so the helper must not copy — a
    silent copy here would double the transient memory of every ETP expert init."""
    tensor = torch.zeros(2, 4, 6)
    assert _Probe(2, 1)._etp_narrow(tensor, 1).data_ptr() >= tensor.data_ptr()
    assert _Probe(2, 1)._etp_narrow(tensor, 1)._base is tensor


def test_indivisible_split_still_raises_through_the_helper():
    """The divisibility check lives in ``_etp_shard_size`` and must not be lost by routing through the
    narrow: a silent floor would drop the top units of every expert."""
    tensor = torch.zeros(2, 4, 7)
    with pytest.raises(ValueError, match="divisible by"):
        _Probe(2, 0)._etp_narrow(tensor, 2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
