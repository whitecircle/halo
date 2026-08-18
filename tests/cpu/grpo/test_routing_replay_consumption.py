#!/usr/bin/env python
"""CPU tests for routing-replay mask consumption integrity
(:class:`~src.trainers.grpo.rollout.routing_replay.RoutingReplayInjector` +
``EPMoELayerBase._maybe_replace_selection``).

Consumption model under test: only non-recompute forwards advance the cursor / monotonic
consumed-total; the cursor wraps to zero on exact exhaustion (whole-pass reuse); GC recomputes
re-read saved slices (LIFO) without consuming. ``disarm()`` must therefore accept exactly the
states where consumed-total is a POSITIVE multiple of the armed size and fail loud otherwise —
in particular the wrap-to-zero must NOT let a never-consumed mask (0 tokens) pass as fully
consumed (both end with cursor == 0).

Run: ``python tests/cpu/grpo/test_routing_replay_consumption.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.gc_scope import EPCheckpointScope
from src.trainers.grpo.rollout.routing_replay import RoutingReplayInjector

ROWS, SEQ, K, E = 2, 4, 2, 8


class _StubEPLayer(EPMoELayerBase):
    """Minimal concrete layer exposing the REAL ``_maybe_replace_selection`` consumption logic —
    bypasses the EP/DeepEP ``__init__`` and sets only the plain attributes the replay path reads."""

    def __init__(self):
        nn.Module.__init__(self)
        self.top_k = K
        self.num_experts = E
        self._forced_topk_indices = None
        self._forced_cursor = 0
        self._forced_consumed_total = 0
        self._capture_routing = False
        self._captured_routing_chunks = []
        self._replay_flip_counts = None

    def forward(self, hidden_states, **kwargs):  # abstract — unused
        raise NotImplementedError


def _armed_injector(layer=None):
    layer = layer or _StubEPLayer()
    injector = RoutingReplayInjector([layer])
    masks = torch.randint(0, E, (ROWS, SEQ, 1, K), dtype=torch.int16)
    injector.arm(masks)
    return injector, layer


def _consume(layer, tokens):
    """Drive one chunk forward's replay consumption through the real selection swap."""
    natural = torch.randint(0, E, (tokens, K), dtype=torch.long)
    layer._maybe_replace_selection(natural)


def test_disarm_raises_on_never_consumed_mask():
    """Armed but never forwarded: cursor is 0 (same as fully-consumed) but consumed-total is 0 —
    disarm must fail loud, not silently accept a gradient forward that never replayed routing."""
    injector, _layer = _armed_injector()
    with pytest.raises(RuntimeError, match="mis-consumed"):
        injector.disarm()


def test_disarm_accepts_exactly_one_full_pass():
    injector, layer = _armed_injector()
    _consume(layer, ROWS * SEQ)
    assert layer._forced_cursor == 0  # wrapped on exact exhaustion
    injector.disarm()  # must not raise
    assert layer._forced_topk_indices is None
    assert layer._forced_consumed_total == 0


def test_disarm_accepts_chunked_full_pass_and_whole_pass_reuse():
    """Row-chunked forwards tiling the mask, then a second full reuse pass (μ>1 wrap) — both fine."""
    injector, layer = _armed_injector()
    for _ in range(ROWS):
        _consume(layer, SEQ)
    _consume(layer, ROWS * SEQ)  # second full pass via the wrap
    assert layer._forced_consumed_total == 2 * ROWS * SEQ
    injector.disarm()  # must not raise


def test_rearm_resets_consumption():
    injector, layer = _armed_injector()
    _consume(layer, ROWS * SEQ)
    injector.arm(torch.randint(0, E, (ROWS, SEQ, 1, K), dtype=torch.int16))
    assert layer._forced_consumed_total == 0
    with pytest.raises(RuntimeError, match="mis-consumed"):
        injector.disarm()


def test_disarm_stays_silent_while_exception_propagates():
    """A real error (OOM/NCCL) mid-microbatch must not be shadowed by the consumption check."""
    injector, _layer = _armed_injector()
    with pytest.raises(ValueError, match="real failure"):
        try:
            raise ValueError("real failure")
        finally:
            injector.disarm()  # partial state, but the propagating exception wins


def test_gc_recompute_does_not_advance_consumption():
    """The checkpointed original forward consumes and saves its slice; the recompute re-reads that
    slice WITHOUT consuming — consumed-total counts each token once."""
    injector, layer = _armed_injector()
    layer.train()
    scope = EPCheckpointScope()
    with scope:  # original forward
        _consume(layer, ROWS * SEQ)
    with scope:  # backward recompute
        _consume(layer, ROWS * SEQ)
    assert layer._forced_consumed_total == ROWS * SEQ
    injector.disarm()  # exactly one pass consumed — must not raise


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
