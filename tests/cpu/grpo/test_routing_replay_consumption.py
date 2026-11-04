#!/usr/bin/env python
"""Routing-replay mask consumption: ``disarm()`` accepts only a positive multiple of the armed size. The cursor wraps
to zero on exact exhaustion, so a never-consumed mask must not pass as fully consumed."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.models.gradient_checkpointing import RecomputeScope
from halo.models.moe.family_specs import MoEFamilySpec
from halo.trainers.grpo.environmental.routing_replay import RoutingReplayInjector

ROWS, SEQ, K, E = 2, 4, 2, 8


class _StubEPLayer(EPMoELayerBase):
    """Exposes the real ``_apply_routing_replay``; bypasses the EP/DeepEP ``__init__``."""

    FAMILY = MoEFamilySpec(model_types=())

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

    def forward(self, hidden_states, **kwargs):  # abstract, unused
        raise NotImplementedError


def _armed_injector(layer=None):
    layer = layer or _StubEPLayer()
    injector = RoutingReplayInjector([layer], engine_layers=1, layer_indices=[0])
    masks = torch.randint(0, E, (ROWS, SEQ, 1, K), dtype=torch.int16)
    injector.arm(masks)
    return injector, layer


def _consume(layer, tokens):
    natural = torch.randint(0, E, (tokens, K), dtype=torch.long)
    layer._apply_routing_replay(natural)


def test_disarm_raises_on_never_consumed_mask():
    """Cursor is 0 as when fully consumed, but consumed-total is 0; accepting it would hide a forward that never
    replayed routing."""
    injector, _layer = _armed_injector()
    with pytest.raises(RuntimeError, match="mis-consumed"):
        injector.disarm()


def test_disarm_accepts_exactly_one_full_pass():
    injector, layer = _armed_injector()
    _consume(layer, ROWS * SEQ)
    assert layer._forced_cursor == 0
    injector.disarm()
    assert layer._forced_topk_indices is None
    assert layer._forced_consumed_total == 0


def test_disarm_accepts_chunked_full_pass_and_whole_pass_reuse():
    injector, layer = _armed_injector()
    for _ in range(ROWS):
        _consume(layer, SEQ)
    _consume(layer, ROWS * SEQ)  # second full pass via the wrap
    assert layer._forced_consumed_total == 2 * ROWS * SEQ
    injector.disarm()


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
            injector.disarm()


def test_gc_recompute_does_not_advance_consumption():
    """The recompute re-reads the saved slice, so consumed-total counts each token once."""
    injector, layer = _armed_injector()
    layer.train()
    scope = RecomputeScope()
    with scope:  # original forward
        _consume(layer, ROWS * SEQ)
    with scope:  # backward recompute
        _consume(layer, ROWS * SEQ)
    assert layer._forced_consumed_total == ROWS * SEQ
    injector.disarm()  # exactly one pass consumed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
