#!/usr/bin/env python
"""The EP checkpoint scope EP's recompute replays from. A second dispatch invalidates the backward node's handle;
the pass counter is the signal, as non-reentrant enables grad on both passes."""

import functools
import gc
import weakref

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from halo.models.gradient_checkpointing import (
    RecomputeScope,
    active_recompute_scope,
    scoped_checkpoint_func,
)

LAYERS, TOKENS, HIDDEN = 6, 16, 8


class _Owner:
    """Slots are keyed by identity."""


def test_first_pass_is_not_recompute_second_is():
    scope = RecomputeScope()
    with scope:
        assert active_recompute_scope() is scope
        assert not scope.is_recompute
    with scope:
        assert scope.is_recompute
    assert active_recompute_scope() is None


def test_recompute_signal_is_independent_of_grad_mode():
    scope = RecomputeScope()
    with torch.enable_grad(), scope:
        assert not scope.is_recompute
    with torch.no_grad(), scope:
        assert scope.is_recompute


def test_named_slots_are_independent():
    owner, scope = _Owner(), RecomputeScope()
    with scope:
        scope.slot(owner, "dispatch")["v"] = 1
        scope.slot(owner, "combine")["v"] = 2
    with scope:
        assert scope.slot(owner, "dispatch")["v"] == 1
        assert scope.slot(owner, "combine")["v"] == 2


def test_concurrent_frames_do_not_share_slots():
    # The 1F1B case: microbatch 2's forward runs between microbatch 1's forward and its recompute.
    owner = _Owner()
    frame1, frame2 = RecomputeScope(), RecomputeScope()
    with frame1:
        frame1.slot(owner, "dispatch")["recv_x"] = "mb1"
    with frame2:
        frame2.slot(owner, "dispatch")["recv_x"] = "mb2"
    with frame1:
        assert frame1.slot(owner, "dispatch")["recv_x"] == "mb1"
    with frame2:
        assert frame2.slot(owner, "dispatch")["recv_x"] == "mb2"


def test_repeated_calls_in_one_pass_replay_in_call_order():
    owner, scope = _Owner(), RecomputeScope()
    with scope:
        scope.slot(owner, "dispatch")["v"] = "call-1"
        scope.slot(owner, "dispatch")["v"] = "call-2"
    with scope:
        assert scope.slot(owner, "dispatch")["v"] == "call-1"
        assert scope.slot(owner, "dispatch")["v"] == "call-2"


def test_recompute_beyond_the_forwards_calls_raises():
    owner, scope = _Owner(), RecomputeScope()
    with scope:
        scope.slot(owner, "dispatch")
    with scope:
        scope.slot(owner, "dispatch")
        with pytest.raises(RuntimeError, match="different path"):
            scope.slot(owner, "dispatch")


def test_nested_scopes_resolve_to_the_innermost():
    outer, inner = RecomputeScope(), RecomputeScope()
    with outer:
        assert active_recompute_scope() is outer
        with inner:
            assert active_recompute_scope() is inner
        assert active_recompute_scope() is outer


def test_scope_pops_even_when_the_body_raises():
    scope = RecomputeScope()
    with pytest.raises(ValueError, match="boom"), scope:
        raise ValueError("boom")
    assert active_recompute_scope() is None


def test_scoped_checkpoint_func_wraps_both_passes():
    seen = []

    def fake_checkpoint(function, *args, **kwargs):
        function(*args, **kwargs)
        return function(*args, **kwargs)

    def body():
        scope = active_recompute_scope()
        seen.append((id(scope), scope.is_recompute))

    scoped_checkpoint_func(fake_checkpoint)(body)
    assert [flag for _, flag in seen] == [False, True]
    assert seen[0][0] == seen[1][0], "the two passes must share one scope"


def test_scoped_checkpoint_func_is_idempotent():
    def fake_checkpoint(function, *args, **kwargs):
        return function(*args, **kwargs)

    once = scoped_checkpoint_func(fake_checkpoint)
    assert scoped_checkpoint_func(once) is once


def test_the_recompute_releases_its_slots():
    owner, scope = _Owner(), RecomputeScope()
    with scope:
        scope.slot(owner, "dispatch")["v"] = 1
    with scope:
        assert scope.slot(owner, "dispatch")["v"] == 1
    with scope, pytest.raises(RuntimeError, match="different path"):
        scope.slot(owner, "dispatch")


class _CachingLayer(nn.Module):
    """As the DeepEP dispatch keeps its receive buffer; records which caches are alive at its own backward."""

    def __init__(self, index: int, caches: list, alive_at_backward: list):
        super().__init__()
        self.index = index
        self.linear = nn.Linear(HIDDEN, HIDDEN)
        self._caches = caches
        self._alive_at_backward = alive_at_backward

    def forward(self, x):
        scope = active_recompute_scope()
        slot = scope.slot(self, "dispatch")
        if scope.is_recompute:
            received = slot["received"] + 0 * x
        else:
            cached = (x * 2).detach().clone()
            slot["received"] = cached
            self._caches.append((self.index, weakref.ref(cached)))
            received = cached + 0 * x
        out = self.linear(received)
        if scope.is_recompute:
            out.register_hook(functools.partial(self._record, self.index))
        return out

    def _record(self, index, grad):
        gc.collect()
        self._alive_at_backward.append((index, _alive(self._caches)))
        return grad


def _alive(caches: list) -> list[int]:
    return sorted(index for index, ref in caches if ref() is not None)


def test_a_reentrant_layers_cache_dies_with_its_recompute():
    """The reentrant checkpoint keeps every scope until the loss drops, so a cache held past its own backward stays
    through every other layer's."""
    caches, alive_at_backward = [], []
    layers = [_CachingLayer(index, caches, alive_at_backward) for index in range(LAYERS)]
    run = scoped_checkpoint_func(functools.partial(checkpoint, use_reentrant=True))
    hidden = torch.randn(TOKENS, HIDDEN, requires_grad=True)
    for layer in layers:
        hidden = run(layer, hidden)
    loss = hidden.sum()
    assert _alive(caches) == list(range(LAYERS))
    loss.backward()
    gc.collect()
    assert alive_at_backward == [(index, list(range(index))) for index in reversed(range(LAYERS))]
    assert _alive(caches) == [], "a replay cache outlived the backward while the loss is still held"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
