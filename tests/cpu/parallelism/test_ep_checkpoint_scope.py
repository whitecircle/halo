#!/usr/bin/env python
"""CPU tests for the EP checkpoint scope — the frame-local cache EP's recompute replays from.

A second DeepEP dispatch during recompute reuses the same ``ElasticBuffer`` and invalidates the
handle the original forward's backward node holds, so the replay cache is load-bearing for
correctness, not just speed. These tests pin the three properties it rests on: the pass counter is
the recompute signal (grad mode is NOT — non-reentrant checkpointing enables grad on both passes),
concurrent frames (pipeline microbatches in flight) never see each other's slots, and a recompute
that diverges from its forward fails loud.

Run: ``pytest tests/cpu/parallelism/test_ep_checkpoint_scope.py``.
"""

import pytest
import torch

from src.distributed.expert_parallel.gc_scope import (
    EPCheckpointScope,
    active_checkpoint_scope,
    scoped_checkpoint_func,
)


class _Owner:
    """Stand-in for an EP MoE layer (slots are keyed by identity)."""


def test_first_pass_is_not_recompute_second_is():
    scope = EPCheckpointScope()
    with scope:
        assert active_checkpoint_scope() is scope
        assert not scope.is_recompute
    with scope:
        assert scope.is_recompute
    assert active_checkpoint_scope() is None


def test_recompute_signal_is_independent_of_grad_mode():
    # Non-reentrant checkpointing runs BOTH passes with grad enabled; the reentrant original forward
    # runs under no_grad. Neither may change the answer.
    scope = EPCheckpointScope()
    with torch.enable_grad(), scope:
        assert not scope.is_recompute
    with torch.no_grad(), scope:
        assert scope.is_recompute


def test_named_slots_are_independent():
    owner, scope = _Owner(), EPCheckpointScope()
    with scope:
        scope.slot(owner, "dispatch")["v"] = 1
        scope.slot(owner, "combine")["v"] = 2
    with scope:
        assert scope.slot(owner, "dispatch")["v"] == 1
        assert scope.slot(owner, "combine")["v"] == 2


def test_concurrent_frames_do_not_share_slots():
    # The 1F1B case: microbatch 2's forward runs between microbatch 1's forward and its recompute.
    owner = _Owner()
    frame1, frame2 = EPCheckpointScope(), EPCheckpointScope()
    with frame1:
        frame1.slot(owner, "dispatch")["recv_x"] = "mb1"
    with frame2:
        frame2.slot(owner, "dispatch")["recv_x"] = "mb2"
    with frame1:
        assert frame1.slot(owner, "dispatch")["recv_x"] == "mb1"
    with frame2:
        assert frame2.slot(owner, "dispatch")["recv_x"] == "mb2"


def test_repeated_calls_in_one_pass_replay_in_call_order():
    owner, scope = _Owner(), EPCheckpointScope()
    with scope:
        scope.slot(owner, "dispatch")["v"] = "call-1"
        scope.slot(owner, "dispatch")["v"] = "call-2"
    with scope:
        assert scope.slot(owner, "dispatch")["v"] == "call-1"
        assert scope.slot(owner, "dispatch")["v"] == "call-2"


def test_recompute_beyond_the_forwards_calls_raises():
    owner, scope = _Owner(), EPCheckpointScope()
    with scope:
        scope.slot(owner, "dispatch")
    with scope:
        scope.slot(owner, "dispatch")
        with pytest.raises(RuntimeError, match="different path"):
            scope.slot(owner, "dispatch")


def test_nested_scopes_resolve_to_the_innermost():
    outer, inner = EPCheckpointScope(), EPCheckpointScope()
    with outer:
        assert active_checkpoint_scope() is outer
        with inner:
            assert active_checkpoint_scope() is inner
        assert active_checkpoint_scope() is outer


def test_scope_pops_even_when_the_body_raises():
    scope = EPCheckpointScope()
    with pytest.raises(ValueError, match="boom"), scope:
        raise ValueError("boom")
    assert active_checkpoint_scope() is None


def test_scoped_checkpoint_func_wraps_both_passes():
    # The wrapper must give the original call and every re-invocation of the same callable one scope.
    seen = []

    def fake_checkpoint(function, *args, **kwargs):
        function(*args, **kwargs)  # original forward
        return function(*args, **kwargs)  # recompute

    def body():
        scope = active_checkpoint_scope()
        seen.append((id(scope), scope.is_recompute))

    scoped_checkpoint_func(fake_checkpoint)(body)
    assert [flag for _, flag in seen] == [False, True]
    assert seen[0][0] == seen[1][0], "the two passes must share one scope"


def test_scoped_checkpoint_func_is_idempotent():
    def fake_checkpoint(function, *args, **kwargs):
        return function(*args, **kwargs)

    once = scoped_checkpoint_func(fake_checkpoint)
    assert scoped_checkpoint_func(once) is once


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
