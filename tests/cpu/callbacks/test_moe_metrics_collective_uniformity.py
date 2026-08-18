"""MoEMetricsCallback's expert-load all-reduce must be reached by every rank, or none.

Neither participation nor payload shape may derive from rank-LOCAL state: a rank whose hook captured
no router logits (a ragged final accumulation cycle, a stage running no MoE forward) then returns
before the collective while its peers enter it — a NCCL hang — and a payload whose expert dimension
is the local max turns that asymmetry into a size mismatch instead.

These tests drive ``on_step_end`` with a stubbed process group and assert on the collectives issued.
"""

from __future__ import annotations

import pytest
import torch

from src.callbacks import moe_metrics
from src.callbacks.moe_metrics import MoEMetricsCallback


class _State:
    def __init__(self, global_step: int):
        self.global_step = global_step


class _RecordingDist:
    """Stands in for ``torch.distributed``, recording every all-reduce payload shape.

    ``get_world_size`` also stands in for the callback's world-size read, which goes through
    :func:`get_global_world_size` rather than this object.
    """

    ReduceOp = torch.distributed.ReduceOp

    def __init__(self, world_size: int = 2):
        self._world_size = world_size
        self.calls: list[tuple[int, ...]] = []

    def get_world_size(self):
        return self._world_size

    def all_reduce(self, tensor, op=None, group=None):
        self.calls.append(tuple(tensor.shape))
        return None


def _install(monkeypatch, stub: _RecordingDist) -> None:
    """Point the callback's collectives AND its world-size read at one stub."""
    monkeypatch.setattr(moe_metrics, "dist", stub)
    monkeypatch.setattr(moe_metrics, "get_global_world_size", stub.get_world_size)


@pytest.fixture
def recording_dist(monkeypatch):
    stub = _RecordingDist()
    _install(monkeypatch, stub)
    return stub


def _callback(counters):
    cb = MoEMetricsCallback(topk=1)
    cb._counters = counters
    return cb


@pytest.mark.parametrize("counters", [[], [None, None]], ids=["never-captured", "hooked-but-idle"])
def test_rank_with_no_counters_still_joins_the_collective(recording_dist, counters):
    """The empty rank must reach the shape-agreement all-reduce, not return ahead of its peers.

    Both spellings of "captured nothing" have to behave alike: no hook ever fired (empty list), and
    routers hooked whose forwards never ran on this rank (a slot per router, all unfilled).
    """
    cb = _callback(counters)

    cb.on_step_end(args=None, state=_State(global_step=1), control=None)

    assert recording_dist.calls, (
        "a rank with no captured router logits issued NO collective on a logging step, while a rank "
        "with counters issues two — that is a NCCL hang"
    )
    assert recording_dist.calls[0] == (2,), "the first collective must be the 2-element dims agreement"


def test_rank_with_counters_agrees_dims_before_reducing_payload(recording_dist):
    cb = _callback([torch.ones(4), torch.ones(4)])

    cb.on_step_end(args=None, state=_State(global_step=1), control=None)

    assert recording_dist.calls[0] == (2,), "dims must be agreed before the payload is built"
    assert len(recording_dist.calls) == 2, "expected dims agreement + payload reduce"


def test_empty_and_populated_ranks_issue_the_same_collective_sequence(recording_dist):
    """The two ranks' collective sequences must be comparable in count, or NCCL desynchronizes."""
    empty = _RecordingDist()
    populated = _RecordingDist()

    import contextlib

    for stub, counters in ((empty, []), (populated, [torch.ones(3)])):
        with pytest.MonkeyPatch.context() as mp:
            _install(mp, stub)
            cb = _callback(counters)
            with contextlib.suppress(Exception):
                cb.on_step_end(args=None, state=_State(global_step=1), control=None)

    assert empty.calls[0] == populated.calls[0] == (2,), (
        f"ranks disagree on the first collective: empty={empty.calls}, populated={populated.calls}"
    )


def test_non_logging_step_issues_no_collective_on_any_rank(recording_dist):
    """The skip gate is rank-uniform trainer state, so skipping entirely is safe — and must be total."""
    cb = MoEMetricsCallback(topk=1, log_every_n_steps=10)
    cb._counters = [torch.ones(4)]

    cb.on_step_end(args=None, state=_State(global_step=3), control=None)

    assert recording_dist.calls == [], "a non-logging step must issue no collective at all"
    assert torch.equal(cb._counters[0], torch.zeros(4)), "counters must still be zeroed each step"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
