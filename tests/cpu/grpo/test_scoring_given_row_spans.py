#!/usr/bin/env python
"""A no-grad pass under routing replay reads its round's row spans once: the scorer takes the table the replay layout was
built from instead of copying the same mask to the host again, and scores bit for bit what it scores on its own."""

import contextlib

import pytest
import torch

from halo.trainers.grpo import logprob_scoring
from halo.trainers.grpo.logprob_scoring import dense_row_spans
from tests.common.length_deal import ragged_round, scoring_host, single_process_accelerator, stacked, tiny_llama


class _Injector:
    """What the pass asks of a routing injector: a capture scope that records the spans it was given."""

    def __init__(self) -> None:
        self.row_spans: list = []

    @contextlib.contextmanager
    def capture(self, rows, width, row_spans, row_order):
        self.row_spans.append(row_spans)
        yield []


def _round():
    return ragged_round([2, 4, 3, 5], [6, 3, 7, 4], vocab=64, seed=1)


def _recompute(monkeypatch, injector):
    calls = []
    read = logprob_scoring.row_spans_on_device
    monkeypatch.setattr(logprob_scoring, "row_spans_on_device", lambda *a, **kw: (calls.append(1), read(*a, **kw))[1])
    host = scoring_host(tiny_llama(64, 0), None, chunks=4, accelerator=single_process_accelerator())
    host._routing_injector = injector
    ids, mask, keep = stacked(_round())
    with torch.no_grad():
        logps, _ = host._recompute_logps_and_routing_masks(ids, mask, keep, None, "train")
    return logps, len(calls), mask


def test_a_replay_pass_copies_the_mask_to_the_host_once_and_scores_the_same(monkeypatch):
    plain, plain_reads, _ = _recompute(monkeypatch, None)
    injector = _Injector()
    replayed, reads, mask = _recompute(monkeypatch, injector)
    assert plain_reads == 1, "the scorer reads its own mask once"
    assert reads == 1, "the replay layout and the scorer read the mask twice"
    assert torch.equal(replayed, plain)
    assert injector.row_spans == [dense_row_spans(mask)[0]]


def test_spans_for_another_batch_raise_instead_of_slicing_the_wrong_tokens():
    host = scoring_host(tiny_llama(64, 0), None, chunks=4, accelerator=single_process_accelerator())
    ids, mask, keep = stacked(_round())
    short = dense_row_spans(mask[:2], keep)
    with pytest.raises(RuntimeError, match="describe another mask"), host._scoring_with_row_spans(short):
        host._row_spans_for(mask, keep)
    assert host._given_row_spans is None


def test_the_given_spans_are_released_when_the_block_raises():
    host = scoring_host(tiny_llama(64, 0), None, chunks=4, accelerator=single_process_accelerator())
    _, mask, keep = stacked(_round())
    with pytest.raises(KeyError), host._scoring_with_row_spans(dense_row_spans(mask, keep)):
        raise KeyError("block failed")
    assert host._row_spans_for(mask, keep)[0] == dense_row_spans(mask, keep)[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
