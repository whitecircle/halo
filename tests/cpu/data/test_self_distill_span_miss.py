#!/usr/bin/env python
"""Completion-span misses in the self-distill label builder: warn, never crash, never train pad.

``build_completion_only_labels`` is shared by the SDPG text collator and by every VLM collator
(:class:`~src.data.collators.vlm.VLMDataCollator` passes the COLLATOR span policy through
``span_policy``). Two defects, both silent or fatal at collate time:

1. Under a policy that bounds a turn by the next marker start with no end-of-sequence fallback —
   exactly the VLM collator's — a row whose assistant turn carries no terminator resolves to marker
   starts and NO ends. Pairing them with ``zip(..., strict=True)`` raises ``ValueError`` on a
   legitimate row, and at collate time that is rank-local: the peers sit in the step's collectives
   until the NCCL watchdog.
2. A row that resolves to no span at all is all-masked with no signal — a zero-loss row, which is
   how a wrong ``assistant_message_template`` burns a whole run unnoticed.

Run: pytest tests/cpu/data/test_self_distill_span_miss.py
"""

import warnings

import pytest
import torch

from src.data.collators.vlm import VLMDataCollator
from src.data.spans import LABEL_IGNORE_INDEX, build_completion_only_labels, resolve_completion_spans

MARKER_IDS = [7, 7]
EOS_ID = 2
RESPONSE_TEMPLATE = "<|assistant|>"

# The policy under test is the collator's own, read off the class so a policy change lands here too.
COLLATOR_POLICY = VLMDataCollator._COMPLETION_SPAN_POLICY


class _FakeTokenizer:
    """Just enough tokenizer for the label builder: marker encoding, eos id, warning decode."""

    pad_token_id = 0
    eos_token_id = EOS_ID

    def encode(self, text, add_special_tokens=False):
        return list(MARKER_IDS)

    def decode(self, ids, **kwargs):
        return " ".join(str(int(token)) for token in ids)


def _labels(row: list[int], span_policy) -> torch.Tensor:
    input_ids = torch.tensor([row], dtype=torch.long)
    return build_completion_only_labels(
        input_ids,
        _FakeTokenizer(),
        RESPONSE_TEMPLATE,
        train_on_completions_only=True,
        attention_mask=torch.ones_like(input_ids),
        span_policy=span_policy,
    )


def test_terminatorless_row_under_collator_policy_warns_instead_of_crashing():
    """Marker present, terminator absent, collator policy → masked row + a warning, not a raise."""
    # The precondition the crash came from, pinned so this test cannot stop covering it: under this
    # policy the row resolves to a start with NO paired end.
    starts, ends = resolve_completion_spans([5, 7, 7, 9], MARKER_IDS, frozenset({EOS_ID}), **COLLATOR_POLICY)
    assert starts and not ends

    with pytest.warns(UserWarning, match="matched but no completion span has a terminator"):
        labels = _labels([5, 7, 7, 9], COLLATOR_POLICY)
    assert torch.equal(labels, torch.full_like(labels, LABEL_IGNORE_INDEX)), (
        "a row with no usable span must train nothing"
    )


def test_missing_marker_warns_instead_of_training_zero_tokens_silently():
    """No marker anywhere (a wrong assistant_message_template) must name the template in a warning."""
    with pytest.warns(UserWarning, match="Could not find response key"):
        labels = _labels([5, 9, 9, EOS_ID], None)
    assert torch.equal(labels, torch.full_like(labels, LABEL_IGNORE_INDEX))


def test_terminatorless_row_still_trains_under_the_self_distill_fallback():
    """The default (self-distill) policy falls back to end-of-sequence, so the SAME row keeps its
    completion — the miss guard must not mask rows that resolve fine."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a warning here would mean the guard fired on a good row
        labels = _labels([5, 7, 7, 9], None)
    # include_marker=False: the span starts after the marker and runs to the last real token.
    assert labels[0].tolist() == [LABEL_IGNORE_INDEX] * 3 + [9]


def test_terminated_row_under_collator_policy_keeps_its_span():
    """The early-out must not change a row that DOES terminate: it stays masked span-wise."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        labels = _labels([5, 7, 7, 9, EOS_ID], COLLATOR_POLICY)
    # include_marker=True: the span covers the marker through the terminator inclusive.
    assert labels[0].tolist() == [LABEL_IGNORE_INDEX, 7, 7, 9, EOS_ID]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
