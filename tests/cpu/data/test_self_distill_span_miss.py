#!/usr/bin/env python
"""Completion-span misses in the self-distill label builder warn, and never crash or train pad. A row with marker
starts and no ends under the collator policy would raise in ``zip(..., strict=True)`` at collate time,
rank-local, hanging the peers; a row with no span is a silent zero-loss row."""

import warnings

import pytest
import torch

from halo.data.collators.vlm import VLMDataCollator
from halo.data.completion_spans import build_completion_only_labels, resolve_completion_spans
from halo.kernels.logprobs import LABEL_IGNORE_INDEX

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
    # The precondition a strict zip crashes on, pinned so this test cannot stop covering it: under this
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
    """The default policy falls back to end-of-sequence, so the same row keeps its completion."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a warning here would mean the guard fired on a good row
        labels = _labels([5, 7, 7, 9], None)
    # include_marker=False: the span starts after the marker and runs to the last real token.
    assert labels[0].tolist() == [LABEL_IGNORE_INDEX] * 3 + [9]


def test_terminated_row_under_collator_policy_keeps_its_span():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        labels = _labels([5, 7, 7, 9, EOS_ID], COLLATOR_POLICY)
    # include_marker=True: the span covers the marker through the terminator inclusive.
    assert labels[0].tolist() == [LABEL_IGNORE_INDEX, 7, 7, 9, EOS_ID]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
