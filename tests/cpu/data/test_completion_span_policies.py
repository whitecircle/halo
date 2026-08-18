#!/usr/bin/env python3
"""Pin the TWO completion-span policies expressible via resolve_completion_spans.

The collators and the self-distill label builder deliberately use DIFFERENT span semantics
(single-homed in resolve_completion_spans, NOT converged):

- collator policy (defaults): span starts AT the marker (marker tokens trained); a turn's
  terminator must precede the next marker start (else the turn is an empty ``-1`` span);
  ``eos_fallback_to_end`` is global (only when the sequence has NO terminator at all).
- self-distill policy (``include_marker=False, bound_by_next_start=False,
  eos_fallback_to_end=True``): span starts AFTER the marker; each span runs to the first
  terminator anywhere after it (consuming intervening markers); a terminator-less span falls
  back to the end of the row.

These tests run both policies on IDENTICAL rows and pin every divergence, plus pin
build_completion_only_labels against an independent reference implementation of the
self-distill policy — either policy drifting fails a test.

Usage:
    python tests/cpu/data/test_completion_span_policies.py
"""

import sys

import pytest
import torch

from src.data.spans import COLLATOR_SPAN_POLICY, build_completion_only_labels, resolve_completion_spans

EOS = 1
M1, M2 = 10, 11
MARKER = [M1, M2]
U, A = 20, 30
IMG = 77
IGNORE = -100


def _collator_spans(sequence, eos_fallback_to_end=False):
    return resolve_completion_spans(sequence, MARKER, frozenset({EOS}), eos_fallback_to_end=eos_fallback_to_end)


def _self_distill_spans(sequence):
    return resolve_completion_spans(
        sequence,
        MARKER,
        frozenset({EOS}),
        eos_fallback_to_end=True,
        include_marker=False,
        bound_by_next_start=False,
    )


def test_marker_inclusion_divergence():
    """Collator spans include the marker tokens; self-distill spans start after them."""
    row = [U, U, M1, M2, A, A, EOS, U, M1, M2, A, A, EOS]

    assert _collator_spans(row) == ([2, 8], [6, 12])
    assert _self_distill_spans(row) == ([4, 10], [6, 12])


def test_bounding_divergence_terminator_less_turn():
    """A turn missing its own EOS: collator yields an empty -1 span (never borrows the next
    turn's EOS); self-distill runs the span THROUGH the next marker to the first EOS anywhere,
    consuming that marker (one span instead of two)."""
    row = [M1, M2, A, M1, M2, A, EOS]

    assert _collator_spans(row) == ([0, 3], [-1, 6])
    assert _self_distill_spans(row) == ([2], [6])


def test_fallback_divergence_no_terminator_at_all():
    """No terminator in the row: collator (padded batches) yields no usable span unless the
    global fallback is on; self-distill always falls back to the row end per span."""
    row = [U, M1, M2, A, A]

    assert _collator_spans(row) == ([1], [])
    assert _collator_spans(row, eos_fallback_to_end=True) == ([1], [4])
    assert _self_distill_spans(row) == ([3], [4])


def test_fallback_scope_divergence_partial_terminators():
    """One turn has an EOS, another doesn't: the collator's fallback is GLOBAL (inert here, so
    the terminator-less turn stays -1); self-distill's is per-span (falls to the row end)."""
    row = [M1, M2, A, EOS, U, M1, M2, A, A]

    assert _collator_spans(row, eos_fallback_to_end=True) == ([0, 5], [3, -1])
    assert _self_distill_spans(row) == ([2, 7], [3, 8])


def _reference_self_distill_labels(input_ids: torch.Tensor, marker: list[int], eos_ids: set[int]) -> torch.Tensor:
    """Independent reference implementation of the self-distill span loop."""
    labels = torch.full_like(input_ids, IGNORE)
    marker_t = torch.tensor(marker)
    m = marker_t.numel()
    for i in range(input_ids.shape[0]):
        row = input_ids[i]
        j = 0
        while j < row.numel() - m + 1:
            if torch.equal(row[j : j + m], marker_t):
                start = j + m
                end = row.numel()
                for k in range(start, row.numel()):
                    if row[k].item() in eos_ids:
                        end = k + 1
                        break
                labels[i, start:end] = row[start:end]
                j = end
            else:
                j += 1
    return labels


class _MarkerTokenizer:
    pad_token_id = 0
    eos_token_id = EOS
    bos_token = None

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(MARKER)

    def decode(self, ids, **kwargs):
        # A no-span row is reported through warn_completion_span_miss, which decodes the offender.
        return " ".join(str(int(token)) for token in ids)


@pytest.mark.parametrize(
    "row",
    [
        [U, U, M1, M2, A, A, EOS, U, M1, M2, A, A, EOS],  # well-formed two turns
        [M1, M2, A, M1, M2, A, EOS],  # first turn terminator-less (marker consumed)
        [U, M1, M2, A, A],  # no terminator at all (fallback to row end)
        [M1, M2, EOS, U, M1, M2, A, EOS, U],  # empty completion + trailing user
        [U, U, U],  # no marker: everything stays masked
    ],
)
def test_build_completion_only_labels_matches_reference(row):
    """The shared span implementation reproduces the self-distill labels bit-for-bit (fails if
    the self-distill policy drifts toward the collator policy)."""
    input_ids = torch.tensor([row], dtype=torch.long)
    labels = build_completion_only_labels(
        input_ids,
        _MarkerTokenizer(),
        response_prompt_template="<marker>",
        train_on_completions_only=True,
        eos_token_ids=frozenset({EOS}),
    )
    expected = _reference_self_distill_labels(input_ids, MARKER, {EOS})
    assert torch.equal(labels, expected), f"row={row}: got {labels.tolist()}, expected {expected.tolist()}"


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_padded_terminator_less_span_never_trains_padding(padding_side):
    """A terminator-less final turn must not train the padding tail.

    The self-distill policy's ``eos_fallback_to_end`` ends such a span at the last position of the
    row; searched over the PADDED row that is a pad token, so every pad became a training target —
    silently undoing the attention-mask pad masking done a few lines earlier.
    """
    real = [U, M1, M2, A, A]  # marker present, NO terminator
    pad_len = 6
    pads = [_MarkerTokenizer.pad_token_id] * pad_len
    if padding_side == "right":
        row, mask = real + pads, [1] * len(real) + [0] * pad_len
    else:
        row, mask = pads + real, [0] * pad_len + [1] * len(real)

    labels = build_completion_only_labels(
        torch.tensor([row], dtype=torch.long),
        _MarkerTokenizer(),
        response_prompt_template="<marker>",
        train_on_completions_only=True,
        attention_mask=torch.tensor([mask], dtype=torch.long),
        eos_token_ids=frozenset({EOS}),
    )[0]

    trained_pads = [i for i, v in enumerate(labels.tolist()) if v != IGNORE and mask[i] == 0]
    assert not trained_pads, f"padding trained at {trained_pads}: {labels.tolist()}"
    # The real completion (after the marker, to the last real token) still trains.
    offset = 0 if padding_side == "right" else pad_len
    assert labels.tolist()[offset + 3 : offset + 5] == [A, A], labels.tolist()


def test_padded_terminated_span_is_unchanged_by_the_pad_guard():
    """The guard must not shorten a normally terminated span in a padded batch."""
    real = [U, M1, M2, A, A, EOS]
    row = real + [_MarkerTokenizer.pad_token_id] * 4
    mask = [1] * len(real) + [0] * 4
    labels = build_completion_only_labels(
        torch.tensor([row], dtype=torch.long),
        _MarkerTokenizer(),
        response_prompt_template="<marker>",
        train_on_completions_only=True,
        attention_mask=torch.tensor([mask], dtype=torch.long),
        eos_token_ids=frozenset({EOS}),
    )[0]
    assert labels.tolist() == [IGNORE, IGNORE, IGNORE, A, A, EOS] + [IGNORE] * 4, labels.tolist()


@pytest.mark.parametrize("span_policy", [None, COLLATOR_SPAN_POLICY], ids=["self_distill", "collator"])
def test_extra_ignore_ids_inside_a_span_stay_masked(span_policy):
    """An extra-ignore id (an image token on the VLM path) sitting INSIDE an assistant span must
    stay masked under both policies.

    The span refill copies from ``input_ids``, so extras masked before it are handed straight back
    as trainable targets and the masking survives only outside the spans — where the row was masked
    anyway. The offline bake applies them after the refill; both paths must agree.
    """
    row = [U, M1, M2, A, IMG, A, EOS]
    labels = build_completion_only_labels(
        torch.tensor([row], dtype=torch.long),
        _MarkerTokenizer(),
        response_prompt_template="<marker>",
        train_on_completions_only=True,
        extra_ignore_token_ids=(IMG,),
        eos_token_ids=frozenset({EOS}),
        span_policy=span_policy,
    )[0].tolist()

    assert labels[4] == IGNORE, f"image token inside the span was restored as a target: {labels}"
    # The rest of the completion still trains — the extras mask must not swallow the span.
    assert labels[3] == A and labels[5] == A, labels


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
