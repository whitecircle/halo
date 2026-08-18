#!/usr/bin/env python3
"""Baked-labels correctness across the offline/runtime seam.

1. Flattening completion mask must detect spans in INPUT_IDS, not labels — precomputed labels
   with the response marker baked to -100 match nothing, silently zeroing the loss.
2. Offline preprocessing must bake the COLLATOR span policy — baked labels have to equal what
   :func:`mask_batch_to_completion_spans` (the runtime collator source of truth) produces, else
   preprocessed and runtime training compute different losses on the same rows.

Usage:
    python tests/cpu/data/test_baked_labels_policy.py
"""

import sys
from unittest.mock import MagicMock

import pytest
import torch

from src.data.collators.packing import DataCollatorWithFlatteningAndCompletionMask
from src.data.pipeline.preprocessing import _completion_only_labels
from src.data.spans import COLLATOR_SPAN_POLICY, build_completion_only_labels, mask_batch_to_completion_spans

PAD = 0
EOS = 1
IGNORE = -100
RESP_1, RESP_2 = 10, 11
IMAGE_TOK = 77
TEMPLATE_IDS = [RESP_1, RESP_2]


def make_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.pad_token_id = PAD
    tok.eos_token_id = EOS
    tok.padding_side = "right"
    tok.encode.return_value = TEMPLATE_IDS
    return tok


def test_flattening_mask_finds_spans_in_input_ids_when_labels_masked():
    """Labels with the marker baked to -100 (offline-masked rows) must still yield spans: a span
    search over labels finds nothing there and zeroes the loss silently."""
    collator = DataCollatorWithFlatteningAndCompletionMask(
        tokenizer=make_tokenizer(),
        response_prompt_template=TEMPLATE_IDS,
        return_flash_attn_kwargs=True,
    )
    features = [
        {
            "input_ids": [20, RESP_1, RESP_2, 30, EOS],
            # Marker + prompt masked in the baked labels; completion tokens kept.
            "labels": [IGNORE, IGNORE, IGNORE, 30, EOS],
        }
    ]
    batch = collator(features)
    labels = batch["labels"][0].tolist()
    assert labels[3] == 30 and labels[4] == EOS, (
        f"completion span must stay trained even when the marker is baked to -100 in labels, got {labels}"
    )


def test_flattening_mask_unchanged_when_labels_are_input_copy():
    """Sanity: labels==input_ids (the non-baked path) behaves exactly as before."""
    collator = DataCollatorWithFlatteningAndCompletionMask(
        tokenizer=make_tokenizer(),
        response_prompt_template=TEMPLATE_IDS,
        return_flash_attn_kwargs=True,
    )
    ids = [20, RESP_1, RESP_2, 30, EOS, 21]
    batch = collator([{"input_ids": ids}])
    labels = batch["labels"][0].tolist()
    # Span = marker..EOS inclusive (collator policy); everything else masked.
    assert labels == [IGNORE, RESP_1, RESP_2, 30, EOS, IGNORE]


def _runtime_collator_labels(input_ids: list[int]) -> list[int]:
    """What the runtime collators produce for one unpadded row (the single source of truth)."""
    ids = torch.tensor([input_ids])
    batch = mask_batch_to_completion_spans(
        {"input_ids": ids, "labels": ids.clone()},
        TEMPLATE_IDS,
        frozenset({EOS}),
        ignore_index=IGNORE,
        train_on_last_assistant_only=False,
        response_prompt_template="<resp>",
        tokenizer=None,
    )
    return batch["labels"][0].tolist()


@pytest.mark.parametrize(
    "input_ids",
    [
        [20, RESP_1, RESP_2, 30, EOS, 21, RESP_1, RESP_2, 31, EOS],
        # Policy-discriminating: the collator TRAINS the marker tokens.
        [20, RESP_1, RESP_2, 30, 31, EOS, 40],
        # Terminator-less final turn: a no-op span, never a fallback that unmasks to the row end.
        [20, RESP_1, RESP_2, 30, EOS, 21, RESP_1, RESP_2, 50],
        # No marker at all: everything masked.
        [20, 21, 22, EOS],
    ],
)
def test_offline_baked_labels_match_runtime_collator(input_ids):
    baked = _completion_only_labels(
        input_ids,
        make_tokenizer(),
        assistant_template="<resp>",
        response_token_ids=[RESP_1, RESP_2],
        eos_token_ids=frozenset({EOS}),
    )
    assert baked == _runtime_collator_labels(input_ids), (
        "offline-baked labels must be identical to the runtime collator's completion mask"
    )


def test_offline_baked_labels_marker_trained():
    """Direct pin of the policy switch: baked labels train the marker tokens (collator policy)
    rather than masking them (the self-distill baking policy)."""
    input_ids = [20, RESP_1, RESP_2, 30, EOS]
    baked = _completion_only_labels(
        input_ids,
        make_tokenizer(),
        assistant_template="<resp>",
        response_token_ids=[RESP_1, RESP_2],
        eos_token_ids=frozenset({EOS}),
    )
    assert baked == [IGNORE, RESP_1, RESP_2, 30, EOS]


def test_offline_baked_labels_extra_ignore_ids_masked():
    """extra_ignore_token_ids (image tokens on the VLM path) stay masked on top of the policy."""
    input_ids = [20, RESP_1, RESP_2, IMAGE_TOK, 30, EOS]
    baked = _completion_only_labels(
        input_ids,
        make_tokenizer(),
        assistant_template="<resp>",
        response_token_ids=[RESP_1, RESP_2],
        extra_ignore_token_ids=(IMAGE_TOK,),
        eos_token_ids=frozenset({EOS}),
    )
    assert baked == [IGNORE, RESP_1, RESP_2, IGNORE, 30, EOS]


@pytest.mark.parametrize(
    "input_ids",
    [
        # Image token INSIDE the completion span — the position a span refill hands back.
        [20, RESP_1, RESP_2, IMAGE_TOK, 30, EOS],
        # Image tokens both outside every span (user turn) and inside one.
        [IMAGE_TOK, 20, RESP_1, RESP_2, 30, IMAGE_TOK, EOS, 21],
    ],
)
def test_runtime_label_builder_matches_offline_bake_with_extras(input_ids):
    """Same row, same policy, same extras: the runtime builder and the offline bake agree.

    Both apply ``extra_ignore_token_ids`` after the span refill; applying them first re-trains every
    extra id that falls inside a completion span, so the two paths would compute different losses on
    the same rows.
    """
    runtime = build_completion_only_labels(
        torch.tensor([input_ids]),
        make_tokenizer(),
        response_prompt_template="<resp>",
        train_on_completions_only=True,
        extra_ignore_token_ids=(IMAGE_TOK,),
        eos_token_ids=frozenset({EOS}),
        span_policy=COLLATOR_SPAN_POLICY,
    )[0].tolist()
    baked = _completion_only_labels(
        input_ids,
        make_tokenizer(),
        assistant_template="<resp>",
        response_token_ids=TEMPLATE_IDS,
        extra_ignore_token_ids=(IMAGE_TOK,),
        eos_token_ids=frozenset({EOS}),
    )
    assert runtime == baked, f"runtime {runtime} != offline {baked}"
    assert IMAGE_TOK not in runtime, f"image token trained as a target: {runtime}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
