#!/usr/bin/env python3
"""
Tests for data collators: completion masking, packing, flattening.

Focuses on verifying train_on_last_assistant_only behavior across all collator
code paths (packed vs non-packed, standard vs flattening).

Usage:
    python tests/data/test_collators.py
"""

import sys
import warnings
from unittest.mock import MagicMock

import pytest
import torch

from src.data.collators.completions_only import DataCollatorForCompletionOnlyLM
from src.data.collators.packing import (
    DataCollatorForCompletionOnlyLMWithPacking,
    DataCollatorWithFlattening,
    DataCollatorWithFlatteningAndCompletionMask,
    DataCollatorWithPacking,
)
from src.data.spans import (
    build_completion_only_labels,
    filter_eos_after_responses,
    find_terminator_positions,
    resolve_eos_token_ids,
    tokenize_response_template,
)

PAD = 0
EOS = 1
RESP_1 = 10
RESP_2 = 11
RESPONSE_TEMPLATE_IDS = [RESP_1, RESP_2]

USER_A, USER_B, USER_C = 20, 21, 22
ASST_A, ASST_B, ASST_C, ASST_D = 30, 31, 32, 33
IGNORE = -100


def make_tokenizer(
    pad_token_id: int = PAD,
    eos_token_id: int = EOS,
    response_template_ids: list[int] | None = None,
) -> MagicMock:
    """Create a mock tokenizer with the minimal interface collators need."""
    tok = MagicMock()
    tok.pad_token_id = pad_token_id
    tok.eos_token_id = eos_token_id
    tok.padding_side = "right"
    tok.model_max_length = 4096

    if response_template_ids is None:
        response_template_ids = RESPONSE_TEMPLATE_IDS
    tok.encode.return_value = response_template_ids

    def _pad(features, **kwargs):
        max_len = max(len(f["input_ids"]) for f in features)
        padded = []
        for f in features:
            ids = list(f["input_ids"])
            labs = list(f.get("labels", ids[:]))
            mask = list(f.get("attention_mask", [1] * len(ids)))
            pad_len = max_len - len(ids)
            ids += [pad_token_id] * pad_len
            labs += [IGNORE] * pad_len
            mask += [0] * pad_len
            padded.append(
                {
                    "input_ids": ids,
                    "labels": labs,
                    "attention_mask": mask,
                }
            )
        return {
            "input_ids": torch.tensor([p["input_ids"] for p in padded]),
            "labels": torch.tensor([p["labels"] for p in padded]),
            "attention_mask": torch.tensor([p["attention_mask"] for p in padded]),
        }

    tok.pad = _pad
    return tok


def make_two_turn_ids():
    """Create input_ids for a 2-turn conversation.

    Structure:
      [USER_A, USER_B, RESP_1, RESP_2, ASST_A, ASST_B, EOS,
       USER_C, RESP_1, RESP_2, ASST_C, ASST_D, EOS]

    Turn 1 assistant tokens: ASST_A, ASST_B (indices 4-5)
    Turn 2 assistant tokens: ASST_C, ASST_D (indices 10-11)
    """
    return [
        USER_A,
        USER_B,
        RESP_1,
        RESP_2,
        ASST_A,
        ASST_B,
        EOS,
        USER_C,
        RESP_1,
        RESP_2,
        ASST_C,
        ASST_D,
        EOS,
    ]


def get_unmasked_positions(labels):
    """Return sorted list of positions where label != -100."""
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    return sorted(i for i, v in enumerate(labels) if v != IGNORE)


def test_completion_only_all_turns():
    """All assistant turns should be unmasked when train_on_last_assistant_only=False."""
    tok = make_tokenizer()
    collator = DataCollatorForCompletionOnlyLM(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
        train_on_last_assistant_only=False,
    )

    ids = make_two_turn_ids()
    examples = [{"input_ids": ids, "attention_mask": [1] * len(ids)}]
    batch = collator.torch_call(examples)

    unmasked = get_unmasked_positions(batch["labels"][0])
    assert 4 in unmasked, f"Turn 1 assistant token missing: {unmasked}"
    assert 5 in unmasked, f"Turn 1 assistant token missing: {unmasked}"
    assert 10 in unmasked, f"Turn 2 assistant token missing: {unmasked}"
    assert 11 in unmasked, f"Turn 2 assistant token missing: {unmasked}"
    assert 0 not in unmasked, f"User token should be masked: {unmasked}"
    assert 1 not in unmasked, f"User token should be masked: {unmasked}"
    print("  PASS: test_completion_only_all_turns")


def test_completion_only_last_turn_only():
    """Only last assistant turn should be unmasked when train_on_last_assistant_only=True."""
    tok = make_tokenizer()
    collator = DataCollatorForCompletionOnlyLM(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
        train_on_last_assistant_only=True,
    )

    ids = make_two_turn_ids()
    examples = [{"input_ids": ids, "attention_mask": [1] * len(ids)}]
    batch = collator.torch_call(examples)

    unmasked = get_unmasked_positions(batch["labels"][0])
    assert 4 not in unmasked, f"Turn 1 should be masked: {unmasked}"
    assert 5 not in unmasked, f"Turn 1 should be masked: {unmasked}"
    assert 10 in unmasked, f"Turn 2 assistant token missing: {unmasked}"
    assert 11 in unmasked, f"Turn 2 assistant token missing: {unmasked}"
    print("  PASS: test_completion_only_last_turn_only")


def test_packing_completion_packed_all_turns():
    """Packed data: all assistant turns unmasked when train_on_last_assistant_only=False."""
    tok = make_tokenizer()
    collator = DataCollatorForCompletionOnlyLMWithPacking(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
        train_on_last_assistant_only=False,
    )

    conv1 = make_two_turn_ids()
    conv2 = [USER_A, RESP_1, RESP_2, ASST_A, EOS]
    packed_ids = conv1 + conv2
    seq_lengths = [len(conv1), len(conv2)]

    examples = [
        {
            "input_ids": packed_ids,
            "attention_mask": [1] * len(packed_ids),
            "seq_lengths": seq_lengths,
        }
    ]
    batch = collator.torch_call(examples)

    unmasked = get_unmasked_positions(batch["labels"][0])
    assert 4 in unmasked, f"Conv1 turn 1 missing: {unmasked}"
    assert 10 in unmasked, f"Conv1 turn 2 missing: {unmasked}"
    offset = len(conv1)
    assert (offset + 3) in unmasked, f"Conv2 assistant missing: {unmasked}"
    print("  PASS: test_packing_completion_packed_all_turns")


def test_packing_completion_packed_last_turn_only():
    """Packed data: only last assistant per conversation when train_on_last_assistant_only=True."""
    tok = make_tokenizer()
    collator = DataCollatorForCompletionOnlyLMWithPacking(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
        train_on_last_assistant_only=True,
    )

    conv1 = make_two_turn_ids()
    conv2 = [USER_A, RESP_1, RESP_2, ASST_A, EOS]
    packed_ids = conv1 + conv2
    seq_lengths = [len(conv1), len(conv2)]

    examples = [
        {
            "input_ids": packed_ids,
            "attention_mask": [1] * len(packed_ids),
            "seq_lengths": seq_lengths,
        }
    ]
    batch = collator.torch_call(examples)

    unmasked = get_unmasked_positions(batch["labels"][0])
    assert 4 not in unmasked, f"Conv1 turn 1 should be masked: {unmasked}"
    assert 5 not in unmasked, f"Conv1 turn 1 should be masked: {unmasked}"
    assert 10 in unmasked, f"Conv1 turn 2 missing: {unmasked}"
    assert 11 in unmasked, f"Conv1 turn 2 missing: {unmasked}"
    offset = len(conv1)
    assert (offset + 3) in unmasked, f"Conv2 assistant missing: {unmasked}"
    print("  PASS: test_packing_completion_packed_last_turn_only")


def test_packing_completion_nonpacked_last_turn_only():
    """Non-packed path: only last turn when train_on_last_assistant_only=True.

    This is the bug that was fixed — _apply_completion_mask_standard was missing
    the train_on_last_assistant_only check.
    """
    tok = make_tokenizer()
    collator = DataCollatorForCompletionOnlyLMWithPacking(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
        train_on_last_assistant_only=True,
    )

    ids = make_two_turn_ids()
    examples = [{"input_ids": ids, "attention_mask": [1] * len(ids)}]
    batch = collator.torch_call(examples)

    unmasked = get_unmasked_positions(batch["labels"][0])
    assert 4 not in unmasked, f"Turn 1 should be masked (BUG if failing): {unmasked}"
    assert 5 not in unmasked, f"Turn 1 should be masked (BUG if failing): {unmasked}"
    assert 10 in unmasked, f"Turn 2 assistant missing: {unmasked}"
    assert 11 in unmasked, f"Turn 2 assistant missing: {unmasked}"
    print("  PASS: test_packing_completion_nonpacked_last_turn_only")


def test_packing_completion_nonpacked_pad_equals_eos():
    """Non-packed path with pad_token_id == eos_token_id: the assistant tokens AND the
    turn-ending EOS must still be trained — not masked away to a zero-loss instance.

    The parent LM collator masks pad positions (which share the EOS id here) to -100 first, so
    ``_apply_completion_mask_standard`` searching the *labels* tensor never finds the EOS: the
    "missing template/EOS" branch fires and the whole sequence is masked (zero tokens trained).
    Detection and the unmasked copy must come from ``input_ids``, which is immune to that masking.
    """
    tok = make_tokenizer(pad_token_id=EOS, eos_token_id=EOS)
    collator = DataCollatorForCompletionOnlyLMWithPacking(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
        train_on_last_assistant_only=False,
    )

    ids = [USER_A, RESP_1, RESP_2, ASST_A, ASST_B, EOS]
    examples = [{"input_ids": ids, "attention_mask": [1] * len(ids)}]
    batch = collator.torch_call(examples)

    unmasked = get_unmasked_positions(batch["labels"][0])
    assert unmasked, f"entire instance was masked — pad==eos bug: {unmasked}"
    assert 3 in unmasked, f"assistant token ASST_A masked: {unmasked}"
    assert 4 in unmasked, f"assistant token ASST_B masked: {unmasked}"
    assert 5 in unmasked, f"turn-ending EOS masked (pad==eos): {unmasked}"
    assert 0 not in unmasked, f"user token should be masked: {unmasked}"
    print("  PASS: test_packing_completion_nonpacked_pad_equals_eos")


def test_packing_completion_packed_pad_equals_eos():
    """PACKED path with pad_token_id == eos_token_id: each document's turn-ending EOS must be
    trained, not masked away.

    ``_mask_sequence`` (the packed path) detects the template/EOS on input_ids and must
    copy the unmasked span from input_ids too, never from the parent-masked ``labels``. When
    pad_token_id == eos_token_id the parent ``DataCollatorForLanguageModeling`` masks every
    EOS-valued position (incl. the real turn-ending EOS) to -100, so copying from labels silently
    drops the EOS — teaching the model not to stop at turn boundaries. The non-packed paths copy
    from input_ids; the packed path must too. The discriminating assertions below are the EOS
    positions (5 and 10): copied from labels they are masked; assistant content tokens (not
    EOS-valued) stay unmasked either way.
    """
    tok = make_tokenizer(pad_token_id=EOS, eos_token_id=EOS)
    collator = DataCollatorForCompletionOnlyLMWithPacking(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
        train_on_last_assistant_only=False,
    )

    doc1 = [USER_A, RESP_1, RESP_2, ASST_A, ASST_B, EOS]  # EOS@5
    doc2 = [USER_C, RESP_1, RESP_2, ASST_C, EOS]  # EOS@10
    packed_ids = doc1 + doc2
    seq_lengths = [len(doc1), len(doc2)]
    examples = [
        {
            "input_ids": packed_ids,
            "attention_mask": [1] * len(packed_ids),
            "seq_lengths": seq_lengths,
        }
    ]
    batch = collator.torch_call(examples)

    unmasked = get_unmasked_positions(batch["labels"][0])
    assert 5 in unmasked, f"doc1 turn-ending EOS masked (pad==eos packed bug): {unmasked}"
    assert 10 in unmasked, f"doc2 turn-ending EOS masked (pad==eos packed bug): {unmasked}"
    assert 3 in unmasked and 4 in unmasked, f"doc1 assistant tokens masked: {unmasked}"
    assert 9 in unmasked, f"doc2 assistant token masked: {unmasked}"
    assert 0 not in unmasked, f"doc1 boundary token should be masked: {unmasked}"
    assert 6 not in unmasked, f"doc2 boundary token should be masked: {unmasked}"
    print("  PASS: test_packing_completion_packed_pad_equals_eos")


def test_packing_completion_consistency():
    """Packed and non-packed paths must agree on which assistant CONTENT tokens are masked.

    Note: the packed path (_mask_sequence) starts unmasking AFTER the response template,
    while the non-packed path (_apply_completion_mask_standard) includes the template.
    This is a pre-existing semantic difference. Here we only check that the actual
    assistant content tokens (not template tokens) are consistently masked/unmasked
    when train_on_last_assistant_only is toggled.
    """
    tok = make_tokenizer()

    turn1_content = {4, 5}
    turn2_content = {10, 11}

    for last_only in [True, False]:
        collator = DataCollatorForCompletionOnlyLMWithPacking(
            response_prompt_template=RESPONSE_TEMPLATE_IDS,
            tokenizer=tok,
            train_on_last_assistant_only=last_only,
        )

        ids = make_two_turn_ids()

        examples_nonpacked = [{"input_ids": ids[:], "attention_mask": [1] * len(ids)}]
        batch_nonpacked = collator.torch_call(examples_nonpacked)
        unmasked_nonpacked = set(get_unmasked_positions(batch_nonpacked["labels"][0]))

        examples_packed = [
            {
                "input_ids": ids[:],
                "attention_mask": [1] * len(ids),
                "seq_lengths": [len(ids)],
            }
        ]
        batch_packed = collator.torch_call(examples_packed)
        unmasked_packed = set(get_unmasked_positions(batch_packed["labels"][0]))

        if last_only:
            assert turn1_content.isdisjoint(unmasked_nonpacked), (
                f"non-packed: turn 1 should be masked with last_only=True: {unmasked_nonpacked}"
            )
            assert turn1_content.isdisjoint(unmasked_packed), (
                f"packed: turn 1 should be masked with last_only=True: {unmasked_packed}"
            )
            assert turn2_content.issubset(unmasked_nonpacked), (
                f"non-packed: turn 2 missing with last_only=True: {unmasked_nonpacked}"
            )
            assert turn2_content.issubset(unmasked_packed), (
                f"packed: turn 2 missing with last_only=True: {unmasked_packed}"
            )
        else:
            assert turn1_content.issubset(unmasked_nonpacked), (
                f"non-packed: turn 1 missing with last_only=False: {unmasked_nonpacked}"
            )
            assert turn1_content.issubset(unmasked_packed), (
                f"packed: turn 1 missing with last_only=False: {unmasked_packed}"
            )
            assert turn2_content.issubset(unmasked_nonpacked), (
                f"non-packed: turn 2 missing with last_only=False: {unmasked_nonpacked}"
            )
            assert turn2_content.issubset(unmasked_packed), (
                f"packed: turn 2 missing with last_only=False: {unmasked_packed}"
            )

    print("  PASS: test_packing_completion_consistency")


def test_packing_position_ids():
    """Packed data should have position_ids that reset per conversation."""
    tok = make_tokenizer()
    collator = DataCollatorWithPacking(tokenizer=tok)

    conv1 = [USER_A, USER_B, ASST_A, EOS]
    conv2 = [USER_C, ASST_B, ASST_C, ASST_D, EOS]
    packed_ids = conv1 + conv2

    examples = [
        {
            "input_ids": packed_ids,
            "attention_mask": [1] * len(packed_ids),
            "seq_lengths": [len(conv1), len(conv2)],
        }
    ]
    batch = collator.torch_call(examples)

    pos_ids = batch["position_ids"][0].tolist()
    assert pos_ids[:4] == [0, 1, 2, 3], f"Conv1 pos_ids wrong: {pos_ids[:4]}"
    assert pos_ids[4:9] == [0, 1, 2, 3, 4], f"Conv2 pos_ids wrong: {pos_ids[4:9]}"
    print("  PASS: test_packing_position_ids")


def test_packing_nonpacked_no_position_ids():
    """Non-packed data should NOT have position_ids (model computes them internally)."""
    tok = make_tokenizer()
    collator = DataCollatorWithPacking(tokenizer=tok)

    ids = [USER_A, USER_B, ASST_A, EOS]
    examples = [{"input_ids": ids, "attention_mask": [1] * len(ids)}]
    batch = collator.torch_call(examples)

    assert "position_ids" not in batch, (
        f"Non-packed data should not have position_ids, got: {batch.get('position_ids')}"
    )
    print("  PASS: test_packing_nonpacked_no_position_ids")


def test_base_packing_collator_restores_eos_labels_when_pad_equals_eos():
    """Base DataCollatorWithPacking, NON-packed raw-labels batch, pad == eos: the parent LM collator
    masks every pad-valued label — including the real turn-ending EOS — so the torch_call wiring
    through restore_eos_labels_when_pad_equals_eos must bring it back at real-token positions while
    true padding stays masked. Drop-the-branch regression, distinct from the completion-masking and
    CP-collator pad==eos tests."""
    tok = make_tokenizer(pad_token_id=EOS, eos_token_id=EOS)
    collator = DataCollatorWithPacking(tokenizer=tok)

    examples = [
        {"input_ids": [USER_A, ASST_A, EOS], "attention_mask": [1, 1, 1]},
        {"input_ids": [USER_B, ASST_B, ASST_C, EOS], "attention_mask": [1, 1, 1, 1]},
    ]
    batch = collator.torch_call(examples)

    assert batch["labels"][0].tolist() == [USER_A, ASST_A, EOS, IGNORE], (
        f"row 0 must train its real EOS and keep the pad masked, got {batch['labels'][0].tolist()}"
    )
    assert batch["labels"][1].tolist() == [USER_B, ASST_B, ASST_C, EOS]
    print("  PASS: test_base_packing_collator_restores_eos_labels_when_pad_equals_eos")


def test_flattening_basic():
    """Basic flattening: multiple examples concatenated, first label gets separator."""
    tok = make_tokenizer()
    collator = DataCollatorWithFlattening(tokenizer=tok)

    ex1 = {"input_ids": [USER_A, ASST_A, EOS], "labels": [USER_A, ASST_A, EOS]}
    ex2 = {"input_ids": [USER_B, ASST_B, EOS], "labels": [USER_B, ASST_B, EOS]}
    batch = collator([ex1, ex2])

    ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()
    assert ids == [USER_A, ASST_A, EOS, USER_B, ASST_B, EOS], f"Wrong ids: {ids}"
    assert labels[0] == IGNORE, f"First label should be separator: {labels}"
    assert labels[3] == IGNORE, f"Second sequence first label should be separator: {labels}"
    assert labels[1] == ASST_A, f"Wrong label at 1: {labels}"
    assert labels[4] == ASST_B, f"Wrong label at 4: {labels}"
    print("  PASS: test_flattening_basic")


def test_flattening_position_ids():
    """Flattening: position_ids should reset per sequence."""
    tok = make_tokenizer()
    collator = DataCollatorWithFlattening(tokenizer=tok)

    ex1 = {"input_ids": [USER_A, ASST_A], "labels": [USER_A, ASST_A]}
    ex2 = {"input_ids": [USER_B, ASST_B, EOS], "labels": [USER_B, ASST_B, EOS]}
    batch = collator([ex1, ex2])

    pos_ids = batch["position_ids"][0].tolist()
    assert pos_ids == [0, 1, 0, 1, 2], f"Wrong pos_ids: {pos_ids}"
    print("  PASS: test_flattening_position_ids")


def test_flattening_cu_seq_lens():
    """Flattening: cu_seq_lens should track cumulative lengths."""
    tok = make_tokenizer()
    collator = DataCollatorWithFlattening(tokenizer=tok, return_flash_attn_kwargs=True)

    ex1 = {"input_ids": [USER_A, ASST_A], "labels": [USER_A, ASST_A]}
    ex2 = {"input_ids": [USER_B, ASST_B, EOS], "labels": [USER_B, ASST_B, EOS]}
    batch = collator([ex1, ex2])

    cu = batch["cu_seq_lens_q"].tolist()
    assert cu == [0, 2, 5], f"Wrong cu_seq_lens: {cu}"
    assert batch["max_length_q"] == 3, f"Wrong max_length: {batch['max_length_q']}"
    print("  PASS: test_flattening_cu_seq_lens")


def test_packing_boundary_labels_masked():
    """Packed data: labels at position_ids==0 must be -100 to prevent cross-doc predictions."""
    tok = make_tokenizer()
    collator = DataCollatorWithPacking(tokenizer=tok)

    conv1 = [USER_A, USER_B, ASST_A, EOS]
    conv2 = [USER_C, ASST_B, ASST_C, ASST_D, EOS]
    packed_ids = conv1 + conv2

    examples = [
        {
            "input_ids": packed_ids,
            "attention_mask": [1] * len(packed_ids),
            "seq_lengths": [len(conv1), len(conv2)],
        }
    ]
    batch = collator.torch_call(examples)

    labels = batch["labels"][0].tolist()
    pos_ids = batch["position_ids"][0].tolist()

    for i, (pos, lab) in enumerate(zip(pos_ids, labels, strict=False)):
        if pos == 0:
            assert lab == IGNORE, (
                f"Label at boundary position {i} should be -100, got {lab}. pos_ids={pos_ids}, labels={labels}"
            )

    assert labels[0] == IGNORE, f"First doc boundary not masked: {labels}"
    assert labels[4] == IGNORE, f"Second doc boundary not masked: {labels}"
    assert labels[1] == USER_B, f"Non-boundary label wrong at 1: {labels}"
    assert labels[5] == ASST_B, f"Non-boundary label wrong at 5: {labels}"
    print("  PASS: test_packing_boundary_labels_masked")


def test_packing_boundary_labels_with_padding():
    """Boundary masking works with different-length examples (padding in batch).

    A B>1 packed batch flattens to [1, total real tokens]: example 1 occupies the row directly
    after example 0, its pad tail dropped by the flatten.
    """
    tok = make_tokenizer()
    collator = DataCollatorWithPacking(tokenizer=tok)

    conv1a = [USER_A, USER_B, ASST_A, EOS]
    conv1b = [USER_C, ASST_B, ASST_C, ASST_D, EOS]
    conv2 = [USER_A, ASST_A, ASST_B, EOS]

    examples = [
        {
            "input_ids": conv1a + conv1b,
            "attention_mask": [1] * (len(conv1a) + len(conv1b)),
            "seq_lengths": [len(conv1a), len(conv1b)],
        },
        {
            "input_ids": conv2,
            "attention_mask": [1] * len(conv2),
            "seq_lengths": [len(conv2)],
        },
    ]
    batch = collator.torch_call(examples)

    assert batch["labels"].shape == (1, 13), f"pads must be dropped from the flatten: {batch['labels'].shape}"
    labels = batch["labels"][0].tolist()
    positions = batch["position_ids"][0].tolist()

    assert labels[0] == IGNORE, f"Ex0 first doc boundary not masked: {labels}"
    assert labels[4] == IGNORE, f"Ex0 second doc boundary not masked: {labels}"

    off = 9
    assert labels[off] == IGNORE, f"Ex1 doc boundary not masked: {labels}"
    assert positions[off : off + len(conv2)] == [0, 1, 2, 3], f"Ex1 positions wrong: {positions}"
    assert PAD not in batch["input_ids"][0].tolist(), f"pads survived the flatten: {batch['input_ids'][0].tolist()}"
    print("  PASS: test_packing_boundary_labels_with_padding")


def test_packing_completion_boundary_labels_masked():
    """Completion-only packing: boundaries must be -100 even in response regions."""
    tok = make_tokenizer()
    collator = DataCollatorForCompletionOnlyLMWithPacking(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
        train_on_last_assistant_only=False,
    )

    conv1 = [USER_A, RESP_1, RESP_2, ASST_A, EOS]
    conv2 = [RESP_1, RESP_2, ASST_B, ASST_C, EOS]
    packed_ids = conv1 + conv2

    examples = [
        {
            "input_ids": packed_ids,
            "attention_mask": [1] * len(packed_ids),
            "seq_lengths": [len(conv1), len(conv2)],
        }
    ]
    batch = collator.torch_call(examples)

    labels = batch["labels"][0].tolist()
    assert labels[0] == IGNORE, f"First doc boundary not masked: {labels}"
    # Index 5 is doc2's start AND a response-template token — boundary masking must still win.
    assert labels[5] == IGNORE, f"Second doc boundary not masked despite being response token: {labels}"
    print("  PASS: test_packing_completion_boundary_labels_masked")


def test_flattening_boundary_labels_masked():
    """Flattening collator: first label of each sequence must be separator (-100)."""
    tok = make_tokenizer()
    collator = DataCollatorWithFlattening(tokenizer=tok)

    ex1 = {"input_ids": [USER_A, ASST_A, EOS], "labels": [USER_A, ASST_A, EOS]}
    ex2 = {"input_ids": [USER_B, ASST_B, EOS], "labels": [USER_B, ASST_B, EOS]}
    ex3 = {"input_ids": [USER_C, ASST_C, EOS], "labels": [USER_C, ASST_C, EOS]}
    batch = collator([ex1, ex2, ex3])

    labels = batch["labels"][0].tolist()
    assert labels[0] == IGNORE, f"Ex1 boundary not masked: {labels}"
    assert labels[3] == IGNORE, f"Ex2 boundary not masked: {labels}"
    assert labels[6] == IGNORE, f"Ex3 boundary not masked: {labels}"
    assert labels[1] == ASST_A, f"Non-boundary label wrong at 1: {labels}"
    assert labels[4] == ASST_B, f"Non-boundary label wrong at 4: {labels}"
    assert labels[7] == ASST_C, f"Non-boundary label wrong at 7: {labels}"
    print("  PASS: test_flattening_boundary_labels_masked")


def test_flattening_completion_boundary_labels_masked():
    """Flattening + completion masking: boundaries must be -100."""
    tok = make_tokenizer()
    collator = DataCollatorWithFlatteningAndCompletionMask(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
        train_on_last_assistant_only=False,
    )

    ids1 = [USER_A, RESP_1, RESP_2, ASST_A, EOS]
    ids2 = [USER_B, RESP_1, RESP_2, ASST_B, EOS]
    features = [
        {"input_ids": ids1, "labels": ids1[:]},
        {"input_ids": ids2, "labels": ids2[:]},
    ]
    batch = collator(features)

    labels = batch["labels"][0].tolist()
    assert labels[0] == IGNORE, f"Ex1 boundary not masked: {labels}"
    assert labels[5] == IGNORE, f"Ex2 boundary not masked: {labels}"
    print("  PASS: test_flattening_completion_boundary_labels_masked")


def test_all_collators_no_cross_doc_training_signal():
    """No collator should produce a valid shifted label that crosses a document boundary.

    Simulates the causal shift (logits[:-1] vs labels[1:]) and checks that for every
    position where shift_labels != -100, both logits[i] and labels[i+1] belong to the
    same document.
    """
    tok = make_tokenizer()

    doc1 = [USER_A, USER_B, ASST_A, EOS]
    doc2 = [USER_C, ASST_B, ASST_C, EOS]

    def get_doc_boundaries(pos_ids_list):
        """Return list of document indices per position from position_ids."""
        doc_idx = 0
        doc_map = []
        for i, p in enumerate(pos_ids_list):
            if p == 0 and i > 0:
                doc_idx += 1
            doc_map.append(doc_idx)
        return doc_map

    collator_pack = DataCollatorWithPacking(tokenizer=tok)
    examples_pack = [
        {
            "input_ids": doc1 + doc2,
            "attention_mask": [1] * (len(doc1) + len(doc2)),
            "seq_lengths": [len(doc1), len(doc2)],
        }
    ]
    batch_pack = collator_pack.torch_call(examples_pack)
    labels_pack = batch_pack["labels"][0].tolist()
    pos_ids_pack = batch_pack["position_ids"][0].tolist()
    doc_map = get_doc_boundaries(pos_ids_pack)

    shift_labels = labels_pack[1:]
    # Anchor: every check below is guarded by `sl != IGNORE`, so an all-masked batch — the collator
    # bug that trains on nothing — would satisfy the loop without entering it once.
    assert any(sl != IGNORE for sl in shift_labels), f"DataCollatorWithPacking masked everything: {labels_pack}"
    for i, sl in enumerate(shift_labels):
        if sl != IGNORE:
            assert doc_map[i] == doc_map[i + 1], (
                f"DataCollatorWithPacking: cross-doc training signal at pos {i}: "
                f"logits from doc {doc_map[i]}, label from doc {doc_map[i + 1]}. "
                f"labels={labels_pack}, pos_ids={pos_ids_pack}"
            )

    collator_flat = DataCollatorWithFlattening(tokenizer=tok)
    ex1 = {"input_ids": doc1, "labels": doc1[:]}
    ex2 = {"input_ids": doc2, "labels": doc2[:]}
    batch_flat = collator_flat([ex1, ex2])
    labels_flat = batch_flat["labels"][0].tolist()
    pos_ids_flat = batch_flat["position_ids"][0].tolist()
    doc_map_flat = get_doc_boundaries(pos_ids_flat)

    shift_labels_flat = labels_flat[1:]
    assert any(sl != IGNORE for sl in shift_labels_flat), (
        f"DataCollatorWithFlattening masked everything: {labels_flat}"
    )
    for i, sl in enumerate(shift_labels_flat):
        if sl != IGNORE:
            assert doc_map_flat[i] == doc_map_flat[i + 1], (
                f"DataCollatorWithFlattening: cross-doc training signal at pos {i}: "
                f"logits from doc {doc_map_flat[i]}, label from doc {doc_map_flat[i + 1]}. "
                f"labels={labels_flat}, pos_ids={pos_ids_flat}"
            )

    print("  PASS: test_all_collators_no_cross_doc_training_signal")


def test_missing_template_warns():
    """Collator should warn when response template is not found."""
    tok = make_tokenizer()
    collator = DataCollatorForCompletionOnlyLM(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
    )

    ids = [USER_A, USER_B, ASST_A, EOS]  # no response template
    examples = [{"input_ids": ids, "attention_mask": [1] * len(ids)}]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        batch = collator.torch_call(examples)
        assert len(w) > 0, "Should have warned about missing template"

    unmasked = get_unmasked_positions(batch["labels"][0])
    assert len(unmasked) == 0, f"All labels should be masked: {unmasked}"
    print("  PASS: test_missing_template_warns")


def test_missing_template_warning_is_length_bounded():
    """The mismatch warning must stay bounded on a long row.

    A wrong assistant_message_template misses on EVERY row of EVERY batch, so an unbounded
    ``tokenizer.decode(input_ids)`` in the message means a full-sequence decode plus a
    max_length-sized log line per row — at 32k context that floods the log and dominates step time.
    """
    tok = make_tokenizer()
    # Real decode, so the assertion measures the message the collator actually builds.
    tok.decode.side_effect = lambda ids, **kwargs: " ".join(f"tok{int(i)}" for i in ids)
    collator = DataCollatorForCompletionOnlyLM(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
    )

    long_ids = [USER_A, USER_B] * 4000  # no response template anywhere
    examples = [{"input_ids": long_ids, "attention_mask": [1] * len(long_ids)}]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batch = collator.torch_call(examples)

    assert len(caught) > 0, "Should have warned about the missing template"
    message = str(caught[-1].message)
    assert len(message) < 1200, f"warning grew with sequence length ({len(message)} chars): {message[:200]}"
    assert "more tokens" in message, "an elided row must say how much was dropped"
    assert get_unmasked_positions(batch["labels"][0]) == [], "an unmatched row must be fully masked"
    print("  PASS: test_missing_template_warning_is_length_bounded")


def test_packing_completion_template_at_boundary():
    """Packed doc starting with response template: assistant tokens must still be unmasked.

    _mask_sequence detects the response template in input_ids, not in
    boundary-masked labels. _handle_packing sets labels=-100 at
    position_ids==0, so a template starting at a document boundary is invisible
    in labels, silently dropping the entire document's response from training.
    """
    tok = make_tokenizer()
    collator = DataCollatorForCompletionOnlyLMWithPacking(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
        train_on_last_assistant_only=False,
    )

    conv1 = [USER_A, RESP_1, RESP_2, ASST_A, EOS]
    conv2 = [RESP_1, RESP_2, ASST_B, ASST_C, EOS]  # template sits ON the doc boundary
    packed_ids = conv1 + conv2

    examples = [
        {
            "input_ids": packed_ids,
            "attention_mask": [1] * len(packed_ids),
            "seq_lengths": [len(conv1), len(conv2)],
        }
    ]
    batch = collator.torch_call(examples)
    labels = batch["labels"][0].tolist()

    offset = len(conv1)
    assert labels[offset] == IGNORE, f"Boundary not masked: {labels}"
    assert labels[offset + 2] == ASST_B, (
        f"Conv2 assistant token ASST_B at {offset + 2} should be unmasked, got {labels[offset + 2]}. labels={labels}"
    )
    assert labels[offset + 3] == ASST_C, (
        f"Conv2 assistant token ASST_C at {offset + 3} should be unmasked, got {labels[offset + 3]}. labels={labels}"
    )
    assert labels[offset + 4] == EOS, (
        f"Conv2 EOS at {offset + 4} should be unmasked, got {labels[offset + 4]}. labels={labels}"
    )
    print("  PASS: test_packing_completion_template_at_boundary")


def test_three_turns_last_only():
    """3-turn conversation: only last turn unmasked with train_on_last_assistant_only=True."""
    tok = make_tokenizer()
    collator = DataCollatorForCompletionOnlyLM(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
        train_on_last_assistant_only=True,
    )

    ids = [
        USER_A,
        RESP_1,
        RESP_2,
        ASST_A,
        EOS,
        USER_B,
        RESP_1,
        RESP_2,
        ASST_B,
        EOS,
        USER_C,
        RESP_1,
        RESP_2,
        ASST_C,
        ASST_D,
        EOS,
    ]
    examples = [{"input_ids": ids, "attention_mask": [1] * len(ids)}]
    batch = collator.torch_call(examples)

    unmasked = get_unmasked_positions(batch["labels"][0])
    assert 3 not in unmasked, f"Turn 1 should be masked: {unmasked}"
    assert 8 not in unmasked, f"Turn 2 should be masked: {unmasked}"
    assert 13 in unmasked, f"Turn 3 assistant missing: {unmasked}"
    assert 14 in unmasked, f"Turn 3 assistant missing: {unmasked}"
    print("  PASS: test_three_turns_last_only")


def test_template_tokens_included_in_training():
    """Response template tokens themselves must be included in the unmasked region.

    All three completion-only collators should unmask from the template start
    (inclusive) through the EOS (inclusive).
    """
    tok = make_tokenizer()
    ids = [USER_A, RESP_1, RESP_2, ASST_A, EOS]

    c1 = DataCollatorForCompletionOnlyLM(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
    )
    batch1 = c1.torch_call([{"input_ids": ids[:], "attention_mask": [1] * len(ids)}])
    u1 = get_unmasked_positions(batch1["labels"][0])
    assert 1 in u1, f"CompletionOnlyLM: template token RESP_1 at 1 should be unmasked: {u1}"
    assert 2 in u1, f"CompletionOnlyLM: template token RESP_2 at 2 should be unmasked: {u1}"

    c2 = DataCollatorForCompletionOnlyLMWithPacking(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
    )
    batch2 = c2.torch_call([{"input_ids": ids[:], "attention_mask": [1] * len(ids)}])
    u2 = get_unmasked_positions(batch2["labels"][0])
    assert 1 in u2, f"WithPacking non-packed: RESP_1 at 1 should be unmasked: {u2}"
    assert 2 in u2, f"WithPacking non-packed: RESP_2 at 2 should be unmasked: {u2}"

    batch3 = c2.torch_call(
        [
            {
                "input_ids": ids[:],
                "attention_mask": [1] * len(ids),
                "seq_lengths": [len(ids)],
            }
        ]
    )
    u3 = get_unmasked_positions(batch3["labels"][0])
    assert 1 in u3, f"WithPacking packed: RESP_1 at 1 should be unmasked: {u3}"
    assert 2 in u3, f"WithPacking packed: RESP_2 at 2 should be unmasked: {u3}"

    c4 = DataCollatorWithFlatteningAndCompletionMask(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
    )
    batch4 = c4([{"input_ids": ids[:], "labels": ids[:]}])
    u4 = get_unmasked_positions(batch4["labels"][0])
    assert 1 in u4, f"Flattening: RESP_1 at 1 should be unmasked: {u4}"
    assert 2 in u4, f"Flattening: RESP_2 at 2 should be unmasked: {u4}"

    print("  PASS: test_template_tokens_included_in_training")


def test_eos_included_in_training():
    """EOS token must be included in the unmasked region for all completion collators."""
    tok = make_tokenizer()
    ids = [USER_A, RESP_1, RESP_2, ASST_A, EOS]

    c1 = DataCollatorForCompletionOnlyLM(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
    )
    batch1 = c1.torch_call([{"input_ids": ids[:], "attention_mask": [1] * len(ids)}])
    u1 = get_unmasked_positions(batch1["labels"][0])
    assert 4 in u1, f"CompletionOnlyLM: EOS at 4 should be unmasked: {u1}"

    c2 = DataCollatorForCompletionOnlyLMWithPacking(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
    )
    batch2 = c2.torch_call([{"input_ids": ids[:], "attention_mask": [1] * len(ids)}])
    u2 = get_unmasked_positions(batch2["labels"][0])
    assert 4 in u2, f"WithPacking non-packed: EOS at 4 should be unmasked: {u2}"

    batch3 = c2.torch_call(
        [
            {
                "input_ids": ids[:],
                "attention_mask": [1] * len(ids),
                "seq_lengths": [len(ids)],
            }
        ]
    )
    u3 = get_unmasked_positions(batch3["labels"][0])
    assert 4 in u3, f"WithPacking packed: EOS at 4 should be unmasked: {u3}"

    c4 = DataCollatorWithFlatteningAndCompletionMask(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
    )
    batch4 = c4([{"input_ids": ids[:], "labels": ids[:]}])
    u4 = get_unmasked_positions(batch4["labels"][0])
    assert 4 in u4, f"Flattening: EOS at 4 should be unmasked: {u4}"

    print("  PASS: test_eos_included_in_training")


def test_exact_labels_completion_only():
    """Verify exact label values for DataCollatorForCompletionOnlyLM.

    input:  [USER_A, USER_B, RESP_1, RESP_2, ASST_A, ASST_B, EOS]
    expect: [-100,   -100,   RESP_1, RESP_2, ASST_A, ASST_B, EOS]
    """
    tok = make_tokenizer()
    collator = DataCollatorForCompletionOnlyLM(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
    )
    ids = [USER_A, USER_B, RESP_1, RESP_2, ASST_A, ASST_B, EOS]
    batch = collator.torch_call([{"input_ids": ids[:], "attention_mask": [1] * len(ids)}])
    labels = batch["labels"][0].tolist()

    expected = [IGNORE, IGNORE, RESP_1, RESP_2, ASST_A, ASST_B, EOS]
    assert labels == expected, f"Expected {expected}, got {labels}"
    print("  PASS: test_exact_labels_completion_only")


def test_empty_response():
    """Response template immediately followed by EOS (no content tokens).

    All completion collators should still unmask the template + EOS.
    """
    tok = make_tokenizer()
    ids = [USER_A, RESP_1, RESP_2, EOS]

    c1 = DataCollatorForCompletionOnlyLM(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
    )
    batch1 = c1.torch_call([{"input_ids": ids[:], "attention_mask": [1] * len(ids)}])
    u1 = get_unmasked_positions(batch1["labels"][0])
    assert 1 in u1 and 2 in u1 and 3 in u1, f"CompletionOnlyLM: empty response should still unmask template+EOS: {u1}"
    assert 0 not in u1, f"User token at 0 should be masked: {u1}"

    c2 = DataCollatorForCompletionOnlyLMWithPacking(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
    )
    batch2 = c2.torch_call(
        [
            {
                "input_ids": ids[:],
                "attention_mask": [1] * len(ids),
                "seq_lengths": [len(ids)],
            }
        ]
    )
    u2 = get_unmasked_positions(batch2["labels"][0])
    # Boundary already masks idx 0; template at 1,2 and EOS at 3.
    assert 1 in u2 and 2 in u2 and 3 in u2, f"WithPacking packed: empty response should unmask template+EOS: {u2}"

    c3 = DataCollatorWithFlatteningAndCompletionMask(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
    )
    batch3 = c3([{"input_ids": ids[:], "labels": ids[:]}])
    u3 = get_unmasked_positions(batch3["labels"][0])
    assert 1 in u3 and 2 in u3 and 3 in u3, f"Flattening: empty response should unmask template+EOS: {u3}"

    print("  PASS: test_empty_response")


def test_truncated_response_no_eos():
    """Response template at end of sequence with no following EOS.

    Packed _mask_sequence falls back to end of sequence; standard collator
    and flattening should mask everything (no matching EOS found).
    """
    tok = make_tokenizer()

    # Both paths fall back to len-1 as the EOS position when no EOS is present.
    c_pack = DataCollatorForCompletionOnlyLMWithPacking(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
    )
    ids_trunc = [USER_A, RESP_1, RESP_2, ASST_A, ASST_B]  # no EOS
    batch = c_pack.torch_call(
        [
            {
                "input_ids": ids_trunc[:],
                "attention_mask": [1] * len(ids_trunc),
                "seq_lengths": [len(ids_trunc)],
            }
        ]
    )
    u = get_unmasked_positions(batch["labels"][0])
    assert 1 in u and 2 in u and 3 in u and 4 in u, (
        f"Packed truncated: expected template+content unmasked via fallback: {u}"
    )

    c_flat = DataCollatorWithFlatteningAndCompletionMask(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
    )
    batch_flat = c_flat([{"input_ids": ids_trunc[:], "labels": ids_trunc[:]}])
    u_flat = get_unmasked_positions(batch_flat["labels"][0])
    assert 1 in u_flat and 2 in u_flat and 3 in u_flat and 4 in u_flat, (
        f"Flattening truncated: expected template+content unmasked via fallback: {u_flat}"
    )

    print("  PASS: test_truncated_response_no_eos")


def test_multi_example_batch_completion_only():
    """Multiple examples in the same batch — each masked independently."""
    tok = make_tokenizer()
    collator = DataCollatorForCompletionOnlyLM(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
    )

    ex1 = [USER_A, RESP_1, RESP_2, ASST_A, EOS]
    ex2 = [USER_A, RESP_1, RESP_2, ASST_A, EOS, USER_B, RESP_1, RESP_2, ASST_B, EOS]

    batch = collator.torch_call(
        [
            {"input_ids": ex1[:], "attention_mask": [1] * len(ex1)},
            {"input_ids": ex2[:], "attention_mask": [1] * len(ex2)},
        ]
    )

    u1 = get_unmasked_positions(batch["labels"][0])
    assert 1 in u1 and 3 in u1 and 4 in u1, f"Ex1 response tokens missing: {u1}"
    assert 0 not in u1, f"Ex1 user token should be masked: {u1}"
    for i in range(5, 10):
        assert i not in u1, f"Ex1 padding at {i} should be masked: {u1}"

    u2 = get_unmasked_positions(batch["labels"][1])
    assert 3 in u2 and 4 in u2, f"Ex2 turn 1 response missing: {u2}"
    assert 8 in u2 and 9 in u2, f"Ex2 turn 2 response missing: {u2}"
    assert 0 not in u2, f"Ex2 user token at 0 should be masked: {u2}"
    assert 5 not in u2, f"Ex2 user token at 5 should be masked: {u2}"
    print("  PASS: test_multi_example_batch_completion_only")


def test_multi_example_batch_last_only():
    """Multi-example batch with last_only: each example's last turn masked independently."""
    tok = make_tokenizer()
    collator = DataCollatorForCompletionOnlyLM(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
        train_on_last_assistant_only=True,
    )

    ex1 = [USER_A, RESP_1, RESP_2, ASST_A, EOS]
    ex2 = [USER_A, RESP_1, RESP_2, ASST_A, EOS, USER_B, RESP_1, RESP_2, ASST_B, EOS]

    batch = collator.torch_call(
        [
            {"input_ids": ex1[:], "attention_mask": [1] * len(ex1)},
            {"input_ids": ex2[:], "attention_mask": [1] * len(ex2)},
        ]
    )

    u1 = get_unmasked_positions(batch["labels"][0])
    assert 3 in u1, f"Ex1 sole turn should be unmasked: {u1}"

    u2 = get_unmasked_positions(batch["labels"][1])
    assert 3 not in u2, f"Ex2 turn 1 should be masked with last_only: {u2}"
    assert 8 in u2, f"Ex2 turn 2 should be unmasked: {u2}"
    assert 9 in u2, f"Ex2 EOS of turn 2 should be unmasked: {u2}"
    print("  PASS: test_multi_example_batch_last_only")


def test_flattening_multi_example_multi_turn():
    """Flattening with multiple multi-turn examples — each example independently masked."""
    tok = make_tokenizer()
    collator = DataCollatorWithFlatteningAndCompletionMask(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
        train_on_last_assistant_only=False,
    )

    ids1 = make_two_turn_ids()  # 13 tokens, 2 turns
    ids2 = [USER_A, RESP_1, RESP_2, ASST_A, EOS]  # 5 tokens, single-turn

    batch = collator(
        [
            {"input_ids": ids1[:], "labels": ids1[:]},
            {"input_ids": ids2[:], "labels": ids2[:]},
        ]
    )

    labels = batch["labels"][0].tolist()
    total = len(ids1) + len(ids2)
    assert len(labels) == total, f"Expected {total} labels, got {len(labels)}"

    assert labels[4] == ASST_A, f"Ex1 turn1 ASST_A at 4: got {labels[4]}"
    assert labels[5] == ASST_B, f"Ex1 turn1 ASST_B at 5: got {labels[5]}"
    assert labels[10] == ASST_C, f"Ex1 turn2 ASST_C at 10: got {labels[10]}"
    assert labels[11] == ASST_D, f"Ex1 turn2 ASST_D at 11: got {labels[11]}"
    assert labels[0] == IGNORE, f"Ex1 boundary at 0: got {labels[0]}"
    assert labels[7] == IGNORE, f"Ex1 user at 7: got {labels[7]}"

    off = len(ids1)
    assert labels[off] == IGNORE, f"Ex2 boundary at {off}: got {labels[off]}"
    assert labels[off + 3] == ASST_A, f"Ex2 ASST_A at {off + 3}: got {labels[off + 3]}"
    assert labels[off + 4] == EOS, f"Ex2 EOS at {off + 4}: got {labels[off + 4]}"

    print("  PASS: test_flattening_multi_example_multi_turn")


def test_flattening_multi_example_last_only():
    """Flattening last_only: each example applies last-only independently."""
    tok = make_tokenizer()
    collator = DataCollatorWithFlatteningAndCompletionMask(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
        train_on_last_assistant_only=True,
    )

    ids1 = make_two_turn_ids()
    ids2 = [USER_A, RESP_1, RESP_2, ASST_B, EOS]

    batch = collator(
        [
            {"input_ids": ids1[:], "labels": ids1[:]},
            {"input_ids": ids2[:], "labels": ids2[:]},
        ]
    )
    labels = batch["labels"][0].tolist()

    assert labels[4] == IGNORE, f"Ex1 turn1 should be masked: {labels[4]}"
    assert labels[5] == IGNORE, f"Ex1 turn1 should be masked: {labels[5]}"
    assert labels[10] == ASST_C, f"Ex1 turn2 ASST_C at 10 should be unmasked: {labels[10]}"
    assert labels[11] == ASST_D, f"Ex1 turn2 ASST_D at 11 should be unmasked: {labels[11]}"

    off = len(ids1)
    assert labels[off + 3] == ASST_B, f"Ex2 ASST_B at {off + 3} should be unmasked: {labels[off + 3]}"
    assert labels[off + 4] == EOS, f"Ex2 EOS at {off + 4} should be unmasked: {labels[off + 4]}"

    print("  PASS: test_flattening_multi_example_last_only")


def test_mixed_packed_nonpacked_batch():
    """Batch with one packed example and one non-packed example flattens to [1, total]:
    the non-packed row is appended after the packed one, both masked per-row pre-flattening."""
    tok = make_tokenizer()
    collator = DataCollatorForCompletionOnlyLMWithPacking(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
    )

    conv_a = [USER_A, RESP_1, RESP_2, ASST_A, EOS]
    conv_b = [USER_B, RESP_1, RESP_2, ASST_B, EOS]
    packed_ids = conv_a + conv_b

    single = [USER_C, RESP_1, RESP_2, ASST_C, ASST_D, EOS]  # no seq_lengths → non-packed row

    batch = collator.torch_call(
        [
            {
                "input_ids": packed_ids[:],
                "attention_mask": [1] * len(packed_ids),
                "seq_lengths": [len(conv_a), len(conv_b)],
            },
            {
                "input_ids": single[:],
                "attention_mask": [1] * len(single),
            },
        ]
    )

    assert batch["labels"].shape == (1, 20), f"mixed packed batch must flatten to [1, 20]: {batch['labels'].shape}"
    u = set(get_unmasked_positions(batch["labels"][0]))

    assert 3 in u, f"Ex1 conv_a ASST_A at 3 missing: {sorted(u)}"
    assert 4 in u, f"Ex1 conv_a EOS at 4 missing: {sorted(u)}"
    assert 8 in u, f"Ex1 conv_b ASST_B at 8 missing: {sorted(u)}"
    assert 9 in u, f"Ex1 conv_b EOS at 9 missing: {sorted(u)}"
    assert 0 not in u, f"Ex1 boundary at 0 should be masked: {sorted(u)}"
    assert 5 not in u, f"Ex1 boundary at 5 should be masked: {sorted(u)}"

    off = 10
    assert off + 3 in u, f"Ex2 ASST_C at {off + 3} missing: {sorted(u)}"
    assert off + 4 in u, f"Ex2 ASST_D at {off + 4} missing: {sorted(u)}"
    assert off + 5 in u, f"Ex2 EOS at {off + 5} missing: {sorted(u)}"
    assert off not in u, f"Ex2 user at {off} should be masked: {sorted(u)}"
    assert u.isdisjoint(range(off + 6, off + 10)), f"Ex2 padding must stay masked: {sorted(u)}"

    print("  PASS: test_mixed_packed_nonpacked_batch")


def test_three_packed_documents():
    """Three documents packed into a single example."""
    tok = make_tokenizer()
    collator = DataCollatorForCompletionOnlyLMWithPacking(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
    )

    doc1 = [USER_A, RESP_1, RESP_2, ASST_A, EOS]
    doc2 = [USER_B, RESP_1, RESP_2, ASST_B, EOS]
    doc3 = [USER_C, RESP_1, RESP_2, ASST_C, EOS]
    packed = doc1 + doc2 + doc3

    batch = collator.torch_call(
        [
            {
                "input_ids": packed[:],
                "attention_mask": [1] * len(packed),
                "seq_lengths": [5, 5, 5],
            }
        ]
    )
    labels = batch["labels"][0].tolist()

    for doc_idx, offset in enumerate([0, 5, 10]):
        assert labels[offset] == IGNORE, f"Doc{doc_idx + 1} boundary at {offset} should be -100: {labels}"
        assert labels[offset + 3] != IGNORE, f"Doc{doc_idx + 1} assistant at {offset + 3} should be unmasked: {labels}"
        assert labels[offset + 4] == EOS, f"Doc{doc_idx + 1} EOS at {offset + 4} should be unmasked: {labels}"

    print("  PASS: test_three_packed_documents")


def test_three_packed_documents_last_only():
    """Three packed docs, each multi-turn. last_only applies per-document."""
    tok = make_tokenizer()
    collator = DataCollatorForCompletionOnlyLMWithPacking(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
        train_on_last_assistant_only=True,
    )

    doc = [
        USER_A,
        RESP_1,
        RESP_2,
        ASST_A,
        EOS,
        USER_B,
        RESP_1,
        RESP_2,
        ASST_B,
        EOS,
    ]
    doc_short = [USER_C, RESP_1, RESP_2, ASST_C, EOS]
    packed = doc + doc_short

    batch = collator.torch_call(
        [
            {
                "input_ids": packed[:],
                "attention_mask": [1] * len(packed),
                "seq_lengths": [len(doc), len(doc_short)],
            }
        ]
    )
    labels = batch["labels"][0].tolist()

    assert labels[3] == IGNORE, f"Doc1 turn1 should be masked: {labels[3]}"
    assert labels[8] == ASST_B, f"Doc1 turn2 ASST_B at 8 should be unmasked: {labels[8]}"
    assert labels[9] == EOS, f"Doc1 turn2 EOS at 9 should be unmasked: {labels[9]}"

    # Doc2's single turn IS its last turn, so last_only keeps it.
    off = len(doc)
    assert labels[off + 3] == ASST_C, f"Doc2 ASST_C at {off + 3} should be unmasked: {labels[off + 3]}"

    print("  PASS: test_three_packed_documents_last_only")


def test_packing_completion_template_at_boundary_last_only():
    """Template at boundary + last_only: assistant tokens must still be found."""
    tok = make_tokenizer()
    collator = DataCollatorForCompletionOnlyLMWithPacking(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
        train_on_last_assistant_only=True,
    )

    conv1 = [USER_A, RESP_1, RESP_2, ASST_A, EOS]
    # conv2 opens with the template ON the doc boundary and has 2 turns.
    conv2 = [
        RESP_1,
        RESP_2,
        ASST_B,
        EOS,
        USER_C,
        RESP_1,
        RESP_2,
        ASST_C,
        EOS,
    ]
    packed = conv1 + conv2

    batch = collator.torch_call(
        [
            {
                "input_ids": packed[:],
                "attention_mask": [1] * len(packed),
                "seq_lengths": [len(conv1), len(conv2)],
            }
        ]
    )
    labels = batch["labels"][0].tolist()
    off = len(conv1)

    assert labels[off + 2] == IGNORE, f"Conv2 turn1 ASST_B at {off + 2} should be masked by last_only: {labels}"
    assert labels[off + 7] == ASST_C, f"Conv2 turn2 ASST_C at {off + 7} should be unmasked: {labels}"
    assert labels[off + 8] == EOS, f"Conv2 turn2 EOS at {off + 8} should be unmasked: {labels}"

    print("  PASS: test_packing_completion_template_at_boundary_last_only")


def test_user_tokens_never_unmasked():
    """Exhaustive check: user/system tokens must always be masked across all collators."""
    tok = make_tokenizer()
    ids = [USER_A, USER_B, USER_C, RESP_1, RESP_2, ASST_A, ASST_B, EOS]
    user_positions = {0, 1, 2}

    c1 = DataCollatorForCompletionOnlyLM(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
    )
    b1 = c1.torch_call([{"input_ids": ids[:], "attention_mask": [1] * len(ids)}])
    u1 = set(get_unmasked_positions(b1["labels"][0]))
    assert user_positions.isdisjoint(u1), f"CompletionOnlyLM: user tokens unmasked: {u1 & user_positions}"

    c2 = DataCollatorForCompletionOnlyLMWithPacking(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
    )
    b2 = c2.torch_call([{"input_ids": ids[:], "attention_mask": [1] * len(ids)}])
    u2 = set(get_unmasked_positions(b2["labels"][0]))
    assert user_positions.isdisjoint(u2), f"WithPacking non-packed: user tokens unmasked: {u2 & user_positions}"

    b3 = c2.torch_call(
        [
            {
                "input_ids": ids[:],
                "attention_mask": [1] * len(ids),
                "seq_lengths": [len(ids)],
            }
        ]
    )
    u3 = set(get_unmasked_positions(b3["labels"][0]))
    assert user_positions.isdisjoint(u3), f"WithPacking packed: user tokens unmasked: {u3 & user_positions}"

    c4 = DataCollatorWithFlatteningAndCompletionMask(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
    )
    b4 = c4([{"input_ids": ids[:], "labels": ids[:]}])
    u4 = set(get_unmasked_positions(b4["labels"][0]))
    assert user_positions.isdisjoint(u4), f"Flattening: user tokens unmasked: {u4 & user_positions}"

    print("  PASS: test_user_tokens_never_unmasked")


def test_no_cross_doc_signal_completion_packed():
    """Completion-only packed collator must also prevent cross-doc training signal.

    Extends test_all_collators_no_cross_doc_training_signal to
    DataCollatorForCompletionOnlyLMWithPacking.
    """
    tok = make_tokenizer()
    collator = DataCollatorForCompletionOnlyLMWithPacking(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
    )

    doc1 = [USER_A, RESP_1, RESP_2, ASST_A, EOS]
    doc2 = [USER_B, RESP_1, RESP_2, ASST_B, EOS]
    packed = doc1 + doc2

    batch = collator.torch_call(
        [
            {
                "input_ids": packed[:],
                "attention_mask": [1] * len(packed),
                "seq_lengths": [5, 5],
            }
        ]
    )
    labels = batch["labels"][0].tolist()
    pos_ids = batch["position_ids"][0].tolist()

    doc_idx = 0
    doc_map = []
    for i, p in enumerate(pos_ids):
        if p == 0 and i > 0:
            doc_idx += 1
        doc_map.append(doc_idx)

    # Causal shift: logits[i] predicts labels[i+1].
    shift_labels = labels[1:]
    # Anchor: the cross-doc check below only runs on unmasked positions, so a collator that masked
    # every label — training on nothing — would pass without entering the loop.
    assert any(sl != IGNORE for sl in shift_labels), f"completion-packed collator masked everything: {labels}"
    for i, sl in enumerate(shift_labels):
        if sl != IGNORE:
            assert doc_map[i] == doc_map[i + 1], (
                f"Cross-doc signal at {i}: logits doc={doc_map[i]}, "
                f"label doc={doc_map[i + 1]}. labels={labels}, pos_ids={pos_ids}"
            )

    print("  PASS: test_no_cross_doc_signal_completion_packed")


def test_flattening_completion_no_cross_doc_signal():
    """Flattening completion collator must prevent cross-doc training signal."""
    tok = make_tokenizer()
    collator = DataCollatorWithFlatteningAndCompletionMask(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
    )

    ids1 = [USER_A, RESP_1, RESP_2, ASST_A, EOS]
    ids2 = [USER_B, RESP_1, RESP_2, ASST_B, EOS]

    batch = collator(
        [
            {"input_ids": ids1[:], "labels": ids1[:]},
            {"input_ids": ids2[:], "labels": ids2[:]},
        ]
    )
    labels = batch["labels"][0].tolist()
    pos_ids = batch["position_ids"][0].tolist()

    doc_idx = 0
    doc_map = []
    for i, p in enumerate(pos_ids):
        if p == 0 and i > 0:
            doc_idx += 1
        doc_map.append(doc_idx)

    shift_labels = labels[1:]
    assert any(sl != IGNORE for sl in shift_labels), f"flattening completion collator masked everything: {labels}"
    for i, sl in enumerate(shift_labels):
        if sl != IGNORE:
            assert doc_map[i] == doc_map[i + 1], (
                f"Cross-doc signal at {i}: logits doc={doc_map[i]}, "
                f"label doc={doc_map[i + 1]}. labels={labels}, pos_ids={pos_ids}"
            )

    print("  PASS: test_flattening_completion_no_cross_doc_signal")


def test_tokenize_response_template_passthrough_ids():
    """Pre-tokenized IDs are returned as a fresh list (validated, not re-encoded)."""
    tok = make_tokenizer()
    out = tokenize_response_template([10, 11], tok)
    assert out == [10, 11]


def test_tokenize_response_template_encodes_string():
    """A string template is encoded via the tokenizer (no special tokens)."""
    tok = make_tokenizer(response_template_ids=[10, 11])
    out = tokenize_response_template("<resp>", tok)
    assert out == [10, 11]
    tok.encode.assert_called_once()
    # add_special_tokens must be False or the template never matches the packed stream.
    assert tok.encode.call_args.kwargs.get("add_special_tokens") is False


def test_tokenize_response_template_empty_raises():
    """An empty template would match every position → ValueError."""
    tok = make_tokenizer()
    with pytest.raises(ValueError, match="empty"):
        tokenize_response_template([], tok)


def test_filter_eos_after_responses_nearest_following():
    """Each response start maps to the nearest EOS strictly after it."""
    assert filter_eos_after_responses([2, 8], [5, 11]) == [5, 11]


def test_filter_eos_after_responses_no_following_eos_returns_sentinel():
    """A response start with no EOS strictly after it returns -1 (an empty unmask span) rather than
    borrowing another turn's EOS (which would unmask non-assistant tokens)."""
    assert filter_eos_after_responses([8], [5]) == [-1]


def test_filter_eos_does_not_cross_into_next_response():
    """Span-leak regression: a turn missing its own EOS must NOT borrow a LATER turn's EOS — the
    search is bounded by the next response start, so an unterminated turn returns -1."""
    # Turn 1 (start 2) must not bind to turn 2's EOS (12) — that span covers the user message.
    assert filter_eos_after_responses([2, 8], [12]) == [-1, 12]
    assert filter_eos_after_responses([2, 8], [6, 12]) == [6, 12]
    # A user-turn EOS (8) between the two assistant turns must not be mis-assigned.
    assert filter_eos_after_responses([2, 9], [6, 8, 13]) == [6, 13]


def test_filter_eos_after_responses_empty_inputs():
    """No starts or no EOS positions → empty result."""
    assert filter_eos_after_responses([], [5]) == []
    assert filter_eos_after_responses([2], []) == []


def test_completion_only_unterminated_turn_does_not_leak_into_user_message():
    """End-to-end span-leak regression: a mid-conversation assistant turn missing its EOS must NOT
    unmask the following user message (the bug bound its start to a later turn's EOS)."""
    tok = make_tokenizer()
    collator = DataCollatorForCompletionOnlyLM(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
        train_on_last_assistant_only=False,
    )
    # Turn-1 EOS removed; USER_C (index 6) sits between the two assistant turns.
    ids = [USER_A, USER_B, RESP_1, RESP_2, ASST_A, ASST_B, USER_C, RESP_1, RESP_2, ASST_C, ASST_D, EOS]
    examples = [{"input_ids": ids, "attention_mask": [1] * len(ids)}]
    batch = collator.torch_call(examples)

    unmasked = get_unmasked_positions(batch["labels"][0])
    assert 6 not in unmasked, f"User message (idx 6) leaked into labels: {unmasked}"
    assert 9 in unmasked and 10 in unmasked, f"Turn 2 assistant tokens missing: {unmasked}"
    print("  PASS: test_completion_only_unterminated_turn_does_not_leak_into_user_message")


def test_build_completion_only_labels_preserves_eos_when_pad_equals_eos():
    """When pad_token_id == eos_token_id, masking must use the attention_mask, not the token value —
    otherwise every REAL eos is erased to -100 and the model never learns to stop."""
    tok = MagicMock()
    tok.pad_token_id = 1
    tok.eos_token_id = 1  # pad == eos (Qwen default)

    # idx 2 is a real EOS in content; idx 3,4 are trailing padding of the same id.
    input_ids = torch.tensor([[5, 6, 1, 1, 1]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]])

    labels = build_completion_only_labels(
        input_ids, tok, response_prompt_template=None, train_on_completions_only=False, attention_mask=attention_mask
    )
    assert labels[0, 2].item() == 1, "real EOS was erased despite attention_mask=1"
    assert labels[0, 3].item() == IGNORE and labels[0, 4].item() == IGNORE, "padding not masked"

    # Without an attention_mask the value-based fallback DOES erase the real eos — that path is
    # reachable only for unpadded single examples.
    labels_fallback = build_completion_only_labels(
        input_ids, tok, response_prompt_template=None, train_on_completions_only=False
    )
    assert labels_fallback[0, 2].item() == IGNORE
    print("  PASS: test_build_completion_only_labels_preserves_eos_when_pad_equals_eos")


# GLM-style markers: tokenizer.eos_token_id (<|endoftext|>) NEVER appears mid-conversation; turns end
# at role markers the model lists in config.eos_token_id instead.
GLM_ENDOFTEXT = 1  # absent from multi-turn sequences
GLM_USER_MARK = 2  # <|user|> — ends an assistant turn that produced a final answer
GLM_OBS_MARK = 3  # <|observation|> — ends an assistant turn that produced a tool call
GLM_EOS_SET = frozenset({GLM_ENDOFTEXT, GLM_USER_MARK, GLM_OBS_MARK})


class _FakeConfig:
    def __init__(self, eos_token_id=None, text_config=None):
        if eos_token_id is not None:
            self.eos_token_id = eos_token_id
        if text_config is not None:
            self.text_config = text_config


def test_resolve_eos_token_ids_folds_config_list():
    """The config's eos_token_id LIST plus the tokenizer's eos become terminators; a distinct
    pad token does NOT (it would close unterminated spans at the first trailing pad)."""
    tok = make_tokenizer(pad_token_id=0, eos_token_id=1)
    assert resolve_eos_token_ids(tok, _FakeConfig(eos_token_id=[1, 2, 3])) == frozenset({1, 2, 3})


def test_resolve_eos_token_ids_single_int_config():
    """A scalar config eos_token_id is folded in (not just lists)."""
    tok = make_tokenizer(pad_token_id=0, eos_token_id=1)
    assert resolve_eos_token_ids(tok, _FakeConfig(eos_token_id=7)) == frozenset({1, 7})


def test_resolve_eos_token_ids_reads_text_config():
    """VLM/composite configs nest eos_token_id under text_config."""
    tok = make_tokenizer(pad_token_id=0, eos_token_id=1)
    cfg = _FakeConfig(text_config=_FakeConfig(eos_token_id=[7, 8]))
    assert resolve_eos_token_ids(tok, cfg) == frozenset({1, 7, 8})


def test_resolve_eos_token_ids_tokenizer_only():
    """No model_config → tokenizer eos only (never the distinct pad)."""
    tok = make_tokenizer(pad_token_id=5, eos_token_id=9)
    assert resolve_eos_token_ids(tok, None) == frozenset({9})


def test_resolve_eos_token_ids_pad_equals_eos_still_terminates():
    """pad == eos stays a terminator via the eos collect (the Qwen/Llama shared-id layout)."""
    tok = make_tokenizer(pad_token_id=9, eos_token_id=9)
    assert resolve_eos_token_ids(tok, None) == frozenset({9})


def test_last_only_unterminated_final_turn_warns_and_masks():
    """train_on_last_assistant_only + a terminator-less FINAL turn: the surviving span is a single
    -1 no-op, so the row trains ZERO tokens — that must WARN, not silently all-mask.

    Regression: the existing missing-template warning did not fire here (response_starts is
    non-empty), so the loss-0 row was invisible.
    """
    tok = make_tokenizer()
    collator = DataCollatorForCompletionOnlyLM(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
        train_on_last_assistant_only=True,
    )
    # Turn 1 terminated (EOS@4); the final turn has NO terminator → only the -1 span survives.
    ids = [USER_A, RESP_1, RESP_2, ASST_A, EOS, USER_B, RESP_1, RESP_2, ASST_B]
    examples = [{"input_ids": ids, "attention_mask": [1] * len(ids)}]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batch = collator.torch_call(examples)

    assert get_unmasked_positions(batch["labels"][0]) == [], "row with only no-op spans must be fully masked"
    assert any("no completion span has a terminator" in str(w.message) for w in caught), (
        f"expected a no-terminator warning for the all-masked row, got: {[str(w.message) for w in caught]}"
    )
    print("  PASS: test_last_only_unterminated_final_turn_warns_and_masks")


def test_completion_only_unterminated_right_padded_row_never_trains_on_pad():
    """A right-padded row whose assistant turn has NO recognized terminator must fall to the
    warn-and-mask path — not close its span at the first trailing pad and train on a pad target."""
    tok = make_tokenizer()  # pad and eos distinct
    collator = DataCollatorForCompletionOnlyLM(response_prompt_template=RESPONSE_TEMPLATE_IDS, tokenizer=tok)
    short_row = [USER_A, RESP_1, RESP_2, ASST_A]  # template present, no EOS anywhere
    long_row = [USER_B, RESP_1, RESP_2, ASST_B, ASST_C, ASST_D, EOS]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batch = collator.torch_call(
            [
                {"input_ids": short_row, "attention_mask": [1] * len(short_row)},
                {"input_ids": long_row, "attention_mask": [1] * len(long_row)},
            ]
        )
    short_labels = batch["labels"][0]
    pad_positions = batch["input_ids"][0] == PAD
    assert (short_labels[pad_positions] == IGNORE).all(), f"trained on pad: {short_labels.tolist()}"
    assert (short_labels == IGNORE).all(), f"unterminated turn must be fully masked: {short_labels.tolist()}"
    # The template DID match, so the warning must blame the missing terminator — telling the user to
    # fix a response key that is already correct sends them to change the one setting that is right.
    messages = [str(w.message) for w in caught]
    assert any("matched but no completion span has a terminator" in m for m in messages), messages
    assert not any("Could not find response key" in m for m in messages), messages


def test_find_terminator_positions():
    """Returns every position whose token is in the terminator set (list or tensor input)."""
    seq = [5, 1, 6, 2, 7, 3, 8]
    assert find_terminator_positions(seq, frozenset({1, 2, 3})) == [1, 3, 5]
    assert find_terminator_positions(torch.tensor(seq), frozenset({2})) == [3]
    assert find_terminator_positions(seq, frozenset()) == []


def _glm_agent_trace():
    """A GLM-style agent trace with NO <|endoftext|>: turn 1 ends at <|observation|> (tool call), turn 2
    ends at <|user|> (final answer).

    idx: 0      1      | 2      3      4     | 5         6      | 7      8      9      10     | 11
    """
    return [
        USER_A,
        USER_B,
        RESP_1,
        RESP_2,
        ASST_A,
        GLM_OBS_MARK,  # ends turn 1 (5)
        USER_C,  # tool result content (6)
        RESP_1,
        RESP_2,
        ASST_C,
        ASST_D,
        GLM_USER_MARK,  # ends turn 2 (11)
    ]


def test_completion_only_stops_at_config_eos_markers():
    """End-to-end: with the config eos set, each assistant turn ends at its role marker (inclusive),
    while user/observation content stays masked — even though tokenizer.eos_token_id is absent. With the
    old single-eos search this sequence trained ZERO tokens (whole instance masked)."""
    tok = make_tokenizer(pad_token_id=0, eos_token_id=GLM_ENDOFTEXT)
    collator = DataCollatorForCompletionOnlyLM(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
        train_on_last_assistant_only=False,
        eos_token_ids=GLM_EOS_SET,
    )
    ids = _glm_agent_trace()
    batch = collator.torch_call([{"input_ids": ids, "attention_mask": [1] * len(ids)}])
    unmasked = set(get_unmasked_positions(batch["labels"][0]))

    assert {2, 3, 4, 5} <= unmasked, f"turn 1 (incl. <|observation|>) not trained: {sorted(unmasked)}"
    assert {7, 8, 9, 10, 11} <= unmasked, f"turn 2 (incl. <|user|>) not trained: {sorted(unmasked)}"
    assert unmasked.isdisjoint({0, 1, 6}), f"non-assistant tokens leaked: {sorted(unmasked)}"
    print("  PASS: test_completion_only_stops_at_config_eos_markers")


def test_completion_only_old_single_eos_would_mask_everything():
    """Guards the regression: searching for ONLY tokenizer.eos_token_id (absent here) finds no turn end,
    so the whole instance is masked. Confirms the GLM trace genuinely exercises the bug."""
    tok = make_tokenizer(pad_token_id=99, eos_token_id=GLM_ENDOFTEXT)  # pad distinct and absent too
    collator = DataCollatorForCompletionOnlyLM(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
        train_on_last_assistant_only=False,
        eos_token_ids=frozenset({GLM_ENDOFTEXT}),  # old behavior: single eos only
    )
    ids = _glm_agent_trace()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        batch = collator.torch_call([{"input_ids": ids, "attention_mask": [1] * len(ids)}])
    assert get_unmasked_positions(batch["labels"][0]) == [], "expected whole-instance mask under single-eos search"
    print("  PASS: test_completion_only_old_single_eos_would_mask_everything")


def test_packing_completion_stops_at_config_eos_markers():
    """Packed path (the SFT grad-test config's collator) ends each turn at its role marker."""
    tok = make_tokenizer(pad_token_id=0, eos_token_id=GLM_ENDOFTEXT)
    collator = DataCollatorForCompletionOnlyLMWithPacking(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
        train_on_last_assistant_only=False,
        eos_token_ids=GLM_EOS_SET,
    )
    conv1 = _glm_agent_trace()
    conv2 = [USER_A, RESP_1, RESP_2, ASST_A, GLM_USER_MARK]
    packed_ids = conv1 + conv2
    examples = [
        {
            "input_ids": packed_ids,
            "attention_mask": [1] * len(packed_ids),
            "seq_lengths": [len(conv1), len(conv2)],
        }
    ]
    batch = collator.torch_call(examples)
    unmasked = set(get_unmasked_positions(batch["labels"][0]))

    assert {2, 3, 4, 5, 7, 8, 9, 10, 11} <= unmasked, f"conv1 turns not fully trained: {sorted(unmasked)}"
    base = len(conv1)
    assert {base + 1, base + 2, base + 3, base + 4} <= unmasked, f"conv2 turn not trained: {sorted(unmasked)}"
    assert unmasked.isdisjoint({0, 1, 6, base}), f"non-assistant tokens leaked: {sorted(unmasked)}"
    print("  PASS: test_packing_completion_stops_at_config_eos_markers")


def test_flattening_completion_stops_at_config_eos_markers():
    """Padding-free flattening path ends each turn at its role marker."""
    tok = make_tokenizer(pad_token_id=0, eos_token_id=GLM_ENDOFTEXT)
    collator = DataCollatorWithFlatteningAndCompletionMask(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=tok,
        eos_token_ids=GLM_EOS_SET,
    )
    ids = _glm_agent_trace()
    batch = collator([{"input_ids": ids, "labels": ids[:]}])
    unmasked = set(get_unmasked_positions(batch["labels"][0]))
    assert {2, 3, 4, 5} <= unmasked, f"turn 1 not trained: {sorted(unmasked)}"
    assert {7, 8, 9, 10, 11} <= unmasked, f"turn 2 not trained: {sorted(unmasked)}"
    assert unmasked.isdisjoint({0, 1, 6}), f"non-assistant tokens leaked: {sorted(unmasked)}"
    print("  PASS: test_flattening_completion_stops_at_config_eos_markers")


def test_build_completion_only_labels_stops_at_config_eos_markers():
    """The shared label builder (preprocessing + self-distill + VLM) ends each turn at a role marker.
    Note: it trains AFTER the marker (not the template tokens), so turn 1 = 4..5, turn 2 = 9..11."""
    tok = make_tokenizer(pad_token_id=0, eos_token_id=GLM_ENDOFTEXT)
    tok.encode.return_value = RESPONSE_TEMPLATE_IDS
    ids = torch.tensor([_glm_agent_trace()])
    labels = build_completion_only_labels(
        ids,
        tok,
        response_prompt_template="<|assistant|>",
        train_on_completions_only=True,
        eos_token_ids=GLM_EOS_SET,
    )
    unmasked = set(get_unmasked_positions(labels[0]))
    assert {4, 5} <= unmasked, f"turn 1 content not trained: {sorted(unmasked)}"
    assert {9, 10, 11} <= unmasked, f"turn 2 content not trained: {sorted(unmasked)}"
    assert unmasked.isdisjoint({0, 1, 6}), f"non-assistant tokens leaked: {sorted(unmasked)}"
    print("  PASS: test_build_completion_only_labels_stops_at_config_eos_markers")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
