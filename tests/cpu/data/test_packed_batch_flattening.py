#!/usr/bin/env python
"""Packed batches must flatten to [1, total_tokens] so transformers' FA packed-sequence path fires.

transformers' ``_is_packed_sequence`` gates the varlen-from-position_ids path on ``batch_size == 1``:
a ``[B>1, L]`` packed batch with no attention_mask falls through to plain dense-causal attention and
documents silently attend across boundaries — the shape any config that packs at
``per_device_train_batch_size > 1`` produces. These tests pin the flattening contract against the REAL
transformers helpers (no reimplementation) and prove loss equivalence: the flattened stream is the
exact concatenation of the per-row (batch-1) collations.

Run: pytest tests/cpu/data/test_packed_batch_flattening.py
"""

import sys
from unittest.mock import MagicMock

import pytest
import torch
from transformers.modeling_flash_attention_utils import (
    _is_packed_sequence,
    prepare_fa_kwargs_from_position_ids,
)

from src.data.collators.packing import (
    DataCollatorForCompletionOnlyLMWithPacking,
    DataCollatorWithPacking,
    flatten_packed_batch,
)

PAD = 0
EOS = 1
RESP_1, RESP_2 = 10, 11
RESPONSE_TEMPLATE_IDS = [RESP_1, RESP_2]
USER_A, USER_B, USER_C = 20, 21, 22
ASST_A, ASST_B, ASST_C = 30, 31, 32
IGNORE = -100


def make_tokenizer(pad_token_id: int = PAD, eos_token_id: int = EOS) -> MagicMock:
    """Mock tokenizer with the minimal interface the LM collators need."""
    tok = MagicMock()
    tok.pad_token_id = pad_token_id
    tok.eos_token_id = eos_token_id
    tok.padding_side = "right"
    tok.model_max_length = 4096
    tok.encode.return_value = RESPONSE_TEMPLATE_IDS

    def _pad(features, **kwargs):
        max_len = max(len(f["input_ids"]) for f in features)
        out = {"input_ids": [], "attention_mask": []}
        for f in features:
            ids = list(f["input_ids"])
            mask = list(f.get("attention_mask", [1] * len(ids)))
            pad_len = max_len - len(ids)
            out["input_ids"].append(ids + [pad_token_id] * pad_len)
            out["attention_mask"].append(mask + [0] * pad_len)
        return {k: torch.tensor(v) for k, v in out.items()}

    tok.pad.side_effect = _pad
    return tok


def _packed_row(*docs: list[int]) -> dict:
    packed = [t for doc in docs for t in doc]
    return {
        "input_ids": packed,
        "attention_mask": [1] * len(packed),
        "seq_lengths": [len(d) for d in docs],
    }


def test_b2_packed_batch_satisfies_transformers_packed_detection():
    """A batch-2 packed collation must come out as [1, N] and fire transformers' own
    ``_is_packed_sequence`` — a [2, L] output fails that check (batch_size gate) and FA then runs
    dense-causal across document boundaries."""
    collator = DataCollatorWithPacking(tokenizer=make_tokenizer())
    row0 = _packed_row([USER_A, ASST_A, EOS, USER_B], [ASST_B, EOS, USER_C, ASST_C])
    row1 = _packed_row([USER_C, ASST_A, EOS, USER_A, ASST_B], [USER_B, ASST_C, EOS])
    batch = collator.torch_call([row0, row1])

    assert batch["input_ids"].shape[0] == 1, (
        f"packed batch must flatten to batch size 1, got shape {tuple(batch['input_ids'].shape)}"
    )
    assert batch["input_ids"].shape[1] == 16, "flattening must preserve every token"
    assert "attention_mask" not in batch, "flattened packed batch must not carry a dense attention_mask"
    assert bool(_is_packed_sequence(batch["position_ids"], batch["input_ids"].shape[0])), (
        "transformers must detect the flattened batch as packed (varlen path)"
    )


def test_flattened_batch_equals_concat_of_single_row_collations():
    """Loss equivalence: collating [row0, row1] together must yield exactly the concatenation of
    collating each row alone (identical input_ids/labels/position_ids streams → identical
    token-level CE over the same non-masked tokens, with the same per-document segmentation)."""
    collator = DataCollatorWithPacking(tokenizer=make_tokenizer())
    # Equal-length rows: no cross-row padding, so concatenation is exact.
    row0 = _packed_row([USER_A, ASST_A, EOS, USER_B], [ASST_B, EOS, USER_C, ASST_C])
    row1 = _packed_row([USER_C, ASST_A, EOS, USER_A, ASST_B], [USER_B, ASST_C, EOS])

    together = collator.torch_call([dict(row0), dict(row1)])
    alone0 = collator.torch_call([dict(row0)])
    alone1 = collator.torch_call([dict(row1)])

    for key in ("input_ids", "labels", "position_ids"):
        expected = alone0[key][0].tolist() + alone1[key][0].tolist()
        assert together[key][0].tolist() == expected, f"{key} diverges from per-row collation"


def test_padding_is_dropped_from_the_flattened_row():
    """A shorter row's pad tail must NOT survive flattening: only real tokens remain, and
    transformers' cu_seqlens derivation sees exactly the documents — no pad segments at all."""
    collator = DataCollatorWithPacking(tokenizer=make_tokenizer())
    row0 = _packed_row([USER_A, ASST_A, EOS, USER_B], [ASST_B, EOS, USER_C, ASST_C])
    row1 = _packed_row([USER_C, ASST_A, EOS, USER_A, ASST_B])  # len 5 → padded to 8, pads dropped
    batch = collator.torch_call([row0, row1])

    assert batch["input_ids"].shape == (1, 13), f"pads must be dropped, got {tuple(batch['input_ids'].shape)}"
    assert PAD not in batch["input_ids"][0].tolist(), "no pad token may survive the flatten"

    (cu_q, _cu_k), _max = prepare_fa_kwargs_from_position_ids(batch["position_ids"])
    assert cu_q.tolist() == [0, 4, 8, 13], f"document boundaries corrupted by flattening: {cu_q.tolist()}"


def test_mostly_pad_row_adds_no_segments():
    """The dataset's final PARTIAL pack, co-batched with a full row, must not explode the segment
    count. Left unflattened, every pad token becomes its own length-1 segment (position_id 0), and
    the varlen kernels pay a fixed cost per segment — a 34k-pad row measured 275x on the FA4
    backward (6.9 s vs 25 ms per layer), serialized onto all ranks by FSDP's per-layer collectives.
    """
    collator = DataCollatorWithPacking(tokenizer=make_tokenizer())
    full = _packed_row([USER_A] * 400, [ASST_A] * 400)
    partial = _packed_row([USER_B, ASST_B, EOS])  # 3 real tokens, 797 pads
    batch = collator.torch_call([full, partial])

    assert batch["input_ids"].shape == (1, 803)
    (cu_q, _cu_k), _max = prepare_fa_kwargs_from_position_ids(batch["position_ids"])
    n_segments = len(cu_q) - 1
    assert n_segments == 3, (
        f"{n_segments} attention segments for 3 documents — pad tokens are becoming length-1 "
        f"segments again, which is the 275x FA4-backward cliff and the 330s training steps"
    )


def test_completion_masked_packed_b2_flattened_spans():
    """Completion masking must run per-row BEFORE flattening: each document's template→EOS span
    lands at the right global offset in the [1, N] output, and the output still fires
    transformers' packed detection."""
    collator = DataCollatorForCompletionOnlyLMWithPacking(
        response_prompt_template=RESPONSE_TEMPLATE_IDS,
        tokenizer=make_tokenizer(),
    )
    doc = [USER_A, RESP_1, RESP_2, ASST_A, EOS]  # span at offsets 1..4 (boundary 0 masked)
    row0 = _packed_row(list(doc), list(doc))
    row1 = _packed_row(list(doc))
    batch = collator.torch_call([row0, row1])

    assert batch["input_ids"].shape == (1, 15)
    assert bool(_is_packed_sequence(batch["position_ids"], batch["input_ids"].shape[0]))
    unmasked = {i for i, v in enumerate(batch["labels"][0].tolist()) if v != IGNORE}
    # Docs start at 0, 5, 10 (row1's pads dropped); each trains template(+1,+2)+ASST(+3)+EOS(+4).
    expected = {1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14}
    assert unmasked == expected, f"span offsets corrupted by flattening: {sorted(unmasked)}"


def test_precomputed_labels_survive_b2_flattening():
    """Offline-baked (ragged) labels must survive a batch-2 packed collation: authoritative masks
    kept, boundary masking applied on top, flattened in row order."""
    collator = DataCollatorWithPacking(tokenizer=make_tokenizer())
    examples = [
        {"input_ids": [5, 6, 7, 8, 9, 12], "labels": [IGNORE, IGNORE, 7, 8, IGNORE, 12], "seq_lengths": [3, 3]},
        {"input_ids": [13, 14, 15], "labels": [13, IGNORE, 15], "seq_lengths": [3]},
    ]
    batch = collator.torch_call(examples)

    assert batch["input_ids"].shape == (1, 9)
    # Row0: baked labels with boundaries (0 and 3) masked; row1: boundary 0 masked, pad tail dropped.
    assert batch["labels"][0].tolist() == [IGNORE, IGNORE, 7, IGNORE, IGNORE, 12, IGNORE, IGNORE, 15]
    assert batch["position_ids"][0].tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2]


def test_flatten_without_real_row_lengths_fails_loudly():
    """``real_lengths`` is required, not optional: a plain reshape keeps every pad, and each pad
    (position_id 0) becomes its own varlen segment — the 275x FA4-backward cliff. A caller that
    omits them must fail at the call, never silently fall back to the padded reshape."""
    batch = {"input_ids": torch.zeros(2, 4, dtype=torch.long)}
    with pytest.raises(TypeError):
        flatten_packed_batch(batch)


def test_b1_single_doc_full_row_stays_dense():
    """A batch-1 pack holding ONE document with no padding is a plain increasing arange —
    transformers correctly treats it as a normal dense row (nothing to isolate)."""
    collator = DataCollatorWithPacking(tokenizer=make_tokenizer())
    batch = collator.torch_call([_packed_row([USER_A, ASST_A, ASST_B, EOS])])
    assert batch["input_ids"].shape == (1, 4)
    assert not bool(_is_packed_sequence(batch["position_ids"], batch["input_ids"].shape[0]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
