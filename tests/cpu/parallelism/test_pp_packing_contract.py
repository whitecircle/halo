#!/usr/bin/env python
"""Packing under pipeline parallelism: fixed shapes without losing cross-document isolation.

PP freezes its P2P buffers on the first step, so every batch must have one shape. Packing normally
flattens ``[B, L] -> [1, B*L]`` because transformers only engages packed-sequence handling at batch
size 1 — but a ``[1, N]`` tensor cannot be microbatched at all (the schedule splits dim 0), and its
width varies with the bin lengths. Under PP the rows stay, and the microbatch split IS the
flattening: each microbatch arrives as ``[1, max_length]``.

These tests pin the three things that make that safe, each of which fails SILENTLY if broken:
the rows survive the collator, the pad tail is an inert document, and ``seq_lengths`` survives
HF's column pruning (without it every packed row collates as ONE document).

Run: pytest tests/cpu/parallelism/test_pp_packing_contract.py
"""

from unittest.mock import MagicMock

import pytest
import torch
from torch.distributed.pipelining.microbatch import split_args_kwargs_into_chunks

from src.data.collators.fixed_shape import FixedShapeCollator, pad_to_fixed_length
from src.data.collators.packing import (
    PAD_TAIL_SEGMENT_CHUNK,
    DataCollatorForCompletionOnlyLMWithPacking,
    DataCollatorWithPacking,
    pad_tail_positions,
)
from src.distributed.pipeline_parallel.runner import resolve_batch_pad_values
from src.kernels.logprobs import loss_token_counts_per_row

PAD = 0
IGNORE = -100
MAX_LENGTH = 16


def make_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.pad_token_id = PAD
    tok.padding_side = "right"
    tok.pad = lambda encoded, **kwargs: _pad(encoded)
    return tok


def _pad(encoded: list[dict]) -> dict:
    """Right-pad to the batch max, mirroring tokenizer.pad(padding=True)."""
    width = max(len(row["input_ids"]) for row in encoded)
    out: dict[str, torch.Tensor] = {}
    for key, fill in (("input_ids", PAD), ("attention_mask", 0)):
        out[key] = torch.tensor([row.get(key, []) + [fill] * (width - len(row.get(key, []))) for row in encoded])
    return out


def packed_rows(*bins: list[int]) -> list[dict]:
    """One dataset row per packed bin; ``seq_lengths`` is what marks it as packed."""
    return [{"input_ids": [i + 100 for i in range(sum(lengths))], "seq_lengths": list(lengths)} for lengths in bins]


def collate(collator, rows: list[dict]) -> dict:
    return collator.torch_call([dict(row) for row in rows])


def test_pp_mode_keeps_rows_so_the_batch_can_be_microbatched():
    """``flatten_to_single_row = False`` must preserve dim 0 — the axis the schedule splits on.

    The default (flattening) collapses the batch to one row, which the pipeline cannot chunk: this
    asserts the torch behavior directly, so a torch upgrade that changed it would surface here
    rather than as a mid-training ``Expecting N arg_mbs but got 1``.
    """
    collator = DataCollatorWithPacking(tokenizer=make_tokenizer())
    rows = packed_rows([4, 4], [3, 5])

    collator.flatten_to_single_row = True
    flat = collate(collator, rows)
    assert flat["input_ids"].shape[0] == 1
    chunks, _ = split_args_kwargs_into_chunks((flat["input_ids"],), {}, 2)
    assert len(chunks) == 1, "a flattened packed batch cannot be split into microbatches"

    collator.flatten_to_single_row = False
    kept = collate(collator, rows)
    assert kept["input_ids"].shape[0] == 2
    chunks, _ = split_args_kwargs_into_chunks((kept["input_ids"],), {}, 2)
    assert len(chunks) == 2
    assert all(chunk[0].shape[0] == 1 for chunk in chunks), "each microbatch must carry exactly one packed row"


def test_ragged_bins_pad_to_one_fixed_shape():
    """Bins are ``<= max_length`` and vary per batch; the pipeline needs one frozen shape.

    Two batches whose packed widths differ must both leave ``FixedShapeCollator`` at
    ``[B, max_length]`` — otherwise the second step writes into P2P buffers sized for the first.
    """
    inner = DataCollatorWithPacking(tokenizer=make_tokenizer())
    inner.flatten_to_single_row = False
    fixed = FixedShapeCollator(inner, MAX_LENGTH, resolve_batch_pad_values(PAD))

    narrow = fixed([dict(row) for row in packed_rows([2, 3], [4, 1])])
    wide = fixed([dict(row) for row in packed_rows([5, 6], [7, 4])])

    assert narrow["input_ids"].shape == (2, MAX_LENGTH)
    assert wide["input_ids"].shape == (2, MAX_LENGTH)
    assert narrow["labels"].shape == wide["labels"].shape


def test_pad_tail_is_an_inert_document():
    """The trailing pad region must contribute no loss, open no attention path into the real
    documents, and stay a HANDFUL of segments.

    The tail's position_ids are a ramp restarting every ``PAD_TAIL_SEGMENT_CHUNK`` tokens: the reset
    at its start seals it off from the last real document, and the chunking is load-bearing — a
    constant-0 tail makes every pad its own varlen segment, and the FA4 backward pays a fixed cost
    per segment (measured 275x, 6.9 s vs 25 ms per layer, on a 34k-pad partial pack). A pad value
    that drifted to the tokenizer pad id would splice the tail onto the last real document instead.
    """
    inner = DataCollatorWithPacking(tokenizer=make_tokenizer())
    inner.flatten_to_single_row = False
    fixed = FixedShapeCollator(inner, MAX_LENGTH, resolve_batch_pad_values(PAD))

    batch = fixed([dict(row) for row in packed_rows([3, 4], [5, 2])])

    assert "attention_mask" not in batch, "a dense mask makes FlashAttention treat the row as padding-only"
    assert torch.equal(
        batch["labels"][batch["position_ids"] == 0],
        torch.full_like(batch["labels"][batch["position_ids"] == 0], IGNORE),
    )
    for row, real_tokens in enumerate((7, 7)):
        assert torch.all(batch["labels"][row, real_tokens:] == IGNORE), "pad tail must carry no loss"
        tail = batch["position_ids"][row, real_tokens:]
        assert tail[0].item() == 0, "the tail must open with a reset — anything else extends the last document"
        expected = pad_tail_positions(tail.numel(), tail.dtype)
        assert torch.equal(tail, expected), "the tail must be the chunked ramp, not per-pad zero segments"
        n_segments = int((batch["position_ids"][row] == 0).sum())
        docs = 2
        assert n_segments <= docs + tail.numel() // PAD_TAIL_SEGMENT_CHUNK + 1, (
            f"{n_segments} segments in one PP row — per-pad segments are the 275x FA4-backward cliff"
        )


def test_row_form_and_flat_form_score_the_same_tokens():
    """The pipeline's loss normalizer must not change because the rows were kept.

    ``loss_token_counts_per_row`` drops one target per row; in the kept-rows form that extra dropped target
    is the next row's first token, which is already ignored (it sits at ``position_ids == 0``). So
    both forms must count identically — if they diverge, PP packing silently rescales the loss.
    """
    tokenizer = make_tokenizer()
    rows = packed_rows([4, 4], [3, 5])

    flat_collator = DataCollatorWithPacking(tokenizer=tokenizer)
    flat_collator.flatten_to_single_row = True
    flat = collate(flat_collator, rows)

    kept_collator = DataCollatorWithPacking(tokenizer=tokenizer)
    kept_collator.flatten_to_single_row = False
    kept = collate(kept_collator, rows)

    assert loss_token_counts_per_row(kept["labels"]).sum() == loss_token_counts_per_row(flat["labels"]).sum()


@pytest.mark.parametrize("collator_cls", [DataCollatorWithPacking, DataCollatorForCompletionOnlyLMWithPacking])
def test_the_column_survives_the_pipeline_wrapper(collator_cls):
    """``seq_lengths`` is what tells the collator a row is packed, and the pin reads the WRAPPED
    collator — so the collator class declaring the column is not enough on its own.

    PP overwrites ``_signature_columns`` with its own pin, and HF prunes every dataset column
    outside it. If ``seq_lengths`` is dropped the collator sees an unpacked row: no resetting
    position_ids, no boundary label masking, and the dense mask retained — the whole row attends as
    ONE document, with no error anywhere.

    ``_setup_pipeline_parallel`` wraps the collator in ``FixedShapeCollator`` first — deliberately,
    so the pin can union the inner collator's columns — and then reads
    ``getattr(self.data_collator, "required_dataset_columns", ())`` off the wrapper. A wrapper that
    does not forward the attribute answers ``()``, HF prunes ``seq_lengths``, and every packed row
    collates as ONE attended document with a finite, plausible loss and no error. Asserting the
    class attribute alone passes in exactly that state, which is why this reproduces the real
    expression instead.
    """
    kwargs = {"tokenizer": make_tokenizer()}
    if collator_cls is DataCollatorForCompletionOnlyLMWithPacking:
        kwargs["response_prompt_template"] = [1, 2]  # required; irrelevant to the column contract
    wrapped = FixedShapeCollator(
        inner=collator_cls(**kwargs), max_length=MAX_LENGTH, pad_values=resolve_batch_pad_values(PAD)
    )
    pinned = getattr(wrapped, "required_dataset_columns", ())
    assert "seq_lengths" in pinned, (
        "the pipeline's fixed-shape wrapper hid the packing collator's column requirement; "
        "HF's column pruning will drop seq_lengths and every packed row will attend as one document"
    )


def test_the_wrapper_reports_nothing_extra_for_a_plain_collator():
    """Anti-over-fitting: forwarding must not invent columns a non-packing collator never declared,
    which would keep dead columns alive through HF's pruning and into the runtime contract."""
    wrapped = FixedShapeCollator(inner=object(), max_length=MAX_LENGTH, pad_values=resolve_batch_pad_values(PAD))
    assert wrapped.required_dataset_columns == ()


def test_over_length_raises_in_each_callers_words():
    """An over-length tensor raises rather than being truncated: the collator says the frozen shape,
    the trainers' own pre-padding the pipeline's fixed shape, each followed by its caller's remedy."""
    batch = {"input_ids": torch.ones(1, MAX_LENGTH + 1, dtype=torch.long)}
    fixed = FixedShapeCollator(lambda features: dict(batch), MAX_LENGTH, resolve_batch_pad_values(PAD))
    with pytest.raises(ValueError, match=rf"tokens but the frozen shape is max_length={MAX_LENGTH};.*Truncate in"):
        fixed([])
    with pytest.raises(ValueError, match=rf"tokens, over the pipeline's fixed shape max_length={MAX_LENGTH};.*remedy"):
        pad_to_fixed_length({"labels": (torch.zeros(2, MAX_LENGTH + 3), IGNORE)}, MAX_LENGTH, "remedy")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
