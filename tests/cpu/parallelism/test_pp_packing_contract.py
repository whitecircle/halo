#!/usr/bin/env python
"""Packing under pipeline parallelism: PP freezes P2P buffers on the first step, so the rows stay and the microbatch
split is the flattening. Each contract here fails silently if broken."""

from unittest.mock import MagicMock

import pytest
import torch
from torch.distributed.pipelining.microbatch import split_args_kwargs_into_chunks

from halo.data.collators.fixed_shape import FixedShapeCollator, pad_to_fixed_length
from halo.data.collators.packing import (
    PAD_TAIL_SEGMENT_CHUNK,
    DataCollatorForCompletionOnlyLMWithPacking,
    DataCollatorWithPacking,
    pad_tail_positions,
)
from halo.distributed.pipeline_parallel.runner import resolve_batch_pad_values
from halo.kernels.logprobs import loss_token_counts_per_row

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
    """Mirrors tokenizer.pad(padding=True)."""
    width = max(len(row["input_ids"]) for row in encoded)
    out: dict[str, torch.Tensor] = {}
    for key, fill in (("input_ids", PAD), ("attention_mask", 0)):
        out[key] = torch.tensor([row.get(key, []) + [fill] * (width - len(row.get(key, []))) for row in encoded])
    return out


def packed_rows(*bins: list[int]) -> list[dict]:
    """``seq_lengths`` marks it as packed."""
    return [{"input_ids": [i + 100 for i in range(sum(lengths))], "seq_lengths": list(lengths)} for lengths in bins]


def collate(collator, rows: list[dict]) -> dict:
    return collator.torch_call([dict(row) for row in rows])


def test_pp_mode_keeps_rows_so_the_batch_can_be_microbatched():
    """A torch upgrade changing this surfaces here rather than as a mid-training ``Expecting N arg_mbs but got 1``."""
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
    """Otherwise the second step writes into P2P buffers sized for the first."""
    inner = DataCollatorWithPacking(tokenizer=make_tokenizer())
    inner.flatten_to_single_row = False
    fixed = FixedShapeCollator(inner, MAX_LENGTH, resolve_batch_pad_values(PAD))

    narrow = fixed([dict(row) for row in packed_rows([2, 3], [4, 1])])
    wide = fixed([dict(row) for row in packed_rows([5, 6], [7, 4])])

    assert narrow["input_ids"].shape == (2, MAX_LENGTH)
    assert wide["input_ids"].shape == (2, MAX_LENGTH)
    assert narrow["labels"].shape == wide["labels"].shape


def test_pad_tail_is_an_inert_document():
    """A constant-0 tail makes every pad a varlen segment, and FA4's backward pays per segment (275x, 6.9 s vs 25 ms
    per layer, on a 34k-pad partial pack)."""
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
    """The extra dropped target in row form is already ignored at ``position_ids == 0``; if counts diverge, PP
    silently rescales the loss."""
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
    """The pin reads the wrapped collator; a wrapper not forwarding the attribute lets HF prune ``seq_lengths``, and
    every row attends as one document."""
    kwargs = {"tokenizer": make_tokenizer()}
    if collator_cls is DataCollatorForCompletionOnlyLMWithPacking:
        kwargs["response_prompt_template"] = [1, 2]  # Required; irrelevant to the column contract.
    wrapped = FixedShapeCollator(
        inner=collator_cls(**kwargs), max_length=MAX_LENGTH, pad_values=resolve_batch_pad_values(PAD)
    )
    pinned = getattr(wrapped, "required_dataset_columns", ())
    assert "seq_lengths" in pinned, (
        "the pipeline's fixed-shape wrapper hid the packing collator's column requirement; "
        "HF's column pruning will drop seq_lengths and every packed row will attend as one document"
    )


def test_the_wrapper_reports_nothing_extra_for_a_plain_collator():
    """Forwarding must not invent columns, which would keep dead columns alive through HF's pruning."""
    wrapped = FixedShapeCollator(inner=object(), max_length=MAX_LENGTH, pad_values=resolve_batch_pad_values(PAD))
    assert wrapped.required_dataset_columns == ()


def test_over_length_raises_in_each_callers_words():
    batch = {"input_ids": torch.ones(1, MAX_LENGTH + 1, dtype=torch.long)}
    fixed = FixedShapeCollator(lambda features: dict(batch), MAX_LENGTH, resolve_batch_pad_values(PAD))
    with pytest.raises(ValueError, match=rf"tokens but the frozen shape is max_length={MAX_LENGTH};.*Truncate in"):
        fixed([])
    with pytest.raises(ValueError, match=rf"tokens, over the pipeline's fixed shape max_length={MAX_LENGTH};.*remedy"):
        pad_to_fixed_length({"labels": (torch.zeros(2, MAX_LENGTH + 3), IGNORE)}, MAX_LENGTH, "remedy")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
