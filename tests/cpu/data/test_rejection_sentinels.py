#!/usr/bin/env python
"""Rejection-sentinel Arrow-type-stability tests.

An all-None rejection sentinel breaks multiprocess tokenization: when one map worker's contiguous
shard is entirely rejected (e.g. a long-form region of a concatenated dataset under a small
max_length) Arrow infers that worker's columns as ``null``, and aligning them with the other
workers' real ``list<int64>`` shards raises
``TypeError: Couldn't cast array of type list<item: int64> to null`` — crashing the whole map. An
all-``[]`` sentinel is equally broken (infers ``list<null>``): the sentinel must carry REAL typed
values, which :func:`create_tokenizer_none_example` guarantees (one-element typed lists,
``attention_mask=[0]``), with :func:`is_valid_example` as the shared drop-predicate. The VLM
preprocessing map instead pins its full output schema via explicit ``features=`` (its vision columns
are legitimately None on text-only rows, so typing can never be left to batch-wise inference there).

Run: pytest tests/cpu/data/test_rejection_sentinels.py
"""

import sys

import pyarrow as pa
import pytest
from datasets import Dataset

from src.data.pipeline.preprocessed_metadata import PreprocessingConfig
from src.data.pipeline.preprocessing import tokenize_dataset, tokenize_vlm_dataset
from src.data.pipeline.processing import process_dataset_with_map_and_filter
from src.data.pipeline.row_processors import (
    create_llm_processor,
    create_tokenizer_none_example,
    is_valid_example,
)
from tests.common.vlm_fakes import FakeVLMProcessorBase


@pytest.fixture(autouse=True)
def _isolated_datasets_cache(tmp_path, monkeypatch):
    """Point the coordinated-op cache at a fresh dir: a stale cache hit would skip the map under
    test (the crash under regression happens inside the map)."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets"))


class _WordTokenizer:
    """Deterministic whitespace tokenizer with the surface the LLM/text processors need."""

    name_or_path = "org/word-tokenizer"
    vocab_size = 32000
    bos_token = None
    bos_token_id = None
    eos_token_id = 7
    chat_template = "{{ messages }}"

    def __len__(self):
        return self.vocab_size

    def __call__(self, text, add_special_tokens=True, truncation=False, padding=False, max_length=None, **kwargs):
        ids = [2 + (i % 97) for i, _ in enumerate(text.split())]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
        return " ".join(m["content"] for m in conversation)

    def decode(self, ids, **kwargs):
        return " ".join("w" for _ in ids)


# The crash shape: worker 0's FIRST writer batch is entirely rejected while real rows land in a
# LATER batch of the SAME worker (an all-rejected shard alone is null-promoted at concatenation).
_WRITER_BATCH = 1000  # datasets' default writer_batch_size
_HALF = _WRITER_BATCH + 100
_NUM_REJECTED = _WRITER_BATCH + 50  # worker 0: batch 1 all-rejected, batch 2 mixes rejected + real
_NUM_VALID = 2 * _HALF - _NUM_REJECTED


def _chat_dataset(num_over_length: int, num_valid: int, over_words: int, valid_words: int) -> Dataset:
    """First ``num_over_length`` rows render to ``over_words`` tokens, the rest to ``valid_words``."""
    rows = [{"messages": [{"role": "user", "content": " ".join(["w"] * over_words)}]} for _ in range(num_over_length)]
    rows += [{"messages": [{"role": "user", "content": " ".join(["w"] * valid_words)}]} for _ in range(num_valid)]
    return Dataset.from_list(rows)


def test_sentinel_is_one_element_typed_row():
    """The sentinel must carry one-element lists of the REAL element type for every tokenizer output
    key — with attention_mask exactly [0] (the drop marker) — never None and never []."""
    sentinel = create_tokenizer_none_example(_WordTokenizer(), truncation=True, padding=False, max_length=8)
    assert set(sentinel) == {"input_ids", "attention_mask"}
    assert sentinel["attention_mask"] == [0]
    assert sentinel["input_ids"] == [2]  # first token of the tokenized sample text
    for values in sentinel.values():
        assert isinstance(values, list) and len(values) == 1
        assert all(isinstance(v, int) for v in values)


def test_sentinel_batch_infers_real_arrow_types():
    """An ENTIRE writer batch of sentinels must still infer real integer-list Arrow columns — the
    type-stability contract. All-None infers ``null`` and all-[] infers ``list<null>``; both crash
    the cast against real ``list<int64>`` batches from other workers."""
    sentinel = create_tokenizer_none_example(_WordTokenizer())
    ds = Dataset.from_list([dict(sentinel) for _ in range(4)])
    for column in ("input_ids", "attention_mask"):
        arrow_type = ds.data.schema.field(column).type
        assert pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type), (
            f"{column} inferred as {arrow_type}, not a list type"
        )
        assert not pa.types.is_null(arrow_type.value_type), (
            f"{column} element type inferred as null — the sentinel is not type-stable"
        )


def test_is_valid_example_predicate():
    """The shared predicate keeps real rows (including padded ones) and drops both sentinel shapes:
    the typed all-zero-attention_mask row and the legacy/VLM None row."""
    assert is_valid_example({"input_ids": [1, 2], "attention_mask": [1, 1]})
    assert is_valid_example({"input_ids": [1, 2, 0], "attention_mask": [1, 1, 0]})  # padded real row
    assert not is_valid_example(create_tokenizer_none_example(_WordTokenizer()))
    assert not is_valid_example({"input_ids": None, "attention_mask": None})
    # Non-sequence filter fields (prompt-style maps) are judged by content alone.
    assert is_valid_example({"prompt": "hello", "answer": "42"}, filter_field="prompt")
    assert not is_valid_example({"prompt": None, "answer": None}, filter_field="prompt")


def test_is_valid_example_blank_string_and_conversation_sentinels():
    """String/conversation prompt columns reject via type-stable BLANK sentinels (the GRPO scripts'
    shape), never None — the predicate must drop them, and an empty/whitespace-only prompt is
    genuinely invalid regardless of how it was produced."""
    assert not is_valid_example({"prompt": ""}, filter_field="prompt")
    assert not is_valid_example({"prompt": "  \n\t"}, filter_field="prompt")
    assert not is_valid_example({"prompt": []}, filter_field="prompt")
    # Blank conversation (the environmental_grpo sentinel): every message content blank.
    assert not is_valid_example({"prompt": [{"role": "user", "content": ""}]}, filter_field="prompt")
    assert not is_valid_example(
        {"prompt": [{"role": "system", "content": " "}, {"role": "user", "content": ""}]}, filter_field="prompt"
    )
    # Image-only content is content-bearing: content-part dicts are not messages.
    assert is_valid_example({"prompt": [{"role": "user", "content": "hi"}]}, filter_field="prompt")
    assert is_valid_example(
        {"prompt": [{"role": "user", "content": [{"type": "image", "image": "img0"}]}]}, filter_field="prompt"
    )
    assert is_valid_example(
        {"prompt": [{"role": "system", "content": ""}, {"role": "user", "content": "hi"}]}, filter_field="prompt"
    )


def test_map_and_filter_survives_all_rejected_writer_batch():
    """num_proc=2 over a dataset whose first worker's FIRST writer batch is entirely over-length:
    the all-None sentinel raised ``TypeError: Couldn't cast array of type list<item: int64> to
    null`` when the same worker later wrote real rows. The typed sentinel keeps every batch's
    schema identical; only the valid rows survive."""
    tokenizer = _WordTokenizer()
    processor = create_llm_processor(tokenizer, max_length=8, conversation_field="messages")
    dataset = _chat_dataset(num_over_length=_NUM_REJECTED, num_valid=_NUM_VALID, over_words=32, valid_words=4)

    result = process_dataset_with_map_and_filter(
        dataset,
        processor,
        remove_columns=["messages"],
        desc="regression tokenize",
        num_proc=2,
    )

    assert len(result) == _NUM_VALID
    for row in result.select(range(0, len(result), 100)):
        assert len(row["input_ids"]) == 4
        assert any(row["attention_mask"])


def test_tokenize_dataset_text_mode_survives_all_rejected_writer_batch():
    """Same writer-batch shape through the preprocessing text path (create_text_processor rejects
    empty documents): the first worker's first writer batch is entirely rejected, the map must not
    crash and the filter must keep only the real documents."""
    dataset = Dataset.from_dict({"text": [""] * _NUM_REJECTED + ["a few real words here"] * _NUM_VALID})
    config = PreprocessingConfig(model_name_or_path="fake/model", mode="text", max_length=16, num_proc=2)

    tokenized = tokenize_dataset(dataset, _WordTokenizer(), config)

    assert len(tokenized) == _NUM_VALID
    for row in tokenized.select(range(0, len(tokenized), 100)):
        assert row["labels"] == row["input_ids"]
        assert any(row["attention_mask"])


def test_tokenize_dataset_all_rejected_still_fails_loud():
    """The zero-surviving-rows guard must still fire on the new sentinel: a split whose every row is
    rejected raises instead of silently training on nothing."""
    dataset = Dataset.from_dict({"text": ["", "", ""]})
    config = PreprocessingConfig(model_name_or_path="fake/model", mode="text", max_length=16, num_proc=1)
    with pytest.raises(ValueError, match="Every example in the 'train' split was dropped"):
        tokenize_dataset(dataset, _WordTokenizer(), config)


def test_map_and_filter_survives_all_rejected_string_writer_batch():
    """GRPO-script shape (rlvr): a STRING prompt column whose first worker's first writer
    batch is entirely rejected. The blank-string sentinel keeps the Arrow schema string-typed (None
    rows inferred a null column and crashed the cast against later real batches), and the filter
    must actually drop the blank rows."""

    def process_prompt_row(row):
        if len(row["text"].split()) > 8:
            return {"prompt": "", "answer": ""}  # the scripts' rejection sentinel
        return {"prompt": row["text"], "answer": "42"}

    dataset = Dataset.from_dict({"text": [" ".join(["w"] * 32)] * _NUM_REJECTED + ["short prompt"] * _NUM_VALID})

    result = process_dataset_with_map_and_filter(
        dataset,
        process_prompt_row,
        filter_field="prompt",
        remove_columns=["text"],
        desc="regression string sentinel",
        num_proc=2,
    )

    assert len(result) == _NUM_VALID
    assert pa.types.is_string(result.data.schema.field("prompt").type)
    for row in result.select(range(0, len(result), 100)):
        assert row["prompt"] == "short prompt"


def test_map_and_filter_survives_all_rejected_conversation_writer_batch():
    """environmental_grpo shape: a MESSAGE-LIST prompt column with the blank-conversation sentinel.
    Same writer-batch crash class as the string column (all-None → null inference), plus the
    sentinel must keep the real struct element type so later real batches cast cleanly."""

    def process_conversation_row(row):
        if len(row["text"].split()) > 8:
            return {"prompt": [{"role": "user", "content": ""}], "answer": row["answer"]}
        return {"prompt": [{"role": "user", "content": row["text"]}], "answer": row["answer"]}

    dataset = Dataset.from_dict(
        {
            "text": [" ".join(["w"] * 32)] * _NUM_REJECTED + ["short prompt"] * _NUM_VALID,
            "answer": ["a"] * (2 * _HALF),
        }
    )

    result = process_dataset_with_map_and_filter(
        dataset,
        process_conversation_row,
        filter_field="prompt",
        remove_columns=["text"],
        desc="regression conversation sentinel",
        num_proc=2,
    )

    assert len(result) == _NUM_VALID
    prompt_type = result.data.schema.field("prompt").type
    assert pa.types.is_list(prompt_type) or pa.types.is_large_list(prompt_type)
    assert pa.types.is_struct(prompt_type.value_type), (
        f"prompt element type inferred as {prompt_type.value_type} — the sentinel is not type-stable"
    )
    for row in result.select(range(0, len(result), 100)):
        assert row["prompt"][0]["content"] == "short prompt"
        assert row["answer"] == "a"


# VLM path: the output schema is pinned via explicit features.


def test_tokenize_vlm_dataset_survives_all_rejected_writer_batch():
    """VLM regression: the first worker's first writer batch is entirely rejected (empty
    conversations → all-None rows), so without the explicitly pinned output features the writer
    inferred null columns and crashed casting the later real batch, exactly like the text path."""
    empty_row = {"conversation": []}
    good_row = {
        "conversation": [
            {"role": "user", "content": "hi there"},
            {"role": "assistant", "content": "ok"},
        ]
    }
    dataset = Dataset.from_list([empty_row] * _NUM_REJECTED + [good_row] * _NUM_VALID)
    config = PreprocessingConfig(
        model_name_or_path="fake/vlm",
        max_length=64,
        conversation_field="conversation",
        is_vlm=True,
        num_proc=2,
    )

    tokenized = tokenize_vlm_dataset(dataset, FakeVLMProcessorBase(), config, split_name="train")

    assert len(tokenized) == _NUM_VALID
    for row in tokenized.select(range(0, len(tokenized), 100)):
        assert len(row["input_ids"]) > 0
        assert row["pixel_values"] is None  # text-only rows keep nullable vision columns
    # The pinned schema decides the Arrow types; batch inference would call pixel_values null here.
    assert not pa.types.is_null(tokenized.data.schema.field("pixel_values").type)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
