#!/usr/bin/env python
"""Rejection sentinels keep Arrow types stable across map workers: an all-None or all-``[]`` row infers ``null`` or
``list<null>``, which raises ``TypeError`` against real ``list<int64>`` shards. :func:`build_rejection_row` and
:func:`blank_conversation` carry real types; the VLM map pins its schema via ``features=``."""

import pyarrow as pa
import pytest
from datasets import Dataset

from halo.configs.grpo_online import OnlineGRPOScriptArguments
from halo.data.pipeline.coordinated_map import coordinated_map_and_filter
from halo.data.pipeline.rejected_rows import blank_conversation, build_rejection_row, is_valid_example
from halo.data.pipeline.text_rows import build_chat_row_processor
from halo.data.preprocessed.metadata import PreprocessingConfig
from halo.data.preprocessed.tokenization import tokenize_dataset, tokenize_vlm_dataset
from scripts.training.grpo.online import build_rlvr_row_fn
from tests.common.vlm_fakes import FakeVLMProcessorBase


@pytest.fixture(autouse=True)
def _isolated_datasets_cache(tmp_path, monkeypatch):
    """A fresh coordinated-op cache: a stale hit would skip the map under test."""
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
        def text(content):
            return content if isinstance(content, str) else " ".join(part["text"] for part in content)

        return " ".join(text(m["content"]) for m in conversation)

    def decode(self, ids, **kwargs):
        return " ".join("w" for _ in ids)


# The crash shape: worker 0's first writer batch is entirely rejected while real rows land in a later batch of the same
# worker (an all-rejected shard alone is null-promoted at concatenation).
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
    """Never None and never []; attention_mask is exactly [0], the drop marker."""
    sentinel = build_rejection_row(_WordTokenizer(), truncation=True, padding=False, max_length=8)
    assert set(sentinel) == {"input_ids", "attention_mask"}
    assert sentinel["attention_mask"] == [0]
    assert sentinel["input_ids"] == [2]  # first token of the tokenized sample text
    for values in sentinel.values():
        assert isinstance(values, list) and len(values) == 1
        assert all(isinstance(v, int) for v in values)


def test_sentinel_batch_infers_real_arrow_types():
    sentinel = build_rejection_row(_WordTokenizer())
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
    """Keeps real rows, padded included, and drops the typed all-zero-mask row and the None row."""
    assert is_valid_example({"input_ids": [1, 2], "attention_mask": [1, 1]})
    assert is_valid_example({"input_ids": [1, 2, 0], "attention_mask": [1, 1, 0]})  # padded real row
    assert not is_valid_example(build_rejection_row(_WordTokenizer()))
    assert not is_valid_example({"input_ids": None, "attention_mask": None})
    # Non-sequence filter fields (prompt-style maps) are judged by content alone.
    assert is_valid_example({"prompt": "hello", "answer": "42"}, filter_field="prompt")
    assert not is_valid_example({"prompt": None, "answer": None}, filter_field="prompt")


def test_is_valid_example_blank_string_and_conversation_sentinels():
    """String and conversation columns reject via type-stable blank sentinels, which the predicate must drop."""
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


def test_blank_conversation_keeps_the_content_shape_and_is_dropped():
    """A string beside the real rows' part lists cannot share one Arrow column, so blanked text parts stay parts. A
    non-text part cannot be blanked into something the filter drops, so it is refused."""
    parts = [{"role": "user", "content": [{"type": "text", "text": "q"}, {"type": "text", "text": "r"}]}]
    blank = blank_conversation(parts)
    assert blank == [{"role": "user", "content": [{"type": "text", "text": ""}, {"type": "text", "text": ""}]}]
    assert not is_valid_example({"prompt": blank}, filter_field="prompt")
    assert is_valid_example({"prompt": parts}, filter_field="prompt")
    assert blank_conversation([{"role": "user", "content": "q", "name": "asker"}]) == [
        {"role": "user", "content": "", "name": "asker"}
    ]
    assert blank_conversation() == [{"role": "user", "content": ""}]
    with pytest.raises(ValueError, match="Only text content can be blanked"):
        blank_conversation([{"role": "user", "content": [{"type": "image", "image": "img0"}]}])


def test_map_and_filter_survives_all_rejected_writer_batch():
    """The first worker's first writer batch is entirely over-length; an all-None sentinel raises when the worker
    later writes real rows."""
    tokenizer = _WordTokenizer()
    processor = build_chat_row_processor(tokenizer, max_length=8, conversation_field="messages")
    dataset = _chat_dataset(num_over_length=_NUM_REJECTED, num_valid=_NUM_VALID, over_words=32, valid_words=4)

    result = coordinated_map_and_filter(
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
    """The same writer-batch shape through the preprocessing text path, where build_text_row_processor rejects empty
    documents."""
    dataset = Dataset.from_dict({"text": [""] * _NUM_REJECTED + ["a few real words here"] * _NUM_VALID})
    config = PreprocessingConfig(model_name_or_path="fake/model", mode="text", max_length=16, num_proc=2)

    tokenized = tokenize_dataset(dataset, _WordTokenizer(), config)

    assert len(tokenized) == _NUM_VALID
    for row in tokenized.select(range(0, len(tokenized), 100)):
        assert row["input_ids"] and all(row["attention_mask"])


def test_tokenize_dataset_all_rejected_still_fails_loud():
    """A split whose every row is rejected raises and does not train on nothing."""
    dataset = Dataset.from_dict({"text": ["", "", ""]})
    config = PreprocessingConfig(model_name_or_path="fake/model", mode="text", max_length=16, num_proc=1)
    with pytest.raises(ValueError, match="Every example in the 'train' split was dropped"):
        tokenize_dataset(dataset, _WordTokenizer(), config)


def test_map_and_filter_survives_all_rejected_string_writer_batch():
    """A string prompt column: None rows infer a null column and crash the cast, and the filter must drop the blank
    rows."""

    def process_prompt_row(row):
        if len(row["text"].split()) > 8:
            return {"prompt": "", "answer": ""}  # the scripts' rejection sentinel
        return {"prompt": row["text"], "answer": "42"}

    dataset = Dataset.from_dict({"text": [" ".join(["w"] * 32)] * _NUM_REJECTED + ["short prompt"] * _NUM_VALID})

    result = coordinated_map_and_filter(
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
    """A message-list column: the sentinel keeps every key of the rejected messages, so later real batches cast
    cleanly."""

    def process_conversation_row(row):
        messages = [{"role": "user", "content": row["text"], "name": "asker"}]
        if len(row["text"].split()) > 8:
            return {"prompt": blank_conversation(messages), "answer": row["answer"]}
        return {"prompt": messages, "answer": row["answer"]}

    dataset = Dataset.from_dict(
        {
            "text": [" ".join(["w"] * 32)] * _NUM_REJECTED + ["short prompt"] * _NUM_VALID,
            "answer": ["a"] * (2 * _HALF),
        }
    )

    result = coordinated_map_and_filter(
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
        assert row["prompt"] == [{"role": "user", "content": "short prompt", "name": "asker"}]
        assert row["answer"] == "a"


def _text_content(text: str) -> str:
    return text


def _parts_content(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


@pytest.mark.parametrize("content", [_text_content, _parts_content], ids=["string-content", "text-parts-content"])
def test_rlvr_rows_survive_an_all_rejected_writer_batch(content):
    """An over-budget row's empty conversation infers ``list<null>`` and the real structs after it fail the cast.
    The blanked conversation keeps the struct, text parts staying parts, and the blank prompt drops the row."""
    args = OnlineGRPOScriptArguments(dataset="org/rlvr", prompt_field="messages", max_prompt_length=8)
    long_turn = [{"role": "user", "content": content(" ".join(["w"] * 32))}]
    short_turn = [{"role": "user", "content": content("short prompt")}]
    dataset = Dataset.from_dict(
        {
            "messages": [long_turn] * _NUM_REJECTED + [short_turn] * _NUM_VALID,
            "answer": [str(i) for i in range(2 * _HALF)],
        }
    )

    result = coordinated_map_and_filter(
        dataset,
        build_rlvr_row_fn(args, _WordTokenizer()),
        filter_field="prompt",
        remove_columns=["messages"],
        desc="regression rlvr sentinel",
        num_proc=2,
    )

    assert len(result) == _NUM_VALID
    conversation_type = result.data.schema.field("conversation").type
    assert pa.types.is_struct(conversation_type.value_type), (
        f"conversation element type inferred as {conversation_type.value_type} — the sentinel is not type-stable"
    )
    for offset in range(0, len(result), 100):
        row = result[offset]
        assert row["prompt"] == "short prompt"
        assert row["conversation"] == short_turn
        assert row["answer"] == str(_NUM_REJECTED + offset)


def test_tokenize_vlm_dataset_survives_all_rejected_writer_batch():
    """Without the pinned output features the writer infers null columns and crashes casting the later real batch,
    like the text path."""
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
    raise SystemExit(pytest.main([__file__, "-v"]))
