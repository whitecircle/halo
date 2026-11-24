#!/usr/bin/env python
"""The VLM renders close a row's last assistant turn where the chat template leaves it open, at the text the
processor tokenizes: the runtime collator, the over-length filter and the offline bake take the same closing token,
the one the template writes after such a turn. Runs on GLM-5.3-Flash's processor, whose template ends a turn only at
the next role marker, and on Qwen3-VL's, which terminates its turns and stays as rendered."""

import json

import numpy as np
import pytest
from datasets import Dataset, DatasetDict
from PIL import Image

from halo.data.collators.vlm import VLMDataCollator
from halo.data.pipeline.vlm_rows import _filter_vlm_over_length, vlm_map_features
from halo.data.preprocessed import tokenization
from halo.data.preprocessed.metadata import PreprocessingConfig
from halo.data.preprocessed.preparation import preprocess_dataset
from halo.data.preprocessed.tokenization import tokenize_vlm_dataset
from halo.data.turn_markers import resolve_eos_token_ids
from halo.data.vlm_render import (
    extract_conversation_images,
    render_vlm_text,
    run_vlm_processor,
    vlm_row_template_kwargs,
)
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from tests.common.models import GLM5_3_FLASH, PINNED_REVISIONS, QWEN3_VL_2B
from tests.common.open_turn_templates import GLM_ANSWER_CLOSER, GLM_ASSISTANT_MARKER, GLM_CALL_CLOSER, TWO_EXCHANGES
from tests.common.tokenizers import load_cached_config, load_cached_processor

MAX_LENGTH = 4096
MARKERS = {GLM5_3_FLASH: GLM_ASSISTANT_MARKER, QWEN3_VL_2B: "<|im_start|>assistant\n"}


@pytest.fixture(autouse=True)
def _isolated_datasets_cache(tmp_path, monkeypatch):
    """Fresh coordinated-op cache per test: a stale cache hit would skip the map under test."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets"))


def _load(model_name):
    revision = PINNED_REVISIONS[model_name]
    return load_cached_processor(model_name, revision=revision), load_cached_config(model_name, revision=revision)


def _image_turns() -> list[dict]:
    image = Image.fromarray(np.random.RandomState(0).randint(0, 255, (100, 140, 3), dtype=np.uint8))
    return [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "What is it?"}]},
        {"role": "assistant", "content": "A noise patch."},
    ]


def _row(conversation) -> dict:
    history, images = extract_conversation_images(conversation)
    return {"history": history, "images": images, "tools_json": None, "reasoning_effort": None}


def _plain(processor, row) -> dict:
    """The row as its template renders it, unpadded: what the processor returns with nothing appended."""
    return run_vlm_processor(processor, [render_vlm_text(processor, row["history"])], [row["images"]], padding=False)


def _collator(model_name, processor, config, *, completions_only: bool, max_length: int = MAX_LENGTH):
    return VLMDataCollator(
        processor,
        processor.tokenizer,
        max_length,
        train_on_completions_only=completions_only,
        response_prompt_template=MARKERS[model_name] if completions_only else None,
        model_config=config,
    )


def _real(batch, key: str, index: int) -> list[int]:
    return batch[key][index][batch["attention_mask"][index].bool()].tolist()


@pytest.mark.parametrize("completions_only", [True, False], ids=["completions-only", "all-tokens"])
def test_the_runtime_collator_closes_an_open_last_turn(completions_only):
    """Each row of a padded batch, the image row included, is its render plus the ``<|user|>`` the template writes after
    an answer, in every per-token column the processor builds, and that stop trains."""
    processor, config = _load(GLM5_3_FLASH)
    tokenizer = processor.tokenizer
    closer = tokenizer.convert_tokens_to_ids(GLM_ANSWER_CLOSER)
    collator = _collator(GLM5_3_FLASH, processor, config, completions_only=completions_only)
    rows = [_row(_image_turns()), _row(TWO_EXCHANGES)]
    batch = collator(rows)
    assert not batch["attention_mask"].all(), "rows of one length would pad nothing"

    for index, row in enumerate(rows):
        plain = _plain(processor, row)
        plain_ids = plain["input_ids"][0].tolist()
        assert plain_ids[-1] not in collator.eos_token_ids, "the template closes its last turn"
        assert _real(batch, "input_ids", index) == [*plain_ids, closer]
        assert _real(batch, "mm_token_type_ids", index) == [*plain["mm_token_type_ids"][0].tolist(), 0]
        assert _real(batch, "labels", index)[-1] == closer, "the stop is not trained"
    trained = tokenizer.decode([token for token in _real(batch, "labels", 1) if token != LABEL_IGNORE_INDEX])
    assert "delta" in trained and (not completions_only or "charlie" not in trained), trained
    assert batch["pixel_values"].shape == _plain(processor, rows[0])["pixel_values"].shape


@pytest.mark.parametrize("tools", [True, False], ids=["tools", "no-tools"])
def test_a_row_ending_on_a_tool_call_closes_with_the_tool_replys_marker(tools):
    """The call is read off the message's folded fields: its turn ends where the template opens the tool reply
    (``<|observation|>``), as the call ends mid-row, not at the ``<|user|>`` after an answer."""
    processor, config = _load(GLM5_3_FLASH)
    tokenizer = processor.tokenizer
    call = {"type": "function", "id": "call_0", "function": {"name": "lookup", "arguments": {"key": "alpha"}}}
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}
    conversation = [*_image_turns()[:1], {"role": "assistant", "content": "", "tool_calls": [call]}]
    row = {**_row(conversation), "tools_json": json.dumps([tool]) if tools else None}
    batch = _collator(GLM5_3_FLASH, processor, config, completions_only=True)([row])
    plain = run_vlm_processor(
        processor, [render_vlm_text(processor, row["history"], **vlm_row_template_kwargs(row))], [row["images"]]
    )["input_ids"][0].tolist()
    closer = tokenizer.convert_tokens_to_ids(GLM_CALL_CLOSER)
    assert tokenizer.decode(plain).endswith("</tool_call>"), tokenizer.decode(plain[-8:])
    assert _real(batch, "input_ids", 0) == [*plain, closer]
    assert _real(batch, "labels", 0)[-1] == closer, "the stop is not trained"


def test_a_row_ending_on_a_user_turn_is_left_as_rendered():
    processor, config = _load(GLM5_3_FLASH)
    row = _row(TWO_EXCHANGES[:3])
    batch = _collator(GLM5_3_FLASH, processor, config, completions_only=False)([row])
    assert batch["input_ids"][0].tolist() == _plain(processor, row)["input_ids"][0].tolist()


@pytest.mark.parametrize("completions_only", [True, False], ids=["completions-only", "all-tokens"])
def test_a_template_that_terminates_its_turns_is_left_as_rendered(completions_only):
    processor, config = _load(QWEN3_VL_2B)
    rows = [_row(_image_turns()), _row(TWO_EXCHANGES)]
    batch = _collator(QWEN3_VL_2B, processor, config, completions_only=completions_only)(rows)
    for index, row in enumerate(rows):
        assert _real(batch, "input_ids", index) == _plain(processor, row)["input_ids"][0].tolist()


def _mapped(*conversations) -> DatasetDict:
    return DatasetDict({"train": Dataset.from_list([_row(c) for c in conversations], features=vlm_map_features())})


def test_the_over_length_filter_measures_the_closed_row():
    """A text row the closing token takes one token over ``max_length`` would pass a filter measuring the bare render,
    then raise in the collator of the rank that drew it."""
    processor, config = _load(GLM5_3_FLASH)
    tokenizer = processor.tokenizer
    eos_token_ids = resolve_eos_token_ids(tokenizer, config)
    bare_length = _plain(processor, _row(TWO_EXCHANGES))["input_ids"].shape[1]
    ds = _mapped(TWO_EXCHANGES, TWO_EXCHANGES[:2])

    kept = _filter_vlm_over_length(ds, processor, tokenizer, bare_length, 1, None, eos_token_ids)["train"]
    assert [len(row["history"]) for row in kept] == [2], "the row was measured before it was closed"
    collator = _collator(GLM5_3_FLASH, processor, config, completions_only=False, max_length=bare_length)
    with pytest.raises(ValueError, match="1 tokens over max_length"):
        collator([_row(TWO_EXCHANGES)])
    with_headroom = _filter_vlm_over_length(ds, processor, tokenizer, bare_length + 1, 1, None, eos_token_ids)
    assert len(with_headroom["train"]) == 2


def test_the_over_length_filter_keeps_a_terminated_row_of_the_budgets_length():
    processor, config = _load(QWEN3_VL_2B)
    tokenizer = processor.tokenizer
    bare_length = _plain(processor, _row(TWO_EXCHANGES))["input_ids"].shape[1]
    eos_token_ids = resolve_eos_token_ids(tokenizer, config)
    kept = _filter_vlm_over_length(_mapped(TWO_EXCHANGES), processor, tokenizer, bare_length, 1, None, eos_token_ids)
    assert len(kept["train"]) == 1


def _bake_config(model_name, *, completions_only: bool, max_length: int = MAX_LENGTH) -> PreprocessingConfig:
    return PreprocessingConfig(
        model_name_or_path=model_name,
        max_length=max_length,
        conversation_field="conversation",
        is_vlm=True,
        num_proc=1,
        train_on_completions_only=completions_only,
        assistant_message_template=MARKERS[model_name] if completions_only else None,
    )


@pytest.mark.parametrize("model_name", [GLM5_3_FLASH, QWEN3_VL_2B])
@pytest.mark.parametrize("completions_only", [True, False], ids=["completions-only", "all-tokens"])
def test_the_offline_bake_closes_a_row_as_the_runtime_collator_does(model_name, completions_only):
    processor, config = _load(model_name)
    eos_token_ids = resolve_eos_token_ids(processor.tokenizer, config)
    dataset = Dataset.from_list([{"conversation": TWO_EXCHANGES}])
    bake_config = _bake_config(model_name, completions_only=completions_only)

    baked = tokenize_vlm_dataset(dataset, processor, bake_config, "train", eos_token_ids=eos_token_ids)
    batch = _collator(model_name, processor, config, completions_only=completions_only)([_row(TWO_EXCHANGES)])

    assert baked[0]["input_ids"] == batch["input_ids"][0].tolist()
    assert baked[0]["labels"] == batch["labels"][0].tolist()
    bare = _plain(processor, _row(TWO_EXCHANGES))["input_ids"][0].tolist()
    closing = [processor.tokenizer.convert_tokens_to_ids(GLM_ANSWER_CLOSER)] if model_name == GLM5_3_FLASH else []
    assert baked[0]["input_ids"] == [*bare, *closing]


def test_the_offline_bake_measures_the_closed_row():
    processor, config = _load(GLM5_3_FLASH)
    eos_token_ids = resolve_eos_token_ids(processor.tokenizer, config)
    bare_length = _plain(processor, _row(TWO_EXCHANGES))["input_ids"].shape[1]
    dataset = Dataset.from_list([{"conversation": TWO_EXCHANGES}, {"conversation": TWO_EXCHANGES[:2]}])
    bake_config = _bake_config(GLM5_3_FLASH, completions_only=False, max_length=bare_length)
    baked = tokenize_vlm_dataset(dataset, processor, bake_config, "train", eos_token_ids=eos_token_ids)
    assert len(baked) == 1 and len(baked[0]["input_ids"]) < bare_length, "the long row was measured before it closed"


def test_a_full_sequence_bake_reads_the_terminators_from_the_model_config(monkeypatch):
    """Once for both splits, completion-only labels or not: the role markers that say whether a turn is closed are
    listed only there."""
    processor, config = _load(GLM5_3_FLASH)
    reads = []

    def _read(*args, **kwargs):
        reads.append(args)
        return config

    monkeypatch.setattr(tokenization.AutoConfig, "from_pretrained", _read)
    split = Dataset.from_list([{"conversation": TWO_EXCHANGES}])
    result = preprocess_dataset(
        DatasetDict({"train": split, "test": split}),
        processor,
        _bake_config(GLM5_3_FLASH, completions_only=False),
    )
    closer = processor.tokenizer.convert_tokens_to_ids(GLM_ANSWER_CLOSER)
    assert [result[split][0]["input_ids"][-1] for split in ("train", "test")] == [closer, closer]
    assert len(reads) == 1, f"the model config was read {len(reads)} times for two splits"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
