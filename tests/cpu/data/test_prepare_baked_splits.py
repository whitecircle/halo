"""A prepared dataset carries every input split it was handed, or refuses the input by name.

The artifact holds only ``train`` and ``test``, and training evaluates on a placeholder cut from train
when ``test`` is absent. A ``{train, validation}`` input would otherwise lose its held-out rows and
evaluate on training data with only a warning; any other extra split would vanish the same way.
"""

import os
import re

import pytest
from accelerate import PartialState
from datasets import Dataset, DatasetDict

from src.data.pipeline.preprocessed_metadata import PreprocessingConfig
from src.data.pipeline.preprocessing import preprocess_dataset
from src.data.sources.loading import load_preprocessed_dataset
from tests.common.models import QWEN3_0_6B
from tests.common.tokenizers import load_cached_tokenizer

_TRAIN = Dataset.from_dict({"text": [f"Training document number {i}." for i in range(12)]})
_HELD_OUT = Dataset.from_dict({"text": [f"Held-out document number {i}." for i in range(3)]})

PartialState()  # the loaders log through accelerate's rank-aware logger


def _prepare(output_dir, splits: dict[str, Dataset], *, num_shards: int = 1) -> None:
    config = PreprocessingConfig(
        model_name_or_path=QWEN3_0_6B, mode="text", max_length=64, num_shards=num_shards, num_proc=1
    )
    tokenizer = load_cached_tokenizer(QWEN3_0_6B)
    preprocess_dataset(DatasetDict(splits), tokenizer, config, output_dir=str(output_dir))


def _decoded_test_rows(output_dir) -> list[str]:
    tokenizer = load_cached_tokenizer(QWEN3_0_6B)
    loaded = load_preprocessed_dataset(str(output_dir))
    return sorted(tokenizer.decode(ids, skip_special_tokens=True) for ids in loaded["test"]["input_ids"])


@pytest.mark.parametrize("num_shards", [1, 2])
def test_a_lone_validation_split_is_baked_as_the_test_split(tmp_path, num_shards):
    _prepare(tmp_path, {"train": _TRAIN, "validation": _HELD_OUT}, num_shards=num_shards)
    assert _decoded_test_rows(tmp_path) == sorted(_HELD_OUT["text"]), (
        "the test split must hold the validation rows, not a placeholder cut from train"
    )
    assert not os.path.exists(tmp_path / "validation")


@pytest.mark.parametrize(
    ("splits", "unbaked"),
    [
        ({"train": _TRAIN, "test": _HELD_OUT, "validation": _HELD_OUT}, "['validation']"),
        ({"train": _TRAIN, "extra": _HELD_OUT}, "['extra']"),
    ],
)
def test_a_split_the_artifact_would_drop_is_refused_by_name(tmp_path, splits, unbaked):
    with pytest.raises(ValueError, match=re.escape(f"carries split(s) {unbaked}")):
        _prepare(tmp_path, splits)
    assert not os.path.exists(tmp_path / "train"), "the refusal must land before anything is written"


def test_a_train_and_test_input_is_baked_as_given(tmp_path):
    _prepare(tmp_path, {"train": _TRAIN, "test": _HELD_OUT})
    assert _decoded_test_rows(tmp_path) == sorted(_HELD_OUT["text"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
