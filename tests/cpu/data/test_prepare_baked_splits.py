"""A prepared dataset bakes the input's held-out split, and names every split it leaves out.

The artifact holds only ``train`` and ``test``, and training evaluates on a placeholder cut from train
when ``test`` is absent. A ``{train, validation}`` input would otherwise lose its held-out rows and
evaluate on training data with only a warning. A split next to the baked pair (GLUE's ``validation``
beside ``test``, imdb's ``unsupervised``) is left out with a warning that names it, not refused.
"""

import logging
import os

import pytest
from accelerate import PartialState
from datasets import Dataset, DatasetDict

from src.data.sources.loading import load_preprocessed_dataset
from tests.common.datasets import write_prepared_text_dataset
from tests.common.models import QWEN3_0_6B
from tests.common.tokenizers import load_cached_tokenizer

_TRAIN = Dataset.from_dict({"text": [f"Training document number {i}." for i in range(12)]})
_HELD_OUT = Dataset.from_dict({"text": [f"Held-out document number {i}." for i in range(3)]})
_EXTRA = Dataset.from_dict({"text": [f"Extra document number {i}." for i in range(2)]})
_PREPROCESSING_LOGGER = "src.data.pipeline.preprocessing"

PartialState()  # the loaders log through accelerate's rank-aware logger


def _decoded_test_rows(output_dir) -> list[str]:
    tokenizer = load_cached_tokenizer(QWEN3_0_6B)
    loaded = load_preprocessed_dataset(str(output_dir))
    return sorted(tokenizer.decode(ids, skip_special_tokens=True) for ids in loaded["test"]["input_ids"])


@pytest.mark.parametrize("num_shards", [1, 2])
def test_a_lone_validation_split_is_baked_as_the_test_split(tmp_path, num_shards):
    write_prepared_text_dataset(
        tmp_path, DatasetDict({"train": _TRAIN, "validation": _HELD_OUT}), num_shards=num_shards
    )
    assert _decoded_test_rows(tmp_path) == sorted(_HELD_OUT["text"]), (
        "the test split must hold the validation rows, not a placeholder cut from train"
    )
    assert not os.path.exists(tmp_path / "validation")


def _unbaked_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING and "not baked" in r.getMessage()]


@pytest.mark.parametrize(
    ("splits", "unbaked"),
    [
        ({"train": _TRAIN, "test": _HELD_OUT, "validation": _EXTRA}, "['validation']"),
        ({"train": _TRAIN, "test": _HELD_OUT, "unsupervised": _EXTRA}, "['unsupervised']"),
        ({"train": _TRAIN, "validation": _HELD_OUT, "unsupervised": _EXTRA}, "['unsupervised']"),
    ],
)
def test_a_split_outside_the_baked_pair_is_named_in_a_warning(tmp_path, caplog, splits, unbaked):
    with caplog.at_level(logging.WARNING, logger=_PREPROCESSING_LOGGER):
        write_prepared_text_dataset(tmp_path, DatasetDict(splits))
    warned = _unbaked_warnings(caplog)
    assert len(warned) == 1 and unbaked in warned[0], warned
    assert _decoded_test_rows(tmp_path) == sorted(_HELD_OUT["text"]), "the extra split must not be baked as test"


@pytest.mark.parametrize("held_out", ["test", "validation"])
def test_no_warning_when_every_split_is_baked(tmp_path, caplog, held_out):
    with caplog.at_level(logging.WARNING, logger=_PREPROCESSING_LOGGER):
        write_prepared_text_dataset(tmp_path, DatasetDict({"train": _TRAIN, held_out: _HELD_OUT}))
    assert _unbaked_warnings(caplog) == []


def test_a_train_and_test_input_is_baked_as_given(tmp_path):
    write_prepared_text_dataset(tmp_path, DatasetDict({"train": _TRAIN, "test": _HELD_OUT}))
    assert _decoded_test_rows(tmp_path) == sorted(_HELD_OUT["text"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
