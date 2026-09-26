"""The default ``--num-shards 1`` artifact trains at any data-parallel size.

A single-shard dataset written in the sharded layout loads through ``ShardedDatasetLoader``, which
refuses fewer shards than the DP size, so the default output of ``prepare_dataset`` trained only on
one GPU. At ``num_shards <= 1`` the splits are saved unsharded instead and every rank loads them whole.
"""

import os

import pytest
from accelerate import PartialState
from datasets import Dataset, DatasetDict

from src.data.pipeline.preprocessed_metadata import PreprocessingConfig
from src.data.pipeline.preprocessing import preprocess_dataset
from src.data.shard_index import SHARD_INDEX_FILE
from src.data.sources.loading import load_preprocessed_dataset
from tests.common.models import QWEN3_0_6B
from tests.common.tokenizers import load_cached_tokenizer

_DOCS = [f"Document number {i} of the raw corpus." for i in range(12)]

PartialState()  # the loaders log through accelerate's rank-aware logger


def _prepare(output_dir, *, num_shards: int, with_test: bool) -> None:
    train = Dataset.from_dict({"text": _DOCS})
    data = DatasetDict({"train": train, "test": train.select(range(4))}) if with_test else train
    config = PreprocessingConfig(
        model_name_or_path=QWEN3_0_6B, mode="text", max_length=64, num_shards=num_shards, num_proc=1
    )
    preprocess_dataset(data, load_cached_tokenizer(QWEN3_0_6B), config, output_dir=str(output_dir))


def test_the_default_single_shard_output_trains_at_dp_2(tmp_path):
    _prepare(tmp_path, num_shards=1, with_test=True)
    assert not os.path.exists(tmp_path / "train" / SHARD_INDEX_FILE), "num_shards=1 must write no shard index"
    for rank in (0, 1):
        loaded = load_preprocessed_dataset(str(tmp_path), data_parallel_rank=rank, data_parallel_size=2)
        assert loaded["train"].num_rows == len(_DOCS), "every rank loads the unsharded dataset whole"
        assert loaded["test"].num_rows == 4


def test_a_train_only_unsharded_output_gets_a_placeholder_test_split(tmp_path):
    _prepare(tmp_path, num_shards=1, with_test=False)
    loaded = load_preprocessed_dataset(str(tmp_path), data_parallel_rank=0, data_parallel_size=2)
    assert loaded["train"].num_rows == len(_DOCS)
    assert loaded["test"].num_rows > 0


def test_a_sharded_output_keeps_the_shard_layout(tmp_path):
    _prepare(tmp_path, num_shards=2, with_test=True)
    assert os.path.exists(tmp_path / "train" / SHARD_INDEX_FILE)
    loaded = load_preprocessed_dataset(str(tmp_path), data_parallel_rank=1, data_parallel_size=2)
    assert 0 < loaded["train"].num_rows < len(_DOCS), "rank 1 loads only its own shard"


def test_a_sharded_output_without_a_test_split_is_refused_before_tokenizing(tmp_path):
    with pytest.raises(ValueError, match="num_shards=2 writes a sharded dataset"):
        _prepare(tmp_path, num_shards=2, with_test=False)
    assert not os.path.exists(tmp_path / "train"), "the refusal must land before anything is written"


def test_a_hub_preprocessed_dataset_is_refused_with_the_download_step():
    with pytest.raises(ValueError, match="hf download org/name --repo-type dataset"):
        load_preprocessed_dataset("org/name")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
