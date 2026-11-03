#!/usr/bin/env python
"""Scripts build ``remove_columns`` from the train split, and ``DatasetDict.map`` applies one list to every split, so a
source whose other split lacks a train column would die in the map after the model load. Such a split maps with the
columns it holds; every other map keeps its one ``DatasetDict.map`` call, and each its cache names."""

import os
import sys
import types
from unittest import mock

import pyarrow as pa
import pytest
from datasets import Dataset, DatasetDict

import scripts.training.grpo.environmental as environmental_script
import scripts.training.grpo.online as online_script
from halo.configs.grpo_environmental import EnvironmentalGRPOScriptArguments
from halo.configs.sft import SFTScriptArguments
from halo.data.pipeline.coordinated_map import coordinated_map
from halo.data.pipeline.map_cache_key import map_cache_file_name
from halo.entrypoints.data_setup import prepare_chat_text_split
from tests.common.models import QWEN3_0_6B
from tests.common.tokenizers import load_cached_tokenizer

_TURNS = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
_DESC = "split column schemas"

_mapped_prompts: list[str] = []


class _ExamplesLogged(Exception):
    """Raised by the stubbed example log, carrying the splits as the trainer would get them."""


@pytest.fixture(autouse=True)
def _isolated_datasets_cache(tmp_path, monkeypatch):
    """A fresh map cache per test: a hit would hide a map that raises."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets"))
    _mapped_prompts.clear()


def _prompt_length(row: dict) -> dict:
    _mapped_prompts.append(row["prompt"])
    return {"length": len(row["prompt"])}


def _mixed_type_value(row: dict) -> dict:
    return {"value": 1 if row["prompt"] == "a" else "text"}


def _train_only_column_rows() -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_dict({"prompt": ["a", "bb"], "source": ["x", "y"]}),
            "test": Dataset.from_dict({"prompt": ["ccc", "dddd"]}),
        }
    )


def _matching_rows() -> DatasetDict:
    rows = {"prompt": ["a", "bb"], "source": ["x", "y"]}
    return DatasetDict({"train": Dataset.from_dict(rows), "test": Dataset.from_dict(rows)})


def _cache_basenames(result: DatasetDict) -> dict[str, str]:
    return {split: os.path.basename(data.cache_files[0]["filename"]) for split, data in result.items()}


def _split_names(cache_name: str) -> dict[str, str]:
    return {split: cache_name.replace(".arrow", f"_{split}.arrow") for split in ("train", "test")}


@pytest.mark.parametrize("remove_columns", [["source"], "source"], ids=["list", "bare_name"])
def test_a_split_lacking_a_listed_column_maps_with_the_columns_it_holds(remove_columns):
    result = coordinated_map(
        _train_only_column_rows(), _prompt_length, desc=_DESC, num_proc=1, remove_columns=remove_columns
    )
    assert {split: data.column_names for split, data in result.items()} == {
        "train": ["prompt", "length"],
        "test": ["prompt", "length"],
    }
    assert result["test"]["length"][:] == [3, 4]


def test_a_map_of_matching_splits_stays_one_call_with_the_one_list():
    dataset = _matching_rows()
    with mock.patch.object(DatasetDict, "map", autospec=True, side_effect=DatasetDict.map) as spied:
        result = coordinated_map(dataset, _prompt_length, desc=_DESC, num_proc=1, remove_columns=["source"])
    assert spied.call_count == 1
    assert spied.call_args.kwargs["remove_columns"] == ["source"]
    assert result["test"].column_names == ["prompt", "length"]


@pytest.mark.parametrize("rows", [_matching_rows, _train_only_column_rows], ids=["matching", "train_only_column"])
def test_the_cache_files_are_named_off_the_listed_columns(rows):
    dataset = rows()
    cache_name = map_cache_file_name("map", _prompt_length, dataset, _DESC, {"remove_columns": ["source"]})
    result = coordinated_map(dataset, _prompt_length, desc=_DESC, num_proc=1, remove_columns=["source"])
    assert _cache_basenames(result) == _split_names(cache_name)


def test_a_rerun_of_a_narrowed_map_reads_each_split_from_its_cache_file():
    coordinated_map(_train_only_column_rows(), _prompt_length, desc=_DESC, num_proc=1, remove_columns=["source"])
    assert sorted(_mapped_prompts) == ["a", "bb", "ccc", "dddd"]
    _mapped_prompts.clear()
    again = coordinated_map(
        _train_only_column_rows(), _prompt_length, desc=_DESC, num_proc=1, remove_columns=["source"]
    )
    assert not _mapped_prompts
    assert again["test"]["length"][:] == [3, 4]


@pytest.mark.parametrize("rows", [_matching_rows, _train_only_column_rows], ids=["matching", "train_only_column"])
def test_a_column_no_split_holds_is_still_refused(rows):
    with pytest.raises(ValueError, match=r"Column to remove \['typo'\] not in the dataset"):
        coordinated_map(rows(), _prompt_length, desc=_DESC, num_proc=1, remove_columns=["source", "typo"])


def test_a_column_only_the_test_split_holds_stays_there():
    dataset = _train_only_column_rows()
    dataset["test"] = dataset["test"].add_column("note", ["n", "m"])
    result = coordinated_map(dataset, _prompt_length, desc=_DESC, num_proc=1, remove_columns=["source"])
    assert result["test"].column_names == ["prompt", "note", "length"]


def test_mixed_type_rows_still_raise_on_the_narrowed_map():
    """``Dataset.map`` alone defaults ``on_mixed_types`` to JSON columns, where ``DatasetDict.map`` raises, so the
    split that lacks a column must not take another path than a matching one."""
    with pytest.raises(pa.ArrowException):
        coordinated_map(
            _train_only_column_rows(), _mixed_type_value, desc=_DESC, num_proc=1, remove_columns=["source"]
        )


def _conversation_rows() -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_dict({"prompt": [_TURNS] * 4, "answer": ["1"] * 4, "source": ["x"] * 4}),
            "test": Dataset.from_dict({"prompt": [_TURNS] * 2, "answer": ["2"] * 2}),
        }
    )


def test_prepare_chat_text_split_takes_a_test_split_lacking_a_train_column():
    args = SFTScriptArguments(dataset="dummy/dataset", conversation_field="prompt")
    train, test = prepare_chat_text_split(
        _conversation_rows().remove_columns("answer"),
        load_cached_tokenizer(QWEN3_0_6B),
        args,
        types.SimpleNamespace(max_length=64, dataset_num_proc=None),
        desc=_DESC,
    )
    assert len(train) == 4 and len(test) == 2
    assert "source" not in train.column_names and "prompt" not in test.column_names
    assert "input_ids" in test.column_names


def test_online_grpo_takes_a_test_split_lacking_a_train_column(tmp_path):
    tokenizer = load_cached_tokenizer(QWEN3_0_6B)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: stub/qwen3-4b\ndataset:\n- dummy/dataset\noutput_dir: {tmp_path / 'out'}\n"
        "bf16: false\nuse_cpu: true\nmax_completion_length: 64\n"
    )
    runtime = types.SimpleNamespace(parallelism_config=types.SimpleNamespace(), model_source="stub/qwen3-4b")

    def stop_at_the_log(datasets, *_args):
        raise _ExamplesLogged(datasets)

    patches = [
        mock.patch.object(online_script, "init_training_script", return_value=runtime),
        mock.patch.object(online_script, "load_script_datasets", return_value=(_conversation_rows(), False)),
        mock.patch.object(
            online_script, "load_model_and_tokenizer", return_value=(types.SimpleNamespace(), tokenizer)
        ),
        mock.patch.object(online_script, "apply_prompt_completion_window", return_value=(tokenizer, None)),
        mock.patch.object(online_script, "configure_adapter_training", return_value=None),
        mock.patch.object(online_script, "log_model_info"),
        mock.patch.object(online_script, "log_script_dataset_examples", side_effect=stop_at_the_log),
        mock.patch("halo.entrypoints.script.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
    ]
    for patch in patches:
        patch.start()
    try:
        with pytest.raises(_ExamplesLogged) as logged:
            online_script.main()
    finally:
        for patch in reversed(patches):
            patch.stop()
    splits = logged.value.args[0]
    assert splits["train"].column_names == splits["test"].column_names
    assert set(splits["test"].column_names) == {"prompt", "conversation", "answer"}
    assert splits["test"]["answer"][:] == ["2", "2"]


def test_environmental_grpo_takes_a_test_split_lacking_a_train_column():
    processed = environmental_script.process_dataset(
        EnvironmentalGRPOScriptArguments(dataset="dummy/dataset"),
        load_cached_tokenizer(QWEN3_0_6B),
        _conversation_rows(),
    )
    assert sorted(processed["test"].column_names) == ["answer", "prompt"]
    assert sorted(processed["train"].column_names) == ["answer", "prompt"]
    assert len(processed["test"]) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
