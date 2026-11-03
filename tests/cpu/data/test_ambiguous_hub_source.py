#!/usr/bin/env python
"""A Hub-shaped dataset path that also names a local directory is refused, naming ``./`` as the way to the
directory. Read as a Hub id, a ``save_to_disk`` tree loads through the json builder, which hands back its
``state.json`` sidecars as rows."""

import re

import pytest
from datasets import Dataset, DatasetDict

from halo.data.sources.loading import load_dataset_from_source, load_dataset_split, load_datasets_auto
from halo.data.sources.paths import reject_ambiguous_hub_source
from tests.common.datasets import write_prepared_text_dataset

_DOCUMENTS = DatasetDict(
    {
        "train": Dataset.from_dict({"text": [f"Training document number {i}." for i in range(8)]}),
        "test": Dataset.from_dict({"text": ["Held-out document."]}),
    }
)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_a_prepared_artifact_under_a_hub_shaped_path_is_refused_and_loads_with_dot_slash(workdir):
    write_prepared_text_dataset(workdir / "preprocessed" / "my_dataset", _DOCUMENTS)

    with pytest.raises(ValueError, match=r"'\./preprocessed/my_dataset'"):
        load_datasets_auto("preprocessed/my_dataset", test_size=None, dataset_ratio=None)

    loaded, preprocessed = load_datasets_auto("./preprocessed/my_dataset", test_size=None, dataset_ratio=None)
    assert preprocessed
    assert {"input_ids", "attention_mask"} <= set(loaded["train"].column_names)
    assert len(loaded["train"]) == len(_DOCUMENTS["train"])


def test_a_saved_raw_dataset_under_a_hub_shaped_path_is_refused_by_every_reader(workdir):
    _DOCUMENTS.save_to_disk(workdir / "data" / "corpus")

    with pytest.raises(ValueError, match="local directory of that name exists"):
        load_dataset_from_source("data/corpus")
    with pytest.raises(ValueError, match="local directory of that name exists"):
        load_datasets_auto("data/corpus", test_size=None, dataset_ratio=None, conversation_field="text")
    with pytest.raises(ValueError, match="local directory of that name exists"):
        load_dataset_split("data/corpus@train", split="train")

    assert load_dataset_from_source("./data/corpus")["train"]["text"] == _DOCUMENTS["train"]["text"]


@pytest.mark.parametrize("spec", ["data/corpus@train", "data/corpus:default@train"])
def test_the_spelling_the_refusal_names_loads_the_directory(workdir, spec):
    """A local path takes no ``:config`` / ``@split`` suffix, so the suggested spelling drops it."""
    _DOCUMENTS.save_to_disk(workdir / "data" / "corpus")
    with pytest.raises(ValueError, match="local directory of that name exists") as refused:
        load_dataset_from_source(spec)

    suggested = re.search(r"Write '([^']+)'", str(refused.value)).group(1)
    assert load_dataset_from_source(suggested)["train"]["text"] == _DOCUMENTS["train"]["text"]


def test_a_hub_shaped_path_without_a_local_directory_stays_a_hub_id(workdir):
    (workdir / "data").mkdir()
    reject_ambiguous_hub_source("data/corpus")
    reject_ambiguous_hub_source("HuggingFaceH4/ultrachat_200k@train_sft")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
