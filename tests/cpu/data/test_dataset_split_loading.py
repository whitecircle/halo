#!/usr/bin/env python
"""``load_dataset_split``, the one split-of-a-source loader the eval, regrade and deduplication tools read through:
each source kind resolves by its spelling, and a mistyped split is refused."""

import json

import pytest
from datasets import Dataset, DatasetDict

from halo.data.sources import loading
from halo.data.sources.loading import load_dataset_split


def test_a_bare_saved_dataset_is_its_one_split(tmp_path):
    path = str(tmp_path / "bare")
    Dataset.from_list([{"prompt": "p", "answer": "a"}]).save_to_disk(path)
    split = load_dataset_split(path, split="test")
    assert isinstance(split, Dataset)
    assert split[0] == {"prompt": "p", "answer": "a"}


def test_a_saved_dataset_dict_gives_the_named_split_and_refuses_another(tmp_path):
    path = str(tmp_path / "dict")
    DatasetDict(
        {"train": Dataset.from_list([{"prompt": "train-row"}]), "test": Dataset.from_list([{"prompt": "test-row"}])}
    ).save_to_disk(path)
    assert load_dataset_split(path, split="test")[0]["prompt"] == "test-row"
    # A mistyped split must not quietly read another one (possibly train).
    with pytest.raises(ValueError, match=r"split 'validation' is not in .*\['train', 'test'\]"):
        load_dataset_split(path, split="validation")


def test_a_local_data_file_is_its_one_split(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps({"prompt": f"row {i}"}) + "\n" for i in range(3)))
    assert load_dataset_split(str(path), split="test")["prompt"] == ["row 0", "row 1", "row 2"]


def test_an_s3_uri_reads_the_named_split_of_what_it_holds(monkeypatch):
    stored = DatasetDict({"train": Dataset.from_list([{"prompt": "s3-train"}])})
    monkeypatch.setattr(
        loading, "load_dataset_from_s3_uri", lambda uri, **_kwargs: stored if uri == "s3://bucket/key" else None
    )
    monkeypatch.setattr(loading, "load_dataset", lambda *a, **k: pytest.fail("an S3 URI reached the Hub loader"))
    assert load_dataset_split("s3://bucket/key", split="train")[0]["prompt"] == "s3-train"
    with pytest.raises(ValueError, match="split 'test' is not in s3://bucket/key"):
        load_dataset_split("s3://bucket/key", split="test")


def test_a_hub_id_is_read_from_the_hub_at_its_config_and_split(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen: dict = {}

    def fake_load_dataset(path, config=None, *, split=None, **_kwargs):
        seen.update(path=path, config=config, split=split)
        return Dataset.from_list([{"prompt": "hub-row"}])

    monkeypatch.setattr(loading, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(loading, "load_from_disk", lambda *a, **k: pytest.fail("a Hub id was read off disk"))
    assert load_dataset_split("org/name", split="test", config="main")[0]["prompt"] == "hub-row"
    assert seen == {"path": "org/name", "config": "main", "split": "test"}


def test_a_missing_local_path_names_what_a_local_source_must_be(tmp_path, monkeypatch):
    """A path that is neither data file nor ``save_to_disk`` directory gets the trainers' loader error, which says
    what each spelling is read as."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"'missing_dir'.* read as a data file; anything else must be a"):
        load_dataset_split("missing_dir", split="test")


def test_a_hub_spec_suffix_is_refused_rather_than_sent_to_the_hub(tmp_path, monkeypatch):
    """A trainer spelling the Hub knows no repo of, whose suffix would contend with the config and split arguments."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(loading, "load_dataset", lambda *a, **k: pytest.fail("a suffixed Hub spec reached the Hub"))
    for spec in ("org/name:cfg", "org/name@train", "org/name:cfg@train"):
        with pytest.raises(ValueError, match=r"suffix; pass the bare 'org/name'"):
            load_dataset_split(spec, split="test")


def test_a_hub_id_a_local_directory_shadows_is_refused(tmp_path, monkeypatch):
    """Classifying by an existence probe reads whatever local copy shadows the id, while the string rule alone
    silently reads the Hub for a caller who meant the directory."""
    shadow = tmp_path / "org" / "name"
    shadow.mkdir(parents=True)
    Dataset.from_list([{"prompt": "local-row"}]).save_to_disk(str(shadow))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(loading, "load_dataset", lambda *a, **k: pytest.fail("a shadowed Hub id was fetched"))
    monkeypatch.setattr(loading, "load_from_disk", lambda *a, **k: pytest.fail("a Hub id was read off disk"))
    with pytest.raises(ValueError, match=r"'\./org/name'"):
        load_dataset_split("org/name", split="test")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
