"""A local path classified as a data file loads, the form the rejection-sampling scripts emit; a
``load_from_disk``-only route reads just ``save_to_disk`` directories."""

import json

import pytest
from datasets import Dataset

from halo.data.sources.loading import load_dataset_from_source
from halo.data.sources.paths import parse_dataset_source

ROWS = [{"prompt": "a", "label": 0}, {"prompt": "b", "label": 1}, {"prompt": "c", "label": 2}]


def test_jsonl_file_loads_the_rows_it_contains(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in ROWS))

    assert parse_dataset_source(str(path))[0] == "local"
    loaded = load_dataset_from_source(str(path))

    train = loaded["train"] if not isinstance(loaded, Dataset) else loaded
    assert len(train) == len(ROWS), f"expected {len(ROWS)} rows, got {len(train)}"
    assert train["prompt"] == ["a", "b", "c"]


@pytest.mark.parametrize("suffix", [".json", ".parquet", ".csv"])
def test_every_advertised_file_extension_loads(tmp_path, suffix):
    """Each extension `parse_dataset_source` accepts needs a builder, or the claim is false."""
    path = tmp_path / f"samples{suffix}"
    dataset = Dataset.from_list(ROWS)
    if suffix == ".json":
        path.write_text(json.dumps(ROWS))
    elif suffix == ".parquet":
        dataset.to_parquet(str(path))
    else:
        dataset.to_csv(str(path))

    loaded = load_dataset_from_source(str(path))
    train = loaded["train"] if not isinstance(loaded, Dataset) else loaded
    assert len(train) == len(ROWS)


def test_saved_directory_still_loads_through_load_from_disk(tmp_path):
    """The file branch must not capture the directory form it shares a source type with."""
    directory = tmp_path / "saved"
    Dataset.from_list(ROWS).save_to_disk(str(directory))

    loaded = load_dataset_from_source(str(directory))
    assert len(loaded) == len(ROWS)


def test_a_missing_data_file_raises_a_named_error(tmp_path):
    """Fail loud and say which forms are legal: a silent empty dataset trains on nothing."""
    with pytest.raises(FileNotFoundError, match="save_to_disk"):
        load_dataset_from_source(str(tmp_path / "absent.jsonl"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
