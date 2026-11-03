"""A sharded split short of the DP degree: an empty eval slice loads, since the trainer's equalization refuses it
once evaluation runs, and an empty train slice raises on every rank at once, since a rank raising alone blocks
its peers in the eval verdict."""

import datetime
import json
import os

import pytest
from accelerate import PartialState
from datasets import Dataset

from halo.data.preprocessed.preparation import shard_dataset
from halo.data.preprocessed.shard_index import SHARD_INDEX_FILE, write_stamped_sidecar
from halo.data.sources.loading import load_datasets
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2

# A diverged load ends in a stuck collective; the suite must not sit on one.
PG_TIMEOUT_SEC = 30

EMPTY_TRAIN_ERROR = "No training data after schema normalization. Please check your dataset."


def _rows(num_rows: int) -> Dataset:
    return Dataset.from_dict(
        {"input_ids": [[i, i + 1] for i in range(num_rows)], "attention_mask": [[1, 1]] * num_rows}
    )


def _save_sharded(directory: str, *, train_shards: int, test_shards: int, train_rows: int = 4) -> str:
    for split, num_shards, num_rows in (("train", train_shards, train_rows), ("test", test_shards, 4)):
        index = shard_dataset(_rows(num_rows), output_dir=directory, split_name=split, num_shards=num_shards)
        write_stamped_sidecar(os.path.join(directory, split, SHARD_INDEX_FILE), index)
    return directory


def _load_on_rank(rank: int, directory: str, results_dir: str, dataset_ratio: float) -> None:
    """Record this rank's load outcome: its split lengths, or the error it raised."""
    PartialState()
    try:
        ds = load_datasets(
            directory,
            test_size=None,
            dataset_ratio=dataset_ratio,
            conversation_field=None,
            data_parallel_rank=rank,
            data_parallel_size=WORLD_SIZE,
        )
        outcome = {"train": len(ds["train"]), "test": len(ds["test"])}
    except ValueError as exc:
        outcome = {"error": str(exc)}
    with open(os.path.join(results_dir, f"{rank}.json"), "w") as handle:
        json.dump(outcome, handle)


def _outcomes(directory: str, tmp_path, dataset_ratio: float = 1) -> list[dict]:
    results_dir = str(tmp_path / "results")
    os.makedirs(results_dir)
    run_gloo_ranks(
        _load_on_rank,
        WORLD_SIZE,
        directory,
        results_dir,
        dataset_ratio,
        pg_timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC),
    )
    outcomes = []
    for rank in range(WORLD_SIZE):
        with open(os.path.join(results_dir, f"{rank}.json")) as handle:
            outcomes.append(json.load(handle))
    return outcomes


def test_a_starved_eval_rank_loads_an_empty_split_with_its_peers(tmp_path):
    directory = _save_sharded(str(tmp_path / "ds"), train_shards=WORLD_SIZE, test_shards=1)

    outcomes = _outcomes(directory, tmp_path)

    assert outcomes[0] == {"train": 2, "test": 4}, outcomes
    assert outcomes[1] == {"train": 2, "test": 0}, outcomes


def test_a_starved_train_rank_raises_on_every_rank(tmp_path):
    # Three train rows over two shards hold 2 and 1; a 0.6 ratio floors them to 1 and 0 rows.
    directory = _save_sharded(str(tmp_path / "ds"), train_shards=WORLD_SIZE, test_shards=WORLD_SIZE, train_rows=3)

    outcomes = _outcomes(directory, tmp_path, dataset_ratio=0.6)

    assert all(outcome == {"error": EMPTY_TRAIN_ERROR} for outcome in outcomes), outcomes


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
