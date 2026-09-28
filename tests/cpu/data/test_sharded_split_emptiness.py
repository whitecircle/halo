"""A sharded eval split short of the data-parallel degree loads empty on the starved rank.

Each data-parallel rank holds its own slice of a sharded dataset. An eval split with fewer shards than
ranks leaves some ranks no rows; that is legitimate at load (the trainer's pre-sharded eval
equalization refuses it world-uniformly once evaluation runs), so the starved rank must load an empty
split with its peers instead of raising while they go on to their next collective.

    python tests/cpu/data/test_sharded_split_emptiness.py
"""

import datetime
import json
import os

import pytest
from accelerate import PartialState
from datasets import Dataset

from src.data.pipeline.preprocessing import shard_dataset
from src.data.shard_index import SHARD_INDEX_FILE
from src.data.sources.loading import load_datasets
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2

# A diverged load ends in a stuck collective; the suite must not sit on one.
PG_TIMEOUT_SEC = 30


def _rows(num_rows: int) -> Dataset:
    return Dataset.from_dict(
        {"input_ids": [[i, i + 1] for i in range(num_rows)], "attention_mask": [[1, 1]] * num_rows}
    )


def _save_sharded(directory: str, *, train_shards: int, test_shards: int) -> str:
    for split, num_shards in (("train", train_shards), ("test", test_shards)):
        index = shard_dataset(_rows(4), output_dir=directory, split_name=split, num_shards=num_shards)
        index.save(os.path.join(directory, split, SHARD_INDEX_FILE))
    return directory


def _load_on_rank(rank: int, directory: str, results_dir: str) -> None:
    """Record this rank's load outcome: its split lengths, or the error it raised."""
    PartialState()
    try:
        ds = load_datasets(
            directory,
            test_size=None,
            dataset_ratio=1,
            conversation_field=None,
            data_parallel_rank=rank,
            data_parallel_size=WORLD_SIZE,
        )
        outcome = {"train": len(ds["train"]), "test": len(ds["test"])}
    except ValueError as exc:
        outcome = {"error": str(exc)}
    with open(os.path.join(results_dir, f"{rank}.json"), "w") as handle:
        json.dump(outcome, handle)


def _outcomes(directory: str, tmp_path) -> list[dict]:
    results_dir = str(tmp_path / "results")
    os.makedirs(results_dir)
    run_gloo_ranks(
        _load_on_rank,
        WORLD_SIZE,
        directory,
        results_dir,
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
