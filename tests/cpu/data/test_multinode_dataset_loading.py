"""Dataset loading across nodes whose copies of the source are their own.

Every rank reads the source itself — a per-node S3 cache, a pre-staged local directory, a cached hub
revision — so one path string can name different bytes on different nodes. Simulated on one host by
giving each rank a working directory holding its "node-local" copy behind a relative path. Proven on
real gloo groups:

1. ranks that must hold the same rows and loaded different ones raise on every rank — a replicated
   load, and the siblings of one sharded replica (TP/CP/ETP partners, pipeline peers) — and ranks
   reading different shard indexes raise, while identical copies whose HF fingerprints differ (a
   data file written at another mtime) load;
2. a whole-source load or a pack that fails on the rank running it first raises its cause on every
   rank, under a shared and a per-node input filesystem, and the waiting rank never repeats the work.

    python tests/cpu/data/test_multinode_dataset_loading.py
"""

import datetime
import json
import os

import pytest
from accelerate import PartialState
from datasets import Dataset, DatasetDict

from src.data.pipeline import processing
from src.data.pipeline.preprocessing import shard_dataset
from src.data.shard_index import SHARD_INDEX_FILE
from src.data.sources import loading
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2

# A diverged load ends in a stuck collective; the suite must not sit on one.
PG_TIMEOUT_SEC = 30

# Two one-rank "nodes" on per-node storage: each rank is its node's local rank 0.
PER_NODE_ENV = {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1", "DIST_SHARED_FILESYSTEM": "0"}

SAVED_SOURCE = "./ds"
ROWS = [f"row {i}" for i in range(12)]
EDITED_ROWS = [*ROWS[:5], "row 5 (re-pushed)", *ROWS[6:]]


def _node_dir(tmp_dir, rank: int) -> str:
    return os.path.join(tmp_dir, f"node{rank}")


def _save_replicated(node: str, rows: list[str]) -> None:
    splits = {"train": Dataset.from_dict({"text": rows}), "test": Dataset.from_dict({"text": rows[:3]})}
    DatasetDict(splits).save_to_disk(os.path.join(node, "ds"))


def _save_sharded(node: str, rows: list[str], num_shards: int) -> None:
    for split in ("train", "test"):
        directory = os.path.join(node, "ds")
        index = shard_dataset(Dataset.from_dict({"text": rows}), num_shards, directory, split_name=split)
        index.save(os.path.join(directory, split, SHARD_INDEX_FILE))


def _record(tmp_dir: str, rank: int, outcome: str) -> None:
    with open(os.path.join(tmp_dir, f"result_{rank}.txt"), "w") as handle:
        handle.write(outcome)


def _count_call(tmp_dir: str, rank: int, name: str) -> None:
    with open(os.path.join(tmp_dir, f"calls_{name}_{rank}.txt"), "a") as handle:
        handle.write("call\n")


def _calls(tmp_dir, rank: int, name: str) -> int:
    path = os.path.join(tmp_dir, f"calls_{name}_{rank}.txt")
    if not os.path.exists(path):
        return 0
    with open(path) as handle:
        return len(handle.readlines())


def _load_worker(rank: int, tmp_dir: str, source: str, data_parallel_size: int) -> None:
    """Load ``source`` from this rank's node directory; record the split length or the error."""
    PartialState()
    os.chdir(_node_dir(tmp_dir, rank))
    real_source_load = loading.load_dataset_from_source

    def counting_source_load(path):
        _count_call(tmp_dir, rank, "source_load")
        return real_source_load(path)

    loading.load_dataset_from_source = counting_source_load
    data_parallel_rank = rank if data_parallel_size == WORLD_SIZE else 0
    try:
        ds, _ = loading._load_dataset_from_path(source, None, data_parallel_rank, data_parallel_size)
        outcome = {"train": len(ds["train"]), "fingerprint": ds["train"]._fingerprint}
    except BaseException as exc:
        outcome = {"error": f"{type(exc).__name__}: {exc}"}
    _record(tmp_dir, rank, json.dumps(outcome))


def _pack_worker(rank: int, tmp_dir: str) -> None:
    """Pack on every rank, the pack raising on the main rank only."""
    PartialState()

    def pack_failing_on_main(dataset, **kwargs):
        _count_call(tmp_dir, rank, "pack")
        if rank == 0:
            raise OSError("No space left on device")
        return dataset

    processing._trl_pack_dataset = pack_failing_on_main
    try:
        processing.pack_dataset_coordinated(Dataset.from_dict({"input_ids": [[1, 2], [3]]}), seq_length=4)
        outcome = "NO RAISE"
    except BaseException as exc:
        outcome = f"{type(exc).__name__}: {exc}"
    _record(tmp_dir, rank, outcome)


def _run(tmp_path, worker, *args, env=None) -> list[str]:
    run_gloo_ranks(
        worker,
        WORLD_SIZE,
        str(tmp_path),
        *args,
        pg_timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC),
        env={"HF_DATASETS_CACHE": str(tmp_path / "hf_datasets"), **(env or {})},
    )
    return [(tmp_path / f"result_{rank}.txt").read_text() for rank in range(WORLD_SIZE)]


def _loads(tmp_path, source: str = SAVED_SOURCE, data_parallel_size: int = 1, env=None) -> list[dict]:
    return [json.loads(outcome) for outcome in _run(tmp_path, _load_worker, source, data_parallel_size, env=env)]


def test_a_replicated_load_whose_node_copies_differ_raises_on_every_rank(tmp_path):
    """Same row count, one row edited on node 1: without the agreement each node trains its own
    version of the corpus and the DP shards no longer partition one dataset."""
    _save_replicated(_node_dir(tmp_path, 0), ROWS)
    _save_replicated(_node_dir(tmp_path, 1), EDITED_ROWS)

    for rank, outcome in enumerate(_loads(tmp_path)):
        assert outcome.get("error", "").startswith("ValueError"), f"rank {rank} trained on: {outcome}"
        assert "loaded different ones" in outcome["error"], f"rank {rank}: {outcome}"


def test_identical_node_copies_load_although_their_hf_fingerprints_differ(tmp_path):
    """Anti-vacuity, and why the identity is read off the rows: a data file pre-staged on each node at
    another mtime gets another HF fingerprint for the very same rows."""
    for rank in range(WORLD_SIZE):
        os.makedirs(_node_dir(tmp_path, rank))
        data_file = os.path.join(_node_dir(tmp_path, rank), "ds.jsonl")
        with open(data_file, "w") as handle:
            handle.writelines(json.dumps({"text": row}) + "\n" for row in ROWS)
        os.utime(data_file, (1_000_000 + rank, 1_000_000 + rank))

    outcomes = _loads(tmp_path, source="./ds.jsonl")

    assert all(outcome.get("train") == len(ROWS) for outcome in outcomes), outcomes
    assert outcomes[0]["fingerprint"] != outcomes[1]["fingerprint"], (
        f"the copies no longer differ in HF fingerprint, so the case proves nothing: {outcomes}"
    )


def test_the_siblings_of_a_sharded_replica_must_hold_the_same_rows(tmp_path):
    """One data-parallel replica over two ranks — TP partners, or two pipeline stages on two nodes.
    Each loads every shard from its own node, and node 1's copy has one row edited under the same
    index: stage 0 would forward one row set while the last stage scores another's labels."""
    _save_sharded(_node_dir(tmp_path, 0), ROWS, num_shards=2)
    _save_sharded(_node_dir(tmp_path, 1), EDITED_ROWS, num_shards=2)
    for split in ("train", "test"):
        index = os.path.join(split, SHARD_INDEX_FILE)
        with open(os.path.join(_node_dir(tmp_path, 0), "ds", index)) as source:
            with open(os.path.join(_node_dir(tmp_path, 1), "ds", index), "w") as target:
                target.write(source.read())

    for rank, outcome in enumerate(_loads(tmp_path, data_parallel_size=1)):
        assert outcome.get("error", "").startswith("ValueError"), f"rank {rank} trained on: {outcome}"
        assert "loaded different ones" in outcome["error"], f"rank {rank}: {outcome}"


def test_ranks_reading_different_shard_indexes_raise(tmp_path):
    """Two data-parallel replicas, nothing shared between them — but the shard assignment is computed
    off each rank's own index, so a node holding a re-prepared copy loads overlapping shards."""
    _save_sharded(_node_dir(tmp_path, 0), ROWS, num_shards=4)
    _save_sharded(_node_dir(tmp_path, 1), ROWS, num_shards=2)

    for rank, outcome in enumerate(_loads(tmp_path, data_parallel_size=WORLD_SIZE)):
        assert outcome.get("error", "").startswith("ValueError"), f"rank {rank} trained on: {outcome}"
        assert "shard index" in outcome["error"], f"rank {rank}: {outcome}"


def test_a_split_only_some_node_copies_carry_raises(tmp_path):
    """Node 1's copy lost its test index (a half-synced directory, a transient read): its ranks would
    skip every coordinated operation over the test split while their peers run them."""
    for rank in range(WORLD_SIZE):
        _save_sharded(_node_dir(tmp_path, rank), ROWS, num_shards=2)
    os.remove(os.path.join(_node_dir(tmp_path, 1), "ds", "test", SHARD_INDEX_FILE))

    for rank, outcome in enumerate(_loads(tmp_path, data_parallel_size=WORLD_SIZE)):
        assert outcome.get("error", "").startswith("ValueError"), f"rank {rank} trained on: {outcome}"
        assert "'test'" in outcome["error"], f"rank {rank} was not told which split: {outcome}"


def test_identical_sharded_copies_load(tmp_path):
    """Anti-vacuity for both sharded checks: matching per-node copies load on every rank."""
    for rank in range(WORLD_SIZE):
        _save_sharded(_node_dir(tmp_path, rank), ROWS, num_shards=2)

    outcomes = _loads(tmp_path, data_parallel_size=1)

    assert all(outcome.get("train") == len(ROWS) for outcome in outcomes), outcomes


def test_a_failed_first_load_raises_everywhere_and_is_not_repeated(tmp_path):
    """Shared input filesystem: rank 0 loads first and fails. The waiting rank must raise the cause
    rather than retry the load itself — at scale, every peer re-downloading a corpus the main rank
    could not — and then go on to the next collective alone."""
    os.makedirs(_node_dir(tmp_path, 0))
    _save_replicated(_node_dir(tmp_path, 1), ROWS)

    outcomes = _loads(tmp_path)

    assert outcomes[0]["error"].startswith("FileNotFoundError"), outcomes[0]
    assert outcomes[1]["error"].startswith("RuntimeError"), outcomes[1]
    assert "Could not load dataset" in outcomes[1]["error"], f"the peer was not told the cause: {outcomes[1]}"
    assert _calls(tmp_path, 1, "source_load") == 0, "the waiting rank repeated the load the main rank failed"


def test_a_node_whose_load_fails_takes_every_node_down(tmp_path):
    """Per-node input filesystem: both nodes load at once, node 1's copy is missing. Node 0 succeeded,
    so without a join it would walk into its next collective alone."""
    _save_replicated(_node_dir(tmp_path, 0), ROWS)
    os.makedirs(_node_dir(tmp_path, 1))

    outcomes = _loads(tmp_path, env=PER_NODE_ENV)

    assert outcomes[1]["error"].startswith("FileNotFoundError"), outcomes[1]
    assert outcomes[0]["error"].startswith("RuntimeError"), outcomes[0]
    assert "Could not load dataset" in outcomes[0]["error"], f"node 0 was not told the cause: {outcomes[0]}"
    assert _calls(tmp_path, 0, "source_load") == _calls(tmp_path, 1, "source_load") == 1


def test_a_failed_pack_raises_everywhere_and_is_not_repeated(tmp_path):
    """The pack is the same shape of work: one rank fills the cache the others read back."""
    outcomes = _run(tmp_path, _pack_worker)

    assert outcomes[0] == "OSError: No space left on device", outcomes[0]
    assert outcomes[1].startswith("RuntimeError") and "No space left on device" in outcomes[1], outcomes[1]
    assert _calls(tmp_path, 1, "pack") == 0, "the waiting rank repeated the pack the main rank failed"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
