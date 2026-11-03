"""Dataset loading across nodes whose copies of the source are their own, simulated by giving each rank a working
directory holding its node-local copy. Ranks that must hold the same rows and diverged raise everywhere, a failed
first load or pack raises its cause everywhere without repeating, and a Hub source resolves once per input scope."""

import datetime
import json
import os
import pickle

import pytest
from accelerate import PartialState
from datasets import Dataset, DatasetDict

from halo.data.pipeline import coordinated_map as coordinated_map_module
from halo.data.preprocessed.preparation import shard_dataset
from halo.data.preprocessed.shard_index import SHARD_INDEX_FILE, write_stamped_sidecar
from halo.data.sources import loading
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2
# Two nodes of two ranks: every input scope holds a follower, and per-node scopes hold copies of their own.
SCOPED_WORLD_SIZE, RANKS_PER_NODE = 4, 2

# A diverged load ends in a stuck collective; the suite must not sit on one.
PG_TIMEOUT_SEC = 30

# Two one-rank nodes on per-node storage: each rank is its node's local rank 0.
PER_NODE_ENV = {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1", "DIST_SHARED_FILESYSTEM": "0"}

SAVED_SOURCE = "./ds"
# Hub-shaped, so the load takes the Hub path; the tests hand it the saved copy under the shared directory.
HUB_SOURCE = "org/corpus"
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
        write_stamped_sidecar(os.path.join(directory, split, SHARD_INDEX_FILE), index)


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


def _hub_copy(tmp_dir, node: int) -> str:
    return os.path.join(tmp_dir, f"hub_copy_node{node}")


def _hub_load_worker(rank: int, tmp_dir: str, follower_open_fails: bool, ranks_per_node: int) -> None:
    """Load a Hub-shaped source that resolves to this rank's node's saved copy; count the resolutions per rank."""
    os.environ.update(LOCAL_RANK=str(rank % ranks_per_node), LOCAL_WORLD_SIZE=str(ranks_per_node))
    PartialState()

    def resolve_from_the_hub(path):
        _count_call(tmp_dir, rank, "source_load")
        return Dataset.load_from_disk(_hub_copy(tmp_dir, rank // ranks_per_node))

    loading.load_dataset_from_source = resolve_from_the_hub
    if follower_open_fails and rank == 1:

        def unreadable(payload):
            raise FileNotFoundError("the load rank's cache file is not on this node")

        loading.pickle = type("Pickle", (), {"dumps": staticmethod(pickle.dumps), "loads": staticmethod(unreadable)})
    try:
        ds, _ = loading._load_dataset_from_path(HUB_SOURCE, None)
        outcome = {
            "train": ds["train"][:]["text"],
            "fingerprint": ds["train"]._fingerprint,
            "files": sorted(entry["filename"] for entry in ds["train"].cache_files),
        }
    except BaseException as exc:
        outcome = {"error": f"{type(exc).__name__}: {exc}"}
    _record(tmp_dir, rank, json.dumps(outcome))


def _pack_worker(rank: int, tmp_dir: str) -> None:
    """Pack on every rank, the pack raising on the main rank only."""
    PartialState()

    real_pack_batches = coordinated_map_module._pack_batches

    def pack_failing_on_main(*args, **kwargs):
        _count_call(tmp_dir, rank, "pack")
        if rank == 0:
            raise OSError("No space left on device")
        return real_pack_batches(*args, **kwargs)

    coordinated_map_module._pack_batches = pack_failing_on_main
    try:
        coordinated_map_module.coordinated_pack(
            Dataset.from_dict({"input_ids": [[1, 2], [3]]}), seq_length=4, num_proc=1
        )
        outcome = "NO RAISE"
    except BaseException as exc:
        outcome = f"{type(exc).__name__}: {exc}"
    _record(tmp_dir, rank, outcome)


def _run(tmp_path, worker, *args, env=None, world_size: int = WORLD_SIZE) -> list[str]:
    run_gloo_ranks(
        worker,
        world_size,
        str(tmp_path),
        *args,
        pg_timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC),
        env={"HF_DATASETS_CACHE": str(tmp_path / "hf_datasets"), **(env or {})},
    )
    return [(tmp_path / f"result_{rank}.txt").read_text() for rank in range(world_size)]


def _loads(tmp_path, source: str = SAVED_SOURCE, data_parallel_size: int = 1, env=None) -> list[dict]:
    return [json.loads(outcome) for outcome in _run(tmp_path, _load_worker, source, data_parallel_size, env=env)]


def test_a_replicated_load_whose_node_copies_differ_raises_on_every_rank(tmp_path):
    """Without the agreement each node trains its own corpus version and the DP shards no longer partition one
    dataset."""
    _save_replicated(_node_dir(tmp_path, 0), ROWS)
    _save_replicated(_node_dir(tmp_path, 1), EDITED_ROWS)

    for rank, outcome in enumerate(_loads(tmp_path)):
        assert outcome.get("error", "").startswith("ValueError"), f"rank {rank} trained on: {outcome}"
        assert "loaded different ones" in outcome["error"], f"rank {rank}: {outcome}"


def test_identical_node_copies_load_although_their_hf_fingerprints_differ(tmp_path):
    """A data file pre-staged at another mtime gets another HF fingerprint for the very same rows, which is why the
    identity is read off the rows."""
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
    """Node 1's copy has one row edited under the same index: stage 0 would forward one row set while the last stage
    scores another's labels."""
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
    """The shard assignment is computed off each rank's own index, so a node holding a re-prepared copy loads
    overlapping shards."""
    _save_sharded(_node_dir(tmp_path, 0), ROWS, num_shards=4)
    _save_sharded(_node_dir(tmp_path, 1), ROWS, num_shards=2)

    for rank, outcome in enumerate(_loads(tmp_path, data_parallel_size=WORLD_SIZE)):
        assert outcome.get("error", "").startswith("ValueError"), f"rank {rank} trained on: {outcome}"
        assert "shard index" in outcome["error"], f"rank {rank}: {outcome}"


def test_a_split_only_some_node_copies_carry_raises(tmp_path):
    """Node 1's ranks would skip every coordinated operation over the test split while their peers run them."""
    for rank in range(WORLD_SIZE):
        _save_sharded(_node_dir(tmp_path, rank), ROWS, num_shards=2)
    os.remove(os.path.join(_node_dir(tmp_path, 1), "ds", "test", SHARD_INDEX_FILE))

    for rank, outcome in enumerate(_loads(tmp_path, data_parallel_size=WORLD_SIZE)):
        assert outcome.get("error", "").startswith("ValueError"), f"rank {rank} trained on: {outcome}"
        assert "'test'" in outcome["error"], f"rank {rank} was not told which split: {outcome}"


def test_identical_sharded_copies_load(tmp_path):
    for rank in range(WORLD_SIZE):
        _save_sharded(_node_dir(tmp_path, rank), ROWS, num_shards=2)

    outcomes = _loads(tmp_path, data_parallel_size=1)

    assert all(outcome.get("train") == len(ROWS) for outcome in outcomes), outcomes


def test_a_failed_first_load_raises_everywhere_and_is_not_repeated(tmp_path):
    """The waiter raises the cause and does not retry: at scale, every peer would re-download a corpus the main rank
    could not."""
    os.makedirs(_node_dir(tmp_path, 0))
    _save_replicated(_node_dir(tmp_path, 1), ROWS)

    outcomes = _loads(tmp_path)

    assert outcomes[0]["error"].startswith("FileNotFoundError"), outcomes[0]
    assert outcomes[1]["error"].startswith("RuntimeError"), outcomes[1]
    assert "Could not load dataset" in outcomes[1]["error"], f"the peer was not told the cause: {outcomes[1]}"
    assert _calls(tmp_path, 1, "source_load") == 0, "the waiting rank repeated the load the main rank failed"


def test_a_node_whose_load_fails_takes_every_node_down(tmp_path):
    """Node 0 succeeded, so without a join it would walk into its next collective alone."""
    _save_replicated(_node_dir(tmp_path, 0), ROWS)
    os.makedirs(_node_dir(tmp_path, 1))

    outcomes = _loads(tmp_path, env=PER_NODE_ENV)

    assert outcomes[1]["error"].startswith("FileNotFoundError"), outcomes[1]
    assert outcomes[0]["error"].startswith("RuntimeError"), outcomes[0]
    assert "Could not load dataset" in outcomes[0]["error"], f"node 0 was not told the cause: {outcomes[0]}"
    assert _calls(tmp_path, 0, "source_load") == _calls(tmp_path, 1, "source_load") == 1


def _hub_loads(
    tmp_path, *, follower_open_fails: bool = False, env=None, world_size: int = WORLD_SIZE, ranks_per_node: int
) -> list[dict]:
    for node in range(world_size // ranks_per_node):
        Dataset.from_dict({"text": ROWS}).save_to_disk(_hub_copy(tmp_path, node))
    outcomes = _run(tmp_path, _hub_load_worker, follower_open_fails, ranks_per_node, env=env, world_size=world_size)
    return [json.loads(outcome) for outcome in outcomes]


@pytest.mark.parametrize("shared", [True, False], ids=["shared", "per-node"])
def test_a_hub_source_is_resolved_once_per_input_scope(tmp_path, shared):
    """Shared: every other rank opens rank 0's files. Per-node: each node's local rank 0 resolves into that node's
    copy, which the node's follower opens."""
    env = None if shared else {"DIST_SHARED_FILESYSTEM": "0"}
    outcomes = _hub_loads(tmp_path, env=env, world_size=SCOPED_WORLD_SIZE, ranks_per_node=RANKS_PER_NODE)

    for rank, outcome in enumerate(outcomes):
        load_rank = 0 if shared else rank - rank % RANKS_PER_NODE
        assert outcome.get("train") == ROWS, f"rank {rank}: {outcome}"
        assert outcome["files"] == outcomes[load_rank]["files"] != [], f"rank {rank} opened another copy: {outcomes}"
        assert outcome["fingerprint"] == outcomes[load_rank]["fingerprint"], f"rank {rank}: {outcomes}"
        resolutions = 1 if rank == load_rank else 0
        assert _calls(tmp_path, rank, "source_load") == resolutions, f"rank {rank} resolved the Hub source"
    assert (outcomes[0]["files"] == outcomes[2]["files"]) is shared, f"the nodes' copies: {outcomes}"


def test_a_follower_that_cannot_open_the_load_ranks_copy_raises_everywhere(tmp_path):
    outcomes = _hub_loads(tmp_path, follower_open_fails=True, ranks_per_node=WORLD_SIZE)

    assert outcomes[1]["error"].startswith("FileNotFoundError"), outcomes[1]
    assert outcomes[0]["error"].startswith("RuntimeError"), outcomes[0]
    assert "not on this node" in outcomes[0]["error"], f"rank 0 was not told the cause: {outcomes[0]}"


def test_a_failed_pack_raises_everywhere_and_is_not_repeated(tmp_path):
    """The pack is the same shape of work: one rank fills the cache the others read back."""
    outcomes = _run(tmp_path, _pack_worker)

    assert outcomes[0] == "OSError: No space left on device", outcomes[0]
    assert outcomes[1].startswith("RuntimeError") and "No space left on device" in outcomes[1], outcomes[1]
    assert _calls(tmp_path, 1, "pack") == 0, "the waiting rank repeated the pack the main rank failed"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
