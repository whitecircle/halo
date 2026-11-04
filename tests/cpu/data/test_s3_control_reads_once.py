"""An S3 dataset's control files are read once per launch: at thousands of ranks the per-rank probes are tens of
thousands of GETs on one prefix. Proven on four gloo ranks as two nodes against a fake S3: rank 0 reads, one rank
per node mirrors, a key rank 0 cannot read is read by every rank, and each served read is a copy."""

import collections
import datetime
import json
import os

import pytest
from accelerate import PartialState
from datasets import Dataset, DatasetDict, load_from_disk

from halo.data.preprocessed.metadata import load_preprocessed_metadata
from halo.data.sources import s3_dataset_store
from halo.data.sources.loading import (
    is_presharded_dataset_load_across_ranks,
    load_datasets_auto,
    share_s3_control_files,
)
from halo.data.sources.sharded_dataset import ShardedDatasetLoader
from tests.common.datasets import write_prepared_text_dataset
from tests.common.gloo import run_gloo_ranks
from tests.common.models import QWEN3_0_6B
from tests.common.tokenizers import load_cached_tokenizer

WORLD_SIZE = 4
RANKS_PER_NODE = 2
PG_TIMEOUT_SEC = 60
BUCKET, KEY = "bucket", "prepared/corpus"
SOURCE = f"s3://{BUCKET}/{KEY}"


def _load_worker(rank: int, tmp_dir: str, rank0_first_read_fails: str | None) -> None:
    """A script's dataset load: the shared read, the presharded probe, the load and the metadata check."""
    os.environ.update(LOCAL_RANK=str(rank % RANKS_PER_NODE), LOCAL_WORLD_SIZE=str(RANKS_PER_NODE))
    PartialState()
    reads, mirror_writes = collections.Counter(), collections.Counter()
    failed_once = set()

    def fake_s3_read(bucket, key):
        reads[key] += 1
        if rank == 0 and key == rank0_first_read_fails and key not in failed_once:
            failed_once.add(key)
            raise OSError("Could not connect to the endpoint URL")
        path = os.path.join(tmp_dir, "s3", bucket, key)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path) as f:
            return json.load(f)

    real_mirror_write = s3_dataset_store._write_control_json_mirror

    def counting_mirror_write(bucket, key, payload):
        mirror_writes[key] += 1
        real_mirror_write(bucket, key, payload)

    s3_dataset_store.read_json_from_s3 = fake_s3_read
    s3_dataset_store._write_control_json_mirror = counting_mirror_write
    s3_dataset_store.HALO_S3_DATASET_CACHE_DIR = os.path.join(tmp_dir, f"node{rank // RANKS_PER_NODE}")
    ShardedDatasetLoader._load_shard_from_s3 = lambda self, shard: load_from_disk(
        os.path.join(tmp_dir, "s3", BUCKET, KEY, shard.path)
    )

    share_s3_control_files(SOURCE)
    presharded = is_presharded_dataset_load_across_ranks(SOURCE, WORLD_SIZE)
    ds, preprocessed = load_datasets_auto(SOURCE, None, None, data_parallel_rank=rank, data_parallel_size=WORLD_SIZE)
    load_preprocessed_metadata(SOURCE)
    outcome = {
        "presharded": presharded,
        "preprocessed": preprocessed,
        "train_rows": len(ds["train"]),
        "reads": dict(reads),
        "mirror_writes": dict(mirror_writes),
    }
    with open(os.path.join(tmp_dir, f"result_{rank}.json"), "w") as f:
        json.dump(outcome, f)


def _run(tmp_path, rank0_first_read_fails: str | None = None) -> list[dict]:
    load_cached_tokenizer(QWEN3_0_6B)  # skips where the artifact's tokenizer is not cached
    rows = [f"document {i} " * 4 for i in range(40)]
    write_prepared_text_dataset(
        tmp_path / "s3" / BUCKET / KEY,
        DatasetDict({"train": Dataset.from_dict({"text": rows}), "test": Dataset.from_dict({"text": rows[:8]})}),
        num_shards=WORLD_SIZE,
    )
    run_gloo_ranks(
        _load_worker,
        WORLD_SIZE,
        str(tmp_path),
        rank0_first_read_fails,
        pg_timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC),
        env={"HF_DATASETS_CACHE": str(tmp_path / "hf_datasets")},
    )
    return [json.loads((tmp_path / f"result_{rank}.json").read_text()) for rank in range(WORLD_SIZE)]


def test_rank_0_reads_the_control_files_once_for_every_rank(tmp_path):
    outcomes = _run(tmp_path)

    control = sorted(outcomes[0]["reads"])
    assert control == [f"{KEY}/metadata.json", f"{KEY}/test/shard_index.json", f"{KEY}/train/shard_index.json"]
    assert all(count == 1 for count in outcomes[0]["reads"].values()), outcomes[0]["reads"]
    for rank in range(1, WORLD_SIZE):
        assert outcomes[rank]["reads"] == {}, f"rank {rank} read S3 itself: {outcomes[rank]['reads']}"
    node_mains = {rank for rank in range(WORLD_SIZE) if rank % RANKS_PER_NODE == 0}
    for rank, outcome in enumerate(outcomes):
        expected = dict.fromkeys(control, 1) if rank in node_mains else {}
        assert outcome["mirror_writes"] == expected, f"rank {rank} mirrored {outcome['mirror_writes']}"
        assert outcome["presharded"] and outcome["preprocessed"] and outcome["train_rows"] > 0, outcome


def test_a_key_rank_0_cannot_read_is_read_by_every_rank(tmp_path):
    train_index = f"{KEY}/train/shard_index.json"
    outcomes = _run(tmp_path, rank0_first_read_fails=train_index)

    for rank, outcome in enumerate(outcomes):
        assert outcome["reads"].get(train_index, 0) >= 1, f"rank {rank} never read the index rank 0 could not"
        assert outcome["reads"].get(f"{KEY}/metadata.json", 0) == (1 if rank == 0 else 0), outcome["reads"]
        assert outcome["presharded"] and outcome["train_rows"] > 0, outcome


def test_a_caller_editing_a_served_read_leaves_the_next_read_whole(tmp_path, monkeypatch):
    stored = {"version": 1, "shards": [{"path": "train/shard_0"}]}
    monkeypatch.setattr(s3_dataset_store, "read_json_from_s3", lambda bucket, key: json.loads(json.dumps(stored)))
    monkeypatch.setattr(s3_dataset_store, "HALO_S3_DATASET_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(s3_dataset_store, "_shared_control_reads", {})
    index = f"{KEY}/train/shard_index.json"
    s3_dataset_store.share_control_files(BUCKET, [index])

    s3_dataset_store.read_control_json_with_cache(BUCKET, index)["shards"].clear()

    assert s3_dataset_store.read_control_json_with_cache(BUCKET, index) == stored


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
