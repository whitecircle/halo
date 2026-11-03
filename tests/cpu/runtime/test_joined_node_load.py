#!/usr/bin/env python
"""``joined_node_load``: the node-throttled eager load, joined across the world on the store, on real gloo groups.
A torn shard raises on every rank; batches' skew is a store wait, where a collective dies at the watchdog."""

import datetime
import os
import time

import pytest
import torch
import torch.distributed as dist

from halo.runtime.load_coordination import joined_node_load
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2
PG_TIMEOUT = datetime.timedelta(seconds=5)
SLOW_BATCH_SEC = 8


class TornShardError(ValueError):
    """Its type must survive the join on the failing rank."""


def _outcome(tmp_dir: str, rank: int, result: str) -> None:
    with open(os.path.join(tmp_dir, f"rank{rank}.txt"), "w") as handle:
        handle.write(result)


def _failing_rank_worker(rank: int, tmp_dir: str) -> None:
    try:
        with joined_node_load("Model load from /ckpt", max_concurrent=1):
            if rank == 1:
                raise TornShardError("model-00002-of-00004.safetensors: incomplete metadata")
        result = "RETURNED"
    except Exception as exc:  # The outcome under test is the raise itself.
        result = f"{type(exc).__name__}: {exc}"
    _outcome(tmp_dir, rank, result)


def _skewed_batches_worker(rank: int, tmp_dir: str) -> None:
    try:
        with joined_node_load("Model load from /ckpt", max_concurrent=1):
            if rank == 1:
                time.sleep(SLOW_BATCH_SEC)
        # The first collective after the load.
        value = torch.ones(1)
        dist.all_reduce(value)
        result = "PASS" if value.item() == WORLD_SIZE else f"FAIL: all_reduce gave {value.item()}"
    except Exception as exc:
        result = f"FAIL: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    _outcome(tmp_dir, rank, result)


def _outcomes(tmp_path) -> list[str]:
    return [(tmp_path / f"rank{rank}.txt").read_text() for rank in range(WORLD_SIZE)]


def test_a_rank_local_load_failure_raises_on_every_rank(tmp_path):
    run_gloo_ranks(_failing_rank_worker, WORLD_SIZE, str(tmp_path))
    peer, failing = _outcomes(tmp_path)
    assert failing.startswith("TornShardError: model-00002"), f"the failing rank lost its own exception: {failing}"
    assert peer.startswith("RuntimeError: Model load from /ckpt failed on 1 of 2 rank(s) [1]"), (
        f"the peer must raise naming the failing rank, not return into the next collective: {peer}"
    )
    assert "TornShardError: model-00002" in peer, f"the peer's error must carry the real cause: {peer}"


def test_the_slowest_batch_is_waited_for_on_the_store_not_in_a_collective(tmp_path):
    run_gloo_ranks(_skewed_batches_worker, WORLD_SIZE, str(tmp_path), pg_timeout=PG_TIMEOUT)
    for rank, result in enumerate(_outcomes(tmp_path)):
        assert result == "PASS", f"rank {rank}: {result}"


def test_no_distributed_runs_the_body_and_keeps_its_exception():
    with pytest.raises(TornShardError):
        with joined_node_load("Model load from /ckpt", max_concurrent=1):
            raise TornShardError("single process")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
