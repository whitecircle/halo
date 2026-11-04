#!/usr/bin/env python
"""``tests.common.gloo.run_gloo_ranks`` keeps the rendezvous store up until every rank has exited, and spawns each
rank with the ``OMP_NUM_THREADS`` torchrun would. A store dying with an early-exiting rank resets its peer's
connection, and a pool ``PartialState`` resized lets two forwards disagree on a loaded xdist host."""

import fcntl
import json
import os
from pathlib import Path

import pytest
import torch
from accelerate import PartialState
from torch.distributed import distributed_c10d as c10d

from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2


def _store_after_rank0_exits(rank: int, tmp_dir: str) -> None:
    """Rank 0 holds a lock only its process exit releases; rank 1 takes it, then sets and reads a key."""
    store = c10d._get_default_store()
    # A raw descriptor, never closed: the lock lasts exactly as long as the process holding it.
    lock = os.open(os.path.join(tmp_dir, "rank0.lock"), os.O_RDWR | os.O_CREAT)
    if rank == 0:
        fcntl.flock(lock, fcntl.LOCK_EX)
        store.set("rank0_locked", "1")
        return
    store.wait(["rank0_locked"])
    fcntl.flock(lock, fcntl.LOCK_EX)
    store.set("after_rank0_exit", "served")
    with open(os.path.join(tmp_dir, "rank1.txt"), "w") as fh:
        fh.write(store.get("after_rank0_exit").decode())


def test_the_store_outlives_the_first_rank_to_exit(tmp_path):
    run_gloo_ranks(_store_after_rank0_exits, WORLD_SIZE, str(tmp_path))
    assert (tmp_path / "rank1.txt").read_text() == "served"


def _thread_counts(rank: int, tmp_dir: str) -> None:
    """The intra-op thread count before and after accelerate's process state, and the OMP_NUM_THREADS the rank sees."""
    before = torch.get_num_threads()
    PartialState()
    record = {"before": before, "after": torch.get_num_threads(), "omp": os.environ.get("OMP_NUM_THREADS")}
    Path(tmp_dir, f"rank{rank}.json").write_text(json.dumps(record))


@pytest.mark.parametrize(("exported", "threads"), [(None, 1), ("3", 3)], ids=["torchrun_default", "caller_export"])
def test_each_rank_starts_with_torchrun_omp_threads_and_partial_state_keeps_them(
    tmp_path, monkeypatch, exported, threads
):
    if exported is None:
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    else:
        monkeypatch.setenv("OMP_NUM_THREADS", exported)
    run_gloo_ranks(_thread_counts, WORLD_SIZE, str(tmp_path))
    records = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(WORLD_SIZE)]
    assert records == [{"before": threads, "after": threads, "omp": str(threads)}] * WORLD_SIZE, records
    assert os.environ.get("OMP_NUM_THREADS") == exported, "the spawn's export leaked into the calling process"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
