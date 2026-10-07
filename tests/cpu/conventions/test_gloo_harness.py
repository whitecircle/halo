#!/usr/bin/env python
"""``tests.common.gloo.run_gloo_ranks`` keeps the rendezvous store up until every rank has exited.

A rank may leave while a peer still has store traffic to send: the main rank of
``fs_aware_main_first`` never waits for its waiters, so it can exit while a waiter is still leaving
the phase. A store that goes down with the exiting rank fails the peer on a reset connection, which a
loaded host turns into an intermittent failure of any test whose ranks finish unevenly. Here rank 0
exits first by construction, and rank 1 uses the store only after.

Run: python tests/cpu/conventions/test_gloo_harness.py
"""

import fcntl
import os

import pytest
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
