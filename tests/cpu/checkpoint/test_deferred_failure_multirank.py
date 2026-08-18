"""A checkpoint write that fails on ONE rank must abort every rank — proven on a real group.

:class:`DeferredRankFailure` exists so a save-rank ENOSPC does not strand its peers in the next
collective. The single-process tests next door pin the local half (record, don't raise, keep the
first cause), but they all take ``reject_across_ranks``'s ``world_size <= 1`` early return, which
degenerates to a plain local raise. Gutting the ``all_gather_object`` and replacing it with
``if self.reason: raise`` passes every one of them — and reinstates exactly the bug the class was
written to prevent, because the ranks that wrote successfully never learn a peer failed.

This is the test that fails for that mutation. Two ranks, real gloo, only rank 1 "fails" its write:
rank 0 must still raise, and its message must carry rank 1's actual errno rather than a timeout.
The process-group timeout is deliberately short so a regression FAILS instead of stalling the suite.

    python tests/cpu/checkpoint/test_deferred_failure_multirank.py
"""

import contextlib
import datetime
import os
import sys

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from src.distributed.runtime import DeferredRankFailure
from tests.common.ports import free_port

WORLD_SIZE = 2
# Far below any plausible real wait: without the gather, the non-failing rank can only end in a gloo
# timeout, and the suite must not sit on it.
PG_TIMEOUT_SEC = 15

ENOSPC = "No space left on device"


def _init(rank: int, port: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=port, RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE))
    dist.init_process_group(
        "gloo", rank=rank, world_size=WORLD_SIZE, timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC)
    )


def _worker(rank: int, tmp_dir: str, failing_rank: int, port: str) -> None:
    """Model the streaming save: every rank stages 'layers', one rank's write fails partway."""
    _init(rank, port)
    guard = DeferredRankFailure("checkpoint write")
    collectives_entered = 0

    try:
        for layer in range(3):
            # Stands in for the per-layer expert all-gather every rank must enter.
            dist.barrier()
            collectives_entered += 1

            def write(layer=layer):
                if rank == failing_rank and layer == 1:
                    raise OSError(28, ENOSPC)

            guard.run(write)
        guard.reject()
        outcome = "NO RAISE"
    except BaseException as e:
        outcome = f"{type(e).__name__}: {e}"

    with open(os.path.join(tmp_dir, f"result_{rank}.txt"), "w") as fh:
        fh.write(f"{collectives_entered}|{outcome}")
    with contextlib.suppress(Exception):
        dist.destroy_process_group()


def _results(tmp_path, failing_rank: int) -> list[tuple[int, str]]:
    # One freshly allocated port per call — a named port races the previous run's lingering TCPStore
    # and every other launch on the host.
    port = str(free_port())
    mp.start_processes(
        _worker, args=(str(tmp_path), failing_rank, port), nprocs=WORLD_SIZE, join=True, start_method="spawn"
    )
    out = []
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"result_{rank}.txt") as fh:
            count, outcome = fh.read().split("|", 1)
        out.append((int(count), outcome))
    return out


def test_one_ranks_write_failure_aborts_every_rank_with_the_cause(tmp_path):
    """The headline behavior. Rank 0's write succeeds; it must still raise, naming rank 1's errno."""
    failing_rank = 1
    results = _results(tmp_path, failing_rank=failing_rank)

    for rank, (collectives, outcome) in enumerate(results):
        # The whole point of deferring: the failing rank keeps entering the collectives.
        assert collectives == 3, f"rank {rank} skipped a collective ({collectives}/3) — peers would hang"
        assert outcome != "NO RAISE", f"rank {rank} sailed past a failed checkpoint write"
        assert "Timeout" not in outcome and "timed out" not in outcome, f"rank {rank} only saw a timeout: {outcome}"
        assert ENOSPC in outcome, f"rank {rank} was not told the real cause: {outcome}"

    assert "1 of 2 rank(s)" in results[0][1], f"the verdict did not identify the failing rank: {results[0][1]}"


def test_a_clean_save_raises_on_nobody(tmp_path):
    """The gather must not be a raise-always: with no failure anywhere, every rank proceeds."""
    results = _results(tmp_path, failing_rank=-1)
    for rank, (collectives, outcome) in enumerate(results):
        assert collectives == 3
        assert outcome == "NO RAISE", f"rank {rank} raised on a clean save: {outcome}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
