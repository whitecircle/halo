"""A checkpoint write failing on one rank aborts every rank, on real gloo. The single-process tests take the
``world_size <= 1`` early return, so a plain local raise passes them and fails here."""

import datetime
import os

import pytest
import torch.distributed as dist

from halo.runtime.consensus import DeferredRankFailure
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2
# Without the gather the non-failing rank can only end in a gloo timeout.
PG_TIMEOUT_SEC = 15

ENOSPC = "No space left on device"


def _worker(rank: int, tmp_dir: str, failing_rank: int) -> None:
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


def _results(tmp_path, failing_rank: int) -> list[tuple[int, str]]:
    run_gloo_ranks(
        _worker, WORLD_SIZE, str(tmp_path), failing_rank, pg_timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC)
    )
    out = []
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"result_{rank}.txt") as fh:
            count, outcome = fh.read().split("|", 1)
        out.append((int(count), outcome))
    return out


def test_one_ranks_write_failure_aborts_every_rank_with_the_cause(tmp_path):
    """Rank 0's write succeeds and it still raises, naming rank 1's errno."""
    failing_rank = 1
    results = _results(tmp_path, failing_rank=failing_rank)

    for rank, (collectives, outcome) in enumerate(results):
        # The failing rank keeps entering the collectives.
        assert collectives == 3, f"rank {rank} skipped a collective ({collectives}/3) — peers would hang"
        assert outcome != "NO RAISE", f"rank {rank} sailed past a failed checkpoint write"
        assert "Timeout" not in outcome and "timed out" not in outcome, f"rank {rank} only saw a timeout: {outcome}"
        assert ENOSPC in outcome, f"rank {rank} was not told the real cause: {outcome}"

    assert "1 of 2 rank(s)" in results[0][1], f"the verdict did not identify the failing rank: {results[0][1]}"


def test_a_clean_save_raises_on_nobody(tmp_path):
    results = _results(tmp_path, failing_rank=-1)
    for rank, (collectives, outcome) in enumerate(results):
        assert collectives == 3
        assert outcome == "NO RAISE", f"rank {rank} raised on a clean save: {outcome}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
