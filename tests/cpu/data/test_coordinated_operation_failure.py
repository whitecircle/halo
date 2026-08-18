"""A per-rank data-pipeline error must abort every rank with the real cause, not hang the job.

``coordinated_dataset_operation`` runs the map/filter on ONE rank while the others wait. Joining the
two phases with a bare barrier would leave a raise in ``operation_fn`` stranding the peers there for
the whole watchdog window, while the failing rank goes straight into ``run_training``'s
``destroy_process_group`` — which blocks too, so the interpreter never prints the traceback. That is
a 20-minute silence for an error a single-rank re-run surfaces in seconds, so the join carries the
failure instead.

Proven on a real gloo group, with a deliberately short process-group timeout so a regression FAILS
rather than stalling the suite: (1) both ranks raise, (2) the non-failing rank is told the actual
cause rather than a timeout, (3) the rank that failed keeps its OWN exception type — the peers take
the uniform ``RuntimeError``, but ``tokenize_vlm_dataset``'s ``NotImplementedError`` capability
refusal is a contract, and (4) a failure on a NON-main rank (a torn cache read) propagates the same
way. The teardown half is pinned separately: ``run_training`` must print the traceback BEFORE its own
blocking teardown.

    python tests/cpu/data/test_coordinated_operation_failure.py
"""

import builtins
import contextlib
import datetime
import os
import sys

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset

from src.data.pipeline.processing import coordinated_dataset_operation
from src.training import environment
from tests.common.ports import free_port

WORLD_SIZE = 2
# One port per test — a reused port collides with the previous run's lingering TCPStore.

# Far below any plausible real barrier wait: without the propagation the waiting rank can only end in
# a gloo timeout, and the suite must not sit on it.
PG_TIMEOUT_SEC = 15

BOOM = "operation_fn blew up"


def _init(rank: int, port: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=port, RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE))
    os.environ["HF_DATASETS_CACHE"] = os.environ.get("TMPDIR", "/tmp") + "/coordinated_op_cache"
    dist.init_process_group(
        "gloo", rank=rank, world_size=WORLD_SIZE, timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC)
    )


def _record(tmp_dir: str, rank: int, text: str) -> None:
    with open(os.path.join(tmp_dir, f"result_{rank}.txt"), "w") as fh:
        fh.write(text)


def _worker(rank: int, tmp_dir: str, failing_rank: int, port: str, exc_name: str) -> None:
    """Fail ``operation_fn`` on ``failing_rank`` only; record what each rank ends up seeing."""
    _init(rank, port)
    dataset = Dataset.from_dict({"text": ["a", "b"]})
    exc_type = getattr(builtins, exc_name)

    def operation_fn(**kwargs):
        if rank == failing_rank:
            raise exc_type(BOOM)
        return dataset

    # BaseException, so a KeyboardInterrupt raised out of the operation is recorded like any other.
    try:
        coordinated_dataset_operation(operation_fn, dataset, operation_name="unit-test op", num_proc=1)
        _record(tmp_dir, rank, "NO RAISE")
    except BaseException as e:
        _record(tmp_dir, rank, f"{type(e).__name__}: {e}")
    with contextlib.suppress(Exception):
        dist.destroy_process_group()


def _results(tmp_path, failing_rank: int, port: str, exc_name: str = "ValueError") -> list[str]:
    mp.start_processes(
        _worker,
        args=(str(tmp_path), failing_rank, port, exc_name),
        nprocs=WORLD_SIZE,
        join=True,
        start_method="spawn",
    )
    out = []
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"result_{rank}.txt") as fh:
            out.append(fh.read())
    return out


def test_main_rank_failure_aborts_every_rank_with_the_cause(tmp_path):
    """Rank 0 owns the map; rank 1 waits. Rank 1 must raise, and its message must name the real
    error — a bare 'barrier timed out' sends the reader hunting the wrong subsystem."""
    failing_rank = 0
    results = _results(tmp_path, failing_rank=failing_rank, port=str(free_port()))

    for rank, result in enumerate(results):
        assert result != "NO RAISE", f"rank {rank} sailed past a failed dataset operation"
        assert "Timeout" not in result and "timed out" not in result, f"rank {rank} only saw a timeout: {result}"
        assert BOOM in result, f"rank {rank} was not told the real cause: {result}"
    # The type, not just the text: callers branch on it (tokenize_vlm_dataset's NotImplementedError).
    assert results[failing_rank].startswith("ValueError:"), (
        f"the failing rank lost its own exception type: {results[failing_rank]}"
    )


def test_non_main_rank_failure_also_aborts_every_rank(tmp_path):
    """The second phase needs the same treatment: only the waiters read the cache back, so a torn or
    unreadable cache file fails on ranks the writer never hears from."""
    failing_rank = 1
    results = _results(tmp_path, failing_rank=failing_rank, port=str(free_port()))

    for rank, result in enumerate(results):
        assert result != "NO RAISE", f"rank {rank} sailed past a failed dataset operation"
        assert BOOM in result, f"rank {rank} was not told the real cause: {result}"
    assert results[failing_rank].startswith("ValueError:"), (
        f"the failing rank lost its own exception type: {results[failing_rank]}"
    )


def test_base_exception_in_the_operation_still_aborts_the_peer(tmp_path):
    """A ``KeyboardInterrupt`` out of the map is the same hang: catching only ``Exception`` would let
    the interrupted rank exit past the join while its peer waits out the process-group timeout."""
    failing_rank = 0
    results = _results(tmp_path, failing_rank=failing_rank, port=str(free_port()), exc_name="KeyboardInterrupt")

    assert results[failing_rank] == f"KeyboardInterrupt: {BOOM}", results[failing_rank]
    peer = results[1 - failing_rank]
    assert peer.startswith("RuntimeError:"), peer
    assert "KeyboardInterrupt" in peer and BOOM in peer, f"the peer was not told the real cause: {peer}"
    assert "Timeout" not in peer and "timed out" not in peer, f"the peer only saw a timeout: {peer}"


SLOW_MAP_PG_TIMEOUT_SEC = 3


def _slow_map_worker(rank: int, tmp_dir: str, port: str) -> None:
    """The writer's map outlives the PG timeout; the waiting rank must survive on the store join."""
    import time

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=port, RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE))
    os.environ["HF_DATASETS_CACHE"] = os.environ.get("TMPDIR", "/tmp") + "/coordinated_op_cache"
    dist.init_process_group(
        "gloo", rank=rank, world_size=WORLD_SIZE, timeout=datetime.timedelta(seconds=SLOW_MAP_PG_TIMEOUT_SEC)
    )
    dataset = Dataset.from_dict({"text": ["a", "b"]})

    def operation_fn(**kwargs):
        if rank == 0:
            time.sleep(SLOW_MAP_PG_TIMEOUT_SEC + 3)
        return dataset

    try:
        coordinated_dataset_operation(operation_fn, dataset, operation_name="slow map", num_proc=1)
        result = "PASS"
    except BaseException as e:
        result = f"FAIL: {type(e).__name__}: {str(e).splitlines()[0][:160]}"
    _record(tmp_dir, rank, result)
    with contextlib.suppress(Exception):
        dist.destroy_process_group()


def test_writer_map_may_outlive_the_process_group_timeout(tmp_path):
    """The store join must let the writer's map run to completion: joins built on
    ``all_gather_object`` keep the waiting rank INSIDE the process group for the whole map, so a
    fresh-cache tokenization past DIST_NCCL_TIMEOUT_MINUTES kills the run blaming the collective."""
    port = str(free_port())
    mp.start_processes(
        _slow_map_worker, args=(str(tmp_path), port), nprocs=WORLD_SIZE, join=True, start_method="spawn"
    )
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"result_{rank}.txt") as fh:
            result = fh.read()
        assert result == "PASS", f"rank {rank}: {result}"


def test_run_training_prints_the_traceback_before_its_blocking_teardown(capfd, monkeypatch):
    """``destroy_process_group`` is a collective and blocks whenever the peers are not also tearing
    down; Python prints an uncaught exception only after the ``finally`` returns. The traceback must
    therefore ALREADY be out when teardown starts, which is what the stderr snapshot taken from
    inside the teardown proves — assert it afterwards and a hanging teardown still passes."""
    at_teardown: list[str] = []
    monkeypatch.setattr(environment, "destroy_all_dispatchers", lambda: at_teardown.append(capfd.readouterr().err))

    @environment.run_training
    def main():
        raise ValueError("the real cause")

    with pytest.raises(ValueError):
        main()

    assert at_teardown, "teardown never ran"
    assert "the real cause" in at_teardown[0], "the traceback was not on stderr before the teardown"
    assert "ValueError" in at_teardown[0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
