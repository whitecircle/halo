"""A per-rank data-pipeline error aborts every rank with the real cause. ``coordinated_dataset_operation`` runs the
map on one rank while the others wait, and a bare barrier would strand them for the whole watchdog window. Proven
on a real gloo group with a short process-group timeout, so a hang fails and does not stall."""

import builtins
import datetime
import os
import time

import pytest
from datasets import Dataset

from halo.data.pipeline.coordinated_map import coordinated_dataset_operation
from halo.entrypoints import script
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2
CACHE_ENV = {"HF_DATASETS_CACHE": os.environ.get("TMPDIR", "/tmp") + "/coordinated_op_cache"}

# Far below any plausible real barrier wait: without the propagation the waiting rank can only end in
# a gloo timeout, and the suite must not sit on it.
PG_TIMEOUT_SEC = 15

BOOM = "operation_fn blew up"


def _record(tmp_dir: str, rank: int, text: str) -> None:
    with open(os.path.join(tmp_dir, f"result_{rank}.txt"), "w") as fh:
        fh.write(text)


def _worker(rank: int, tmp_dir: str, failing_rank: int, exc_name: str) -> None:
    """Fail ``operation_fn`` on ``failing_rank`` only; record what each rank ends up seeing."""
    dataset = Dataset.from_dict({"text": ["a", "b"]})
    exc_type = getattr(builtins, exc_name)

    def operation_fn(**kwargs):
        if rank == failing_rank:
            raise exc_type(BOOM)
        return dataset

    # BaseException, so a KeyboardInterrupt raised out of the operation is recorded like any other.
    try:
        coordinated_dataset_operation(
            operation_fn, dataset, operation_name="unit-test op", num_proc=1, cache_file_name="cache-op.arrow"
        )
        _record(tmp_dir, rank, "NO RAISE")
    except BaseException as e:
        _record(tmp_dir, rank, f"{type(e).__name__}: {e}")


def _results(tmp_path, failing_rank: int, exc_name: str = "ValueError") -> list[str]:
    run_gloo_ranks(
        _worker,
        WORLD_SIZE,
        str(tmp_path),
        failing_rank,
        exc_name,
        pg_timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC),
        env=CACHE_ENV,
    )
    out = []
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"result_{rank}.txt") as fh:
            out.append(fh.read())
    return out


def test_main_rank_failure_aborts_every_rank_with_the_cause(tmp_path):
    """A bare 'barrier timed out' sends the reader hunting the wrong subsystem."""
    failing_rank = 0
    results = _results(tmp_path, failing_rank=failing_rank)

    for rank, result in enumerate(results):
        assert result != "NO RAISE", f"rank {rank} sailed past a failed dataset operation"
        assert "Timeout" not in result and "timed out" not in result, f"rank {rank} only saw a timeout: {result}"
        assert BOOM in result, f"rank {rank} was not told the real cause: {result}"
    # The type, not just the text: callers branch on it (tokenize_vlm_dataset's NotImplementedError).
    assert results[failing_rank].startswith("ValueError:"), (
        f"the failing rank lost its own exception type: {results[failing_rank]}"
    )


def test_non_main_rank_failure_also_aborts_every_rank(tmp_path):
    """Only the waiters read the cache back, so a torn cache file fails on ranks the writer never hears from."""
    failing_rank = 1
    results = _results(tmp_path, failing_rank=failing_rank)

    for rank, result in enumerate(results):
        assert result != "NO RAISE", f"rank {rank} sailed past a failed dataset operation"
        assert BOOM in result, f"rank {rank} was not told the real cause: {result}"
    assert results[failing_rank].startswith("ValueError:"), (
        f"the failing rank lost its own exception type: {results[failing_rank]}"
    )


def test_base_exception_in_the_operation_still_aborts_the_peer(tmp_path):
    """Catching only ``Exception`` lets the interrupted rank exit past the join while its peer waits out the
    timeout."""
    failing_rank = 0
    results = _results(tmp_path, failing_rank=failing_rank, exc_name="KeyboardInterrupt")

    assert results[failing_rank] == f"KeyboardInterrupt: {BOOM}", results[failing_rank]
    peer = results[1 - failing_rank]
    assert peer.startswith("RuntimeError:"), peer
    assert "KeyboardInterrupt" in peer and BOOM in peer, f"the peer was not told the real cause: {peer}"
    assert "Timeout" not in peer and "timed out" not in peer, f"the peer only saw a timeout: {peer}"


SLOW_MAP_PG_TIMEOUT_SEC = 3


def _slow_map_worker(rank: int, tmp_dir: str) -> None:
    """The writer's map outlives the PG timeout; the waiting rank must survive on the store join."""
    dataset = Dataset.from_dict({"text": ["a", "b"]})

    def operation_fn(**kwargs):
        if rank == 0:
            time.sleep(SLOW_MAP_PG_TIMEOUT_SEC + 3)
        return dataset

    try:
        coordinated_dataset_operation(
            operation_fn, dataset, operation_name="slow map", num_proc=1, cache_file_name="cache-slow.arrow"
        )
        result = "PASS"
    except BaseException as e:
        result = f"FAIL: {type(e).__name__}: {str(e).splitlines()[0][:160]}"
    _record(tmp_dir, rank, result)


def test_writer_map_may_outlive_the_process_group_timeout(tmp_path):
    """Joins on ``all_gather_object`` keep the waiter inside the process group for the whole map, so a fresh-cache
    tokenization past DIST_NCCL_TIMEOUT_MINUTES would kill the run."""
    run_gloo_ranks(
        _slow_map_worker,
        WORLD_SIZE,
        str(tmp_path),
        pg_timeout=datetime.timedelta(seconds=SLOW_MAP_PG_TIMEOUT_SEC),
        env=CACHE_ENV,
    )
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"result_{rank}.txt") as fh:
            result = fh.read()
        assert result == "PASS", f"rank {rank}: {result}"


def _record_teardown(monkeypatch) -> list[str]:
    torn_down: list[str] = []
    monkeypatch.setattr(script, "destroy_all_dispatchers", lambda: torn_down.append("dispatchers"))
    monkeypatch.setattr(script.dist, "is_available", lambda: True)
    monkeypatch.setattr(script.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(script.dist, "destroy_process_group", lambda: torn_down.append("process_group"))
    return torn_down


def test_with_distributed_teardown_skips_the_collective_teardown_when_main_raises(monkeypatch):
    """Both teardown halves are collectives, and a rank failing mid-step has peers inside the step's own collective,
    so teardown would park it in the NCCL watchdog."""
    torn_down = _record_teardown(monkeypatch)

    @script.with_distributed_teardown
    def main():
        raise ValueError("the real cause")

    with pytest.raises(ValueError, match="the real cause"):
        main()

    assert torn_down == [], f"teardown collectives ran on the failure path: {torn_down}"


def test_with_distributed_teardown_tears_down_dispatchers_before_the_process_group_on_success(monkeypatch):
    """Gin frees a DeepEP buffer through the group communicator, so the order matters."""
    torn_down = _record_teardown(monkeypatch)

    @script.with_distributed_teardown
    def main():
        return "done"

    assert main() == "done"
    assert torn_down == ["dispatchers", "process_group"]


def test_with_distributed_teardown_tears_down_before_an_early_stop_exits(monkeypatch):
    """Every rank leaves an early-stopped run at the same step, so the collectives meet; skipped, a reference
    model's DeepEP buffer outlives the process group."""
    torn_down = _record_teardown(monkeypatch)

    @script.with_distributed_teardown
    def main():
        raise script.EarlyStopExit("GRPO training stopped early at step 3 of 5.")

    with pytest.raises(SystemExit, match="stopped early at step 3 of 5"):
        main()
    assert torn_down == ["dispatchers", "process_group"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
