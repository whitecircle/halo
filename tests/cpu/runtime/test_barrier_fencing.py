#!/usr/bin/env python3
"""A rank-gated write failing on one rank fails every rank, on a real group: ``barrier_on_exit`` joins the failure
where the barrier stood. Otherwise peers sit in the next collective until the NCCL watchdog, with no diagnostic."""

import datetime
import json
import os

import pytest
import torch
import torch.distributed as dist

import halo.model_loading.parallel.whole_model as whole_model
from halo.runtime import consensus, ranks
from halo.runtime.consensus import barrier_on_exit
from halo.runtime.filesystem import fs_aware_makedirs
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2
FAILING_RANK = 1
# A rank that returned instead of joining would reach the next fence alone.
PG_TIMEOUT = datetime.timedelta(seconds=30)
ENOSPC = "No space left on device"
JIT_FAILURE = "FA4 JIT compile failed"
# A regular file, so a directory under it cannot be made.
BLOCKING_FILE = "not_a_directory"


def test_the_writers_own_exception_propagates():
    """Callers match on the writer's exception type."""
    with pytest.raises(OSError, match=ENOSPC):
        with barrier_on_exit("Writing scheduler.pt"):
            raise OSError(28, ENOSPC)


def test_the_join_runs_after_the_body(monkeypatch):
    """A fence before the write would publish a half-written checkpoint."""
    order = []
    monkeypatch.setattr(consensus, "reject_across_ranks", lambda reason, what: order.append(("join", reason, what)))
    with barrier_on_exit("Writing scheduler.pt"):
        order.append("write")
    assert order == ["write", ("join", None, "Writing scheduler.pt")]


def test_the_fence_is_inert_without_a_process_group():
    assert not ranks.dist.is_initialized()
    with barrier_on_exit("Writing scheduler.pt"):
        pass  # Reaching here without raising is the assertion.


def _fa4_compile_failing_on_one_rank(_model, *, dtype) -> None:
    del dtype
    if dist.get_rank() == FAILING_RANK:
        raise RuntimeError(JIT_FAILURE)


def _no_flex_compile(_model, *, dtype) -> None:
    del dtype


def _fence_outcomes(rank: int, tmp_dir: str) -> None:
    """A clean fence last proves the ranks are still in step after three joined failures."""
    outcomes: dict[str, str] = {}

    def record(name, fence) -> None:
        try:
            fence()
            outcomes[name] = "returned"
        except BaseException as exc:  # Recorded per rank: the outcome on every rank is the subject.
            outcomes[name] = f"{type(exc).__name__}: {exc}"

    def failing_write() -> None:
        with barrier_on_exit("Writing scheduler.pt"):
            if rank == FAILING_RANK:
                raise OSError(28, ENOSPC)

    def clean_write() -> None:
        with barrier_on_exit("Writing scheduler.pt"):
            pass

    # Global rank 0 is the shared-filesystem writer, so the failure sits on the other rank.
    record("fs_aware_makedirs", lambda: fs_aware_makedirs(os.path.join(tmp_dir, BLOCKING_FILE, "cache")))
    record("barrier_on_exit", failing_write)
    whole_model.warm_fa4_kernels = _fa4_compile_failing_on_one_rank
    whole_model.warm_flex_sliding_kernels = _no_flex_compile
    record("warm-up", lambda: whole_model.warm_attention_kernels(None, dtype=torch.bfloat16))
    record("clean", clean_write)
    with open(os.path.join(tmp_dir, f"outcomes_{rank}.json"), "w") as fh:
        json.dump(outcomes, fh)


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory) -> list[dict[str, str]]:
    tmp_dir = tmp_path_factory.mktemp("fences")
    (tmp_dir / BLOCKING_FILE).write_text("")
    run_gloo_ranks(_fence_outcomes, WORLD_SIZE, str(tmp_dir), pg_timeout=PG_TIMEOUT)
    return [json.loads((tmp_dir / f"outcomes_{rank}.json").read_text()) for rank in range(WORLD_SIZE)]


@pytest.mark.parametrize(
    ("fence", "failing_rank", "writer_error", "cause"),
    [
        pytest.param("barrier_on_exit", FAILING_RANK, "OSError", ENOSPC, id="barrier_on_exit"),
        pytest.param("fs_aware_makedirs", 0, "NotADirectoryError", BLOCKING_FILE, id="fs_aware_makedirs"),
        pytest.param("warm-up", FAILING_RANK, "RuntimeError", JIT_FAILURE, id="attention warm-up"),
    ],
)
def test_one_ranks_failure_raises_on_every_rank_naming_it(outcomes, fence, failing_rank, writer_error, cause):
    failing, peer = outcomes[failing_rank][fence], outcomes[1 - failing_rank][fence]
    assert failing.startswith(f"{writer_error}:") and cause in failing, failing
    assert peer.startswith("RuntimeError:"), f"the peer of a failed {fence} saw: {peer}"
    assert f"1 of {WORLD_SIZE} rank(s) [{failing_rank}]" in peer and cause in peer, peer


def test_a_clean_fence_raises_on_nobody(outcomes):
    """The join must not raise always."""
    assert [rank_outcomes["clean"] for rank_outcomes in outcomes] == ["returned"] * WORLD_SIZE


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
