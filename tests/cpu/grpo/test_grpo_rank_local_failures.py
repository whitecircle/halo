#!/usr/bin/env python
"""A GRPO step that fails on one rank between two collectives must raise on EVERY rank.

Each case is a rank-local failure ahead of a collective the peers are already headed for, so a lone
raise leaves them there until the watchdog, with a traceback naming the collective instead of the cause:

* component start (``_init_async_components``): every rank builds its own rollout manager and, under
  ``isr_engine_reference``, its own score-only clients, whose constructor checks a server this rank may
  not reach, and the main process its weight-sync client; the start's verdict and then the client
  formation's join them.
* the fenced weight sync: the collective gather runs on every rank and can fail on a non-main one.
* the completions record: only the writer rank writes it, and the step's next collective follows.

Proven on a real 2-rank gloo group with the external pieces (Ray, the actors, the engine) stubbed, so a
regression is the hang it would be in production, bounded by the group timeout.

    python tests/cpu/grpo/test_grpo_rank_local_failures.py
"""

import datetime
import os
import time
import types

import pytest

import src.trainers.grpo.rollout.async_rollouts as async_mod
from src.trainers.grpo.rollout.async_rollouts import AsyncRolloutMixin
from src.trainers.grpo.rollout.completions_logging import emit_completion_artifacts
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2
FAILING_RANK = 1
# A stuck peer must surface as a failed assertion, not a wedged suite; generous for an 8-way xdist host.
PG_TIMEOUT = datetime.timedelta(seconds=90)
_PUSH_SECONDS = 0.05


class _Manager:
    """A rollout manager whose actor start fails where the scenario says, and records the pause credit."""

    max_concurrent = 4

    def __init__(self, fail_start: bool):
        self._fail_start = fail_start
        self.credited: list[float] = []

    async def start(self):
        if self._fail_start:
            raise RuntimeError("no node can place the rollout actors")

    def warn_if_servers_unreachable_from_actors(self, multinode: bool) -> None:
        pass

    def begin_engine_pause(self) -> None:
        pass

    def end_engine_pause(self, seconds: float) -> None:
        self.credited.append(seconds)


def _scorer_cls(fail: bool):
    class _Scorer:
        BACKEND_NAME = "vLLM"

        def __init__(self, base_url, connection_timeout):
            if fail:
                raise ConnectionError(f"rollout server {base_url} unreachable from this node")
            self.session = types.SimpleNamespace(close=lambda: None)

    return _Scorer


class _Host(AsyncRolloutMixin):
    """The real component start and fenced sync over stubbed Ray, actors and engine."""

    def __init__(self, rank: int, fail_client: bool = False):
        self._fail_client = fail_client
        self.accelerator = types.SimpleNamespace(is_main_process=rank == 0, num_processes=WORLD_SIZE)
        self.async_config = types.SimpleNamespace(
            ray_address=None,
            num_rollout_workers=1,
            max_concurrent_rollouts=4,
            eval_rollout_batch_size=None,
            isr_engine_reference=True,
            rollout_backend="vllm",
            rollout_connection_timeout=1.0,
            get_server_urls=lambda: ["http://10.0.0.1:8000"],
            get_rollout_config=lambda **kwargs: None,
        )
        self.state = types.SimpleNamespace(global_step=5)
        self._environment_spec, self._env_config_dict = "env", {}
        self._rollout_manager = None
        self._engine_rescore_clients = None
        self._rollout_start_callback = None
        self._rollout_stop_token_ids = None
        self._loop = None
        self.callbacks: list = []

    def _resolve_reasoning_end_token_id(self):
        return None

    def _init_weight_sync_client(self):
        if self._fail_client and self.accelerator.is_main_process:
            raise ConnectionError("the NCCL group to the engine did not form")

    def _train_loader_batch_size(self) -> int:
        return 8

    def add_callback(self, callback):
        self.callbacks.append(callback)


def _record(tmp_dir: str, rank: int, step) -> None:
    """Run ``step`` and write what it raised (or ``NO RAISE``): a rank that never returns writes nothing."""
    try:
        step()
        result = "NO RAISE"
    except Exception as e:
        result = f"{type(e).__name__}: {e}"
    with open(os.path.join(tmp_dir, f"result_{rank}.txt"), "w") as fh:
        fh.write(result)


def _start_worker(rank: int, tmp_dir: str, failure: str, failing_rank: int) -> None:
    failing = rank == failing_rank
    async_mod.ray = types.SimpleNamespace(is_initialized=lambda: True)
    async_mod.RolloutManager = lambda **kwargs: _Manager(fail_start=failing and failure == "actors")
    async_mod.resolve_weight_sync_client = lambda backend: _scorer_cls(failing and failure == "rescore_client")
    _record(tmp_dir, rank, _Host(rank, fail_client=failing and failure == "sync_client")._init_async_components)


def _sync_worker(rank: int, tmp_dir: str) -> None:
    host = _Host(rank)
    host._rollout_manager = _Manager(fail_start=False)

    def push(force=False):
        if rank == FAILING_RANK:
            raise RuntimeError("expert gather refused a DTensor on this rank")
        time.sleep(_PUSH_SECONDS)
        return True

    host._sync_weights_to_engine = push
    _record(tmp_dir, rank, host._sync_weights_to_engine_fenced)
    with open(os.path.join(tmp_dir, f"credit_{rank}.txt"), "w") as fh:
        fh.write(" ".join(str(seconds) for seconds in host._rollout_manager.credited))


def _completions_worker(rank: int, tmp_dir: str) -> None:
    # A file where the completions directory goes: the writer's makedirs fails, as on a read-only mount.
    blocked = os.path.join(tmp_dir, "blocked")
    with open(blocked, "w"):
        pass
    trainer = types.SimpleNamespace(
        state=types.SimpleNamespace(global_step=7),
        args=types.SimpleNamespace(output_dir=os.path.join(blocked, "out"), report_to=[]),
        model=types.SimpleNamespace(training=True),
        num_completions_to_print=1,
        log_unique_prompts=False,
        _logs={
            "prompt": ["p"],
            "completion": ["c"],
            "rewards": {"environment_reward": [1.0]},
            "advantages": [0.5],
            "extra": {},
            "images": [],
        },
    )
    _record(tmp_dir, rank, lambda: emit_completion_artifacts(trainer, console=False, save=True))


def _results(tmp_path) -> dict[int, str]:
    results = {}
    for rank in range(WORLD_SIZE):
        path = tmp_path / f"result_{rank}.txt"
        results[rank] = path.read_text() if path.exists() else "NO RESULT (the rank never returned)"
    return results


def _assert_raised_everywhere(results: dict[int, str], cause: str, failing_rank: int) -> None:
    for rank, result in results.items():
        assert result != "NO RAISE", f"rank {rank} sailed past the failure: {results}"
        assert cause in result, f"rank {rank} was not told the real cause: {results}"
        assert f"rank {failing_rank}" in result, f"rank {rank} was not told which rank failed: {results}"


@pytest.mark.parametrize(
    ("failure", "cause", "failing_rank"),
    [
        ("rescore_client", "unreachable from this node", FAILING_RANK),
        ("actors", "no node can place the rollout actors", FAILING_RANK),
        ("sync_client", "the NCCL group to the engine did not form", 0),
    ],
)
def test_a_rank_local_component_start_failure_raises_on_every_rank(tmp_path, failure, cause, failing_rank):
    """Rank 1 cannot reach the server its re-score client checks, or cannot place its actors, while rank 0
    starts cleanly and heads into the start verdict; or the main process's weight-sync client fails, which
    the formation's verdict right after it carries."""
    run_gloo_ranks(_start_worker, WORLD_SIZE, str(tmp_path), failure, failing_rank, pg_timeout=PG_TIMEOUT)
    _assert_raised_everywhere(_results(tmp_path), cause, failing_rank)


def test_the_rescore_clients_are_built_at_component_start(monkeypatch):
    """At start, under the start verdict: the batch build that reads them sits between collectives that
    could no longer report a server this rank cannot reach."""
    monkeypatch.setattr(async_mod, "ray", types.SimpleNamespace(is_initialized=lambda: True))
    monkeypatch.setattr(async_mod, "RolloutManager", lambda **kwargs: _Manager(fail_start=False))
    monkeypatch.setattr(async_mod, "resolve_weight_sync_client", lambda backend: _scorer_cls(False))
    host = _Host(0)
    host._init_async_components()
    assert len(host._engine_rescore_clients) == 1

    host.async_config.isr_engine_reference = False
    host._rollout_manager = host._engine_rescore_clients = None
    host._init_async_components()
    assert host._engine_rescore_clients is None, "a run without the re-score built its clients"


def test_a_non_main_rank_weight_sync_failure_raises_on_every_rank(tmp_path):
    """The gather inside the push runs on every rank, so a non-main rank can fail it alone while the main
    process heads for the barrier after the sync. The pause credit stays the main process's push
    duration, on every rank."""
    run_gloo_ranks(_sync_worker, WORLD_SIZE, str(tmp_path), pg_timeout=PG_TIMEOUT)
    _assert_raised_everywhere(_results(tmp_path), "expert gather refused a DTensor", FAILING_RANK)
    credits = [float((tmp_path / f"credit_{rank}.txt").read_text()) for rank in range(WORLD_SIZE)]
    assert credits[0] == credits[1] >= _PUSH_SECONDS, f"the ranks credited different pauses: {credits}"


def test_a_completions_writer_failure_raises_on_every_rank(tmp_path):
    """Only the writer rank (rank 0, shared output filesystem) touches the disk."""
    run_gloo_ranks(_completions_worker, WORLD_SIZE, str(tmp_path), pg_timeout=PG_TIMEOUT)
    _assert_raised_everywhere(_results(tmp_path), "NotADirectoryError", failing_rank=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
