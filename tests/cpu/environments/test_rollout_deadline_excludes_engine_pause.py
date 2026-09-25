#!/usr/bin/env python
"""The episode and request deadlines count engine-serving time, not the seconds a weight sync held the
engines paused.

A sync freezes every in-flight generation (vLLM ``mode=keep``) for the whole push. A wall-clock
deadline charges that window to the episode, and a request that expires inside it is re-issued while
its frozen original still completes server-side: the same turn is sampled twice. ``RolloutManager``
keeps the paused-seconds clock the trainer credits after each sync, the actors read a Ray copy of it,
and ``_await_with_deadline`` re-arms against it.

    python tests/cpu/environments/test_rollout_deadline_excludes_engine_pause.py
"""

import asyncio
from types import SimpleNamespace

import pytest

from src.configs.rollout_config import RolloutConfig
from src.environments import ray_actors
from src.environments.base import Trajectory
from src.environments.episode import RolloutResult
from src.environments.ray_actors import EnginePauseClock, EnvironmentActor, RolloutManager, _await_with_deadline

_TIMEOUT_S = 0.2
_EPISODE_S = 0.3


class _SlowRef:
    """Stand-in for a Ray ObjectRef that resolves after ``delay`` seconds."""

    def __init__(self, delay: float):
        self._delay = delay

    def __await__(self):
        return asyncio.sleep(self._delay, result="done").__await__()


class _PausedClock:
    """A paused-seconds clock that reads 0 at the episode start and ``credit`` from then on."""

    def __init__(self, credit: float):
        self.credit = credit
        self.reads = 0

    def __call__(self) -> float:
        self.reads += 1
        return 0.0 if self.reads == 1 else self.credit


def test_a_pause_equal_to_the_timeout_does_not_expire_the_episode():
    clock = _PausedClock(credit=_TIMEOUT_S)
    assert asyncio.run(_await_with_deadline(_SlowRef(_EPISODE_S), _TIMEOUT_S, clock)) == "done"
    assert clock.reads >= 2, "the deadline never re-armed against the credit"


def test_without_a_pause_the_same_episode_expires():
    with pytest.raises(TimeoutError, match="deadline"):
        asyncio.run(_await_with_deadline(_SlowRef(_EPISODE_S), _TIMEOUT_S, _PausedClock(credit=0.0)))


def test_a_pause_shorter_than_the_overrun_still_expires():
    with pytest.raises(TimeoutError):
        asyncio.run(_await_with_deadline(_SlowRef(_EPISODE_S), _TIMEOUT_S, _PausedClock(credit=0.02)))


def test_the_episodes_own_timeout_error_is_not_mistaken_for_the_deadline():
    class _FailingRef:
        def __await__(self):
            yield from asyncio.sleep(0).__await__()
            raise TimeoutError("the episode's own")

    with pytest.raises(TimeoutError, match="the episode's own"):
        asyncio.run(_await_with_deadline(_FailingRef(), 5.0, _PausedClock(credit=1.0)))


def test_a_result_that_lands_while_the_clock_is_read_is_returned_not_expired():
    """The deadline runs out and the work finishes during the clock read that settles the expiry: the
    finished result stands rather than being dropped for an expiry that no longer holds."""

    async def scenario():
        landed = asyncio.Event()

        async def work():
            await landed.wait()
            return "done"

        async def a_read_during_which_the_work_lands():
            landed.set()
            await asyncio.sleep(0)
            return 0.0

        return await _await_with_deadline(work(), 0.01, a_read_during_which_the_work_lands)

    assert asyncio.run(scenario()) == "done"


def _manager(episode_timeout: float) -> RolloutManager:
    return RolloutManager(
        num_workers=1,
        env_type="native_math",
        env_config={},
        server_urls=["http://x"],
        rollout_config=RolloutConfig(episode_timeout=episode_timeout),
    )


def test_manager_pause_window_counts_while_open_and_takes_the_measured_figure_when_closed(monkeypatch):
    manager = _manager(1.0)
    assert manager.paused_seconds == 0.0
    ticks = iter([100.0, 103.0])
    # The module's ``time`` name, not the stdlib clock: only the pause window reads it, and an
    # unexpected read exhausts the ticks instead of passing on a real elapsed time.
    monkeypatch.setattr(ray_actors, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    manager.begin_engine_pause()
    assert manager.paused_seconds == 3.0, "an open window counts while it is open"
    manager.end_engine_pause(5.0)
    assert manager.paused_seconds == 5.0, "the forwarding rank's measured push replaces the open window"
    manager.end_engine_pause(2.0)
    assert manager.paused_seconds == 7.0, "credits accumulate across syncs"


async def test_an_episode_spanning_an_open_pause_is_not_cancelled():
    manager = _manager(_TIMEOUT_S)

    def _remote(*, prompt, context, server_url, config):
        async def _episode():
            await asyncio.sleep(_EPISODE_S)
            return RolloutResult(prompt=prompt, trajectory=Trajectory(done=True), success=True)

        return _episode()

    manager._actors = [SimpleNamespace(run_episode=SimpleNamespace(remote=_remote))]
    manager._started = True

    manager.begin_engine_pause()
    (result,) = await manager.collect_rollouts(["p"])
    manager.end_engine_pause(_EPISODE_S)

    assert result.error is None, result.error
    assert result.success


class _FrozenSession:
    """An ``aiohttp.ClientSession`` stand-in whose every request answers after ``delay`` seconds; the
    first request starts ``on_first_request``, so a scripted sync is timed from the request, not the
    episode's reset."""

    def __init__(self, delay: float, on_first_request):
        self.delay = delay
        self.posts = 0
        self._on_first_request = on_first_request

    def post(self, url, json):  # noqa: A002 — aiohttp's own keyword
        self.posts += 1
        if self.posts == 1:
            self._on_first_request()
        return self

    async def __aenter__(self):
        await asyncio.sleep(self.delay)
        body = {"choices": [{"message": {"content": "final answer: 4"}, "finish_reason": "stop"}], "usage": {}}

        async def _json():
            return body

        return SimpleNamespace(status=200, json=_json)

    async def __aexit__(self, *exc_info):
        return False


def _ray_handle(clock: EnginePauseClock):
    """A Ray actor-handle stand-in: ``paused_seconds.remote()`` is an awaitable of the value at the call."""

    def _remote():
        future = asyncio.get_running_loop().create_future()
        future.set_result(clock.paused_seconds())
        return future

    return SimpleNamespace(paused_seconds=SimpleNamespace(remote=_remote))


async def _episode_across_a_pause(
    *, request_delay: float, request_timeout: float, pause_from: float, pause_s: float, max_retries: int
):
    """One real actor episode against a server answering after ``request_delay``, while a sync holds the
    engines paused from ``pause_from`` after the first request for ``pause_s``; returns the result and
    the requests issued."""
    clock = EnginePauseClock()
    cls = EnvironmentActor.__ray_metadata__.modified_class
    actor = cls.__new__(cls)
    actor.__init__(actor_id=0, env_type="native_math", env_config={"max_turns": 3}, pause_clock=_ray_handle(clock))
    syncs: list[asyncio.Task] = []

    async def _sync():
        await asyncio.sleep(pause_from)
        clock.begin()
        await asyncio.sleep(pause_s)
        clock.end(pause_s)

    session = _FrozenSession(request_delay, lambda: syncs.append(asyncio.create_task(_sync())))

    async def _client():
        return session

    actor._get_http_client = _client
    config = RolloutConfig(request_timeout=request_timeout, max_retries=max_retries, retry_base_wait=0.0)
    result = await actor.run_episode("2+2?", {"answer": "4"}, "http://x", config)
    for sync in syncs:
        sync.cancel()
    return result, session.posts


async def test_a_request_frozen_across_a_sync_is_sampled_once():
    # Wall clock: the 1.0 s request overruns its 0.5 s deadline. Serving time is 0.2 s — the engine sat
    # paused for the other 0.8 s, and without the credit the turn is re-issued and sampled twice.
    result, posts = await _episode_across_a_pause(
        request_delay=1.0, request_timeout=0.5, pause_from=0.1, pause_s=0.8, max_retries=2
    )

    assert result.error is None, result.error
    assert posts == 1, "the frozen request expired and its turn was generated again"
    assert result.episode_length == 1
    assert result.requests_expired_in_sync == 0


async def test_a_request_that_overruns_its_credit_is_counted_as_expired_in_sync():
    # 0.2 s of pause stretches the 0.3 s deadline to 0.5 s; a request that never answers still expires,
    # in flight across the sync. Its retry sees no pause and expires uncounted.
    result, posts = await _episode_across_a_pause(
        request_delay=30.0, request_timeout=0.3, pause_from=0.05, pause_s=0.2, max_retries=1
    )

    assert posts == 2
    assert result.error and "DeadlineExpired: request exceeded" in result.error
    assert result.requests_expired_in_sync == 1


async def test_a_last_try_that_expires_in_sync_is_counted_too():
    # No retry left: the expiry ends the episode, and the turn the engine had started is still lost.
    result, posts = await _episode_across_a_pause(
        request_delay=30.0, request_timeout=0.3, pause_from=0.05, pause_s=0.2, max_retries=0
    )

    assert posts == 1
    assert result.error and "DeadlineExpired: request exceeded" in result.error
    assert result.requests_expired_in_sync == 1


async def test_a_request_outside_any_sync_expires_uncounted():
    result, posts = await _episode_across_a_pause(
        request_delay=30.0, request_timeout=0.2, pause_from=30.0, pause_s=0.0, max_retries=0
    )

    assert posts == 1
    assert result.error and "DeadlineExpired: request exceeded" in result.error
    assert result.requests_expired_in_sync == 0


def test_the_manager_feeds_every_pause_window_to_the_actors_clock():
    manager = _manager(1.0)
    calls: list[tuple] = []
    manager._actor_pause_clock = SimpleNamespace(
        begin=SimpleNamespace(remote=lambda: calls.append(("begin",))),
        end=SimpleNamespace(remote=lambda seconds: calls.append(("end", seconds))),
    )

    manager.begin_engine_pause()
    manager.end_engine_pause(2.5)

    assert calls == [("begin",), ("end", 2.5)], "the actors' requests would never see this sync's pause"
    assert manager.paused_seconds == 2.5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
