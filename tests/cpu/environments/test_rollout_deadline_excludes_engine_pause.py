#!/usr/bin/env python
"""The episode deadline counts engine-serving time, not the seconds a weight sync held the engines paused.

A sync freezes every in-flight generation (vLLM ``mode=keep``) for the whole push; a wall-clock
deadline charged that window to the episode, and on expiry the retry re-issued a request whose
frozen original still completed server-side. ``RolloutManager`` keeps the paused-seconds clock the
trainer credits after each sync, and ``_await_with_deadline`` re-arms against it.

    python tests/cpu/environments/test_rollout_deadline_excludes_engine_pause.py
"""

import asyncio
from types import SimpleNamespace

import pytest

from src.configs.rollout_config import RolloutConfig
from src.environments import ray_actors
from src.environments.base import Trajectory
from src.environments.episode import RolloutResult
from src.environments.ray_actors import RolloutManager, _await_with_deadline

# The episode overruns the deadline by half a second, and a credit equal to the timeout clears it by
# another half: a stalled event loop on a loaded runner must not flip either verdict.
_TIMEOUT_S = 1.0
_EPISODE_S = 1.5
_OVERRUN_S = _EPISODE_S - _TIMEOUT_S


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
        asyncio.run(_await_with_deadline(_SlowRef(_EPISODE_S), _TIMEOUT_S, _PausedClock(credit=_OVERRUN_S / 5)))


def test_the_episodes_own_timeout_error_is_not_mistaken_for_the_deadline():
    class _FailingRef:
        def __await__(self):
            yield from asyncio.sleep(0).__await__()
            raise TimeoutError("the episode's own")

    with pytest.raises(TimeoutError, match="the episode's own"):
        asyncio.run(_await_with_deadline(_FailingRef(), 5.0, _PausedClock(credit=1.0)))


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
