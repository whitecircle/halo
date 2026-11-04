"""``_await_with_deadline``: ``request_timeout`` bounds one HTTP call, while a wedged tool or sandbox hangs the
episode, the awaiting rank never reaches the next collective and every peer blocks behind it."""

import asyncio
from dataclasses import fields

import pytest

from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.environments.ray_actors import _await_with_deadline
from halo.environments.rollout_config import RolloutConfig


class _SlowRef:
    """Stand-in for a Ray ObjectRef that never resolves in time."""

    def __init__(self, delay: float):
        self._delay = delay

    def __await__(self):
        return asyncio.sleep(self._delay, result="done").__await__()


def test_fast_episode_returns_its_result():
    assert asyncio.run(_await_with_deadline(_SlowRef(0.0), timeout=5.0)) == "done"


def test_wedged_episode_raises_timeout_instead_of_hanging():
    with pytest.raises(TimeoutError):
        asyncio.run(_await_with_deadline(_SlowRef(30.0), timeout=0.05))


def test_deadline_is_enforced_promptly():
    async def _timed():
        loop = asyncio.get_running_loop()
        start = loop.time()
        with pytest.raises(TimeoutError):
            await _await_with_deadline(_SlowRef(30.0), timeout=0.05)
        return loop.time() - start

    assert asyncio.run(_timed()) < 5.0  # bounded by the deadline, not by the episode


def test_episode_timeout_is_configurable_and_defaults_sanely():
    """Identity: a side re-declaring its own literal compares equal until the other moves, letting a straggler outlive
    the NCCL watchdog."""
    async_default = next(f for f in fields(AsyncRolloutConfig) if f.name == "episode_timeout").default
    assert async_default is RolloutConfig.episode_timeout
    assert RolloutConfig().episode_timeout == 1200.0
    assert RolloutConfig(episode_timeout=42.0).episode_timeout == 42.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
