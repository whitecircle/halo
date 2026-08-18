"""Wall-clock deadline for a rollout episode (``_await_with_deadline``).

``request_timeout`` bounds one HTTP call; a wedged tool/sandbox hangs the episode itself. Without an
episode deadline the awaiting rank never reaches the next collective and every peer blocks behind it
(observed: three ranks spinning in ``dist.all_reduce`` while one waited on a rollout forever).
"""

import asyncio
import sys
from dataclasses import fields

import pytest

from src.configs.async_training_config import AsyncTrainingConfig
from src.configs.rollout_config import DEFAULT_EPISODE_TIMEOUT_SECONDS
from src.environments.ray_actors import RolloutConfig, _await_with_deadline


class _SlowRef:
    """Stand-in for a Ray ObjectRef that never resolves in time."""

    def __init__(self, delay: float):
        self._delay = delay

    def __await__(self):
        return asyncio.sleep(self._delay, result="done").__await__()


def test_fast_episode_returns_its_result():
    assert asyncio.run(_await_with_deadline(_SlowRef(0.0), timeout=5.0)) == "done"


def test_wedged_episode_raises_timeout_instead_of_hanging():
    """The regression this guards: an episode that never finishes must not block forever."""
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
    """Both configs must read the ONE shared default, which AsyncTrainingConfig validates against the
    NCCL watchdog. Identity, not equality: a side that re-declares its own literal compares equal
    today and drifts the moment the other moves, letting a straggler outlive the watchdog."""
    async_default = next(f for f in fields(AsyncTrainingConfig) if f.name == "episode_timeout").default
    assert async_default is DEFAULT_EPISODE_TIMEOUT_SECONDS
    assert RolloutConfig().episode_timeout is DEFAULT_EPISODE_TIMEOUT_SECONDS
    assert DEFAULT_EPISODE_TIMEOUT_SECONDS == 1200.0
    assert RolloutConfig(episode_timeout=42.0).episode_timeout == 42.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
