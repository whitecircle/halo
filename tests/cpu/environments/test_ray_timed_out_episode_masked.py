#!/usr/bin/env python
"""A timed-out episode must be an INVALID row, not a valid zero-reward group member.

Both deadlines surface as a bare ``TimeoutError`` — ``asyncio.wait_for`` on the episode deadline,
aiohttp's total timeout on the last retry — and ``str(TimeoutError())`` is ``""``. An error field
built from it reads as no error to ``rollout_valid_mask``, so every timed-out episode entered the
GRPO baseline at reward 0 and an all-timed-out step never tripped the empty-step halt.

    python tests/cpu/environments/test_ray_timed_out_episode_masked.py
"""

import asyncio
from types import SimpleNamespace

import pytest
import torch

import src.environments.ray_actors as ray_actors
from src.configs.rollout_config import RolloutConfig
from src.environments.ray_actors import EnvironmentActor, RolloutManager
from src.trainers.grpo.environmental import rollout_valid_mask

_CPU = torch.device("cpu")


def _actor():
    cls = EnvironmentActor.__ray_metadata__.modified_class
    actor = cls.__new__(cls)
    actor.__init__(actor_id=0, env_type="native_math", env_config={"max_turns": 3})
    return actor


class _NeverReturningActor:
    """Ray-handle stand-in whose episode outlives any deadline."""

    def __init__(self):
        self.run_episode = SimpleNamespace(remote=self._remote)

    def _remote(self, *, prompt, context, server_url, config):
        async def _episode():
            await asyncio.sleep(60.0)

        return _episode()


async def test_a_request_timeout_inside_the_episode_is_a_masked_error():
    actor = _actor()

    async def _client(timeout):
        return None

    async def _timed_out_generate(client, url, messages, config, reasoning_effort=None):
        raise TimeoutError()  # aiohttp's total-timeout expiry: the builtin, with an empty message

    actor._get_http_client = _client
    actor._generate = _timed_out_generate

    result = await actor.run_episode("2+2?", {"answer": "4"}, "http://x", RolloutConfig(max_retries=0))

    assert result.error, "an empty error string is a valid zero-reward row to the trainer"
    assert "TimeoutError" in result.error
    assert result.trajectory.info["error"] == result.error
    assert rollout_valid_mask([result], _CPU).tolist() == [False]


async def test_an_episode_past_its_deadline_is_a_masked_error(monkeypatch):
    # The deadline path cancels the Ray task; a fake handle has no ObjectRef to cancel.
    monkeypatch.setattr(ray_actors.ray, "cancel", lambda ref, force=False: None)
    manager = RolloutManager(
        num_workers=1,
        env_type="native_math",
        env_config={},
        server_urls=["http://x"],
        rollout_config=RolloutConfig(episode_timeout=0.05),
    )
    manager._actors = [_NeverReturningActor()]
    manager._started = True

    (result,) = await manager.collect_rollouts(["p"])

    assert result.error and "TimeoutError" in result.error
    assert rollout_valid_mask([result], _CPU).tolist() == [False]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
