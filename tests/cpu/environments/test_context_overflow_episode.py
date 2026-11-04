#!/usr/bin/env python
"""A turn the served window refuses is the one lost turn an episode's own length causes. Errored out, the longest failures
never carried their negative signal, while the eval path graded the same episode on what it earned. SGLang refuses any
prompt plus ``max_tokens`` past the window, so a shipped 131k window at 30k per turn hits it ~101k in."""

import pytest

from halo.environments import ray_actors
from halo.environments.episode_keys import CONTEXT_OVERFLOW_KEY, EPISODE_ERROR_KEY
from halo.environments.rollout_config import RolloutConfig
from tests.cpu.environments.test_effort_binding import _actor_text_turn, _actor_tool_turn, _PlainEnv

_OVERFLOW = ray_actors.RolloutHTTPError(
    400, "sglang", "The input (101000 tokens) is longer than the model's context length (131072 tokens)."
)
_MALFORMED = ray_actors.RolloutHTTPError(400, "vllm", '"auto" tool choice requires --enable-auto-tool-choice')


async def _run(turns: list) -> ray_actors.RolloutResult:
    cls = ray_actors.EnvironmentActor.__ray_metadata__.modified_class
    actor = cls.__new__(cls)
    actor.__init__(actor_id=0, env_type=(_PlainEnv, {}), env_config={})
    served = iter(turns)

    async def fake_generate(
        client, url, messages, config, reasoning_effort=None, reasoning_budget=None, budget_soft=False
    ):
        turn = next(served)
        if isinstance(turn, Exception):
            raise turn
        return turn

    actor._http_client = None
    actor._generate = fake_generate
    return await actor.run_episode(
        "solve it", {}, "http://x", RolloutConfig(model_name="m", max_tokens=16, max_retries=0)
    )


async def test_a_turn_the_window_refuses_keeps_the_row_and_the_reward():
    result = await _run([_actor_tool_turn(), _OVERFLOW])
    assert result.error is None and result.counts_toward_baseline
    assert result.trajectory.truncated and result.episode_length == 1
    assert result.trajectory.info[CONTEXT_OVERFLOW_KEY] is True
    assert EPISODE_ERROR_KEY in result.trajectory.info, "a lost driver's stamp keeps the turn-overflow price off it"
    assert result.metrics["episode/context_overflow"] == 1.0


async def test_any_other_refusal_still_loses_the_episode():
    """A malformed request is no fact about the policy; its reward would train on an engine's refusal."""
    result = await _run([_actor_tool_turn(), _MALFORMED])
    assert result.error is not None and not result.counts_toward_baseline


async def test_a_clean_episode_reads_zero():
    result = await _run([_actor_tool_turn(), _actor_text_turn()])
    assert result.metrics["episode/context_overflow"] == 0.0
    assert CONTEXT_OVERFLOW_KEY not in result.trajectory.info


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
