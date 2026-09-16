#!/usr/bin/env python
"""An engine abort is re-issued, never stepped: the fragment is the engine's doing, not the policy's.

SGLang's sync pause aborts every in-flight request, and ``finish_reason: abort`` is an engine-cut
reason — stepping the env with it spends the episode's length-cutoff recovery cap and, past the cap,
truncates the episode, all for a cut the model did not make. The actor re-issues the turn for the
same observation, bounded by ``max_retries`` aborts per turn; past that the episode errors into a
masked row.

    python tests/cpu/environments/test_ray_engine_abort_retry.py
"""

import pytest

from src.configs.rollout_config import RolloutConfig
from src.environments.episode import TurnGeneration
from src.environments.ray_actors import EnvironmentActor
from src.inference.response import FINISH_REASON_ABORT

_OBSERVATION = [{"role": "user", "content": "2+2?"}]


def _actor():
    cls = EnvironmentActor.__ray_metadata__.modified_class
    actor = cls.__new__(cls)
    actor.__init__(actor_id=0, env_type="native_math", env_config={"max_turns": 3})
    return actor


def _scripted(actor, finish_reasons: list[str | None]):
    """Make the actor's engine call answer ``finish_reasons`` in order; ``None`` is a completed turn."""
    served: list[list[dict]] = []

    async def _client(timeout):
        return None

    async def _generate(client, url, messages, config, reasoning_effort=None):
        served.append(messages)
        reason = finish_reasons[len(served) - 1]
        if reason == FINISH_REASON_ABORT:
            return TurnGeneration("partial reas", [], "", 3, finish_reason=reason)
        return TurnGeneration("final answer: 4", [], "", 5, finish_reason=reason)

    actor._get_http_client = _client
    actor._generate = _generate
    return served


async def test_an_aborted_turn_is_re_issued_for_the_same_observation():
    actor = _actor()
    served = _scripted(actor, [FINISH_REASON_ABORT, None])

    result = await actor.run_episode("2+2?", {"answer": "4"}, "http://x", RolloutConfig(max_retries=1))

    assert result.error is None, result.error
    assert served == [_OBSERVATION, _OBSERVATION], "the re-issue must replay the SAME observation"
    assert result.episode_length == 1, "the fragment never reached the env"
    assert result.trajectory.info.get("length_cutoff_turns", 0) == 0, "an abort must not spend a recovery"
    assert result.generation_tokens == 5, "an aborted fragment is not part of the episode's generation"


async def test_aborts_past_max_retries_error_the_episode():
    actor = _actor()
    served = _scripted(actor, [FINISH_REASON_ABORT, FINISH_REASON_ABORT, None])

    result = await actor.run_episode("2+2?", {"answer": "4"}, "http://x", RolloutConfig(max_retries=1))

    assert len(served) == 2, "max_retries bounds the re-issues"
    assert result.error and "aborted the same turn 2 times" in result.error
    assert result.total_reward == 0.0 and not result.success


async def test_max_retries_zero_tolerates_no_abort():
    actor = _actor()
    served = _scripted(actor, [FINISH_REASON_ABORT, None])

    result = await actor.run_episode("2+2?", {"answer": "4"}, "http://x", RolloutConfig(max_retries=0))

    assert len(served) == 1
    assert result.error and "aborted the same turn 1 times" in result.error


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
