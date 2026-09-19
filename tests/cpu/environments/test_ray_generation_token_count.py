#!/usr/bin/env python
"""A completion without ``usage.completion_tokens`` counts its captured token ids, never a word count.

``len(text.split())`` fabricated a length that fed ``episode/generation_tokens``; the captured ids are
the only honest length, and their absence is 0 plus one warning.

    python tests/cpu/environments/test_ray_generation_token_count.py
"""

import logging
from types import SimpleNamespace

import pytest

import src.environments.ray_actors as ray_actors
from src.configs.rollout_config import RolloutConfig
from src.environments.ray_actors import EnvironmentActor


class _Session:
    """An ``aiohttp.ClientSession`` stand-in serving one canned /v1/chat/completions body."""

    def __init__(self, body: dict):
        self._body = body

    def post(self, url, json):  # noqa: A002 — aiohttp's own keyword
        return self

    async def __aenter__(self):
        return SimpleNamespace(status=200, json=self._json, text=self._text)

    async def __aexit__(self, *exc_info):
        return False

    async def _json(self):
        return self._body

    async def _text(self):
        return ""


def _actor():
    cls = EnvironmentActor.__ray_metadata__.modified_class
    actor = cls.__new__(cls)
    actor.__init__(actor_id=0, env_type="native_math", env_config={"max_turns": 3})
    actor._tools_schema = None
    return actor


_NO_USAGE_BODY = {
    "choices": [
        {
            "message": {"content": "four words in here"},
            "finish_reason": "stop",
            "logprobs": {
                "content": [{"token": "token_id:11", "logprob": -0.1}, {"token": "token_id:12", "logprob": -0.2}]
            },
        }
    ]
}


@pytest.fixture
def _rearm_warning():
    ray_actors._COMPLETION_TOKENS_MISSING_WARNED.discard("vllm")
    yield
    ray_actors._COMPLETION_TOKENS_MISSING_WARNED.discard("vllm")


async def test_captured_ids_are_the_length_when_usage_is_missing(_rearm_warning, caplog):
    actor = _actor()
    messages = [{"role": "user", "content": "2+2?"}]
    with caplog.at_level(logging.WARNING, logger=ray_actors.__name__):
        captured = await actor._generate(
            _Session(_NO_USAGE_BODY), "server:8000", messages, RolloutConfig(max_retries=0, capture_token_ids=True)
        )
        uncaptured = await actor._generate(
            _Session(_NO_USAGE_BODY), "server:8000", messages, RolloutConfig(max_retries=0, capture_token_ids=False)
        )

    assert captured.tokens == 2, "two sampled ids were captured; the four-word text is not a length"
    assert uncaptured.tokens == 0, "nothing captured, nothing counted"
    warned = [r for r in caplog.records if "usage.completion_tokens" in r.getMessage()]
    assert len(warned) == 1, "the missing count is announced once per backend, not per turn"


async def test_a_reported_usage_count_wins():
    body = {**_NO_USAGE_BODY, "usage": {"completion_tokens": 7}}
    generation = await _actor()._generate(
        _Session(body), "server:8000", [{"role": "user", "content": "2+2?"}], RolloutConfig(max_retries=0)
    )
    assert generation.tokens == 7


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
