#!/usr/bin/env python
"""CPU tests: the generative judge — request shape, grading prompt, verdict parsing, scoring and the
failure semantics (a failed or unparseable verdict is a ``None`` score, never an exception).

Run: python tests/cpu/rewards/test_judge.py  (or pytest)
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from src.rewards import judge as judge_module
from src.rewards.judge import GenerativeJudge, grading_prompt, judge_api_key, parse_verdict
from src.rewards.samples import ScoringSample
from src.rewards.spec import JudgeTerm, Requirement

REQUIREMENTS = (
    Requirement(name="correctness", description="The final answer is correct."),
    Requirement(name="clarity", description="The explanation is easy to follow.", weight=0.5),
)
SAMPLE = ScoringSample(
    prompt=[{"role": "system", "content": "Be terse."}, {"role": "user", "content": "What is 2+2?"}],
    completion=[{"role": "assistant", "content": "It is 4."}],
    reference="4",
)
PERFECT_VERDICT = json.dumps({"scores": {"correctness": 10, "clarity": 10}, "rationale": ""})


def _term(**overrides) -> JudgeTerm:
    return JudgeTerm(name="quality", requirements=REQUIREMENTS, **overrides)


def _reply(content, *, tokens=12, cost=0.0003):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(completion_tokens=tokens, cost=cost),
    )


class _FakeClient:
    """Stands in for ``AsyncOpenAI``: records every request, answers from a script."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def close(self):
        self.closed = True


class _Yield:
    """One loop tick. Not ``asyncio.sleep``: two tests here monkeypatch the module's ``sleep``."""

    def __await__(self):
        yield


class _OverlappingClient(_FakeClient):
    """Suspends inside every request and records how many were in flight at that moment."""

    def __init__(self, replies):
        super().__init__(replies)
        self.inflight: list[int] = []
        self._open = 0

    async def _create(self, **kwargs):
        self._open += 1
        self.inflight.append(self._open)
        try:
            await _Yield()
            return await super()._create(**kwargs)
        finally:
            self._open -= 1


def _judge(term, replies) -> tuple[GenerativeJudge, _FakeClient]:
    judge = GenerativeJudge(term)
    client = _FakeClient(replies)
    judge._create_client = lambda: client
    return judge, client


def _overlapping_judge(term, count: int) -> tuple[GenerativeJudge, _OverlappingClient]:
    judge = GenerativeJudge(term)
    client = _OverlappingClient([_reply(PERFECT_VERDICT) for _ in range(count)])
    judge._create_client = lambda: client
    return judge, client


def test_request_carries_the_term_shape():
    judge, client = _judge(
        _term(), [_reply(json.dumps({"scores": {"correctness": 10, "clarity": 10}, "rationale": "ok"}))]
    )
    asyncio.run(judge.score([SAMPLE]))
    (request,) = client.requests
    assert request["model"] == "openai/gpt-5.6-luna"
    assert request["reasoning_effort"] == "medium"
    assert request["max_completion_tokens"] == 8192
    assert request["timeout"] == 120.0
    assert "temperature" not in request
    schema = request["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert set(schema["schema"]["properties"]["scores"]["properties"]) == {"correctness", "clarity"}
    assert schema["schema"]["properties"]["scores"]["additionalProperties"] is False
    assert [message["role"] for message in request["messages"]] == ["system", "user"]


def test_optional_request_fields_follow_the_term():
    term = _term(reasoning_effort=None, temperature=0.0, structured_output=False, max_tokens=999, request_timeout=7.5)
    judge, client = _judge(term, [_reply('{"scores": {"correctness": 5, "clarity": 5}, "rationale": ""}')])
    asyncio.run(judge.score([SAMPLE]))
    (request,) = client.requests
    assert "reasoning_effort" not in request and "response_format" not in request
    assert request["temperature"] == 0.0 and request["max_completion_tokens"] == 999 and request["timeout"] == 7.5


def test_grading_prompt_shows_task_reference_response_and_rubric():
    prompt = grading_prompt(_term(instructions="Ignore formatting."), SAMPLE)
    assert "# Task\nWhat is 2+2?" in prompt
    assert "Be terse." not in prompt  # the system prompt is the policy's steer, not the task
    assert "# Reference answer\n4" in prompt
    assert "# Response\nIt is 4." in prompt
    assert "1. correctness: The final answer is correct." in prompt
    assert "2. clarity: The explanation is easy to follow." in prompt
    assert "from 0 (not met) to 10 (fully met)" in prompt
    assert "# Grading instructions\nIgnore formatting." in prompt
    assert '"correctness": <integer>, "clarity": <integer>' in prompt


def test_reference_is_withheld_when_disabled_and_absent():
    assert "Reference" not in grading_prompt(_term(include_reference=False), SAMPLE)
    unreferenced = ScoringSample(prompt=SAMPLE.prompt, completion=SAMPLE.completion)
    assert "Reference" not in grading_prompt(_term(), unreferenced)


def test_full_transcript_renders_every_turn_with_tool_calls():
    sample = ScoringSample(
        prompt=[{"role": "user", "content": "Run it."}],
        completion=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "run", "arguments": '{"code": "print(1)"}'}}],
            },
            {"role": "tool", "name": "run", "content": "1"},
            {"role": "assistant", "content": "Done."},
        ],
    )
    final = grading_prompt(_term(), sample)
    assert "# Response\nDone." in final and "print(1)" not in final
    full = grading_prompt(_term(transcript="full"), sample)
    assert '-> run({"code": "print(1)"})' in full and "[tool:run]\n1" in full and "Done." in full


def test_long_response_is_cut_with_a_marker():
    sample = ScoringSample(prompt=SAMPLE.prompt, completion=[{"role": "assistant", "content": "x" * 100}])
    prompt = grading_prompt(_term(max_transcript_chars=40), sample)
    assert "x" * 40 + "\n…[truncated 60 chars]" in prompt and "x" * 41 not in prompt


def test_score_is_the_weighted_fraction_with_diagnostics():
    reply = _reply(
        json.dumps({"scores": {"correctness": 8, "clarity": 10}, "rationale": "Right and clear."}),
        tokens=33,
        cost=0.002,
    )
    judge, _ = _judge(_term(), [reply])
    (result,) = asyncio.run(judge.score([SAMPLE]))
    assert result.score == pytest.approx((8 * 1.0 + 10 * 0.5) / (10 * 1.5))
    assert result.metrics == {
        "judge/quality/correctness": 0.8,
        "judge/quality/clarity": 1.0,
        "judge/quality/completion_tokens": 33.0,
        "judge/quality/cost_usd": 0.002,
    }
    assert result.detail == "Right and clear." and result.error is None


def test_out_of_scale_requirement_scores_are_clamped_in_the_diagnostics():
    """A judge that answers outside ``[0, scale]`` must not log 1.2 for a 12/10: the diagnostic is
    clamped like the score, so the W&B panel and the reward agree."""
    reply = _reply(json.dumps({"scores": {"correctness": 12, "clarity": -3}, "rationale": "off scale"}))
    judge, _ = _judge(_term(), [reply])
    (result,) = asyncio.run(judge.score([SAMPLE]))
    assert result.metrics["judge/quality/correctness"] == 1.0
    assert result.metrics["judge/quality/clarity"] == 0.0
    assert result.score == pytest.approx(10 / 15)


@pytest.mark.parametrize(
    "content",
    [
        "I would rate this highly.",
        '{"scores": {"correctness": 8}}',
        '{"scores": {"correctness": "eight", "clarity": 10}}',
        '{"scores": {"correctness": true, "clarity": 10}}',
        "",
    ],
)
def test_unparseable_verdicts_score_none_with_the_reason(content):
    judge, _ = _judge(_term(), [_reply(content)])
    (result,) = asyncio.run(judge.score([SAMPLE]))
    assert result.score is None and result.error.startswith("unparseable judge reply")


def test_verdict_embedded_in_prose_and_numeric_strings_parse():
    content = 'Verdict follows.\n{"scores": {"correctness": "9", "clarity": 7.5}, "rationale": "fine"}\nBye.'
    assert parse_verdict(content, _term()) == ({"correctness": 9.0, "clarity": 7.5}, "fine")


def test_an_upstream_rate_limit_in_a_200_reply_is_retried(monkeypatch):
    waits = []

    async def no_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(judge_module.asyncio, "sleep", no_sleep)
    limited = SimpleNamespace(
        choices=[], error={"code": 429, "message": "temporarily rate-limited upstream"}, usage=None
    )
    good = _reply(json.dumps({"scores": {"correctness": 10, "clarity": 10}, "rationale": "ok"}))
    judge, client = _judge(_term(), [limited, limited, good])
    (result,) = asyncio.run(judge.score([SAMPLE]))
    assert result.score == 1.0 and len(client.requests) == 3
    assert waits == [2.0, 4.0]  # exponential backoff between the retried attempts


def test_a_reply_without_choices_scores_none_when_not_retryable_or_exhausted(monkeypatch):
    async def no_sleep(seconds):
        pass

    monkeypatch.setattr(judge_module.asyncio, "sleep", no_sleep)
    judge, client = _judge(_term(), [SimpleNamespace(choices=[], error={"code": 400, "message": "bad"}, usage=None)])
    (result,) = asyncio.run(judge.score([SAMPLE]))
    assert (
        result.score is None and "no choices" in result.error and "400" in result.error and len(client.requests) == 1
    )
    exhausted = [SimpleNamespace(choices=[], error={"code": 503}, usage=None)] * 5
    judge, client = _judge(_term(), exhausted)
    (result,) = asyncio.run(judge.score([SAMPLE]))
    assert result.score is None and "503" in result.error and len(client.requests) == 5


def test_unparseable_reply_names_the_finish_reason():
    reply = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=""), finish_reason="length")], usage=None
    )
    judge, _ = _judge(_term(), [reply])
    (result,) = asyncio.run(judge.score([SAMPLE]))
    assert result.score is None and "finish_reason='length'" in result.error


def test_metric_keys_are_fixed_per_term():
    assert GenerativeJudge(_term()).metric_keys == (
        "judge/quality/correctness",
        "judge/quality/clarity",
        "judge/quality/completion_tokens",
    )


def test_request_failure_scores_none_without_raising():
    judge, _ = _judge(_term(), [RuntimeError("503 upstream")])
    (result,) = asyncio.run(judge.score([SAMPLE]))
    assert result.score is None and result.error == "request failed: RuntimeError: 503 upstream"


def test_samples_score_concurrently_in_order():
    replies = [_reply(json.dumps({"scores": {"correctness": s, "clarity": s}, "rationale": ""})) for s in (10, 0, 5)]
    judge, client = _judge(_term(max_concurrency=2), replies)
    results = asyncio.run(judge.score([SAMPLE, SAMPLE, SAMPLE]))
    assert [round(r.score, 2) for r in results] == [1.0, 0.0, 0.5]
    assert len(client.requests) == 3


def test_the_cap_bounds_the_requests_in_flight():
    judge, client = _overlapping_judge(_term(max_concurrency=2), count=6)
    results = asyncio.run(judge.score([SAMPLE] * 6))
    assert [r.score for r in results] == [1.0] * 6
    assert max(client.inflight) == 2, "max_concurrency bounds the judge calls in flight"


def test_one_judge_reused_on_a_second_loop_rebinds_its_semaphore():
    """A semaphore binds to the loop it first blocks on: the launch probe and the Ray actor run on
    different loops, so a judge that kept the first one raises ``bound to a different event loop``."""
    judge, client = _overlapping_judge(_term(max_concurrency=1), count=4)
    first = asyncio.run(judge.score([SAMPLE, SAMPLE]))
    second = asyncio.run(judge.score([SAMPLE, SAMPLE]))
    assert [r.score for r in first + second] == [1.0] * 4
    assert max(client.inflight) == 1, "both runs must contend, or the rebinding branch is never reached"


def test_verify_probes_through_a_fresh_client_and_closes_it():
    judge, client = _judge(
        _term(), [_reply(json.dumps({"scores": {"correctness": 10, "clarity": 10}, "rationale": ""}))]
    )
    asyncio.run(judge.verify())
    assert client.closed and judge._client is None
    assert "ready" in client.requests[0]["messages"][1]["content"]


def test_verify_raises_on_a_failed_or_unparseable_probe():
    judge, client = _judge(_term(), [_reply("no json here")])
    with pytest.raises(RuntimeError, match="probe failed: unparseable"):
        asyncio.run(judge.verify())
    assert client.closed
    judge, _ = _judge(_term(), [ConnectionError("refused")])
    with pytest.raises(RuntimeError, match="request failed"):
        asyncio.run(judge.verify())


def test_api_key_comes_from_the_term_variable_then_the_hosted_chain(monkeypatch):
    for name in ("OPENAI_API_KEY_MY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY_MY"):
        judge_api_key(_term(api_key_env="OPENAI_API_KEY_MY"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-chain")
    assert judge_api_key(_term(api_key_env="OPENAI_API_KEY_MY")) == "sk-or-chain"
    monkeypatch.setenv("OPENAI_API_KEY_MY", "sk-or-mine")
    assert judge_api_key(_term(api_key_env="OPENAI_API_KEY_MY")) == "sk-or-mine"


def test_client_is_built_against_the_term_endpoint(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    captured = {}

    def fake_create(base_url, api_key_override):
        captured.update(base_url=base_url, api_key_override=api_key_override)
        return _FakeClient([])

    monkeypatch.setattr(judge_module, "create_openai_client", fake_create)
    GenerativeJudge(_term(base_url="http://judge:8000/v1"))._create_client()
    assert captured == {"base_url": "http://judge:8000/v1", "api_key_override": "sk-or-test"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
