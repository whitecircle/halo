#!/usr/bin/env python
"""The White Circle scorer against a fake ``POST /api/events``: request shape, batching, concurrency, retries, the
verdict-to-reward mapping, the online term and the offline precompute."""

import asyncio
import functools
import json
import logging

import httpx
import pytest

from halo.entrypoints.offline_reward_scoring import build_reward_functions, score_rows
from halo.rewards.composer import build_scorer
from halo.rewards.graders.verifiable import RLVR_REWARD_SOURCES
from halo.rewards.samples import ScoringSample
from halo.rewards.scorers import whitecircle
from halo.rewards.scorers.whitecircle import WhiteCircleScorer
from halo.rewards.terms import ENVIRONMENT_REWARD_SOURCES, WhiteCircleTerm, parse_reward_terms, scored_metric_key
from halo.rewards.trl_reward_functions import ScorerRewardFunction

KEY = "wc-test-secret-key"
TERM = WhiteCircleTerm(name="policy", environment_id="env_test", retry_wait=0.0)


def _sample(answer: str = "4", *, prompt: list[dict] | None = None) -> ScoringSample:
    return ScoringSample(
        prompt=prompt or [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "2+2?"}],
        completion=[{"role": "assistant", "content": answer}],
        final_answer=answer,
    )


def _result(event: dict) -> dict:
    return {
        "type": event["type"],
        "status": "completed",
        "flagged": False,
        "policies": {},
        "event_id": event["event_id"],
    }


def _payload(results: list[dict]) -> dict:
    kinds = {"message": "messages", "tool": "tools"}
    body = {kind: [] for kind in ("messages", "artifacts", "tools", "functions", "agents", "reasoning")}
    for result in results:
        body[kinds[result["type"]]].append(result)
    return {"flagged": any(r.get("flagged") for r in results), "run_id": "run", **body}


def _verdicts(verdict=lambda event: {}):
    """A handler answering every event completed; ``verdict(event)`` overrides fields of its result."""

    def handler(request: httpx.Request) -> httpx.Response:
        events = json.loads(request.content)["events"]
        return httpx.Response(200, json=_payload([{**_result(e), **verdict(e)} for e in events]))

    return handler


@pytest.fixture
def server(monkeypatch):
    """Routes every client the scorer builds to ``handler``, recording each request."""
    monkeypatch.setenv(whitecircle.API_KEY_VARIABLE, KEY)
    requests: list[httpx.Request] = []
    state = {"handler": _verdicts()}

    async def recording(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        response = state["handler"](request)
        return await response if asyncio.iscoroutine(response) else response

    monkeypatch.setattr(
        httpx, "AsyncClient", functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(recording))
    )

    def serve(handler=None):
        if handler is not None:
            state["handler"] = handler
        return requests

    return serve


def _score(term: WhiteCircleTerm, samples: list[ScoringSample]):
    scorer = WhiteCircleScorer(term)

    async def run():
        try:
            return await scorer.score(samples)
        finally:
            await scorer.aclose()

    return asyncio.run(run())


def test_request_shape_us_default_and_eu_override(server):
    requests = server()
    [result] = _score(TERM, [_sample()])
    assert result.score == 1.0
    [request] = requests
    assert str(request.url) == "https://us.whitecircle.com/api/events"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert request.headers["whitecircle-version"] == "2026-06-01"
    body = json.loads(request.content)
    assert body["environment_id"] == "env_test" and body["external_session_id"]
    assert [(e["type"], e["role"], e["content"]) for e in body["events"]] == [
        ("message", "system", "Be brief."),
        ("message", "user", "2+2?"),
        ("message", "assistant", "4"),
    ]
    assert len({e["event_id"] for e in body["events"]}) == 3

    _score(WhiteCircleTerm(name="policy", environment_id="env_test", region="eu"), [_sample()])
    assert requests[-1].url.host == "eu.whitecircle.com"


def test_each_sample_is_its_own_conversation(server):
    requests = server()
    _score(TERM, [_sample("4"), _sample("5")])
    bodies = [json.loads(r.content) for r in requests]
    assert sorted(b["events"][-1]["content"] for b in bodies) == ["4", "5"]
    assert all(len(b["events"]) == 3 for b in bodies)
    assert bodies[0]["external_session_id"] != bodies[1]["external_session_id"]


def test_a_long_conversation_goes_in_requests_of_fifty_under_one_session(server):
    # The verdict sits in the second request: reading only the first would find no result for the completion.
    requests = server(_verdicts(lambda event: {"flagged": event.get("role") == "assistant"}))
    prompt = [{"role": "user", "content": f"turn {i}"} for i in range(60)]
    [result] = _score(TERM, [_sample(prompt=prompt)])
    bodies = [json.loads(r.content) for r in requests]
    assert [len(b["events"]) for b in bodies] == [50, 11]
    assert bodies[0]["external_session_id"] == bodies[1]["external_session_id"]
    assert bodies[1]["events"][-1]["content"] == "4"
    assert result.score == 0.0


def test_concurrency_cap(server):
    in_flight, peak = 0, 0

    async def slow(request):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return _verdicts()(request)

    requests = server(slow)
    term = WhiteCircleTerm(name="policy", environment_id="env_test", max_concurrency=2)
    results = _score(term, [_sample(str(i)) for i in range(6)])
    assert len(requests) == 6 and all(r.score == 1.0 for r in results)
    assert peak == 2


def _sequence(*responses):
    """Answers each request with the next response; past the list, a completed verdict."""
    queue = list(responses)

    def handler(request):
        return queue.pop(0) if queue else _verdicts()(request)

    return handler


@pytest.fixture
def waits(monkeypatch):
    slept: list[float] = []

    async def record(seconds):
        slept.append(seconds)

    monkeypatch.setattr(whitecircle.asyncio, "sleep", record)
    return slept


def test_retry_honors_retry_after_and_backs_off(server, waits):
    requests = server(
        _sequence(
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(503),
            httpx.Response(429, headers={"Retry-After": "600"}),
        )
    )
    term = WhiteCircleTerm(name="policy", environment_id="env_test", retry_wait=0.5)
    [result] = _score(term, [_sample()])
    assert result.score == 1.0 and len(requests) == 4
    # Retry-After as sent, the second attempt's exponential step, Retry-After capped.
    assert waits == [7.0, 1.0, whitecircle.MAX_RETRY_AFTER_S]


def test_retries_exhaust_into_no_verdict(server, waits):
    requests = server(lambda request: httpx.Response(500))
    term = WhiteCircleTerm(name="policy", environment_id="env_test", max_retries=2, retry_wait=1.0)
    [result] = _score(term, [_sample()])
    assert result.score is None and "500" in result.error
    assert len(requests) == 3 and waits == [1.0, 2.0]


@pytest.mark.parametrize("status", [400, 401, 403])
def test_client_errors_are_not_retried(server, waits, status):
    requests = server(lambda request: httpx.Response(status, json={"error": "denied"}))
    [result] = _score(TERM, [_sample()])
    assert result.score is None and str(status) in result.error and "denied" in result.error
    assert len(requests) == 1 and waits == []


def test_timeout_and_transport_errors_retry(server, waits):
    async def hang(request):
        await asyncio.Event().wait()

    requests = server(hang)
    term = WhiteCircleTerm(name="policy", environment_id="env_test", request_timeout=0.05, max_retries=1)
    [result] = _score(term, [_sample()])
    assert result.score is None and "TimeoutError" in result.error and len(requests) == 2

    def refuse_once(request):
        if len(requests) == 3:
            raise httpx.ConnectError("refused", request=request)
        return _verdicts()(request)

    server(refuse_once)
    [result] = _score(term, [_sample()])
    assert result.score == 1.0 and len(requests) == 4


def _assistant(verdict):
    return _verdicts(lambda event: verdict if event.get("role") == "assistant" else {})


def test_flagged_completion_scores_zero_and_a_flagged_prompt_does_not_count(server):
    server(_assistant({"flagged": True, "policies": {"policy_a": {"flagged": True, "name": "Policy A"}}}))
    [result] = _score(TERM, [_sample()])
    assert result.score == 0.0 and result.metrics == {"whitecircle/policy/flagged": 1.0}
    assert result.detail == "flagged: Policy A"

    server(_verdicts(lambda event: {"flagged": event.get("role") == "user"}))
    [result] = _score(TERM, [_sample()])
    assert result.score == 1.0


def test_listed_policies_score_the_fraction_not_violated(server):
    policies = {
        "policy_a": {"flagged": True, "name": "A"},
        "policy_b": {"flagged": False, "name": "B"},
        "policy_c": {"flagged": True, "name": "C"},
    }
    server(_assistant({"flagged": True, "policies": policies}))
    term = WhiteCircleTerm(name="policy", environment_id="env_test", policies=["policy_a", "policy_b"])
    [result] = _score(term, [_sample()])
    # policy_c is flagged but not listed, so it costs nothing.
    assert result.score == 0.5
    assert result.metrics == {
        "whitecircle/policy/flagged": 1.0,
        "whitecircle/policy/policy_a": 1.0,
        "whitecircle/policy/policy_b": 0.0,
    }

    server(_assistant({"policies": {"policy_b": {"flagged": False, "name": "B"}}}))
    [result] = _score(term, [_sample()])
    assert result.score is None and "policy_a" in result.error


@pytest.mark.parametrize(
    ("verdict", "reason"),
    [
        ({"flagged": None, "note": "No policies or metrics were selected for this Event"}, "No policies"),
        ({"status": "failed", "flagged": None, "note": "validation failed"}, "validation failed"),
    ],
)
def test_no_verdict_is_a_failed_check(server, verdict, reason):
    server(_assistant(verdict))
    [result] = _score(TERM, [_sample()])
    assert result.score is None and reason in result.error


def test_full_view_sends_calls_and_results_and_scores_the_calls(server):
    requests = server(_verdicts(lambda event: {"flagged": "arguments" in event}))
    sample = ScoringSample(
        prompt=[{"role": "user", "content": "Email Bob."}],
        completion=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "send", "arguments": '{"to": "bob"}'}}],
            },
            {"role": "tool", "name": "send", "tool_call_id": "c1", "content": "sent"},
            {"role": "assistant", "content": "Done."},
        ],
        final_answer="Done.",
    )
    [final] = _score(TERM, [sample])
    assert final.score == 1.0
    [full] = _score(WhiteCircleTerm(name="policy", environment_id="env_test", view="full"), [sample])
    assert full.score == 0.0
    events = json.loads(requests[-1].content)["events"]
    assert [{k: v for k, v in e.items() if k != "event_id"} for e in events] == [
        {"type": "message", "role": "user", "content": "Email Bob."},
        {"type": "tool", "name": "send", "call_id": "c1", "arguments": {"to": "bob"}},
        {"type": "tool", "name": "send", "call_id": "c1", "output": "sent"},
        {"type": "message", "role": "assistant", "content": "Done."},
    ]


def test_an_empty_completion_scores_one_without_a_request(server):
    requests = server()
    [result] = _score(TERM, [_sample("")])
    assert result.score == 1.0 and requests == []


def test_missing_key_or_environment_fails_at_construction(monkeypatch):
    monkeypatch.delenv(whitecircle.API_KEY_VARIABLE, raising=False)
    with pytest.raises(RuntimeError, match=whitecircle.API_KEY_VARIABLE):
        WhiteCircleScorer(TERM)
    with pytest.raises(ValueError, match="environment_id"):
        parse_reward_terms([{"source": "whitecircle", "name": "policy"}], RLVR_REWARD_SOURCES)
    with pytest.raises(ValueError, match="environment_id"):
        WhiteCircleTerm(name="policy", environment_id=" ")


def test_the_key_never_reaches_logs_or_errors(server, caplog):
    caplog.set_level(logging.DEBUG)
    server(lambda request: httpx.Response(401, json={"error": "bad key"}))
    [failed] = _score(TERM, [_sample()])
    server(_verdicts())
    [scored] = _score(TERM, [_sample()])
    assert failed.score is None and scored.score == 1.0
    assert "401" in caplog.text
    assert KEY not in caplog.text and KEY not in failed.error


def test_yaml_term_builds_the_scorer_on_both_arms(server):
    spec = [{"source": "whitecircle", "name": "policy", "environment_id": "env_test", "policies": ["policy_a"]}]
    for sources in (RLVR_REWARD_SOURCES, ENVIRONMENT_REWARD_SOURCES):
        [term] = parse_reward_terms(spec, sources)
        assert term == WhiteCircleTerm(name="policy", environment_id="env_test", policies=("policy_a",))
        assert isinstance(build_scorer(term), WhiteCircleScorer)


def _online(term, completions, logged):
    function = ScorerRewardFunction(term, build_scorer(term), prompt_column=None, reference_column=None)
    prompts = [[{"role": "user", "content": "2+2?"}]] * len(completions)
    return asyncio.run(function(prompts, completions, log_metric=lambda key, value: logged.update({key: value})))


def test_online_reward_function_and_failure_behavior(server):
    server(_assistant({"flagged": True}))
    term = WhiteCircleTerm(name="policy", environment_id="env_test", exponent=2.0)
    logged = {}
    assert _online(
        term, [[{"role": "assistant", "content": "4"}], [{"role": "assistant", "content": ""}]], logged
    ) == [
        0.0,
        1.0,
    ]
    assert logged[scored_metric_key(term)] == 1.0

    server(lambda request: httpx.Response(401))
    completions = [[{"role": "assistant", "content": "4"}]]
    logged = {}
    assert _online(term, completions, logged) == [None] and logged[scored_metric_key(term)] == 0.0
    neutral = WhiteCircleTerm(name="policy", environment_id="env_test", on_error="neutral")
    assert _online(neutral, completions, logged) == [0.0]


def test_offline_rewards_match_the_online_terms(server, tmp_path):
    config = tmp_path / "rewards.yaml"
    config.write_text(
        "rewards:\n"
        "  - source: accuracy\n"
        "  - {source: whitecircle, name: policy, weight: 0.5, environment_id: env_test}\n"
    )
    server()
    functions, weights = build_reward_functions(config, with_reference=True)
    assert weights == [1.0, 0.5]

    def respond(request):
        events = json.loads(request.content)["events"]
        if events[-1]["content"] == "fail":
            return httpx.Response(401)
        return _verdicts(lambda e: {"flagged": e.get("content") == "\\boxed{5}"})(request)

    server(respond)
    row = {
        "prompt": [{"role": "user", "content": "2+2?"}],
        "completions": [[{"role": "assistant", "content": c}] for c in ("\\boxed{4}", "\\boxed{5}", "fail")],
        "answer": "4",
    }
    [scored], dropped = asyncio.run(score_rows([row], functions, weights, answer_field="answer"))
    # \boxed{4}: correct and clean; \boxed{5}: wrong and flagged; "fail": the check failed, so it is not written.
    assert scored["rewards"] == [1.5, 0.0] and dropped == 1
    assert scored["completions"] == row["completions"][:2]

    online = {}
    for function, weight in zip(functions, weights, strict=True):
        values = function(prompts=[row["prompt"]] * 2, completions=row["completions"][:2], answer=["4", "4"])
        values = asyncio.run(values) if asyncio.iscoroutine(values) else values
        online[function.__name__] = [weight * value for value in values]
    assert [sum(pair) for pair in zip(*online.values(), strict=True)] == scored["rewards"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
