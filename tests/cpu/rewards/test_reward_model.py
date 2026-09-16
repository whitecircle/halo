#!/usr/bin/env python
"""CPU tests: the served reward-model scorer against fake vLLM and SGLang classify routes.

Run: python tests/cpu/rewards/test_reward_model.py  (or pytest)
"""

import asyncio
import json
import math

import httpx
import pytest

from src.rewards.reward_model import ServedRewardModel
from src.rewards.samples import ScoringSample
from src.rewards.spec import RewardModelTerm


class _Template:
    """A tokenizer stand-in: renders every message as ``role: content`` on its own line and tokenizes a
    text to its character codes, refusing to add special tokens."""

    def apply_chat_template(self, messages, tokenize):
        assert tokenize is False
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def __call__(self, texts, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [[ord(c) for c in text] for text in texts]}


def _sample(text: str) -> ScoringSample:
    return ScoringSample(
        prompt=[{"role": "user", "content": "Hi"}], completion=[{"role": "assistant", "content": text}]
    )


def _scorer(term: RewardModelTerm, handler) -> tuple[ServedRewardModel, list[httpx.Request]]:
    scorer = ServedRewardModel(term)
    scorer._tokenizer = _Template()
    requests: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    scorer._create_client = lambda: httpx.AsyncClient(transport=httpx.MockTransport(recording))
    return scorer, requests


def _vllm_handler(logits_by_text):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        # Out of order on purpose: the client must sort by index, not trust arrival order.
        rows = [
            {"index": i, "probs": [logits_by_text[text], 0.0], "label": None, "num_classes": 2}
            for i, text in enumerate(body["input"])
        ]
        return httpx.Response(200, json={"data": list(reversed(rows)), "model": body["model"], "usage": {}})

    return handler


def test_vllm_request_and_parsing():
    term = RewardModelTerm(name="pref", url="http://rm:8100/", model="org/rm", batch_size=8)
    scorer, requests = _scorer(
        term, _vllm_handler({"user: Hi\nassistant: good": 2.0, "user: Hi\nassistant: bad": -2.0})
    )
    results = asyncio.run(scorer.score([_sample("good"), _sample("bad")]))
    (request,) = requests
    assert str(request.url) == "http://rm:8100/classify"
    assert json.loads(request.content) == {
        "model": "org/rm",
        "input": ["user: Hi\nassistant: good", "user: Hi\nassistant: bad"],
        "use_activation": False,
        "add_special_tokens": False,
    }
    assert scorer.metric_keys == ("reward_model/pref/logit",)
    assert [r.score for r in results] == [pytest.approx(1 / (1 + math.exp(-2))), pytest.approx(1 / (1 + math.exp(2)))]
    assert results[0].metrics == {"reward_model/pref/logit": 2.0}


def test_sglang_request_and_parsing():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200, json=[{"embedding": [0.5 * i], "meta_info": {}} for i, _ in enumerate(body["input_ids"])]
        )

    term = RewardModelTerm(name="pref", url="http://rm:30000", model="org/rm", backend="sglang", label_index=0)
    scorer, requests = _scorer(term, handler)
    results = asyncio.run(scorer.score([_sample("a"), _sample("b"), _sample("c")]))
    assert str(requests[0].url) == "http://rm:30000/classify"
    # Token ids of the rendered text, without added special tokens: TRL's in-process reward-model input.
    assert json.loads(requests[0].content) == {
        "input_ids": [[ord(c) for c in f"user: Hi\nassistant: {t}"] for t in ("a", "b", "c")]
    }
    assert [r.metrics["reward_model/pref/logit"] for r in results] == [0.0, 0.5, 1.0]
    assert results[0].score == 0.5


def test_label_index_picks_the_head_output():
    """A 2-class RM whose positive label is index 1: reading index 0 would reward the negative logit."""
    term = RewardModelTerm(name="pref", url="http://rm", model="m", label_index=1)
    scorer, _ = _scorer(term, lambda request: httpx.Response(200, json={"data": [{"index": 0, "probs": [9.0, -9.0]}]}))
    (result,) = asyncio.run(scorer.score([_sample("x")]))
    assert result.metrics["reward_model/pref/logit"] == -9.0
    assert result.score == pytest.approx(1 / (1 + math.exp(9)))


def test_sglang_single_object_payload_is_one_result():
    scorer, _ = _scorer(
        RewardModelTerm(name="pref", url="http://rm", model="m", backend="sglang"),
        lambda request: httpx.Response(200, json={"embedding": [3.0], "meta_info": {}}),
    )
    (result,) = asyncio.run(scorer.score([_sample("x")]))
    assert result.metrics["reward_model/pref/logit"] == 3.0


def test_batches_respect_batch_size_and_keep_order():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        seen.append(len(texts))
        return httpx.Response(200, json={"data": [{"index": i, "probs": [float(t[-1])]} for i, t in enumerate(texts)]})

    term = RewardModelTerm(name="pref", url="http://rm", model="m", batch_size=2, max_concurrency=1)
    scorer, _ = _scorer(term, handler)
    results = asyncio.run(scorer.score([_sample(str(i)) for i in range(5)]))
    assert sorted(seen) == [1, 2, 2]
    assert [r.metrics["reward_model/pref/logit"] for r in results] == [0.0, 1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(500, text="boom"), "request failed"),
        (httpx.Response(200, json={"data": []}), "0 outputs for 1 inputs"),
        (httpx.Response(200, json={"data": [{"index": 0, "probs": []}]}), "label_index 0 is out of range"),
        (
            httpx.Response(
                200, content=b'{"data": [{"index": 0, "probs": [NaN]}]}', headers={"content-type": "application/json"}
            ),
            "non-finite",
        ),
    ],
)
def test_bad_responses_score_none_without_raising(response, message):
    scorer, _ = _scorer(RewardModelTerm(name="pref", url="http://rm", model="m"), lambda request: response)
    (result,) = asyncio.run(scorer.score([_sample("x")]))
    assert result.score is None and message in result.error


def test_verify_scores_a_probe_and_raises_on_failure():
    ok, _ = _scorer(
        RewardModelTerm(name="pref", url="http://rm", model="m"),
        lambda r: httpx.Response(200, json={"data": [{"index": 0, "probs": [0.1]}]}),
    )
    asyncio.run(ok.verify())
    bad, _ = _scorer(
        RewardModelTerm(name="pref", url="http://rm", model="m"), lambda r: httpx.Response(404, text="no route")
    )
    with pytest.raises(RuntimeError, match="probe failed"):
        asyncio.run(bad.verify())


def test_verify_probes_through_a_fresh_client_and_closes_it():
    """The launch probe runs on a different loop than the Ray actor, and a cached AsyncClient carries a
    pool bound to the loop it was built on — so the probe must build its own and close it."""
    scorer, _ = _scorer(
        RewardModelTerm(name="pref", url="http://rm", model="m"),
        lambda r: httpx.Response(200, json={"data": [{"index": 0, "probs": [0.1]}]}),
    )
    build, created = scorer._create_client, []

    def tracked() -> httpx.AsyncClient:
        created.append(client := build())
        return client

    scorer._create_client = tracked
    asyncio.run(scorer.verify())
    assert len(created) == 1 and created[0].is_closed and scorer._client is None


def test_transcript_view_selects_what_is_rendered():
    rendered = []

    def handler(request: httpx.Request) -> httpx.Response:
        rendered.extend(json.loads(request.content)["input"])
        return httpx.Response(200, json={"data": [{"index": 0, "probs": [0.0]}]})

    sample = ScoringSample(
        prompt=[{"role": "user", "content": "Hi"}],
        completion=[
            {"role": "assistant", "content": "step"},
            {"role": "tool", "content": "out"},
            {"role": "assistant", "content": "final"},
        ],
    )
    final, _ = _scorer(RewardModelTerm(name="p", url="http://rm", model="m"), handler)
    asyncio.run(final.score([sample]))
    full, _ = _scorer(RewardModelTerm(name="p", url="http://rm", model="m", transcript="full"), handler)
    asyncio.run(full.score([sample]))
    assert rendered == ["user: Hi\nassistant: final", "user: Hi\nassistant: step\ntool: out\nassistant: final"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
