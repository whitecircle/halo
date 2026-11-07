#!/usr/bin/env python
"""The startup checks score centering adds. Candidate log-probs reach the objective only from an SGLang server that
returns K of them as a distribution holding the sampled token, so vLLM, a run without the sampled ids, an image without
SGLang's flat arrays and a reply that is no such distribution are each refused before a rollout trains on a wrong tail
mass."""

import ast

import numpy as np
import pytest

import tests.common.fake_engines as fake_engines
from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.trainers.grpo.environmental.environmental_rules import reject_unsupported_score_centering
from halo.trainers.grpo.rollout.engine_preflight import ServerRequirements, verify_server_flags
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from tests.common.fake_engines import FakeSGLangServer
from tests.common.repo import REPO_ROOT

K = 32


@pytest.fixture
def server():
    server = FakeSGLangServer()
    yield server
    server.close()


def _verify_candidates(server, k: int = K) -> None:
    verify_server_flags(
        [server.url],
        ServerRequirements(is_moe=False, sampled_token_ids=True, candidate_logprobs=k),
        backend=SGLangWeightSyncClient.BACKEND_KEY,
        connection_timeout=5.0,
    )


def test_a_server_that_returns_k_candidates_passes(server):
    _verify_candidates(server)


def test_an_sglang_image_without_the_flat_arrays_is_refused(server):
    server.flat_top_logprobs = False
    with pytest.raises(ValueError, match="candidate log-probs per generated token"):
        _verify_candidates(server)


@pytest.mark.parametrize(
    "reply",
    [
        lambda k, sampled: ([sampled + 1 + j for j in range(k)], [float(np.log(0.5 / k))] * k),
        lambda k, sampled: ([sampled] + [sampled + 1 + j for j in range(k - 1)], [float(np.log(0.9))] * k),
    ],
    ids=["sampled token missing", "mass above one"],
)
def test_candidates_that_are_no_distribution_holding_the_sampled_token_are_refused(server, monkeypatch, reply):
    monkeypatch.setattr(fake_engines, "candidate_distribution", reply)
    with pytest.raises(ValueError, match="not the sampled token's distribution"):
        _verify_candidates(server)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"rollout_backend": "vllm"}, "65% of rollout throughput at K=16 and 80% at K=32"),
        ({"train_on_sampled_tokens": False}, "train_on_sampled_tokens: false"),
    ],
)
def test_vllm_and_a_run_without_sampled_ids_are_refused(overrides, match):
    config = AsyncRolloutConfig(
        score_centering_top_k=16, **{"rollout_backend": "sglang", "train_on_sampled_tokens": True, **overrides}
    )
    with pytest.raises(ValueError, match=match):
        reject_unsupported_score_centering(config)
    reject_unsupported_score_centering(AsyncRolloutConfig(**overrides))


def test_the_environmental_script_refuses_before_its_preflights_and_arms_the_probe():
    """A script that drops the early refusal sends a vLLM run into the candidate probe, and one that drops the keyword
    leaves an image without the arrays unrefused."""
    tree = ast.parse((REPO_ROOT / "scripts/training/grpo/environmental.py").read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    first_line = {}
    for node in calls:
        first_line.setdefault(ast.unparse(node.func), node.lineno)
    assert first_line["reject_unsupported_score_centering"] < first_line["verify_server_flags"]
    requirements = [
        {k.arg: ast.unparse(k.value) for k in node.keywords}
        for node in calls
        if ast.unparse(node.func) == "ServerRequirements"
    ]
    assert any("async_config.score_centering_top_k" in r.get("candidate_logprobs", "") for r in requirements)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
