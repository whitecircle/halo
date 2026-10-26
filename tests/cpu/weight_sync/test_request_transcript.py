#!/usr/bin/env python
"""What each engine's weight-sync client sends through its public lifecycle, pinned against the
request-transcript golden: every request with its route, body, timeout and session, what goes on the
NCCL wire, the local group's fate and the at-exit hooks, in scenarios succeeding and failing at each step.

Run: python tests/cpu/weight_sync/test_request_transcript.py
"""

import dataclasses
import json
from collections import deque

import pytest
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from halo.subclass_registry import iter_subclasses
from halo.trainers.grpo.rollout.trl_vllm_client import TrlVLLMClient
from halo.weight_sync.clients import base as client_base
from halo.weight_sync.clients.base import WeightSyncClientBase
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from tests.common.goldens.weight_sync_transcript import ENGINES, GOLDEN_PATH, SCENARIOS, scenario_record

GOLDEN = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
SCENARIO = {scenario.key: scenario for scenario in SCENARIOS}


def _routes(record: list[dict]) -> list[tuple[str, list[str]]]:
    """Each step with the method and path of every request it sent."""
    return [(step["step"], [" ".join(line.split(" ", 2)[:2]) for line in step["requests"]]) for step in record]


def test_the_scenarios_are_the_golden_scenarios():
    assert list(GOLDEN) == list(ENGINES)
    assert all(list(GOLDEN[engine]) == list(SCENARIO) for engine in ENGINES)


@pytest.mark.parametrize(
    ("engine", "scenario"),
    [pytest.param(engine, scenario, id=f"{engine}-{scenario.key}") for engine in ENGINES for scenario in SCENARIOS],
)
def test_the_client_sends_the_golden_transcript(engine, scenario):
    assert scenario_record(engine, scenario) == GOLDEN[engine][scenario.key]


def test_a_retried_session_request_counts_every_receipt():
    """Anti-vacuity: the session retries below the call the recorder sees, so only the engine's receipt
    count shows the retry a refused quiesce takes."""

    def refuse_the_first_pause(engine, server):
        server.replies["/pause"] = deque([503])

    scenario = dataclasses.replace(SCENARIO["at exit after a sync"], script=refuse_the_first_pause)
    record = scenario_record("vllm", scenario)
    golden = GOLDEN["vllm"]["at exit after a sync"]
    [finish_step] = [index for index, step in enumerate(record) if step["step"] == "finish_weight_update"]
    first, *rest = record[finish_step]["requests"]
    assert first == golden[finish_step]["requests"][0] + " received=2"
    assert rest == golden[finish_step]["requests"][1:]


def test_a_collective_rpc_sent_through_the_retrying_session_breaks_the_transcript(monkeypatch):
    """Anti-vacuity: a retry of a blocking engine call queues a second collective RPC and wedges the
    engine, so routing the single-shot posts through the session must read as a different transcript
    although every route is the same."""
    request = WeightSyncClientBase.request

    def over_the_session(self, method, path, *, retry, **kwargs):
        return request(self, method, path, retry=True, **kwargs)

    monkeypatch.setattr(WeightSyncClientBase, "request", over_the_session)
    record = scenario_record("vllm", SCENARIO["dense sync"])
    golden = GOLDEN["vllm"]["dense sync"]
    assert _routes(record) == _routes(golden), "premise: the mutant sends the same routes"
    assert record != golden


def test_a_session_that_no_longer_retries_a_server_error_breaks_the_transcript(monkeypatch):
    """Anti-vacuity: the session's retry policy is recorded field by field, so dropping 500 from the codes
    it retries must read as a different transcript although every request still goes out once."""

    def without_500(max_retries: Retry) -> HTTPAdapter:
        codes = [code for code in max_retries.status_forcelist if code != 500]
        return HTTPAdapter(max_retries=max_retries.new(status_forcelist=codes))

    monkeypatch.setattr(client_base, "HTTPAdapter", without_500)
    record = scenario_record("vllm", SCENARIO["dense sync"])
    golden = GOLDEN["vllm"]["dense sync"]
    assert _routes(record) == _routes(golden), "premise: the mutant sends the same routes"
    assert record != golden


def test_an_exit_hook_that_destroys_the_local_group_breaks_the_transcript(monkeypatch):
    """Anti-vacuity: at interpreter exit the engine never enters the destroy, so the hook must only forget
    the local group; one that destroys it reads as a different transcript."""
    close = SGLangWeightSyncClient.close_communicator
    monkeypatch.setattr(SGLangWeightSyncClient, "close_communicator", lambda self, **kwargs: close(self))
    record = scenario_record("sglang", SCENARIO["at exit after a sync"])
    golden = GOLDEN["sglang"]["at exit after a sync"]
    assert _routes(record) == _routes(golden), "premise: the mutant sends the same routes"
    assert record[-1]["transport"] == ["destroy group"] and golden[-1]["transport"] == ["forget group"]


def test_no_client_overrides_the_chunk_accounting():
    """Every chunk is counted, and its failure judged, in ``WeightSyncClientBase._transmit_chunk``: the
    count behind ``can_replay_sync`` and the verdict that keeps a partly written engine paused. A client
    overriding it would bypass both while every transcript above still passes."""
    clients = [cls for cls in iter_subclasses(WeightSyncClientBase) if cls.__module__.startswith("halo.")]
    assert {VLLMWeightSyncClient, SGLangWeightSyncClient, TrlVLLMClient} <= set(clients)
    assert [cls.__qualname__ for cls in clients if "_transmit_chunk" in vars(cls)] == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
