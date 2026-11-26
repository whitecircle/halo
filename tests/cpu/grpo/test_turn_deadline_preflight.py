#!/usr/bin/env python
"""A turn that runs to ``rollout_max_tokens`` must be able to finish inside ``request_timeout``: a timed-out episode
trains as a masked row, so the batch would keep only the fast ones."""

import ast
import logging
import types

import pytest

import halo.trainers.grpo.rollout.engine_preflight as preflight
from halo.trainers.grpo.rollout.engine_preflight import _verify_turn_deadline_on_rank0, probe_generation_rate
from tests.common.repo import REPO_ROOT


class _Reply:
    def __init__(self, tokens: int):
        self.tokens = tokens

    def json(self):
        return {"usage": {"completion_tokens": self.tokens}}


class _Client:
    """One timed completion per scripted ``(tokens, seconds)`` pair, on a clock the test advances."""

    def __init__(self, clock, script):
        self.clock, self.script, self.bodies = clock, list(script), []

    def served_model_id(self):
        return "served"

    def request(self, method, path, **kwargs):
        tokens, seconds = self.script.pop(0)
        self.bodies.append((method, path, kwargs["json"]))
        self.clock.now += seconds
        return _Reply(tokens)


@pytest.fixture
def clock(monkeypatch):
    now = types.SimpleNamespace(now=0.0)
    monkeypatch.setattr(preflight.time, "perf_counter", lambda: now.now)
    return now


def test_the_rate_is_the_better_of_two_timed_completions(clock):
    client = _Client(clock, [(64, 1.0), (64, 0.5)])
    assert probe_generation_rate(client) == pytest.approx(128.0)
    ((method, path, body), _) = client.bodies
    assert (method, path) == ("POST", "/v1/completions")
    assert body["max_tokens"] == 64 and body["ignore_eos"] is True and body["temperature"] == 0.0


def test_a_reply_too_short_to_time_counts_for_nothing(clock):
    assert probe_generation_rate(_Client(clock, [(3, 0.01), (5, 0.01)])) is None
    assert probe_generation_rate(_Client(clock, [(3, 0.01), (64, 2.0)])) == pytest.approx(32.0)


def _verify(monkeypatch, rate, *, max_tokens, timeout):
    monkeypatch.setattr(preflight, "_probe_server", lambda *args, **kwargs: rate)
    _verify_turn_deadline_on_rank0(object, ["http://s0"], max_tokens, timeout)


def test_a_turn_well_inside_the_deadline_passes_quietly(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        _verify(monkeypatch, 100.0, max_tokens=8192, timeout=300.0)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_turn_that_a_loaded_server_would_miss_warns_with_the_timeout_to_set(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        _verify(monkeypatch, 100.0, max_tokens=8192, timeout=120.0)
    (message,) = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert "request_timeout=120s" in message and "at least 164" in message and "masked row" in message


def test_a_turn_an_idle_server_cannot_finish_is_refused(monkeypatch):
    with pytest.raises(ValueError, match=r"Even an idle rollout server.*at least 164"):
        _verify(monkeypatch, 100.0, max_tokens=8192, timeout=30.0)


def test_a_server_that_could_not_be_timed_is_left_to_the_run(monkeypatch):
    _verify(monkeypatch, None, max_tokens=10**6, timeout=1.0)


def test_the_environmental_script_runs_the_check_with_the_turn_cap_and_timeout():
    """A script that drops the call leaves a run whose long turns all time out to find out at the batch."""
    module = ast.parse((REPO_ROOT / "scripts/training/grpo/environmental.py").read_text())
    calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "verify_turn_deadline"
    ]
    assert len(calls) == 1
    assert [ast.unparse(arg) for arg in calls[0].args[1:]] == [
        "async_config.rollout_max_tokens",
        "async_config.request_timeout",
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
