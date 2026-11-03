#!/usr/bin/env python
"""Online GRPO's vLLM generation timeout must stay below the NCCL collective watchdog. The POST blocks every peer at
the next collective, so the client's default derives from the ``nccl_timeout()`` resolver instead of a literal."""

import logging

import pytest

from halo.env import nccl_timeout, watchdog_bounded_seconds
from halo.trainers.grpo.rollout.trl_vllm_client import TrlVLLMClient
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from tests.common.weight_sync import offline_client


class _NoChoices:
    def json(self):
        return {"choices": []}


@pytest.fixture
def generation_timeout(monkeypatch):
    """The deadline ``TrlVLLMClient.generate`` puts on its ``/v1/completions`` request, through a real client; a
    structural read of the source would pass against a restored literal."""

    def resolve(**env):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        calls = []

        def request(method, path, *, timeout, **_):
            calls.append((path, timeout))
            return _NoChoices()

        client = offline_client(TrlVLLMClient)
        client.request = request
        client.generate([[1, 2]])
        ((path, deadline),) = calls
        assert path == "/v1/completions"
        return deadline

    return resolve


def test_default_generation_timeout_clears_the_default_watchdog(monkeypatch, generation_timeout):
    monkeypatch.delenv("HALO_VLLM_GENERATION_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("DIST_NCCL_TIMEOUT_MINUTES", raising=False)

    timeout = generation_timeout()

    assert timeout == watchdog_bounded_seconds()
    assert timeout < nccl_timeout().total_seconds(), (
        "a generation call that reaches the watchdog aborts every peer's collective before it returns"
    )


def test_a_lowered_watchdog_lowers_the_generation_timeout(monkeypatch, generation_timeout):
    monkeypatch.delenv("HALO_VLLM_GENERATION_TIMEOUT_SECONDS", raising=False)

    default_watchdog = generation_timeout(DIST_NCCL_TIMEOUT_MINUTES="30")
    lowered = generation_timeout(DIST_NCCL_TIMEOUT_MINUTES="10")

    assert lowered < default_watchdog
    assert lowered < 10 * 60, "the HTTP timeout is unreachable under a 10-minute watchdog"


def test_an_explicit_generation_timeout_still_wins(monkeypatch, generation_timeout):
    """An operator who raised the watchdog can raise this too."""
    assert generation_timeout(HALO_VLLM_GENERATION_TIMEOUT_SECONDS="900.5") == 900.5


@pytest.mark.parametrize(("seconds", "warned"), [("600", True), ("599", False)], ids=["at-watchdog", "below"])
def test_a_generation_timeout_at_the_watchdog_is_warned_about(monkeypatch, caplog, seconds, warned):
    """An override the watchdog would abort first names itself and the knob to raise instead."""
    monkeypatch.setenv("DIST_NCCL_TIMEOUT_MINUTES", "10")
    monkeypatch.setenv("HALO_VLLM_GENERATION_TIMEOUT_SECONDS", seconds)

    with caplog.at_level(logging.WARNING):
        offline_client(TrlVLLMClient)

    warning = f"HALO_VLLM_GENERATION_TIMEOUT_SECONDS={seconds}s is at or above the NCCL collective watchdog (600s)"
    assert (warning in caplog.text) is warned
    assert ("DIST_NCCL_TIMEOUT_MINUTES" in caplog.text) is warned


def test_the_weight_sync_client_alone_never_warns(monkeypatch, caplog):
    """Async GRPO's weight-sync client generates nothing under this deadline, so it names no knob."""
    monkeypatch.setenv("DIST_NCCL_TIMEOUT_MINUTES", "10")
    monkeypatch.setenv("HALO_VLLM_GENERATION_TIMEOUT_SECONDS", "600")

    with caplog.at_level(logging.WARNING):
        offline_client(VLLMWeightSyncClient)

    assert "HALO_VLLM_GENERATION_TIMEOUT_SECONDS" not in caplog.text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
