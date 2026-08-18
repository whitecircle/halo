#!/usr/bin/env python
"""Online GRPO's vLLM generation timeout must stay below the NCCL collective watchdog.

The generation POST is single-shot and blocking, so every peer sits at the next collective for its
whole duration. Env-GRPO validates its rollout timeouts against ``resolve_nccl_timeout_minutes()``;
online GRPO has no such guard, so the client's default is derived from the same resolver instead of
spelled as a literal — otherwise lowering ``DIST_NCCL_TIMEOUT_MINUTES`` leaves an HTTP timeout the
watchdog fires long before.

    python tests/cpu/grpo/test_generation_timeout_watchdog.py
"""

import pytest

from src.distributed.nccl.clients.base import BaseWeightSyncClient
from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient
from src.env import resolve_nccl_timeout_minutes, watchdog_bounded_seconds


@pytest.fixture
def generation_timeout(monkeypatch):
    """Resolve the client's generation timeout without touching the network.

    Drives the real ``VLLMWeightSyncClient.__init__`` line, with only the base client's socket/HTTP
    setup stubbed out — a structural read of the source would pass against a restored literal.
    """
    monkeypatch.setattr(BaseWeightSyncClient, "__init__", lambda self, **kwargs: None)

    def resolve(**env):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return VLLMWeightSyncClient(base_url="http://localhost:8000")._generation_timeout

    return resolve


def test_default_generation_timeout_clears_the_default_watchdog(monkeypatch, generation_timeout):
    monkeypatch.delenv("HALO_VLLM_GENERATION_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("DIST_NCCL_TIMEOUT_MINUTES", raising=False)

    timeout = generation_timeout()

    assert timeout == watchdog_bounded_seconds()
    assert timeout < resolve_nccl_timeout_minutes() * 60, (
        "a generation call that reaches the watchdog aborts every peer's collective before it returns"
    )


def test_a_lowered_watchdog_lowers_the_generation_timeout(monkeypatch, generation_timeout):
    """The regression a literal default hides: shrinking the watchdog must shrink the HTTP bound."""
    monkeypatch.delenv("HALO_VLLM_GENERATION_TIMEOUT_SECONDS", raising=False)

    default_watchdog = generation_timeout(DIST_NCCL_TIMEOUT_MINUTES="30")
    lowered = generation_timeout(DIST_NCCL_TIMEOUT_MINUTES="10")

    assert lowered < default_watchdog
    assert lowered < 10 * 60, "the HTTP timeout is unreachable under a 10-minute watchdog"


def test_an_explicit_generation_timeout_still_wins(monkeypatch, generation_timeout):
    """The knob is an override, not a ceiling — an operator who raised the watchdog can raise it."""
    assert generation_timeout(HALO_VLLM_GENERATION_TIMEOUT_SECONDS="900.5") == 900.5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
