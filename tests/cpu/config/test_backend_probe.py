#!/usr/bin/env python
"""The launch-time probe of an external backend: rank 0 probes, every rank raises its verdict. A probe on every rank
hits the backend world-size times, and a rank-0 raise alone leaves the peers waiting in the next collective."""

import ast

import pytest

from halo.entrypoints.script import verify_backend_on_rank0
from halo.runtime import consensus
from tests.common.repo import REPO_ROOT

_PROBING_SCRIPTS = ("scripts/training/grpo/online.py", "scripts/training/grpo/environmental.py")
_PROBE_METHODS = frozenset({"verify", "verify_backend"})


def test_a_failing_probe_raises_naming_what_was_probed():
    def probe():
        raise ConnectionError("connection refused")

    with pytest.raises(RuntimeError, match=r"^reward term 'judge' probe failed: connection refused$") as raised:
        verify_backend_on_rank0(probe, "reward term 'judge'")
    assert isinstance(raised.value.__cause__, ConnectionError), "rank 0 must keep the probe's own traceback"


def test_a_passing_probe_returns():
    calls = []
    verify_backend_on_rank0(lambda: calls.append(True), "environment backend")
    assert calls == [True]


def test_an_async_verifier_is_run_to_completion():
    """The scorers' ``verify`` is ``async def``: calling it only builds the coroutine."""
    awaited = []

    async def passing():
        awaited.append(True)

    async def failing():
        raise ValueError("unparseable judge reply")

    verify_backend_on_rank0(passing, "reward term 'judge'")
    assert awaited == [True]
    with pytest.raises(RuntimeError, match="unparseable judge reply"):
        verify_backend_on_rank0(failing, "reward term 'judge'")


@pytest.mark.parametrize("rank0_verdict", [None, "reward term 'judge' probe failed: 401 Unauthorized"])
def test_a_peer_rank_skips_the_probe_and_raises_rank0s_verdict(monkeypatch, rank0_verdict):
    """Off rank 0 the probe never runs; the peers raise rank 0's message, word for word, or return with it."""
    sent = []

    def broadcast(value):
        sent.append(value)
        return rank0_verdict

    monkeypatch.setattr(consensus, "is_global_main_process", lambda: False)
    monkeypatch.setattr(consensus, "broadcast_from_rank0", broadcast)

    def probe():
        raise AssertionError("a peer rank must not probe the backend")

    if rank0_verdict is None:
        verify_backend_on_rank0(probe, "reward term 'judge'")
    else:
        with pytest.raises(RuntimeError) as raised:
            verify_backend_on_rank0(probe, "reward term 'judge'")
        assert str(raised.value) == rank0_verdict
    assert sent == [None], "every rank must enter the broadcast, the peers with no verdict of their own"


@pytest.mark.parametrize("script", _PROBING_SCRIPTS)
def test_the_scripts_probe_their_backends_only_through_the_rank0_seam(script):
    tree = ast.parse((REPO_ROOT / script).read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    direct = [
        ast.unparse(call)
        for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr in _PROBE_METHODS
    ]
    assert not direct, f"{script} probes a backend outside verify_backend_on_rank0: {direct}"
    handed = [
        call.args[0]
        for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "verify_backend_on_rank0" and call.args
    ]
    assert handed, f"{script} no longer probes its backend at launch"
    assert all(isinstance(probe, ast.Attribute) and probe.attr in _PROBE_METHODS for probe in handed), [
        ast.unparse(probe) for probe in handed
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
