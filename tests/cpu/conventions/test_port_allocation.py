#!/usr/bin/env python
"""``tests.common.ports.free_port`` hands out ports nothing else on the host can take first.

A port is released between the probe and its consumer's bind. Under pytest-xdist that window loses
two races: a concurrent worker is handed the same port, or an outbound connection's kernel-picked
local port lands on it, and the rank-0 TCPStore dies with ``EADDRINUSE``. These pins hold the two
properties that close the window, the probe that skips a port something already holds, and the
random start that keeps two sessions on one network apart.

Run: python tests/cpu/conventions/test_port_allocation.py
"""

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tests.common import ports
from tests.common.ports import POOL_FLOOR, ephemeral_floor, free_port, worker_port_block
from tests.common.utils import REPO_ROOT

_WORKERS = 4
_DRAWS = 8
_SESSIONS = 3
_ALLOCATE = "from tests.common.ports import free_port; print(free_port())"
# Every container runs its session as PID 1, so a start derived from the pid repeats across them.
_ALLOCATE_AS_PID_1 = "import os; os.getpid = lambda: 1; " + _ALLOCATE


def _allocate(script: str = _ALLOCATE, **xdist_env: str) -> subprocess.CompletedProcess:
    """Run ``script`` in a fresh interpreter, posing as the xdist worker ``xdist_env`` describes."""
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, **xdist_env},
        check=False,
    )


def _allocate_only(port: int) -> subprocess.CompletedProcess:
    """``free_port()`` from a one-port slice holding exactly ``port``: one worker per pool port."""
    return _allocate(
        PYTEST_XDIST_WORKER_COUNT=str(ephemeral_floor() - POOL_FLOOR), PYTEST_XDIST_WORKER=f"gw{port - POOL_FLOOR}"
    )


def _claim_own_port(sock: socket.socket) -> int:
    """Bind ``sock`` in this worker's own slice, where no concurrent worker allocates."""
    for port in worker_port_block():
        try:
            sock.bind(("", port))
        except OSError:
            continue
        return port
    pytest.fail("no bindable port in this worker's slice")


def test_ports_lie_below_the_ephemeral_range():
    """The kernel picks every ``bind(0)`` port and outbound local port from the ephemeral range."""
    kernel_floor = int(Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split()[0])
    outside = [port for port in (free_port() for _ in range(_DRAWS)) if not POOL_FLOOR <= port < kernel_floor]
    assert not outside, f"ports {outside} fall outside the pool [{POOL_FLOOR}, {kernel_floor})"


def test_concurrent_xdist_workers_never_share_a_port(monkeypatch):
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", str(_WORKERS))
    slices = []
    for worker in range(_WORKERS):
        monkeypatch.setenv("PYTEST_XDIST_WORKER", f"gw{worker}")
        own = set(worker_port_block())
        drawn = {free_port() for _ in range(_DRAWS)}
        assert drawn <= own, f"gw{worker} drew {sorted(drawn - own)} outside its own slice"
        slices.append(own)
    overlaps = [(a, b) for a in range(_WORKERS) for b in range(a + 1, _WORKERS) if slices[a] & slices[b]]
    assert not overlaps, f"worker slices overlap: {overlaps}"


def test_sessions_that_share_a_pid_draw_apart():
    """Two GPU sessions on the host network, each PID 1 of its container, must not walk one sequence."""
    runs = [_allocate(_ALLOCATE_AS_PID_1) for _ in range(_SESSIONS)]
    assert all(run.returncode == 0 for run in runs), [run.stderr for run in runs]
    ports = {int(run.stdout) for run in runs}
    assert len(ports) > 1, f"{_SESSIONS} sessions with one pid all drew port {ports}"


def test_a_listening_port_is_never_handed_out():
    """A one-port slice whose port has a listener fails loud, and yields that port once it closes."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        port = _claim_own_port(holder)
        holder.listen()
        held = _allocate_only(port)
        assert held.returncode != 0, f"free_port() handed out {held.stdout.strip()}, held by a listener"
        assert "issued or held" in held.stderr, held.stderr

    released = _allocate_only(port)
    assert released.returncode == 0, released.stderr
    assert int(released.stdout) == port


def test_a_port_in_time_wait_is_never_handed_out():
    """A closed TCPStore (a SO_REUSEADDR listener) leaves TIME_WAIT behind, which a consumer that binds
    without SO_REUSEADDR cannot get past."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = _claim_own_port(server)
        server.listen()
        with socket.create_connection(("127.0.0.1", port)):
            accepted, _ = server.accept()
            accepted.close()  # the side that closes first holds TIME_WAIT
    result = _allocate_only(port)
    assert result.returncode != 0, f"free_port() handed out {result.stdout.strip()}, still in TIME_WAIT"
    assert "issued or held" in result.stderr, result.stderr


def test_an_ephemeral_range_that_covers_the_pool_names_the_fix(monkeypatch):
    """A host whose ephemeral range starts at or below the pool has nowhere safe to draw from."""
    monkeypatch.setattr(ports, "ephemeral_floor", lambda: POOL_FLOOR)
    with pytest.raises(RuntimeError, match=r'ip_local_port_range="32768 60999"'):
        free_port()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
