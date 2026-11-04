#!/usr/bin/env python
"""``tests.common.ports.free_port`` hands out ports nothing else on the host can take first. A port is released
between the probe and its consumer's bind, where a concurrent worker or an outbound connection can take it and
the TCPStore dies with ``EADDRINUSE``."""

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tests.common import ports
from tests.common.ports import free_port, resolve_port_pool, worker_port_block
from tests.common.repo import REPO_ROOT

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
    pool = resolve_port_pool(int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")))
    return _allocate(PYTEST_XDIST_WORKER_COUNT=str(len(pool)), PYTEST_XDIST_WORKER=f"gw{port - pool.start}")


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
    drawn = [free_port() for _ in range(_DRAWS)]
    assert len(set(drawn)) == _DRAWS, "a session reissued a port"
    assert all(1024 <= port < kernel_floor and not 10000 <= port < 20000 for port in drawn), drawn


@pytest.mark.parametrize("floor", [16000, 32768])
def test_concurrent_xdist_workers_never_share_a_port(monkeypatch, floor):
    monkeypatch.setattr(ports, "ephemeral_floor", lambda: floor)
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", str(_WORKERS))
    slices = []
    for worker in range(_WORKERS):
        monkeypatch.setenv("PYTEST_XDIST_WORKER", f"gw{worker}")
        own = set(worker_port_block())
        drawn = {free_port() for _ in range(_DRAWS)}
        assert len(drawn) == _DRAWS, "a session reissued a port"
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


@pytest.mark.parametrize(
    ("floor", "workers", "expected"),
    [
        (32768, 4, range(20000, 32768)),
        (16000, 4, range(1024, 10000)),
        (20000, 4, range(1024, 10000)),
        (20003, 4, range(1024, 10000)),
        (10000, 4, range(1024, 10000)),
        (9000, 4, range(1024, 9000)),
        (1028, 4, range(1024, 1028)),
    ],
)
def test_pool_selection_respects_ephemeral_and_reserved_ranges(monkeypatch, floor, workers, expected):
    monkeypatch.setattr(ports, "ephemeral_floor", lambda: floor)
    pool = resolve_port_pool(workers)
    assert pool == expected
    assert len(pool) >= workers
    assert all(1024 <= port < floor and not 10000 <= port < 20000 for port in pool)


@pytest.mark.parametrize(("floor", "workers"), [(1024, 1), (1027, 4), (16000, 8977)])
def test_no_safe_worker_slice_names_the_fix(monkeypatch, floor, workers):
    monkeypatch.setattr(ports, "ephemeral_floor", lambda: floor)
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", str(workers))
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    with pytest.raises(RuntimeError, match=r'ip_local_port_range="32768 60999"'):
        free_port()


def test_low_pool_worker_slices_stay_nonempty_at_the_boundary(monkeypatch):
    monkeypatch.setattr(ports, "ephemeral_floor", lambda: 1028)
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "4")
    blocks = []
    for worker in range(4):
        monkeypatch.setenv("PYTEST_XDIST_WORKER", f"gw{worker}")
        blocks.append(list(worker_port_block()))
    assert blocks == [[1024], [1025], [1026], [1027]]


@pytest.mark.parametrize("workers", [0, -1])
def test_invalid_worker_count_fails_before_partitioning(monkeypatch, workers):
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", str(workers))
    with pytest.raises(ValueError, match="worker count must be positive"):
        worker_port_block()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
