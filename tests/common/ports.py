"""Collision-free port allocation for tests that bind a listener; a hardcoded port races every other launch on the
host. The pool lies below the kernel's ephemeral range, so only an explicit bind can take a port. Each xdist worker has
its own slice, and the scan starts at a random offset, so sessions sharing a network namespace rarely collide."""

import os
import secrets
import socket
from pathlib import Path

# Prefer the band above Ray's default workers; lower ephemeral floors need the non-Ray low band.
POOL_FLOOR = 20000
_UNPRIVILEGED_FLOOR = 1024
_LOW_POOL_CEILING = 10000
_EPHEMERAL_RANGE = Path("/proc/sys/net/ipv4/ip_local_port_range")

# Never re-issued within a session.
_ISSUED: set[int] = set()


def ephemeral_floor() -> int:
    return int(_EPHEMERAL_RANGE.read_text().split()[0])


def resolve_port_pool(workers: int) -> range:
    """An unprivileged pool outside the kernel's ephemeral range and Ray's default worker ports."""
    if workers < 1:
        raise ValueError("The test worker count must be positive")
    floor = ephemeral_floor()
    pool = range(POOL_FLOOR, floor)
    if len(pool) < workers:
        pool = range(_UNPRIVILEGED_FLOOR, min(_LOW_POOL_CEILING, floor))
    if len(pool) < workers:
        raise RuntimeError(
            f"net.ipv4.ip_local_port_range starts at {floor}, which leaves {workers} test worker(s) no port "
            "outside privileged, ephemeral and Ray worker ranges. Move the range up to the Linux default: "
            '`sysctl -w net.ipv4.ip_local_port_range="32768 60999"` on the host (a --network host '
            "container shares it), or "
            '`docker run --sysctl net.ipv4.ip_local_port_range="32768 60999"` for a container of its own.'
        )
    return pool


def worker_port_block() -> range:
    """The slice of the pool this process allocates from, disjoint across one xdist session's workers."""
    workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
    pool = resolve_port_pool(workers)
    # A restarted worker gets a fresh id past the count; wrap it back into the pool.
    index = int(os.environ.get("PYTEST_XDIST_WORKER", "gw0").removeprefix("gw")) % workers
    size = len(pool) // workers
    start = pool.start + index * size
    return range(start, start + size)


def _bindable(port: int) -> bool:
    # No SO_REUSEADDR: a port still in TIME_WAIT fails a consumer that binds without it.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("", port))
        except OSError:
            return False
    return True


def free_port() -> int:
    """A port no other worker of this session can be handed, and that nothing holds right now."""
    block = worker_port_block()
    # Not `random`: a test that seeds it would pin every session to one sequence.
    offset = secrets.randbelow(len(block))
    for step in range(len(block)):
        port = block[(offset + step) % len(block)]
        if port not in _ISSUED and _bindable(port):
            _ISSUED.add(port)
            return port
    raise RuntimeError(f"every port in {block.start}-{block.stop - 1} is issued or held")
