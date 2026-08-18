"""Collision-free master-port allocation for torchrun-launched tests.

The pytest launcher (``tests/gpu/conftest.py``) asks for a port per run via :func:`free_port` and
passes it as ``--master_port``. A test that hardcodes a port races every other launch on the host
(two tests back-to-back, or two CI shards) into a rendezvous failure with ``[Errno 98] Address
already in use``. Standalone, ``torchrun`` picks its own.

The "bind to :0, read the port, close" approach has a small TOCTOU window, since another process can
take the port between the close and ``torchrun`` re-binding it. :func:`free_port` narrows it by
excluding ports seen in this process and by scanning the ephemeral range, which suffices for
serial/low-concurrency test launching.
"""

import contextlib
import socket

# Ports already handed out in this process; not re-issued within a session even
# where the OS would allow the re-bind.
_ISSUED: set[int] = set()


def free_port() -> int:
    """Return a currently-free TCP port on localhost.

    Binds an ephemeral socket to ``:0``, lets the OS pick a port, records it so
    the same port is not returned twice in this process, and returns it.
    """
    for _ in range(64):
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", 0))
            port = s.getsockname()[1]
        if port not in _ISSUED:
            _ISSUED.add(port)
            return port
    # 64 binds all collided with ports already issued.
    raise RuntimeError("could not allocate a free port after 64 attempts")
