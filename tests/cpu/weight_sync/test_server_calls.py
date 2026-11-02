#!/usr/bin/env python
"""``AsyncCall``: a blocking server call on a background thread, joined within the bound its caller names.

Every caller bounds its own wait (a chunk RPC by ``CHUNK_SYNC_TIMEOUT_S``), so the join takes no default that
could silently apply another call's bound; a call still running past it raises rather than returning as done.

    python tests/cpu/weight_sync/test_server_calls.py
"""

import threading

import pytest

from halo.weight_sync.clients.server_calls import AsyncCall


def test_a_wait_names_its_own_bound():
    call = AsyncCall(lambda: 1, name="probe")
    with pytest.raises(TypeError):
        call.wait()
    call.wait(timeout=5)
    assert call.result == 1


def test_a_call_still_running_past_its_bound_raises_and_an_error_is_reraised():
    release = threading.Event()
    blocked = AsyncCall(release.wait, name="/update_weights")
    with pytest.raises(TimeoutError, match="/update_weights still blocked after 0s"):
        blocked.wait(timeout=0.05)
    release.set()

    def fail():
        raise RuntimeError("engine rejected the chunk")

    with pytest.raises(RuntimeError, match="engine rejected the chunk"):
        AsyncCall(fail).wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
