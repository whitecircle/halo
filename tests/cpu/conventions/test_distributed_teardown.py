#!/usr/bin/env python
"""``teardown_distributed`` frees every DeepEP buffer before destroying the process group. Gin frees a buffer's heap
through the group communicator, so a later free faults the GPU suites at exit."""

import pytest

from tests.common import distributed as distributed_helpers


def test_teardown_frees_ep_buffers_before_destroying_the_group(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(distributed_helpers.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_helpers.dist, "destroy_process_group", lambda: calls.append("destroy_group"))
    monkeypatch.setattr(
        distributed_helpers.ep_dispatcher, "destroy_all_dispatchers", lambda: calls.append("free_ep_buffers")
    )

    distributed_helpers.teardown_distributed()

    assert calls == ["free_ep_buffers", "destroy_group"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
