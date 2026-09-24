#!/usr/bin/env python
"""``teardown_distributed`` frees every DeepEP buffer before destroying the process group.

Gin frees a buffer's symmetric heap through the group communicator, so a free skipped or moved after
``destroy_process_group`` faults the GPU suites at exit.

Run: pytest tests/cpu/conventions/test_distributed_teardown.py
"""

import pytest

from tests.common import distributed as distributed_helpers


def test_teardown_frees_ep_buffers_before_destroying_the_group(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(distributed_helpers.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_helpers.dist, "destroy_process_group", lambda: calls.append("destroy_group"))
    monkeypatch.setattr(distributed_helpers, "destroy_all_dispatchers", lambda: calls.append("free_ep_buffers"))

    distributed_helpers.teardown_distributed()

    assert calls == ["free_ep_buffers", "destroy_group"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
