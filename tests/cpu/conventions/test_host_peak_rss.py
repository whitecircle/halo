"""The peak host memory the gathered-save benchmark reports follows what the process holds, and restarts from
the current size on a reset, so a save's peak is not the model load's.

Run: python tests/cpu/conventions/test_host_peak_rss.py
"""

import pytest

from tests.common.memory import host_peak_rss_gib, reset_host_peak_rss

BLOCK_GIB = 0.25


def test_the_peak_follows_an_allocation_and_restarts_on_a_reset():
    reset_host_peak_rss()
    before = host_peak_rss_gib()
    block = b"x" * int(BLOCK_GIB * 1024**3)
    grown = host_peak_rss_gib()
    del block

    reset_host_peak_rss()
    restarted = host_peak_rss_gib()

    assert grown - before >= 0.95 * BLOCK_GIB
    assert restarted <= grown - 0.95 * BLOCK_GIB


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
