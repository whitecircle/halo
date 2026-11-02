"""The peak host memory the gathered-save benchmark reports follows what the process holds, and restarts from
the current size on a reset, so a save's peak is not the model load's.

The probe runs in a fresh interpreter pinned to one CPU: the kernel restarts the peak from its per-CPU batched
resident-set count, which a long-lived process whose threads ran on many CPUs (an xdist worker) can leave hundreds of
MiB above what it holds on a many-CPU host, enough to hide the whole block.

Run: python tests/cpu/conventions/test_host_peak_rss.py
"""

import pytest

from tests.common.repo import probe_findings

MARKER = "FINDINGS:"
BLOCK_GIB = 0.25

_PROBE = """
import os

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

from tests.common.memory import host_peak_rss_gib, reset_host_peak_rss

reset_host_peak_rss()
before = host_peak_rss_gib()
block = b"x" * int(BLOCK_GIB * 1024**3)
grown = host_peak_rss_gib()
del block
held = host_peak_rss_gib()

reset_host_peak_rss()
restarted = host_peak_rss_gib()

findings = []
if grown - before < 0.95 * BLOCK_GIB:
    findings.append(f"the peak grew {grown - before:.3f} GiB over a {BLOCK_GIB} GiB block")
if held < grown - 0.05 * BLOCK_GIB:
    findings.append(f"the peak fell from {grown:.3f} to {held:.3f} GiB when the block was freed")
if restarted > grown - 0.95 * BLOCK_GIB:
    findings.append(f"the reset left the peak at {restarted:.3f} GiB after it reached {grown:.3f} GiB")
print(MARKER + "|".join(findings))
"""


def test_the_peak_follows_an_allocation_and_restarts_on_a_reset():
    findings = probe_findings(f"BLOCK_GIB = {BLOCK_GIB}\nMARKER = {MARKER!r}\n{_PROBE}", MARKER)
    assert not findings, "\n".join(findings)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
