#!/usr/bin/env python
"""A flash run builds its attention masks without transformers' packed-sequence probe.

transformers' mask construction tests ``position_ids`` for packed documents whenever it builds a causal or
sliding-window mask from them without a padding mask (``find_packed_sequence_indices``, a device read at the start
of every forward), and ``flash_attention_mask`` discards the packed mask that test feeds. Across the tiny roster of
packing families (:mod:`tests.common.packed_flash_varlen`, one fresh interpreter each), a packed flash training step
must make no host read in mask construction and train bit-identically to the same step with the probe, while an
SDPA forward of the same rows still runs it.

    python tests/cpu/models/test_flash_mask_packed_probe.py
"""

import pytest

from tests.common.packed_flash_varlen import FINDINGS_MARKER, PACKING_FAMILIES, REACH_PREFIX, SKIP_PREFIX
from tests.common.repo import probe_findings

# Probe reads per forward with the probe in place: one per causal mask, one more for a sliding-window mask.
_PROBE_READS = {"dense:qwen3": 1, "dense:gemma3_text": 2}


def _report(family: str) -> list[str]:
    return probe_findings(
        f"from tests.common.packed_flash_varlen import mask_probe_findings, report; report(mask_probe_findings({family!r}))",
        FINDINGS_MARKER,
    )


def _reach(report: list[str]) -> int:
    return int(next(entry for entry in report if entry.startswith(REACH_PREFIX)).removeprefix(REACH_PREFIX))


@pytest.mark.parametrize("family", sorted(PACKING_FAMILIES))
def test_a_packed_flash_step_builds_its_masks_without_the_probe(family):
    report = _report(family)
    if report and report[0].startswith(SKIP_PREFIX):
        pytest.skip(report[0].removeprefix(SKIP_PREFIX).strip())
    failures = [entry for entry in report if not entry.startswith(REACH_PREFIX)]
    assert not failures, "\n".join(failures)
    assert _reach(report) > 0


@pytest.mark.parametrize("family", sorted(_PROBE_READS))
def test_the_sweep_reaches_a_causal_and_a_sliding_window_mask(family):
    """Anti-vacuity: with the probe in place, these families' masks read the device once per mask built."""
    report = _report(family)
    assert not report[0].startswith(SKIP_PREFIX), report
    assert _reach(report) == _PROBE_READS[family], report


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
