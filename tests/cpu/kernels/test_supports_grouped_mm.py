#!/usr/bin/env python
"""CPU tests for the grouped-GEMM capability predicate.

The answer picks the expert weight layout, so every rank must reach the same one: a capability read
that fails on one rank has to raise there, never fall back to the per-expert loop while its peers
take grouped GEMM and later hang in the expert all-gather.

    pytest -m cpu tests/cpu/kernels/test_supports_grouped_mm.py
"""

from __future__ import annotations

import pytest
import torch

from halo.hardware import supports_grouped_mm


def _capability(monkeypatch, capability) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def get_device_capability(*_args, **_kwargs):
        if isinstance(capability, Exception):
            raise capability
        return capability

    monkeypatch.setattr(torch.cuda, "get_device_capability", get_device_capability)


def test_no_cuda_reports_unsupported(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert supports_grouped_mm() is False


@pytest.mark.parametrize(
    ("capability", "expected"), [((8, 0), False), ((8, 9), False), ((9, 0), True), ((10, 3), True)]
)
def test_capability_threshold_is_sm90(monkeypatch, capability, expected):
    _capability(monkeypatch, capability)
    assert supports_grouped_mm() is expected


def test_failed_capability_read_raises(monkeypatch):
    _capability(monkeypatch, RuntimeError("CUDA error: unspecified launch failure"))
    with pytest.raises(RuntimeError, match="unspecified launch failure"):
        supports_grouped_mm()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
