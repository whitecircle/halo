#!/usr/bin/env python
"""AdamWBF16 (fused kernel and eager step) and Muon's two kernels against float64 references written from the update
equations. In a fresh process, since ``TRITON_INTERPRET`` is read at kernel definition."""

import pytest

from tests.common.repo import probe_findings

_MARKER = "FLOAT64_OPTIMIZER_MISMATCHES:"


def _probe(check: str) -> str:
    return f"""
import os
os.environ["TRITON_INTERPRET"] = "1"
from tests.common.float64_optimizer_reference import {check}
print({_MARKER!r} + "|".join({check}()))
"""


def test_adamw_bf16_steps_land_in_their_float64_brackets_interpreted():
    """Steps 1 and 3, lr 0 and 1e-3, weight decay 0 and 0.1, eps 1e-8 and 1; ``lr = 0`` leaves every weight as it was."""
    assert probe_findings(_probe("adamw_float64_mismatches"), _MARKER) == []


def test_adamw_bf16_stochastic_rounding_is_unbiased_interpreted():
    assert probe_findings(_probe("adamw_rounding_bias_mismatches"), _MARKER) == []


def test_muon_kernels_land_in_their_float64_brackets_interpreted():
    assert probe_findings(_probe("muon_float64_mismatches"), _MARKER) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
