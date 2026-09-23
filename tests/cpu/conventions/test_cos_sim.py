#!/usr/bin/env python
"""``tests.common.utils.cos_sim`` raises on an operand with no direction instead of scoring it.

The GPU correctness suites gate gradient and weight agreement on this helper, and they run only on the
GPU tiers. A cosine that scores a zero-norm operand passes one side of every threshold: 1.0 matches a
dead gradient against a live one, and 0.0 lets a negative control whose gradient vanished read as
decorrelated. NaN slips past the ``<`` and ``min`` trackers that report the worst tensor. These pins
keep those outcomes out of the helper on the CPU tier.

Run: python tests/cpu/conventions/test_cos_sim.py
"""

import math

import pytest
import torch

from tests.common.utils import cos_sim

SEED = 0


@pytest.mark.parametrize("pair", ["dead_vs_live", "live_vs_dead", "both_dead"])
def test_a_zero_norm_operand_raises(pair):
    live, dead = torch.randn(64, generator=torch.Generator().manual_seed(SEED)), torch.zeros(64)
    a, b = {"dead_vs_live": (dead, live), "live_vs_dead": (live, dead), "both_dead": (dead, dead)}[pair]
    with pytest.raises(ValueError, match="^layers.0.router.weight: cosine of a zero-norm"):
        cos_sim(a, b, "layers.0.router.weight")


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_operand_raises(bad):
    live = torch.randn(64, generator=torch.Generator().manual_seed(SEED))
    corrupted = live.clone()
    corrupted[3] = bad
    with pytest.raises(ValueError, match="non-finite"):
        cos_sim(corrupted, live)


@pytest.mark.parametrize("scale", [1.0, 1e-7, 1e-12])
def test_small_live_tensors_keep_their_direction(scale):
    """No epsilon floor: a norm product under 1e-12, or a norm under 1e-8, still compares by direction."""
    a = torch.randn(4, 16, generator=torch.Generator().manual_seed(SEED)) * scale
    assert math.isclose(cos_sim(a, a), 1.0, rel_tol=1e-5)
    assert math.isclose(cos_sim(a, -a), -1.0, rel_tol=1e-5)


def test_bf16_operands_compare_flat_in_fp32():
    a = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.bfloat16)
    b = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.bfloat16)
    assert math.isclose(cos_sim(a, b), 1 / math.sqrt(2), rel_tol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
