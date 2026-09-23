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
import re

import pytest
import torch

from tests.common.utils import cos_sim

SEED = 0
LABEL = "model.layers.0.mlp.router.weight"


def _live(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(SEED))


@pytest.mark.parametrize("pair", ["dead_vs_live", "live_vs_dead", "both_dead"])
def test_a_zero_norm_operand_raises(pair):
    live, dead = _live(64), torch.zeros(64)
    a, b = {"dead_vs_live": (dead, live), "live_vs_dead": (live, dead), "both_dead": (dead, dead)}[pair]
    with pytest.raises(ValueError, match="^" + re.escape(f"{LABEL}: cosine of a zero-norm")):
        cos_sim(a, b, label=LABEL)


@pytest.mark.parametrize("corrupted_operand", [0, 1])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_operand_raises(bad, corrupted_operand):
    operands = [_live(64), _live(64)]
    operands[corrupted_operand][3] = bad
    with pytest.raises(ValueError, match="^" + re.escape(f"{LABEL}: cosine of a non-finite")):
        cos_sim(*operands, label=LABEL)


def test_a_finite_tensor_at_the_bf16_range_limit_is_scored():
    """A sink neutralized to the bf16 minimum is finite; its square overflows fp32, not fp64."""
    floor = torch.full((64,), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16)
    assert math.isclose(cos_sim(floor, floor, label=LABEL), 1.0, rel_tol=1e-12)


@pytest.mark.parametrize("scale", [1.0, 1e-7, 1e-12])
def test_small_live_tensors_keep_their_direction(scale):
    """No epsilon floor: a norm product under 1e-12, or a norm under 1e-8, still compares by direction."""
    a = _live(4, 16) * scale
    assert math.isclose(cos_sim(a, a, label=LABEL), 1.0, rel_tol=1e-5)
    assert math.isclose(cos_sim(a, -a, label=LABEL), -1.0, rel_tol=1e-5)


def test_bf16_operands_compare_flat():
    a = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.bfloat16)
    b = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.bfloat16)
    assert math.isclose(cos_sim(a, b, label=LABEL), 1 / math.sqrt(2), rel_tol=1e-12)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
