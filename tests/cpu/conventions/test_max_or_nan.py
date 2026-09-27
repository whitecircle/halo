#!/usr/bin/env python
"""``tests.common.utils.max_or_nan`` carries a NaN through to the value a bound is checked on.

The GPU correctness suites track their worst cross-rank or cross-step difference with this helper and
run only on the GPU tiers. The builtin ``max`` keeps a NaN only when it comes first, so a NaN anywhere
else would vanish and the worst difference would read as the largest finite one. These pins keep that
outcome out of the helper on the CPU tier.

Run: python tests/cpu/conventions/test_max_or_nan.py
"""

import math

import pytest

from tests.common.utils import max_or_nan

FINITE = [0.25, 3.0, 1.5]


@pytest.mark.parametrize("position", range(len(FINITE) + 1))
def test_a_nan_anywhere_is_the_result(position):
    values = [*FINITE[:position], math.nan, *FINITE[position:]]
    assert math.isnan(max_or_nan(values))
    assert math.isnan(max_or_nan(iter(values), default=0.0))


def test_finite_values_give_their_maximum():
    assert max_or_nan(FINITE) == 3.0
    assert max_or_nan(iter(FINITE), default=0.0) == 3.0


def test_infinity_is_a_value_not_a_nan():
    assert max_or_nan([*FINITE, math.inf]) == math.inf
    assert max_or_nan([-math.inf, *FINITE]) == 3.0


def test_an_empty_input_returns_the_default_or_raises_without_one():
    assert max_or_nan([], default=0.0) == 0.0
    with pytest.raises(ValueError):
        max_or_nan([])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
