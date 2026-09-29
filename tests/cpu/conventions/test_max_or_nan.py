#!/usr/bin/env python
"""``tests.common.utils.max_or_nan`` and ``min_or_nan`` carry a NaN through to the value a bound is
checked on.

The GPU correctness suites track their worst cross-rank or cross-step difference with these helpers
and run only on the GPU tiers. The builtin ``max`` and ``min`` keep a NaN only when it comes first, so
a NaN anywhere else would vanish and the worst difference would read as the extreme finite one. These
pins keep that outcome out of the helpers on the CPU tier.

Run: python tests/cpu/conventions/test_max_or_nan.py
"""

import math

import pytest

from tests.common.utils import max_or_nan, min_or_nan

FINITE = [0.25, 3.0, 1.5]
# Each helper with its extreme over FINITE and the infinity that is its extreme over any finite values.
HELPERS = {"max": (max_or_nan, 3.0, math.inf), "min": (min_or_nan, 0.25, -math.inf)}


@pytest.mark.parametrize("name", HELPERS)
@pytest.mark.parametrize("position", range(len(FINITE) + 1))
def test_a_nan_anywhere_is_the_result(name, position):
    helper, _, _ = HELPERS[name]
    values = [*FINITE[:position], math.nan, *FINITE[position:]]
    assert math.isnan(helper(values))
    assert math.isnan(helper(iter(values), default=0.0))


@pytest.mark.parametrize("name", HELPERS)
def test_finite_values_give_their_extreme(name):
    helper, extreme, _ = HELPERS[name]
    assert helper(FINITE) == extreme
    assert helper(iter(FINITE), default=0.0) == extreme


@pytest.mark.parametrize("name", HELPERS)
def test_infinity_is_a_value_not_a_nan(name):
    helper, extreme, infinity = HELPERS[name]
    assert helper([*FINITE, infinity]) == infinity
    assert helper([-infinity, *FINITE]) == extreme


@pytest.mark.parametrize("name", HELPERS)
def test_an_empty_input_returns_the_default_or_raises_without_one(name):
    helper, _, _ = HELPERS[name]
    assert helper([], default=0.0) == 0.0
    with pytest.raises(ValueError):
        helper([])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
