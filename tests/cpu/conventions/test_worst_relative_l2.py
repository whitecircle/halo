#!/usr/bin/env python
"""``tests.common.numerics.worst_relative_l2``: the per-tensor relative error a GPU suite bounds two runs' gradients
by.

The step-guard suite's norm-only rows compare step 1's gradients against a clipping run's with it, and run only on
the GPU tiers. A gradient synced twice or not at all may sit in a tensor too small to move an aggregate error (a
per-head norm's weight beside an embedding), so these pin that the worst tensor decides, that comparing nothing or
mismatched keys fails a bound, and that a NaN or a zero-norm reference cannot read as a match.

Run: python tests/cpu/conventions/test_worst_relative_l2.py
"""

import math

import pytest
import torch

from tests.common.numerics import relative_l2, worst_relative_l2


def _grads() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {"embed": torch.randn(16384, 64), "q_norm": torch.randn(64), "bias": torch.randn(8)}


def test_identical_dicts_differ_by_zero():
    grads = _grads()
    assert worst_relative_l2({k: v.clone() for k, v in grads.items()}, grads)[0] == 0.0


def test_a_small_tensor_summed_twice_is_the_worst_where_the_aggregate_barely_moves():
    reference = _grads()
    doubled = {**reference, "q_norm": 2 * reference["q_norm"]}
    worst, key = worst_relative_l2(doubled, reference)
    assert key == "q_norm" and worst == pytest.approx(1.0)
    assert relative_l2(doubled, reference) < 1e-2, "premise: the aggregate error hides the doubled tensor"


def test_bf16_rounding_noise_stays_far_under_a_doubled_tensor():
    reference = _grads()
    rounded = {k: v.bfloat16().float() for k, v in reference.items()}
    assert worst_relative_l2(rounded, reference)[0] < 5e-3


@pytest.mark.parametrize(
    "actual",
    [
        pytest.param({}, id="nothing_compared"),
        pytest.param({"embed": torch.ones(2)}, id="a_key_missing"),
        pytest.param({"embed": torch.ones(2), "q_norm": torch.ones(2), "extra": torch.ones(2)}, id="a_key_extra"),
    ],
)
def test_mismatched_keys_fail_any_bound(actual):
    reference = {"embed": torch.ones(2), "q_norm": torch.ones(2)}
    assert worst_relative_l2(actual, reference) == (math.inf, None)


def test_an_empty_reference_fails_any_bound():
    assert worst_relative_l2({}, {}) == (math.inf, None)


def test_a_nan_is_the_result_wherever_it_sits():
    reference = _grads()
    for key in reference:
        poisoned = {**reference, key: reference[key].clone()}
        poisoned[key][0] = math.nan
        worst, worst_key = worst_relative_l2(poisoned, reference)
        assert math.isnan(worst) and worst_key == key


def test_a_zero_norm_reference_matches_only_zeros():
    reference = {"dead": torch.zeros(3), "live": torch.ones(3)}
    assert worst_relative_l2({"dead": torch.zeros(3), "live": torch.ones(3)}, reference)[0] == 0.0
    assert worst_relative_l2({"dead": torch.full((3,), 1e-12), "live": torch.ones(3)}, reference) == (math.inf, "dead")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
