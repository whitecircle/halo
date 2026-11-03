#!/usr/bin/env python
"""Offline-GRPO group advantages: each method's map, the clip, and integral reward columns.
0/1 integer rewards stay int64, so the in-place emphasis multiply raises ``UFuncTypeError`` inside ``datasets.map``."""

import statistics

import pytest
import torch

from halo.trainers.grpo.objective.advantages import STD_EPS, compute_group_advantages, group_relative_advantages

METHODS = ["z_norm", "minmax", "quantile_norm", "quantile_uniform", "robust"]

# Distinct rewards in ascending order, so their ranks are 1..4; spread unevenly, so the methods disagree.
_ASYMMETRIC = [0.0, 0.2, 0.3, 1.0]
_RANKS = [1, 2, 3, 4]
_EXPECTED = {
    "z_norm": [(r - 0.375) / (statistics.stdev(_ASYMMETRIC) + STD_EPS) for r in _ASYMMETRIC],
    "minmax": [-1.0, -0.6, -0.4, 1.0],
    "quantile_norm": [statistics.NormalDist().inv_cdf((k - 0.5) / 4) for k in _RANKS],
    "quantile_uniform": [-1.0, -1 / 3, 1 / 3, 1.0],
    # Median 0.25, interquartile range 0.475 - 0.15 by linear interpolation.
    "robust": [(r - 0.25) / 0.325 for r in _ASYMMETRIC],
}


@pytest.mark.parametrize("method", METHODS)
def test_each_method_maps_an_asymmetric_group_to_its_own_advantages(method):
    assert compute_group_advantages(_ASYMMETRIC, method, best_completion_emphasis=0.0) == pytest.approx(
        _EXPECTED[method]
    )


def test_advantages_are_clipped_to_ten_either_way():
    """An outlier on each side of a tight interquartile range, 400 ranges out."""
    rewards = [-100.0, 0.0, 0.1, 0.2, 0.3, 100.0]
    assert compute_group_advantages(rewards, "robust", best_completion_emphasis=0.0) == pytest.approx(
        [-10.0, -0.6, -0.2, 0.2, 0.6, 10.0]
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("rewards", [[1, 1, 1], [0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 0]])
def test_integer_rewards_with_best_completion_emphasis(method, rewards):
    advantages = compute_group_advantages(rewards, method, best_completion_emphasis=1.5)
    assert len(advantages) == len(rewards)
    assert all(isinstance(a, float) for a in advantages)


@pytest.mark.parametrize("method", METHODS)
def test_integer_and_float_rewards_agree(method):
    as_int = compute_group_advantages([0, 1, 1, 0], method, best_completion_emphasis=1.5)
    as_float = compute_group_advantages([0.0, 1.0, 1.0, 0.0], method, best_completion_emphasis=1.5)
    assert as_int == pytest.approx(as_float)


def test_degenerate_integer_group_emphasis_is_inert():
    for method in ("minmax", "robust"):
        assert compute_group_advantages([1, 1, 1], method, best_completion_emphasis=3.0) == pytest.approx(
            [0.0, 0.0, 0.0]
        )


def test_emphasis_still_scales_the_best_completion():
    plain = compute_group_advantages([0, 0, 1], "minmax", best_completion_emphasis=0.0)
    emphasized = compute_group_advantages([0, 0, 1], "minmax", best_completion_emphasis=2.0)
    best = plain.index(max(plain))
    assert emphasized[best] == pytest.approx(plain[best] * 2.0)


def test_offline_z_norm_matches_the_torch_group_z_norm():
    """``np.std`` defaults to ``ddof=0``, ``Tensor.std()`` to ``correction=1``: an implicit default is ~6% off at n=8.
    The same rewards would train at different advantage scales depending on the trainer."""
    rewards = [0.0, 0.25, 0.75, 1.0, 0.5, 0.0, 1.0, 0.5]

    offline = compute_group_advantages(rewards, "z_norm", best_completion_emphasis=0.0)
    online = group_relative_advantages(
        torch.tensor(rewards, dtype=torch.float64), num_generations=len(rewards), scale_rewards="group"
    )

    assert offline == pytest.approx(online.tolist(), rel=1e-9, abs=1e-9)
    # Pins the shared epsilon: a divergent one would shift every advantage by more than this.
    assert STD_EPS == 1e-4


def test_auto_emphasis_scales_by_the_population_std_under_every_method():
    """The torch-parity correction belongs to the z_norm divisor only.
    ``ddof=1`` here would move runs that never opted in: +6.9% on the std at ``num_generations=8``, best row only."""
    rewards = [0.0, 1.0]  # population std 0.5; the sample std would be 0.7071
    expected_factor = 3.0 + 2.0 * 0.5 / 1.5

    plain = compute_group_advantages(rewards, "minmax", best_completion_emphasis=0.0)
    auto = compute_group_advantages(rewards, "minmax", best_completion_emphasis="auto")
    best = plain.index(max(plain))

    assert auto[best] == pytest.approx(plain[best] * expected_factor)
    assert auto[best] != pytest.approx(plain[best] * (3.0 + 2.0 * 0.7071067811865476 / 1.7071067811865476))


@pytest.mark.parametrize("method", METHODS)
def test_single_completion_group_has_zero_advantage(method):
    """No method has an ``n == 1`` case; dividing by ``n - 1`` or a zero IQR would surface here as NaN."""
    assert compute_group_advantages([0.5], method, best_completion_emphasis=3.0) == [0.0]


@pytest.mark.parametrize("method", ["minmax", "quantile_uniform", "robust"])
def test_two_completion_group_spans_the_full_range(method):
    """``quantile_uniform`` gets there through ``(rank - 1) / (n - 1)``, with no ``n == 2`` special case."""
    assert compute_group_advantages([0.0, 1.0], method, best_completion_emphasis=0.0) == [-1.0, 1.0]
    assert compute_group_advantages([1.0, 0.0], method, best_completion_emphasis=0.0) == [1.0, -1.0]


def test_two_completion_normal_methods_are_symmetric():
    """The two normal-quantile methods stay antisymmetric at n=2 (±std, ±ppf(0.75))."""
    assert compute_group_advantages([0.0, 1.0], "z_norm", 0.0) == pytest.approx(
        [-0.7070067953266834, 0.7070067953266834]
    )
    assert compute_group_advantages([0.0, 1.0], "quantile_norm", 0.0) == pytest.approx(
        [-0.6744897501960817, 0.6744897501960817]
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("group", [1, 2, 3])
def test_non_finite_rewards_are_refused(method, bad, group):
    """Unrefused, a bad row trains as an average completion or poisons its siblings' gradients, by group size."""
    rewards = [0.0, 1.0, 0.5][: group - 1] + [bad]
    with pytest.raises(ValueError, match="Non-finite reward"):
        compute_group_advantages(rewards, method, best_completion_emphasis=0.0)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("emphasis", [0.0, 0.5, 1.0])
def test_emphasis_at_or_below_one_is_inert(method, emphasis):
    """``0.0`` is not special-cased to ``1.0``."""
    rewards = [0.0, 0.25, 1.0, 1.0]
    assert compute_group_advantages(rewards, method, emphasis) == compute_group_advantages(rewards, method, 1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
