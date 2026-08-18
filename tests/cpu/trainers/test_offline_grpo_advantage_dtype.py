#!/usr/bin/env python
"""Offline-GRPO group advantages must survive integral reward columns.

Verifiable rewards are commonly 0/1 integers. ``np.array`` on such a list keeps an int64 dtype, the
degenerate-group branches return ``np.zeros_like`` (still int64), and the in-place best-completion
emphasis multiply then raises ``UFuncTypeError`` inside ``datasets.map`` — where the traceback
points at the map worker, not the reward column.
"""

import pytest
import torch

from src.trainers.grpo.objective.advantages import STD_EPS, group_relative_advantages
from src.trainers.grpo.offline import compute_group_advantages

METHODS = ["z_norm", "minmax", "quantile_norm", "quantile_uniform", "robust"]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("rewards", [[1, 1, 1], [0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 0]])
def test_integer_rewards_with_best_completion_emphasis(method, rewards):
    """Integral rewards + emphasis must not raise, and must return floats."""
    advantages = compute_group_advantages(rewards, method, best_completion_emphasis=1.5)
    assert len(advantages) == len(rewards)
    assert all(isinstance(a, float) for a in advantages)


@pytest.mark.parametrize("method", METHODS)
def test_integer_and_float_rewards_agree(method):
    """The int and float spellings of the same rewards must give identical advantages."""
    as_int = compute_group_advantages([0, 1, 1, 0], method, best_completion_emphasis=1.5)
    as_float = compute_group_advantages([0.0, 1.0, 1.0, 0.0], method, best_completion_emphasis=1.5)
    assert as_int == pytest.approx(as_float)


def test_degenerate_integer_group_emphasis_is_inert():
    """An all-equal integer group has zero advantages; the emphasis multiply must leave them zero."""
    for method in ("minmax", "robust"):
        assert compute_group_advantages([1, 1, 1], method, best_completion_emphasis=3.0) == pytest.approx(
            [0.0, 0.0, 0.0]
        )


def test_emphasis_still_scales_the_best_completion():
    """The fix must not disable the emphasis on a group that does have spread."""
    plain = compute_group_advantages([0, 0, 1], "minmax", best_completion_emphasis=0.0)
    emphasized = compute_group_advantages([0, 0, 1], "minmax", best_completion_emphasis=2.0)
    best = plain.index(max(plain))
    assert emphasized[best] == pytest.approx(plain[best] * 2.0)


def test_offline_z_norm_matches_the_torch_group_z_norm():
    """The two GRPO z-norms must agree numerically.

    ``np.std`` defaults to ``ddof=0`` while ``Tensor.std()`` is ``correction=1``, so an implicit
    default makes offline's divisor smaller by ``sqrt((n-1)/n)`` — ~6% at n=8, inside the
    ``num_generations`` range GRPO runs at — and the same rewards train at different advantage scales
    depending on which trainer read them. The epsilon must be the same shared constant too.
    """
    rewards = [0.0, 0.25, 0.75, 1.0, 0.5, 0.0, 1.0, 0.5]

    offline = compute_group_advantages(rewards, "z_norm", best_completion_emphasis=0.0)
    online = group_relative_advantages(
        torch.tensor(rewards, dtype=torch.float64), num_generations=len(rewards), scale_rewards="group"
    )

    assert offline == pytest.approx(online.tolist(), rel=1e-9, abs=1e-9)
    # Pins the shared epsilon: a divergent one would shift every advantage by more than this.
    assert STD_EPS == 1e-4


def test_auto_emphasis_scales_by_the_population_std_under_every_method():
    """The torch-parity correction belongs to the z_norm DIVISOR only.

    ``best_completion_emphasis="auto"`` multiplies the best row under every advantage method, and
    ``group_relative_advantages`` has no emphasis term to agree with — so an ``ddof=1`` std here would
    move the advantages of ``minmax``/``quantile_*``/``robust`` runs that never opted into the
    alignment (+6.9% on the std at ``num_generations=8``, applied to the best completion only).
    """
    rewards = [0.0, 1.0]  # population std 0.5; the sample std would be 0.7071
    expected_factor = 3.0 + 2.0 * 0.5 / 1.5

    plain = compute_group_advantages(rewards, "minmax", best_completion_emphasis=0.0)
    auto = compute_group_advantages(rewards, "minmax", best_completion_emphasis="auto")
    best = plain.index(max(plain))

    assert auto[best] == pytest.approx(plain[best] * expected_factor)
    assert auto[best] != pytest.approx(plain[best] * (3.0 + 2.0 * 0.7071067811865476 / 1.7071067811865476))


@pytest.mark.parametrize("method", METHODS)
def test_single_completion_group_has_zero_advantage(method):
    """A one-completion group has no spread, so every method must return exactly ``[0.0]``.

    No method carries a hand-written ``n == 1`` case any more — the general rank/IQR branches produce
    this. A regression that divides by ``n - 1`` or by a zero IQR would surface here as NaN.
    """
    assert compute_group_advantages([0.5], method, best_completion_emphasis=3.0) == [0.0]


@pytest.mark.parametrize("method", ["minmax", "quantile_uniform", "robust"])
def test_two_completion_group_spans_the_full_range(method):
    """A two-completion group maps loser to -1 and winner to +1 exactly, in either order.

    ``quantile_uniform`` reaches this through the general ``(rank - 1) / (n - 1)`` scale rather than
    an ``n == 2`` special case, so the equality is what pins the two as interchangeable.
    """
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
    """A NaN/Inf reward must fail loud rather than be absorbed.

    It used to be absorbed into a ``0.0``/``±1`` advantage in one-and two-completion groups while
    already NaN-ing the batch at three — so the same bad row trained as an average completion or
    poisoned every sibling's gradient depending only on how many completions it happened to have.
    """
    rewards = [0.0, 1.0, 0.5][: group - 1] + [bad]
    with pytest.raises(ValueError, match="Non-finite reward"):
        compute_group_advantages(rewards, method, best_completion_emphasis=0.0)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("emphasis", [0.0, 0.5, 1.0])
def test_emphasis_at_or_below_one_is_inert(method, emphasis):
    """Any emphasis ≤ 1 leaves the advantages untouched — ``0.0`` is not special-cased to ``1.0``."""
    rewards = [0.0, 0.25, 1.0, 1.0]
    assert compute_group_advantages(rewards, method, emphasis) == compute_group_advantages(rewards, method, 1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
