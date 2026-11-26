"""The packed-gradient verdict against synthetic gradients: the floor is the per-row bf16 error from the fp32
reference, and a packed error is judged against it."""

import pytest
import torch

from tests.common.packed_grad_floor import GRAD_FLOOR_FACTOR, judge_packed_grads

FLOOR = 1e-2


def _unit_noise(seed: int, like: torch.Tensor) -> torch.Tensor:
    noise = torch.randn(like.shape, generator=torch.Generator().manual_seed(seed))
    return noise / noise.norm() * like.norm()


def _verdict(packed_ratio: float, unscaled_scale: float = 4.0):
    reference = [torch.randn(64, 32, generator=torch.Generator().manual_seed(0)), torch.randn(128)]
    per_row = [ref + FLOOR * _unit_noise(1, ref) for ref in reference]
    packed = [ref + packed_ratio * FLOOR * _unit_noise(2, ref) for ref in reference]
    reorder = [row + 1e-4 * _unit_noise(3, row) for row in per_row]
    unscaled = [ref + (unscaled_scale - 1.0) * ref + FLOOR * _unit_noise(2, ref) for ref in reference]
    return judge_packed_grads(per_row, reorder, packed, unscaled, reference)


def test_floor_is_the_per_row_distance_from_the_reference():
    verdict = _verdict(1.0)
    assert verdict.floor == pytest.approx(FLOOR, rel=1e-3)
    assert verdict.order_floor == pytest.approx(1e-4, rel=0.2)


def test_packed_error_within_twice_the_floor_passes():
    verdict = _verdict(GRAD_FLOOR_FACTOR * 0.95)
    assert verdict.inside_the_floor
    assert verdict.points_the_same_way


def test_packed_error_at_three_times_the_floor_fails():
    assert not _verdict(3.0).inside_the_floor


def test_a_scaled_gradient_bites_the_control():
    assert _verdict(1.0, unscaled_scale=4.0).control_bites


def test_a_control_close_to_the_packed_gradient_fails_the_control():
    assert not _verdict(1.0, unscaled_scale=1.0).control_bites


def test_a_control_at_four_times_the_noise_fails_the_five_fold_margin():
    assert not _verdict(1.0, unscaled_scale=1.0 + 4.0 * FLOOR).control_bites


def test_an_identical_unscaled_window_changes_nothing():
    reference = [torch.randn(64, generator=torch.Generator().manual_seed(0))]
    assert judge_packed_grads(reference, reference, reference, reference, reference).rescale_changes_nothing


def test_a_rescaled_window_is_not_identical():
    assert not _verdict(1.0, unscaled_scale=1.5).rescale_changes_nothing


def test_an_unscaled_gap_within_the_repeat_noise_changes_nothing():
    packed = [torch.randn(64, generator=torch.Generator().manual_seed(0))]
    repeat = [packed[0] + 2e-2 * _unit_noise(1, packed[0])]
    unscaled = [packed[0] + 2e-2 * _unit_noise(2, packed[0])]
    assert judge_packed_grads(packed, packed, packed, unscaled, None, repeat=repeat).rescale_changes_nothing
    assert not judge_packed_grads(packed, packed, packed, unscaled, None).rescale_changes_nothing


def test_a_rescale_past_the_repeat_noise_is_caught():
    packed = [torch.randn(64, generator=torch.Generator().manual_seed(0))]
    repeat = [packed[0] + 2e-2 * _unit_noise(1, packed[0])]
    unscaled = [1.5 * packed[0]]
    assert not judge_packed_grads(packed, packed, packed, unscaled, None, repeat=repeat).rescale_changes_nothing


def test_without_a_reference_the_order_floor_is_the_floor():
    gradient = [torch.randn(64, generator=torch.Generator().manual_seed(0))]
    reorder = [gradient[0] + 1e-2 * _unit_noise(1, gradient[0])]
    verdict = judge_packed_grads(gradient, reorder, gradient, gradient, None)
    assert verdict.floor == verdict.order_floor > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
