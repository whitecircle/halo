#!/usr/bin/env python
"""The packed-vs-per-row routing comparison: near-tie flips pass, a wide-margin flip or too high a rate fails."""

import pytest
import torch

from tests.common.routing import FLIP_RATE_MAX, NEAR_TIE_MARGIN_MAX, check_flips_are_near_ties, router_margins

ROWS, SEQ, LAYERS, TOP_K = 4, 50, 2, 2


def _reference() -> torch.Tensor:
    ids = torch.tensor([0, 1]).expand(ROWS, SEQ, LAYERS, TOP_K).clone()
    return ids.to(torch.int16)


def _margins(value: float) -> torch.Tensor:
    return torch.full((ROWS, SEQ, LAYERS), value)


def _flip(captured: torch.Tensor, count: int) -> torch.Tensor:
    captured = captured.clone()
    captured.view(-1, TOP_K)[:count, 1] = 5
    return captured


def test_near_tie_flips_pass_and_report_their_margin():
    margins = _margins(0.2)
    margins.view(-1)[:2] = NEAR_TIE_MARGIN_MAX / 2
    reference = _reference()
    stats = check_flips_are_near_ties(_flip(reference, 2), reference, margins)
    assert stats["flip_rate"] == pytest.approx(2 / reference[..., 0].numel())
    assert stats["widest_flip_margin"] == pytest.approx(NEAR_TIE_MARGIN_MAX / 2)
    assert stats["near_tie_share"] == pytest.approx(2 / reference[..., 0].numel())


def test_a_flip_in_a_different_order_is_not_a_flip():
    reference = _reference()
    reordered = reference.flip(-1)
    assert check_flips_are_near_ties(reordered, reference, _margins(0.2))["flip_rate"] == 0.0


def test_a_wide_margin_flip_fails():
    margins = _margins(0.2)
    margins.view(-1)[0] = NEAR_TIE_MARGIN_MAX / 2
    reference = _reference()
    captured = _flip(reference, 2)
    with pytest.raises(AssertionError, match="wide-margin"):
        check_flips_are_near_ties(captured, reference, margins)


def test_a_rate_above_the_cap_fails_even_when_every_flip_is_a_near_tie():
    reference = _reference()
    count = int(FLIP_RATE_MAX * reference[..., 0].numel()) + 1
    with pytest.raises(AssertionError, match="cap"):
        check_flips_are_near_ties(_flip(reference, count), reference, _margins(NEAR_TIE_MARGIN_MAX / 2))


def test_positions_outside_the_reference_do_not_count():
    reference = _reference()
    reference[0] = -1
    margins = _margins(0.2)
    captured = reference.clone()
    assert check_flips_are_near_ties(captured, reference, margins)["flip_rate"] == 0.0
    with pytest.raises(AssertionError, match="different positions"):
        check_flips_are_near_ties(_reference(), reference, margins)


def test_router_margin_is_the_probability_gap_at_the_cut():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
    assert router_margins(logits, 1).item() == pytest.approx(0.2)
    assert router_margins(logits, 2).item() == pytest.approx(0.1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
