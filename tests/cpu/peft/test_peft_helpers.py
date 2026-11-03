#!/usr/bin/env python
"""``tests.common.adapters.assert_adapters_moved`` fails the adapter states that should fail it.
``require_all`` also fails a run that left one adapter where it started, which the default lets through."""

import pytest
import torch

from tests.common.adapters import assert_adapters_moved

_BEFORE = {
    "q_proj.lora_A.default.weight": torch.ones(2, 4),
    "q_proj.lora_B.default.weight": torch.zeros(4, 2),
    "v_proj.lora_A.default.weight": torch.ones(2, 4),
    "v_proj.lora_B.default.weight": torch.zeros(4, 2),
}


def _moved(*names: str, value: float = 1.0) -> dict[str, torch.Tensor]:
    """``_BEFORE`` with ``value`` added to the named adapters."""
    return {name: tensor + value if name in names else tensor.clone() for name, tensor in _BEFORE.items()}


@pytest.mark.parametrize("require_all", [False, True])
def test_every_adapter_moved_passes(require_all):
    ok, why = assert_adapters_moved(_BEFORE, _moved(*_BEFORE), require_all=require_all)

    assert ok, why


def test_an_adapter_left_where_it_started_fails_only_under_require_all():
    after = _moved("q_proj.lora_A.default.weight", "q_proj.lora_B.default.weight", "v_proj.lora_A.default.weight")

    assert assert_adapters_moved(_BEFORE, after)[0]
    ok, why = assert_adapters_moved(_BEFORE, after, require_all=True)
    assert not ok
    assert "v_proj.lora_B.default.weight" in why


def test_an_adapter_missing_after_training_counts_as_unmoved_under_require_all():
    after = _moved(*_BEFORE)
    del after["v_proj.lora_A.default.weight"]

    assert not assert_adapters_moved(_BEFORE, after, require_all=True)[0]


@pytest.mark.parametrize("require_all", [False, True])
def test_no_lora_b_moved_fails(require_all):
    after = _moved("q_proj.lora_A.default.weight", "v_proj.lora_A.default.weight")

    ok, why = assert_adapters_moved(_BEFORE, after, require_all=require_all)
    assert not ok
    assert "lora_B" in why


@pytest.mark.parametrize("require_all", [False, True])
def test_a_non_finite_adapter_fails(require_all):
    ok, why = assert_adapters_moved(_BEFORE, _moved(*_BEFORE, value=float("nan")), require_all=require_all)

    assert not ok
    assert "non-finite" in why


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
