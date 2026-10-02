#!/usr/bin/env python
"""Shared CP scoring oracles distinguish routed and dense parameters."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.common.cp_grpo import ep_gradient_parameter_names, gradient_agreement, optimizer_step_agreement
from tests.common.tolerances import TOL


def test_ep_parameter_classification_uses_identity_and_keeps_shared_experts_dense():
    model = nn.Module()
    model.ordinary_name = nn.Linear(2, 2, bias=False)
    model.other_name = nn.Linear(2, 2, bias=False)
    model.shared_expert = nn.Linear(2, 2, bias=False)
    model.router_named_dense_projection = nn.Linear(2, 2, bias=False)
    layer = SimpleNamespace(
        expert_named_params=lambda: [("expert", model.ordinary_name.weight)],
        _live_router_module=lambda: model.other_name,
    )
    assert ep_gradient_parameter_names(model, [layer]) == {"ordinary_name.weight", "other_name.weight"}


def test_ep_bounds_accept_routing_noise_without_weakening_dense_cp_bounds():
    expected = {"parameter": torch.tensor([1.0, 0.0])}
    actual = {"parameter": torch.tensor([1.14, 0.37])}
    keys, direction, norm, _, _ = gradient_agreement(actual, expected)
    assert keys and not direction and not norm
    bounds = {"cosine_min": TOL.ep_grad_cosine_min, "norm_ratio_band": TOL.ep_grad_norm_ratio_band}
    keys, direction, norm, _, _ = gradient_agreement(actual, expected, **bounds)
    assert keys and direction and norm
    initial = {"parameter": torch.zeros(2)}
    assert not optimizer_step_agreement(actual, expected, initial)
    assert optimizer_step_agreement(actual, expected, initial, **bounds)


@pytest.mark.parametrize("bad_gradient", [[2.0, 0.0], [0.0, 1.0]])
def test_ep_bounds_still_reject_missing_gradient_average_and_wrong_direction(bad_gradient):
    _, direction, norm, _, _ = gradient_agreement(
        {"parameter": torch.tensor(bad_gradient)},
        {"parameter": torch.tensor([1.0, 0.0])},
        cosine_min=TOL.ep_grad_cosine_min,
        norm_ratio_band=TOL.ep_grad_norm_ratio_band,
    )
    assert not (direction and norm)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
