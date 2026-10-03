"""The GPU oracle must preserve SR order/groups and grade the independent applied update."""

import pytest
import torch

from src.optimizers.adamw_bf16 import AdamWBF16
from tests.common.cp_grpo import optimizer_step_agreement, same_layout_optimizer_step


def _fixture():
    model = torch.nn.ParameterDict(
        {
            "first": torch.nn.Parameter(torch.full((256,), 0.5, dtype=torch.bfloat16)),
            "second": torch.nn.Parameter(torch.full((256,), 1.0, dtype=torch.bfloat16)),
            "router": torch.nn.Parameter(torch.linspace(-0.5, 0.5, 256)),
        }
    )
    generator = torch.Generator().manual_seed(27)
    gradients = {name: torch.randn(parameter.shape, generator=generator) * 0.1 for name, parameter in model.items()}
    optimizer = AdamWBF16(
        [
            {"params": [model["second"], model["first"]], "lr": 0.001, "weight_decay": 0.03},
            {"params": [model["router"]], "lr": 0.004, "betas": (0.8, 0.95), "eps": 1e-6},
        ],
        use_triton=False,
    )
    initial = {name: parameter.detach().float().clone() for name, parameter in model.items()}
    return model, optimizer, gradients, initial


def test_same_layout_step_preserves_groups_order_and_does_not_touch_policy():
    model, optimizer, gradients, initial = _fixture()
    # Policy gradients intentionally differ: the oracle must not read them.
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    expected, oracle_optimizer, clones = same_layout_optimizer_step(model, optimizer, gradients)
    assert not optimizer.state
    assert oracle_optimizer._use_triton == optimizer._use_triton
    assert all(torch.equal(parameter.float(), initial[name]) for name, parameter in model.items())
    assert [list(group["params"]) for group in oracle_optimizer.param_groups] == [
        [clones["second"], clones["first"]],
        [clones["router"]],
    ]
    for parameter, name in ((model["first"], "first"), (model["second"], "second"), (model["router"], "router")):
        parameter.grad = gradients[name].to(parameter.dtype)
    optimizer.step()
    actual = {name: parameter.detach().float().clone() for name, parameter in model.items()}
    assert all(torch.equal(actual[name], expected[name]) for name in expected)
    assert optimizer_step_agreement(actual, expected, initial)
    assert not optimizer_step_agreement(initial, expected, initial)


def test_reversed_reference_gradient_fails_applied_update_oracle():
    model, optimizer, gradients, initial = _fixture()
    expected, _, _ = same_layout_optimizer_step(model, optimizer, gradients)
    for name, parameter in model.items():
        parameter.grad = -gradients[name].to(parameter.dtype)
    optimizer.step()
    actual = {name: parameter.detach().float().clone() for name, parameter in model.items()}
    assert not optimizer_step_agreement(actual, expected, initial)


def test_same_layout_step_rejects_stale_optimizer_and_missing_reference_parameter():
    model, optimizer, gradients, _ = _fixture()
    with pytest.raises(KeyError, match="second"):
        same_layout_optimizer_step(model, optimizer, {})
    optimizer.step()
    with pytest.raises(ValueError, match="fresh AdamWBF16"):
        same_layout_optimizer_step(model, optimizer, gradients)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
