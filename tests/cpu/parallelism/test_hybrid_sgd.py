"""Native SGD and derived storage-rounding controls for the hybrid CUDA comparison."""

from dataclasses import replace

import pytest
import torch

from tests.common.hybrid_cp import hybrid_sgd_agreement, hybrid_sgd_step, hybrid_tensor_agreement


def _native_step(before, gradient, learning_rate=0.1):
    parameter = torch.nn.Parameter(before.clone())
    parameter.grad = gradient.clone()
    torch.optim.SGD([parameter], lr=learning_rate).step()
    return hybrid_sgd_step(before, gradient, parameter.detach(), learning_rate)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_sgd_proof_uses_native_dtype_and_keeps_zero_updates_exact(dtype):
    before = torch.tensor([0.5, -0.7, 1.0], dtype=dtype)
    gradient = torch.tensor([0.3, 0.0, -0.2], dtype=dtype)
    first = _native_step(before, gradient)
    assert hybrid_sgd_agreement(first, first)[0]
    assert first.delta.dtype == first.ideal.dtype == first.rounding.dtype == torch.float64
    assert first.delta[1] == first.ideal[1] == first.rounding[1] == 0


def test_one_ulp_update_difference_is_accepted_only_by_its_derived_rounding_budget():
    """A one-ULP delta mismatch is admissible only when independently predicted rounding covers it."""
    before = torch.ones(1)
    first_gradient = torch.tensor([26.49 * 2**-24 / 0.1])
    second_gradient = torch.tensor([26.51 * 2**-24 / 0.1])
    assert hybrid_tensor_agreement(first_gradient, second_gradient, gradient=True)[0]
    first, second = (_native_step(before, gradient) for gradient in (first_gradient, second_gradient))
    assert float((first.delta - second.delta).abs()) == 2**-24
    assert not hybrid_tensor_agreement(first.delta, second.delta, gradient=True)[0]
    passed, metrics = hybrid_sgd_agreement(first, second)
    assert passed
    assert metrics["ideal_gradient_difference"] < metrics["absolute_l2"]
    assert metrics["absolute_l2"] <= metrics["derived_rounding_budget"]


@pytest.mark.parametrize(
    "omitted", ("model.layers.0.linear_attn.A_log", "model.layers.0.linear_attn.dt_bias", "lm_head.weight")
)
def test_sgd_proof_rejects_one_skipped_parameter_even_when_its_update_is_small(omitted):
    """Small updates to decay, timestep bias and head weights must remain observable to the SGD proof."""
    before = {
        "model.layers.0.linear_attn.A_log": torch.ones(8),
        "model.layers.0.linear_attn.dt_bias": torch.full((8,), 0.25),
        "lm_head.weight": torch.full((4, 4), 0.7),
    }
    gradients = {name: torch.full_like(value, 1.58e-5) for name, value in before.items()}
    snapshots = {name: _native_step(value, gradients[name]) for name, value in before.items()}
    for snapshot in snapshots.values():
        assert hybrid_sgd_agreement(snapshot, snapshot)[0]
    with pytest.raises(AssertionError):
        hybrid_sgd_step(before[omitted], gradients[omitted], before[omitted], 0.1)
    changed = replace(snapshots[omitted], delta=torch.zeros_like(snapshots[omitted].delta))
    assert not hybrid_sgd_agreement(changed, snapshots[omitted])[0]


@pytest.mark.parametrize("fault", ("nonfinite", "dtype", "shape", "missing"))
def test_sgd_snapshot_and_pair_refuse_invalid_captures(fault):
    """Missing pairs fail comparison; malformed native snapshots cannot certify an update."""
    before, gradient = torch.ones(4), torch.full((4,), 0.01)
    reference = _native_step(before, gradient)
    if fault == "missing":
        assert not hybrid_sgd_agreement(None, reference)[0]
        return
    changed = gradient.clone()
    if fault == "nonfinite":
        changed[0] = float("nan")
    elif fault == "dtype":
        changed = changed.double()
    else:
        changed = changed[:2]
    with pytest.raises(AssertionError, match="snapshots"):
        hybrid_sgd_step(before, changed, before, 0.1)


@pytest.mark.parametrize("arm", ("actual", "expected"))
@pytest.mark.parametrize("fault", ("nonfinite_ideal", "broadcastable_delta"))
def test_sgd_pair_refuses_malformed_proof_fields(arm, fault):
    """Neither proof arm may turn nonfinite ideals or broadcasting into an admissible update budget."""
    reference = _native_step(torch.ones(4), torch.full((4,), 0.01))
    changed = (
        replace(reference, ideal=torch.full_like(reference.ideal, float("inf")))
        if fault == "nonfinite_ideal"
        else replace(reference, delta=reference.delta[:1])
    )
    actual, expected = (changed, reference) if arm == "actual" else (reference, changed)
    assert hybrid_sgd_agreement(actual, expected) == (False, {"finite_and_same_shape": False})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
