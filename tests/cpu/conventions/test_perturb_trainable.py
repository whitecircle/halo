#!/usr/bin/env python
"""``tests.common.numerics.perturb_trainable``: the deterministic weight perturbation a GPU suite's measured floor
starts from.

The GPU suites run it on every rank of a sharded model, so these pin what they rely on: a perturbation the dtype can
hold changes stored elements by about its relative size and reports how many, one below half a ULP changes none and
says so, ranks holding the same values perturb them alike, a DTensor's local shard is what changes, and a frozen
parameter is left alone.

Run: python tests/cpu/conventions/test_perturb_trainable.py
"""

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor

from tests.common.numerics import perturb_trainable

# The size the GPU floor arm uses, and one under half a bf16 ULP everywhere (2**-9 at the top of a binade).
FLOOR = 4e-3
BELOW_HALF_ULP = 5e-4
# At FLOOR about half of a bf16 tensor's elements move by one ULP; far fewer would leave the floor arm near its
# reference.
FLOOR_MOVED_FRACTION_MIN = 0.25


def _linear(dtype: torch.dtype, seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(64, 64).to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_a_holdable_perturbation_moves_elements_by_about_its_size(dtype):
    model = _linear(dtype)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    changed = perturb_trainable(model, FLOOR)
    moved = sum(int((p != before[name]).sum()) for name, p in model.named_parameters())
    assert changed == moved > FLOOR_MOVED_FRACTION_MIN * sum(p.numel() for p in model.parameters())
    for name, param in model.named_parameters():
        relative = ((param.float() - before[name].float()).abs() / before[name].float().abs().clamp_min(1e-30)).max()
        assert relative <= 2 * FLOOR + torch.finfo(dtype).eps, name


def test_a_perturbation_under_half_a_bf16_ulp_changes_nothing_and_says_so():
    model = _linear(torch.bfloat16)
    before = [p.detach().clone() for p in model.parameters()]
    assert perturb_trainable(model, BELOW_HALF_ULP) == 0
    assert all(torch.equal(p, b) for p, b in zip(model.parameters(), before, strict=True))


def test_copies_of_one_tensor_perturb_alike():
    """Replicas on different ranks hold equal values; the factor reads only the value, so they stay equal."""
    first, second = _linear(torch.bfloat16), _linear(torch.bfloat16)
    perturb_trainable(first, FLOOR)
    perturb_trainable(second, FLOOR)
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters(), strict=True))


def test_a_frozen_parameter_is_left_alone():
    model = _linear(torch.bfloat16)
    model.bias.requires_grad_(False)
    bias = model.bias.detach().clone()
    weight = model.weight.detach().clone()
    perturb_trainable(model, FLOOR)
    assert torch.equal(model.bias, bias)
    assert not torch.equal(model.weight, weight)


def test_a_dtensor_parameter_changes_its_local_shard(tmp_path):
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path}/pg_init")
    try:
        mesh = init_device_mesh("cpu", (1,))
        model = nn.Module()
        torch.manual_seed(0)
        weight = torch.randn(64, 64).bfloat16()
        model.register_parameter("weight", nn.Parameter(distribute_tensor(weight, mesh, [Replicate()])))
        assert isinstance(model.weight, DTensor)
        before = model.weight.to_local().detach().clone()
        assert perturb_trainable(model, FLOOR) > 0
        assert not torch.equal(model.weight.to_local(), before)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
