#!/usr/bin/env python
"""``build_tensor_type_grouped_optimizer`` splits (weight decay, dtype, tensor type) groups, since DTensors and
plain EP tensors cannot share a foreach/fused update."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
from transformers import Trainer, TrainingArguments

from halo.optimizers.param_groups import build_tensor_type_grouped_optimizer
from halo.trainers.construction.optimizer_selection import build_optimizer

DECAYED = ["w_dtensor", "w_plain_fp32"]
STEPS = 3
# fp32 fused and single-tensor AdamW differ by the fused kernel's operation order only.
FUSED_MATCH_ATOL = 1e-6


@pytest.fixture
def mesh(tmp_path):
    # A fixed MASTER_PORT collides with any concurrently running suite.
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path}/pg_init")
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


def _model(mesh, *, bf16_plain: bool = True) -> nn.Module:
    """As FSDP2 leaves the non-EP and EP params."""
    generator = torch.Generator().manual_seed(0)
    model = nn.Module()
    weight = torch.randn(4, 4, generator=generator)
    model.register_parameter("w_dtensor", nn.Parameter(distribute_tensor(weight, mesh, [Replicate()])))
    bias = torch.randn(4, generator=generator)
    model.register_parameter("b_plain", nn.Parameter(bias.bfloat16() if bf16_plain else bias))
    model.register_parameter("w_plain_fp32", nn.Parameter(torch.randn(4, 4, generator=generator)))
    return model


def _build(model, optim: str):
    args = TrainingArguments(output_dir="unused", optim=optim, learning_rate=1e-2, weight_decay=0.1, report_to="none")
    return build_optimizer(
        model,
        args,
        DECAYED,
        bf16_optimizer=False,
        config=SimpleNamespace(fp32_non_ep_params=True),
        has_ep_layers=True,
        get_optimizer_cls_and_kwargs=Trainer.get_optimizer_cls_and_kwargs,
    )


def _local(param: torch.Tensor) -> torch.Tensor:
    return param.to_local() if isinstance(param, DTensor) else param


def test_groups_split_by_tensor_type_and_dtype(mesh):
    model = _model(mesh)
    assert isinstance(model.w_dtensor, DTensor)
    optimizer = build_tensor_type_grouped_optimizer(
        model,
        SimpleNamespace(weight_decay=0.1),
        DECAYED,
        lambda a, m: (torch.optim.SGD, {"lr": 0.1, "foreach": True, "fused": False}),
    )

    groups = optimizer.param_groups
    # (wd=0.1, fp32, DTensor), (wd=0.0, bf16, plain), (wd=0.1, fp32, plain), never merged.
    assert len(groups) == 3
    for group in groups:
        assert group["foreach"] is False
        assert group["fused"] is False
        kinds = {isinstance(p, DTensor) for p in group["params"]}
        assert len(kinds) == 1, "a group mixes DTensor and plain tensors"
    assert sorted(g["weight_decay"] for g in groups) == [0.0, 0.1, 0.1]


@pytest.mark.parametrize(("optim", "fused"), [("adamw_torch_fused", True), ("adamw_torch", False)])
def test_every_group_is_fused_exactly_when_the_args_ask(mesh, optim, fused):
    groups = _build(_model(mesh), optim).param_groups
    assert len(groups) == 3
    assert [(group["fused"], group["foreach"]) for group in groups] == [(fused, False)] * 3


def test_a_fused_step_matches_the_single_tensor_step(mesh):
    models = {optim: _model(mesh, bf16_plain=False) for optim in ("adamw_torch_fused", "adamw_torch")}
    optimizers = {optim: _build(model, optim) for optim, model in models.items()}
    assert {group["fused"] for group in optimizers["adamw_torch_fused"].param_groups} == {True}
    initial = {name: _local(p).detach().clone() for name, p in models["adamw_torch"].named_parameters()}
    generator = torch.Generator().manual_seed(1)
    for _ in range(STEPS):
        grads = {name: torch.randn(p.shape, generator=generator) for name, p in initial.items()}
        for optim, model in models.items():
            for name, param in model.named_parameters():
                grad = grads[name]
                param.grad = distribute_tensor(grad, mesh, [Replicate()]) if isinstance(param, DTensor) else grad
            optimizers[optim].step()
            optimizers[optim].zero_grad(set_to_none=True)

    fused = dict(models["adamw_torch_fused"].named_parameters())
    for name, param in models["adamw_torch"].named_parameters():
        assert not torch.equal(_local(param), initial[name]), f"{name} never moved"
        torch.testing.assert_close(_local(fused[name]), _local(param), rtol=0, atol=FUSED_MATCH_ATOL, msg=name)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
