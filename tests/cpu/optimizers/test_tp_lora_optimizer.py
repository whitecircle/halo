#!/usr/bin/env python
"""Native TP LoRA's stock optimizer must step both shards and plain replicas."""

import os
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from transformers import Trainer, TrainingArguments
from transformers.training_args import OptimizerNames

from src.optimizers.adamw_bf16 import AdamWBF16
from src.trainers.mixins.base import DistributedTrainerMixin
from tests.common.gloo import run_gloo_ranks


class _OptimizerTrainer(DistributedTrainerMixin, Trainer):
    def __init__(self, model, args, optimizer=None):
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.optimizer_cls_and_kwargs = None
        self._bf16_optimizer = False
        self._native_tp_lora = True
        self._ep_config = None
        self.parallelism_config = SimpleNamespace(
            fp32_non_ep_params=False,
            is_tp_mode=True,
            ep_group_size=1,
            needs_ep_wrappers=False,
        )


def _adapter_model(mesh):
    model = nn.Module()
    for name, a_placement, b_placement in (("q_proj", None, Shard(0)), ("o_proj", Shard(1), None)):
        layer = nn.Module()
        for factor_name, shape, placement in (("lora_A", (2, 4), a_placement), ("lora_B", (4, 2), b_placement)):
            weight = torch.full(shape, 0.5)
            if mesh is not None and placement is not None:
                weight = distribute_tensor(weight, mesh, [placement])
            layer.register_parameter(factor_name, nn.Parameter(weight))
        model.add_module(name, layer)
    model.register_parameter("base_weight", nn.Parameter(torch.ones(4, 4), requires_grad=False))
    return model


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _optimizer_worker(rank, world_size, args):
    torch.set_num_threads(1)
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    model = _adapter_model(mesh)
    trainer = _OptimizerTrainer(model, args)
    requested_class, requested_kwargs = Trainer.get_optimizer_cls_and_kwargs(args, model)
    if args.optim == OptimizerNames.ADAMW_TORCH_FUSED:
        assert requested_class is torch.optim.AdamW
        assert Trainer.get_optimizer_cls_and_kwargs(args)[1]["fused"] is True
        assert requested_kwargs["fused"] is False
        assert requested_kwargs["foreach"] is False

    optimizer = trainer.create_optimizer()
    assert type(optimizer) is requested_class
    assert trainer.create_optimizer() is optimizer
    assert trainer._tp_grad_sync_hook_registered
    trainable = [param for param in model.parameters() if param.requires_grad]
    assert len(trainable) == 4
    assert sum(isinstance(param, DTensor) for param in trainable) == 2
    assert {id(param) for group in optimizer.param_groups for param in group["params"]} == {
        id(param) for param in trainable
    }
    assert len(optimizer.param_groups) == 2
    for group in optimizer.param_groups:
        assert group["foreach"] is False
        assert group["fused"] is False
        assert group["lr"] == args.learning_rate
        assert group["weight_decay"] == args.weight_decay
        assert len({isinstance(param, DTensor) for param in group["params"]}) == 1

    _assert_step_updates_factors(optimizer, model, rank)
    reference = _adapter_model(None)
    reference_optimizer = requested_class(
        [param for param in reference.parameters() if param.requires_grad],
        **{**requested_kwargs, "foreach": False, "fused": False, "weight_decay": args.weight_decay},
    )
    _assign_factor_grads(reference)
    reference_optimizer.step()
    for (name, param), (reference_name, reference_param) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == reference_name
        updated = param.full_tensor() if isinstance(param, DTensor) else param
        torch.testing.assert_close(updated, reference_param, rtol=0, atol=0)


def _assert_step_updates_factors(optimizer, model, rank):
    trainable = [param for param in model.parameters() if param.requires_grad]
    before = [_local(param).detach().clone() for param in trainable]
    frozen_before = model.base_weight.detach().clone()
    _assign_factor_grads(model)
    optimizer.step()
    for param, original in zip(trainable, before, strict=True):
        assert torch.all(_local(param) < original), f"rank {rank}: adapter factor did not update"
    torch.testing.assert_close(model.base_weight, frozen_before, rtol=0, atol=0)


def _assign_factor_grads(model):
    for param in model.parameters():
        if not param.requires_grad:
            continue
        gradient = torch.linspace(0.25, 0.75, param.numel(), dtype=param.dtype).reshape(param.shape)
        param.grad = (
            distribute_tensor(gradient, param.device_mesh, param.placements)
            if isinstance(param, DTensor)
            else gradient
        )


def _supplied_optimizer_worker(rank, world_size, args):
    torch.set_num_threads(1)
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    model = _adapter_model(mesh)
    trainable = [param for param in model.parameters() if param.requires_grad]
    trainer = _OptimizerTrainer(model, args)
    trainer.optimizer_cls_and_kwargs = (torch.optim.SGD, {"lr": args.learning_rate})
    with pytest.raises(ValueError, match="Native TP LoRA does not support optimizer_cls_and_kwargs"):
        trainer.create_optimizer()
    assert trainer.optimizer is None

    for options in (
        {"fused": True, "foreach": False},
        {"fused": False, "foreach": True},
        {"fused": None, "foreach": False},
        {"fused": False, "foreach": None},
    ):
        supplied = torch.optim.AdamW(trainable, lr=args.learning_rate, **options)
        trainer = _OptimizerTrainer(model, args, supplied)
        with pytest.raises(ValueError, match="supplied optimizer's fused/foreach group mixing"):
            trainer.create_optimizer()
        assert trainer.optimizer is supplied
        assert not supplied.state

    for optimizer_class, kwargs in (
        (torch.optim.AdamW, {"fused": False, "foreach": False}),
        (AdamWBF16, {"use_triton": False}),
    ):
        supplied = optimizer_class(trainable, lr=args.learning_rate, **kwargs)
        trainer = _OptimizerTrainer(model, args, supplied)
        assert trainer.create_optimizer() is supplied
        assert trainer.create_optimizer() is supplied
        _assert_step_updates_factors(supplied, model, rank)

    supplied = torch.optim.AdamW(trainable, lr=args.learning_rate, fused=True)
    trainer = _OptimizerTrainer(model, args, supplied)
    trainer._native_tp_lora = False
    assert trainer.create_optimizer() is supplied

    trainer = _OptimizerTrainer(model, args)
    trainer._native_tp_lora = False
    trainer.optimizer_cls_and_kwargs = (torch.optim.SGD, {"lr": args.learning_rate, "foreach": False})
    optimizer = trainer.create_optimizer()
    assert type(optimizer) is torch.optim.SGD
    _assert_step_updates_factors(optimizer, model, rank)


def _arguments(tmp_path, optim=None):
    kwargs = {} if optim is None else {"optim": optim}
    return TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
        learning_rate=0.03,
        weight_decay=0.2,
        max_grad_norm=1.0,
        report_to="none",
        **kwargs,
    )


def _run_workers(worker, args):
    env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    with patch.dict(os.environ, env):
        run_gloo_ranks(worker, 2, 2, args, pg_timeout=timedelta(seconds=60), env=env)


@pytest.mark.parametrize("optim", [None, "sgd"])
def test_native_tp_lora_builds_requested_stock_optimizer(tmp_path, optim):
    args = _arguments(tmp_path, optim)
    if optim is None:
        assert args.optim is OptimizerNames.ADAMW_TORCH_FUSED
    _run_workers(_optimizer_worker, args)


def test_native_tp_lora_supplied_optimizer_contracts(tmp_path):
    _run_workers(_supplied_optimizer_worker, _arguments(tmp_path))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
