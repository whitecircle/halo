#!/usr/bin/env python
"""An ``fp32_router`` EP layer feeds its router at the dtype the live weight holds at the forward, which
``experts_fsdp_managed`` does not decide: FSDP2 skips one-rank jobs and frozen routers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_leaves

from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.fsdp import _apply_fsdp2, apply_fsdp2_for_dp
from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.parallelism.mode_setup import ModeSetupMixin
from tests.common.parallelism import make_parallelism_config
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

BF16, FP32 = torch.bfloat16, torch.float32
# ``a @ b`` reaches a torch function mode as ``Tensor.matmul``.
_PROJECTIONS = (F.linear, torch.matmul, torch.Tensor.matmul)
# Casts a router may apply to its own weight before projecting.
_CASTS = (torch.Tensor.to, torch.Tensor.type, torch.Tensor.type_as, torch.Tensor.float, torch.Tensor.bfloat16)


class _RouterProjections(TorchFunctionMode):
    """The parameters as each EP layer's forward sees them, since FSDP2 swaps in its unsharded ones."""

    def __init__(self):
        super().__init__()
        # Held, since a freed cast copy's id could be reused by an unrelated tensor.
        self.router_tensors: dict[int, torch.Tensor] = {}
        self.weight_dtypes: set[torch.dtype] = set()
        self.projections: list[tuple[torch.dtype, set[torch.dtype]]] = []
        self.routed_layers: set[int] = set()

    def watch(self, model: nn.Module) -> None:
        for _, layer in named_ep_layers(model):
            layer.register_forward_pre_hook(self._capture_router)

    def _capture_router(self, layer: nn.Module, _args) -> None:
        params = list(layer._live_router_module().parameters())
        self.router_tensors.update({id(p): p for p in params})
        self.weight_dtypes |= {p.dtype for p in params}
        self.routed_layers.add(id(layer))

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _PROJECTIONS:
            tensors = [t for t in tree_leaves((args, kwargs)) if isinstance(t, torch.Tensor)]
            weights = [t for t in tensors if id(t) in self.router_tensors]
            if weights:
                others = {t.dtype for t in tensors if id(t) not in self.router_tensors and t.is_floating_point()}
                self.projections.append((weights[0].dtype, others))
        result = func(*args, **kwargs)
        if func in _CASTS and id(args[0]) in self.router_tensors and isinstance(result, torch.Tensor):
            self.router_tensors[id(result)] = result
        return result


def _one_rank_config(**fields) -> ParallelismConfig:
    return make_parallelism_config(world_size=1, gpus_per_node=1, ep_fp32_router=True, **fields)


def _ep_model(family: str, ep_config: EPConfig) -> nn.Module:
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_FAMILIES[family]).to(BF16).train()
    apply_ep_layers(model, ep_config)
    if all(layer._ROUTER_ATTR is None for _, layer in named_ep_layers(model)):
        pytest.skip(f"{family}'s router sits outside the EP layer: fp32_router has no effect there")
    return model


def _forward_recording(model: nn.Module) -> _RouterProjections:
    recorder = _RouterProjections()
    recorder.watch(model)
    ids = torch.arange(3, 15).view(1, -1)
    with recorder:
        model(input_ids=ids, labels=ids.clone())
    return recorder


def _assert_router_projects_at(recorder: _RouterProjections, model: nn.Module, dtype: torch.dtype) -> None:
    assert recorder.routed_layers == {id(layer) for _, layer in named_ep_layers(model)}, "an EP layer never ran"
    assert recorder.weight_dtypes == {dtype}, f"premise: the router weight is {recorder.weight_dtypes} at the forward"
    assert recorder.projections, "no projection took a router weight — the operand check below would be vacuous"
    mismatched = [(weight, others) for weight, others in recorder.projections if others - {weight}]
    assert not mismatched, f"router projections met operands of another dtype: {mismatched}"


def _fsdp2_wrap(model: nn.Module, mesh, ignored_params: list[nn.Parameter] | None = None) -> None:
    """Per-layer FSDP2 under the production bf16 policy."""
    _apply_fsdp2(
        model, mesh, compute_dtype=BF16, fp32_reduce=False, topology="one rank", ignored_params=ignored_params
    )
    assert any(isinstance(module, FSDPModule) for module in model.modules()), "premise: FSDP2 wrapped the model"


@pytest.fixture
def single_rank_mesh(tmp_path):
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("shard_ep1_experts", [True, False], ids=["experts-fsdp-managed", "fsdp-ignored"])
@pytest.mark.parametrize("family", TINY_MOE_FAMILIES)
def test_a_router_no_fsdp2_group_casts_projects_in_fp32(family, shard_ep1_experts):
    """Under the managed verdict the DP wrap at width one wraps nothing, so the router keeps its fp32 master."""
    config = _one_rank_config(fsdp_shard_ep1_experts=shard_ep1_experts)
    assert config.experts_fsdp_managed is shard_ep1_experts
    model = _ep_model(family, config.ep_config())
    assert not apply_fsdp2_for_dp(model, config.stage_world_size, compute_dtype=BF16, fp32_reduce=False)
    _assert_router_projects_at(_forward_recording(model), model, FP32)


@pytest.mark.parametrize("family", TINY_MOE_FAMILIES)
def test_a_router_fsdp2_unshards_projects_at_the_policy_dtype(family, single_rank_mesh):
    config = _one_rank_config(fsdp_shard_ep1_experts=True)
    model = _ep_model(family, config.ep_config())
    _fsdp2_wrap(model, single_rank_mesh)
    _assert_router_projects_at(_forward_recording(model), model, BF16)


@pytest.mark.parametrize("family", TINY_MOE_FAMILIES)
def test_a_frozen_router_the_wrap_leaves_out_projects_in_fp32(family, single_rank_mesh):
    """It shares no trainable parameter's dtype, so the FSDP2 exclusions leave it out and no policy casts it."""
    config = _one_rank_config(fsdp_shard_ep1_experts=True)
    model = _ep_model(family, config.ep_config())
    ep_layers = [layer for _, layer in named_ep_layers(model)]
    routers = [p for layer in ep_layers for p in layer._live_router_module().requires_grad_(False).parameters()]
    trainer = SimpleNamespace(model=model, parallelism_config=config, _ep_layers=lambda: ep_layers)
    trainer._dtype_excluded_params = lambda: ModeSetupMixin._dtype_excluded_params(trainer)
    excluded = ModeSetupMixin._fsdp_excluded_params(trainer)
    left_out = {id(p) for p in excluded}
    assert all(id(p) in left_out for p in routers), "premise: the exclusions leave every frozen router param out"
    _fsdp2_wrap(model, single_rank_mesh, excluded)
    _assert_router_projects_at(_forward_recording(model), model, FP32)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
