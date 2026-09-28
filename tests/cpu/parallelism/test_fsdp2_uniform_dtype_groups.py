#!/usr/bin/env python
"""Every FSDP2 shard group holds one trainable dtype: a mixed layer splits into nested groups.

FSDP2 asserts one original dtype per group at the first forward. ``fp32_router`` at ep1 under
``fsdp_shard_ep1_experts`` puts an fp32 router in its decoder layer's group beside bf16 parameters, so
``apply_fsdp2_per_layer`` gives such a submodule a group of its own first. The split decision is
checked on small module trees; the real wrap runs on a single-rank gloo mesh over a tiny Qwen3 MoE
whose routers hold fp32 masters, with the router called as a module and read in place (as the LFM2,
DeepSeek-V4 and Inkling EP wrappers read ``gate.weight``). There the router's gradient must equal a
plain bf16 model's bit for bit, and each premise is shown: FSDP2 refuses the unsplit layer, and a
router read in place meets its sharded DTensor unless the layer's forward unshards it.

Run: pytest tests/cpu/parallelism/test_fsdp2_uniform_dtype_groups.py
"""

import copy
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from peft.utils import ModulesToSaveWrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

import src.distributed.fsdp as fsdp
from src.distributed.fsdp import IdentityParamSet
from tests.common.models import TINY_QWEN3_MOE_CONFIG

BF16, FP32 = torch.bfloat16, torch.float32
HIDDEN = 8
# The toolkit's bf16 policy: compute and reduce in bf16 over bf16 or fp32 masters.
POLICY = MixedPrecisionPolicy(param_dtype=BF16, reduce_dtype=BF16)
ROUTER_ACCESS = ("called", "read_in_place")


def _linear(dtype: torch.dtype, *, trainable: bool = True) -> nn.Linear:
    linear = nn.Linear(HIDDEN, HIDDEN, bias=False).to(dtype)
    linear.weight.requires_grad_(trainable)
    return linear


class _Experts(nn.Module):
    """An EP-wrapper stand-in: bare expert parameters on the module itself, the router as a child."""

    def __init__(self, router: nn.Module | None = None, router_param: nn.Parameter | None = None):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(2, HIDDEN, HIDDEN, dtype=BF16))
        if router is not None:
            self.gate = router
        if router_param is not None:
            self.router_weight = router_param


def _layer(mlp: nn.Module) -> nn.Module:
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = _linear(BF16)
    layer.norm = _linear(BF16)
    layer.mlp = mlp
    return layer


def _split(layer: nn.Module, left_out=()) -> list[nn.Module]:
    return fsdp._uniform_dtype_split(layer, IdentityParamSet(left_out))


def test_uniform_layer_takes_no_extra_group():
    assert _split(_layer(_Experts(router=_linear(BF16)))) == []


def test_fp32_router_takes_its_own_group():
    router = _linear(FP32)
    assert _split(_layer(_Experts(router=router))) == [router]


def test_frozen_and_left_out_parameters_do_not_split():
    assert _split(_layer(_Experts(router=_linear(FP32, trainable=False)))) == []
    router = _linear(FP32)
    assert _split(_layer(_Experts(router=router)), left_out=[router.weight]) == []


def test_router_owning_no_parameter_splits_whole():
    """Zaya's router holds its projections in child modules: one group for the router, not one each."""
    router = nn.Module()
    router.down_proj = _linear(FP32)
    router.router_mlp = nn.Sequential(_linear(FP32), _linear(FP32))
    assert _split(_layer(_Experts(router=router))) == [router]


def test_modules_to_save_copy_splits_at_the_copy():
    """The trainable fp32 copy is the smallest submodule holding every fp32 parameter; the frozen
    original stays with the layer."""
    wrapper = ModulesToSaveWrapper(_linear(FP32), "default")
    wrapper.original_module.requires_grad_(False)
    copy_module = wrapper.modules_to_save["default"].requires_grad_(True)
    assert _split(_layer(_Experts(router=wrapper))) == [copy_module]


def test_submodule_owning_another_dtype_nests_innermost_first():
    """A module owning fp32 parameters with a bf16 child: the child splits from the module, the
    module from the layer, and the child is wrapped first."""
    inner = _linear(BF16)
    outer = nn.Module()
    outer.scale = nn.Parameter(torch.ones(HIDDEN, dtype=FP32))
    outer.proj = inner
    assert _split(_layer(outer)) == [inner, outer]


def test_layer_keeps_the_dtype_needing_fewest_groups():
    """Two fp32 submodules and one bf16: the layer stays fp32 and the bf16 one splits off."""
    layer = nn.Module()
    layer.a, layer.b, layer.c = _linear(FP32), _linear(FP32), _linear(BF16)
    assert _split(layer) == [layer.c]


def test_module_owning_two_trainable_dtypes_is_refused():
    """A bare fp32 router parameter on the module holding the bf16 experts cannot be separated."""
    layer = _layer(_Experts(router_param=nn.Parameter(torch.zeros(4, HIDDEN, dtype=FP32))))
    with pytest.raises(RuntimeError, match=r"_Experts .* directly owns trainable parameters of dtypes"):
        _split(layer)


@pytest.fixture
def single_rank_mesh(tmp_path):
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


def _read_router_in_place(block: nn.Module) -> None:
    """Route off ``block.gate.weight`` without calling the router module."""

    def forward(hidden_states):
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        probs = F.linear(flat, block.gate.weight).softmax(-1, dtype=FP32)
        weights, experts = probs.topk(block.gate.top_k, dim=-1)
        return block.experts(flat, experts, weights.to(flat.dtype)).view_as(hidden_states)

    block.forward = forward


def _moe_pair(access: str) -> tuple[nn.Module, nn.Module]:
    """A tiny bf16 Qwen3 MoE and a copy whose routers hold fp32 masters of the same values."""
    torch.manual_seed(0)
    reference = Qwen3MoeForCausalLM(Qwen3MoeConfig(**TINY_QWEN3_MOE_CONFIG)).to(BF16).train()
    model = copy.deepcopy(reference)
    for layer in model.model.layers:
        layer.mlp.gate.float()
    if access == "read_in_place":
        for m in (reference, model):
            for layer in m.model.layers:
                _read_router_in_place(layer.mlp)
    return reference, model


def _batch() -> dict[str, torch.Tensor]:
    ids = torch.arange(6, 18).view(1, -1)
    return {"input_ids": ids, "labels": ids.clone()}


def _wrap(model: nn.Module, mesh) -> None:
    fsdp.apply_fsdp2_per_layer(model, mesh, POLICY, False, IdentityParamSet())


@pytest.mark.parametrize("access", ROUTER_ACCESS)
def test_fp32_router_trains_as_its_own_group(access, single_rank_mesh):
    reference, model = _moe_pair(access)
    _wrap(model, single_rank_mesh)

    layers = list(model.model.layers)
    units = [m for m in model.modules() if isinstance(m, FSDPModule)]
    assert units == [model, model.model, *(u for layer in layers for u in (layer, layer.mlp.gate))]
    for layer in layers:
        weight = layer.mlp.gate.weight
        assert isinstance(weight, DTensor) and weight.dtype == FP32

    loss = model(**_batch()).loss
    loss.backward()
    reference_loss = reference(**_batch()).loss
    reference_loss.backward()
    assert torch.equal(loss.detach(), reference_loss.detach())
    for layer, reference_layer in zip(layers, reference.model.layers, strict=True):
        grad = layer.mlp.gate.weight.grad
        assert isinstance(grad, DTensor) and grad.dtype == FP32
        assert torch.equal(grad.full_tensor(), reference_layer.mlp.gate.weight.grad.float())


def test_uniform_model_takes_no_extra_group(single_rank_mesh):
    reference, _ = _moe_pair("called")
    _wrap(reference, single_rank_mesh)
    units = [m for m in reference.modules() if isinstance(m, FSDPModule)]
    assert units == [reference, reference.model, *reference.model.layers]


def test_premise_fsdp2_refuses_the_unsplit_layer(single_rank_mesh):
    _, model = _moe_pair("called")
    with patch.object(fsdp, "_uniform_dtype_split", return_value=[]):
        _wrap(model, single_rank_mesh)
    with pytest.raises(AssertionError, match="uniform original parameter dtype"):
        model(**_batch())


def test_premise_router_read_in_place_needs_the_layer_unshard(single_rank_mesh):
    _, model = _moe_pair("read_in_place")
    with patch.object(fsdp, "_unshard_with_layer"):
        _wrap(model, single_rank_mesh)
    with pytest.raises(RuntimeError, match="mixed torch.Tensor and DTensor"):
        model(**_batch())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
