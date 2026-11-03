#!/usr/bin/env python
"""``_cast_peft_params_to_compute_dtype`` must not downcast EP-owned fp32 trainables; non-EP adapters are still cast.
A bf16 downcast crashes the fp32 router forward or silently negates ``ep_fp32_experts``."""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from halo.distributed.expert_parallel.families.lfm2 import EPLfm2MoELayer
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.parallelism import single_process_ep_config

E, H, M, K = 4, 8, 16, 2  # experts, hidden, intermediate, top_k


class _Lfm2Gate(nn.Module):
    # The wrapper reads the routing constants off the gate without defaults, as on Lfm2MoeTopKRouter.
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(E, H))
        self.num_experts = E
        self.top_k = K
        self.norm_topk_prob = True
        self.routed_scaling_factor = 1.0
        self.use_expert_bias = False

    def forward(self, x):
        return F.linear(x, self.weight)


class _Lfm2Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _Lfm2Gate()
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(torch.randn(E, 2 * M, H))
        self.experts.down_proj = nn.Parameter(torch.randn(E, H, M))
        self.top_k = K
        self.norm_topk_prob = True
        self.routed_scaling_factor = 1.0
        self.use_expert_bias = False


class _TinyMoEModel(nn.Module):
    """One LoRA-targetable linear + one real EP layer, mirroring an attention-PEFT + EP run."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(H, H, bias=False)
        self.moe = EPLfm2MoELayer(_Lfm2Block(), single_process_ep_config(E, experts_fsdp_managed=False)).cpu()

    def forward(self, x):  # pragma: no cover - never called
        return self.moe(self.proj(x))


def _fake_trainer(model: nn.Module) -> types.SimpleNamespace:
    """Exposes what the cast touches; the production methods are bound below, so the cast is the real mixin code."""
    me = types.SimpleNamespace(
        model=model,
        _ep_config=None,
        _param_id_caches={},
        parallelism_config=types.SimpleNamespace(fp32_non_ep_params=False),
        args=types.SimpleNamespace(bf16=True, fp16=False),
    )
    me._ep_layers = types.MethodType(DistributedTrainerMixin._ep_layers, me)
    me._memoized_param_ids = types.MethodType(DistributedTrainerMixin._memoized_param_ids, me)
    me._get_ep_param_ids = types.MethodType(DistributedTrainerMixin._get_ep_param_ids, me)
    return me


def test_cast_skips_ep_owned_fp32_trainables_but_casts_adapters():
    torch.manual_seed(0)
    peft_model = get_peft_model(
        _TinyMoEModel(), LoraConfig(r=4, lora_alpha=8, target_modules=["proj"], task_type=None)
    )
    moe = peft_model.base_model.model.moe

    # Held in fp32 deliberately; in production PEFT's global freeze is re-lifted for them.
    moe.gate.weight.requires_grad_(True)
    moe.gate_up_proj.requires_grad_(True)
    assert moe.gate.weight.dtype == torch.float32
    assert moe.gate_up_proj.dtype == torch.float32

    me = _fake_trainer(peft_model)
    DistributedTrainerMixin._cast_peft_params_to_compute_dtype(me)

    assert moe.gate.weight.dtype == torch.float32, (
        "EP router trainable was downcast to bf16 — fp32 router input would crash F.linear at step 1"
    )
    assert moe.gate_up_proj.dtype == torch.float32, (
        "EP expert trainable was downcast to bf16 — silently negates ep_fp32_experts"
    )

    lora_dtypes = {p.dtype for n, p in peft_model.named_parameters() if "lora_" in n and p.requires_grad}
    assert lora_dtypes == {torch.bfloat16}, f"non-EP LoRA adapters not cast to bf16: {lora_dtypes}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
