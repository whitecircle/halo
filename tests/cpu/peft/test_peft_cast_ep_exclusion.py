#!/usr/bin/env python
"""``_cast_peft_params_to_compute_dtype`` must not downcast EP-owned fp32 trainables.

EP layers are FSDP-ignored at ``ep_group_size>1``, so the one-dtype-per-``fully_shard``-group
constraint the cast serves never applies to them — they own their precision. ``ep_fp32_router`` with
a PEFT ``modules_to_save`` router copy, and ``ep_fp32_experts`` with grouped expert-LoRA, hold EP
trainables in fp32 deliberately; a bf16 downcast crashes the fp32 router forward (``F.linear`` dtype
mismatch at step 1) or silently negates the experts flag. Non-EP adapter params must still be cast
(that is the constraint the method exists for).

Uses a real EP family layer (LFM2) at single-process ep1 and a real ``PeftModel`` wrap, so the
exclusion is exercised by module membership on production structures, not name matching.

Run: ``python tests/cpu/peft/test_peft_cast_ep_exclusion.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from peft import LoraConfig, get_peft_model

from src.distributed.expert_parallel.layers.lfm2 import EPLfm2MoELayer
from src.trainers.mixins.base import DistributedTrainerMixin
from tests.common.parallelism import single_process_ep_config

E, H, M, K = 4, 8, 16, 2  # experts, hidden, intermediate, top_k


class _Lfm2Gate(nn.Module):
    # Mirrors Lfm2MoeTopKRouter's attribute surface: the wrapper reads the routing constants off
    # the gate without defaults, so the stub must carry them.
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
        self.moe = EPLfm2MoELayer(_Lfm2Block(), single_process_ep_config(E, fsdp_shard_ep1_experts=False)).cpu()

    def forward(self, x):  # pragma: no cover - never called
        return self.moe(self.proj(x))


def _fake_trainer(model: nn.Module) -> types.SimpleNamespace:
    """Minimal stand-in exposing what the cast touches; the production methods are bound below so
    the EP-module discovery and the cast itself are the real mixin code, not a mock."""
    me = types.SimpleNamespace(
        model=model,
        _ep_config=None,
        parallelism_config=types.SimpleNamespace(fp32_non_ep_params=False),
        args=types.SimpleNamespace(bf16=True, fp16=False),
    )
    me._find_ep_modules = types.MethodType(DistributedTrainerMixin._find_ep_modules, me)
    return me


def test_cast_skips_ep_owned_fp32_trainables_but_casts_adapters():
    PartialState()  # the cast logs through accelerate's logger
    torch.manual_seed(0)
    peft_model = get_peft_model(
        _TinyMoEModel(), LoraConfig(r=4, lora_alpha=8, target_modules=["proj"], task_type=None)
    )
    moe = peft_model.base_model.model.moe

    # The EP layer holds these in fp32 deliberately (ep_fp32_router modules_to_save copy /
    # ep_fp32_experts grouped expert-LoRA); PEFT's global freeze is re-lifted for them in production.
    moe.gate.weight.requires_grad_(True)
    moe.gate_up_proj.requires_grad_(True)
    assert moe.gate.weight.dtype == torch.float32
    assert moe.gate_up_proj.dtype == torch.float32

    me = _fake_trainer(peft_model)
    DistributedTrainerMixin._cast_peft_params_to_compute_dtype(me)

    # EP-owned trainables survive in fp32 — the crash/negation the exclusion prevents.
    assert moe.gate.weight.dtype == torch.float32, (
        "EP router trainable was downcast to bf16 — fp32 router input would crash F.linear at step 1"
    )
    assert moe.gate_up_proj.dtype == torch.float32, (
        "EP expert trainable was downcast to bf16 — silently negates ep_fp32_experts"
    )

    # Non-EP adapters are still aligned to the compute dtype (the constraint the cast serves).
    lora_dtypes = {p.dtype for n, p in peft_model.named_parameters() if "lora_" in n and p.requires_grad}
    assert lora_dtypes == {torch.bfloat16}, f"non-EP LoRA adapters not cast to bf16: {lora_dtypes}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
