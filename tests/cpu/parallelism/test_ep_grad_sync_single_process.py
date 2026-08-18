#!/usr/bin/env python
"""CPU tests: EP grad-sync hooks must be no-ops on a single-process launch.

A bare ``python`` run (dist never initialized) still builds EP wrappers at ``ep_size=1`` (grouped
GEMM default); with ``fsdp_shard_ep1_experts=False`` the layer registers its router/expert
post-accumulate hooks. A router hook built without consulting the live process state issues
``dist.all_reduce(group=None)``, and the first backward crashes with "Default process group has not
been initialized". The hooks must detect the single-process state and no-op.

Run: ``python tests/cpu/parallelism/test_ep_grad_sync_single_process.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.distributed.expert_parallel.grad_sync import (
    create_expert_grad_hook,
    create_router_grad_hook,
    has_grad_sync_peers,
)
from src.distributed.expert_parallel.layers.lfm2 import EPLfm2MoELayer
from tests.common.parallelism import single_process_ep_config

E, H, M, K = 4, 8, 16, 2  # experts, hidden, intermediate, top_k


def _param_with_grad() -> nn.Parameter:
    param = nn.Parameter(torch.randn(4, 4))
    param.grad = torch.randn(4, 4)
    return param


def test_router_hook_noop_without_process_group():
    """The router hook must not touch dist collectives when no process group exists."""
    ep_config = single_process_ep_config(E, fsdp_shard_ep1_experts=False)
    assert not has_grad_sync_peers(ep_config), "a single-process run must report no grad-sync peers"

    param = _param_with_grad()
    before = param.grad.clone()
    create_router_grad_hook(ep_config)(param)  # a live collective: ValueError from all_reduce(group=None)
    assert torch.equal(param.grad, before), "single-process router hook must leave the grad untouched"


def test_expert_hook_noop_without_process_group():
    param = _param_with_grad()
    before = param.grad.clone()
    create_expert_grad_hook(single_process_ep_config(E, fsdp_shard_ep1_experts=False))(param)
    assert torch.equal(param.grad, before), "single-process expert hook must leave the grad untouched"


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


def test_first_backward_survives_single_process_launch():
    """End-to-end over the crash path: a real EP family layer built at ep1 single-process with
    ``fsdp_shard_ep1_experts=False`` registers its grad-sync hooks; the first backward fires them.
    A live collective in the router hook raises here."""
    torch.manual_seed(0)
    layer = EPLfm2MoELayer(_Lfm2Block(), single_process_ep_config(E, fsdp_shard_ep1_experts=False)).cpu()
    layer.train()

    out = layer(torch.randn(2, 3, H))
    out.sum().backward()  # a live collective raises "Default process group has not been initialized"

    assert layer.gate.weight.grad is not None, "router grad missing after backward"
    assert layer.gate_up_proj.grad is not None, "expert grad missing after backward"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
