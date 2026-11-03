#!/usr/bin/env python
"""EP grad sync is a no-op on a single-process launch, which still builds EP layers at ``ep_size=1``.
Otherwise the first backward crashes with "Default process group has not been initialized"."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from halo.distributed.expert_parallel.families.lfm2 import EPLfm2MoELayer
from tests.common.parallelism import single_process_ep_config

E, H, M, K = 4, 8, 16, 2  # experts, hidden, intermediate, top_k


def test_a_single_process_run_has_no_grad_sync_peers():
    assert not single_process_ep_config(E, experts_fsdp_managed=False).has_grad_sync_peers


def test_expert_sweep_not_armed_without_process_group():
    """The sweep's collectives have no group."""
    ep_config = single_process_ep_config(E, experts_fsdp_managed=False)
    assert not ep_config.runs_post_backward_sweep
    assert not ep_config.sweep_reduces_across_replicas


class _Lfm2Gate(nn.Module):
    # Mirrors Lfm2MoeTopKRouter: the wrapper reads the routing constants off the gate without defaults.
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
    torch.manual_seed(0)
    layer = EPLfm2MoELayer(_Lfm2Block(), single_process_ep_config(E, experts_fsdp_managed=False)).cpu()
    layer.train()

    out = layer(torch.randn(2, 3, H))
    out.sum().backward()  # A live collective raises "Default process group has not been initialized".

    assert layer.gate.weight.grad is not None, "router grad missing after backward"
    assert layer.gate_up_proj.grad is not None, "expert grad missing after backward"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
