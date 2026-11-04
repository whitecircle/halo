#!/usr/bin/env python
"""``apply_ep_layers`` with zero patched layers raises under EP/ETP, or experts run replicated and un-synced at a DP
size assuming they are sharded. Grouped-GEMM-only wrapping is an optimization, so there it only warns."""

from __future__ import annotations

import pytest
import torch.nn as nn

from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.expert_parallel.parallelize import apply_ep_layers


def _model_without_moe() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))


def _ep_config(ep_size: int, expert_tp_size: int = 1) -> EPConfig:
    world = ep_size * expert_tp_size
    return EPConfig(
        ep_size=ep_size,
        stage_world_size=world,
        nvlink_domain_size=world,
        expert_tp_size=expert_tp_size,
        use_grouped_gemm=False,
        experts_fsdp_managed=False,
    )


@pytest.mark.parametrize(("ep_size", "expert_tp_size"), [(2, 1), (1, 2)])
def test_distributed_ep_with_zero_patched_layers_raises(ep_size, expert_tp_size):
    cfg = _ep_config(ep_size, expert_tp_size)
    assert cfg.ep_group_size > 1
    with pytest.raises(ValueError, match="NO MoE layer was patched"):
        apply_ep_layers(_model_without_moe(), cfg)


def test_grouped_gemm_only_with_zero_patched_layers_warns_and_returns(caplog):
    cfg = _ep_config(ep_size=1)
    assert cfg.ep_group_size == 1
    model = _model_without_moe()
    with caplog.at_level("WARNING"):
        returned = apply_ep_layers(model, cfg)
    assert returned is model
    assert any("No MoE layers found" in rec.message for rec in caplog.records)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
