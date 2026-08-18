#!/usr/bin/env python
"""CPU tests for the zero-patched-layers outcome of ``patch_moe_model_for_ep``.

When EP/ETP distribution is requested but NO MoE block matches a registered wrapper, the run would
otherwise proceed with replicated, un-synced experts while reporting a data-parallel size that
assumes they are sharded — it must RAISE (this mirrors the TP loader's zero-sharded-modules gate
and the CP patcher's zero-wrapped-layers gate).

Grouped-GEMM-only wrapping (``ep_group_size == 1``) is an optimization, so an unwrapped family there
keeps the stock HF expert compute and only warns.

Run: ``python tests/cpu/parallelism/test_ep_patch_zero_layers.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.patching import patch_moe_model_for_ep


def _model_without_moe() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))


def _ep_config(ep_size: int, expert_tp_size: int = 1) -> EPConfig:
    world = ep_size * expert_tp_size
    return EPConfig(
        ep_size=ep_size,
        world_size=world,
        gpus_per_node=world,
        expert_tp_size=expert_tp_size,
        use_grouped_gemm=False,
    )


@pytest.mark.parametrize(("ep_size", "expert_tp_size"), [(2, 1), (1, 2)])
def test_distributed_ep_with_zero_patched_layers_raises(ep_size, expert_tp_size):
    cfg = _ep_config(ep_size, expert_tp_size)
    assert cfg.ep_group_size > 1
    with pytest.raises(ValueError, match="NO MoE layer was patched"):
        patch_moe_model_for_ep(_model_without_moe(), cfg)


def test_grouped_gemm_only_with_zero_patched_layers_warns_and_returns(caplog):
    cfg = _ep_config(ep_size=1)
    assert cfg.ep_group_size == 1
    model = _model_without_moe()
    with caplog.at_level("WARNING"):
        returned = patch_moe_model_for_ep(model, cfg)
    assert returned is model
    assert any("No MoE layers found" in rec.message for rec in caplog.records)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
