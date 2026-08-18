#!/usr/bin/env python
"""CPU tests for the EPConfig expert-divisibility contract.

``EPConfig.finalize_expert_assignment`` must REJECT ``num_experts % ep_size != 0``: DeepEP dispatch maps
tokens to ranks by a uniform expert→rank division, so an uneven per-rank assignment could not be
honored — tokens would reach the wrong rank's experts or be silently dropped by the local
valid-index filter (``EPMoELayerBase._sort_tokens_for_grouped_mm``). An uneven split hands low ranks
one extra expert, silently.

Run: ``python tests/cpu/parallelism/test_ep_config_expert_divisibility.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest

from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.lazy_loader import EPWeightPlanner


def _config(ep_size: int, **kwargs) -> EPConfig:
    # Distributed is not initialized → single-process mode (no process groups created).
    return EPConfig(ep_size=ep_size, world_size=ep_size, gpus_per_node=ep_size, **kwargs)


def test_dividing_num_experts_finalizes_uniform_assignment():
    cfg = _config(ep_size=8)
    cfg.finalize_expert_assignment(128)
    assert cfg.num_experts == 128
    assert cfg.experts_per_rank == 16
    assert cfg.expert_start_idx == 0  # dispatch_ep_rank 0 in single-process mode
    assert cfg.expert_end_idx == 16


def test_assignment_is_uniform_per_dispatch_rank():
    cfg = _config(ep_size=8)
    cfg.dispatch_ep_rank = 3
    cfg.finalize_expert_assignment(128)
    assert cfg.experts_per_rank == 16
    assert cfg.expert_start_idx == 48
    assert cfg.expert_end_idx == 64


@pytest.mark.parametrize(("ep_size", "num_experts"), [(8, 100), (4, 6), (8, 12)])
def test_non_dividing_num_experts_raises(ep_size: int, num_experts: int):
    cfg = _config(ep_size=ep_size)
    with pytest.raises(ValueError, match="divisible by ep_size"):
        cfg.finalize_expert_assignment(num_experts)
    # Nothing was partially finalized.
    assert cfg.num_experts is None
    assert cfg.experts_per_rank is None


def test_divisor_is_ep_size_not_ep_group_size_under_expert_tp():
    # ETP: distribution size stays ep_size (2); ep_group_size = 4 would wrongly reject 6 experts.
    cfg = EPConfig(ep_size=2, expert_tp_size=2, world_size=4, gpus_per_node=4)
    cfg.finalize_expert_assignment(6)
    assert cfg.experts_per_rank == 3
    with pytest.raises(ValueError, match="divisible by ep_size"):
        cfg.finalize_expert_assignment(7)


def test_expert_assignment_fields_exist_before_finalization():
    """All four expert-assignment fields are declared at construction, not conjured by
    ``finalize_expert_assignment`` — so reading one before finalization sees ``None`` rather than
    raising ``AttributeError``, and the loader can tell "not finalized" from "no EP"."""
    cfg = _config(ep_size=2)

    assert (cfg.num_experts, cfg.experts_per_rank, cfg.expert_start_idx, cfg.expert_end_idx) == (
        None,
        None,
        None,
        None,
    )


def test_planner_refuses_an_unfinalized_config():
    """An unfinalized config must not plan a load: every expert key would classify as REPLICATE, so
    each rank would materialize the full unsharded expert set on a job that asked for EP. "Shard no
    experts" is spelled ``ep_config=None``, which stays legal."""
    with pytest.raises(RuntimeError, match="not finalized"):
        EPWeightPlanner(_config(ep_size=2))

    assert EPWeightPlanner(None).start is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
