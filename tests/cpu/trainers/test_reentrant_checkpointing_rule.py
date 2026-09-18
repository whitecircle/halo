#!/usr/bin/env python
"""Which runs force reentrant gradient checkpointing before the trainer enables it.

Non-reentrant checkpointing re-runs the checkpointed forward lazily and checks every recomputed tensor's
metadata against the forward's. A MoE routes in that recompute: the router runs on a hidden state most
attention kernels recompute nondeterministically, a near-tie pick flips, and the routing tensors change
shape — a rejected recompute, with or without EP wrappers. CP's all-to-alls and EP's DeepEP barriers
have their own reasons. Pipeline parallelism is the one mode that needs the non-reentrant form.

Run: pytest tests/cpu/trainers/test_reentrant_checkpointing_rule.py
"""

import pytest
from transformers import Qwen3Config, Qwen3MoeConfig

from src.trainers.mixins.base import forces_reentrant_checkpointing
from tests.common.models import TINY_QWEN3_CONFIG, TINY_QWEN3_MOE_CONFIG
from tests.common.parallelism import make_parallelism_config

DENSE = Qwen3Config(**TINY_QWEN3_CONFIG)
MOE = Qwen3MoeConfig(**TINY_QWEN3_MOE_CONFIG)


def _config(**kwargs):
    kwargs.setdefault("world_size", 4)
    kwargs.setdefault("gpus_per_node", 4)
    return make_parallelism_config(**kwargs)


def test_a_dense_data_parallel_run_keeps_the_configured_form():
    assert forces_reentrant_checkpointing(_config(), DENSE) is False
    assert forces_reentrant_checkpointing(_config(), None) is False


def test_a_moe_forces_reentrant_even_without_expert_parallelism():
    """The stock-experts path (``use_grouped_gemm: false``, no EP wrappers) routes in its recompute too."""
    assert forces_reentrant_checkpointing(_config(), MOE) is True


@pytest.mark.parametrize("axis", ["ep_size", "cp_size"])
def test_expert_and_context_parallelism_force_reentrant(axis):
    assert forces_reentrant_checkpointing(_config(**{axis: 2}), DENSE) is True


def test_pipeline_parallelism_keeps_non_reentrant_even_for_a_moe():
    assert forces_reentrant_checkpointing(_config(pp_size=2, gpus_per_node=2), MOE) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
