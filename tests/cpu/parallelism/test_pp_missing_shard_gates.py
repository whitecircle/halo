#!/usr/bin/env python
"""The PP stage loader's missing-shard gate for per-expert fusion tasks.

Per-expert checkpoint keys never ride through the plan list (they map to no fused model key), so the
plan-level missing-shard gate cannot see them. Ungated, a wrong-topology per-node resume of an
INDIVIDUAL-format checkpoint dies rank-local in the fuser's ``safe_open`` between two collectives,
stranding the other stages until the NCCL watchdog. :func:`_missing_fusion_shards_reason` routes that
verdict through ``reject_across_ranks`` instead.

Run: ``python tests/cpu/parallelism/test_pp_missing_shard_gates.py`` (or ``pytest -m cpu``).
"""

import sys

import pytest

from src.distributed.pipeline_parallel.lazy_loader import _missing_fusion_shards_reason


def _task(shard_file: str, expert_idx: int = 0):
    """One fusion task in the fuser's shape: (model_key, fusion_type, {idx: {suffix: (disk, shard)}})."""
    return (
        "model.layers.0.mlp.experts.gate_up_proj",
        "gate_up",
        {expert_idx: {"gate_proj.weight": (f"model.layers.0.mlp.experts.{expert_idx}.gate_proj.weight", shard_file)}},
    )


def test_missing_fusion_shard_is_rejected_with_the_wrong_topology_message():
    reason = _missing_fusion_shards_reason([_task("model-00002.safetensors")], ["model-00002.safetensors"], "/ckpt", 1)
    assert reason is not None
    assert "model-00002.safetensors" in reason
    assert "PP stage 1" in reason
    assert "topology" in reason


def test_present_shards_pass():
    assert (
        _missing_fusion_shards_reason([_task("model-00001.safetensors")], ["model-00002.safetensors"], "/c", 0) is None
    )


def test_no_tasks_or_no_missing_files_pass():
    assert _missing_fusion_shards_reason(None, ["model-00002.safetensors"], "/c", 0) is None
    assert _missing_fusion_shards_reason([], ["model-00002.safetensors"], "/c", 0) is None
    assert _missing_fusion_shards_reason([_task("model-00001.safetensors")], [], "/c", 0) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
