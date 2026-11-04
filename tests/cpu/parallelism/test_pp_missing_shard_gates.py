#!/usr/bin/env python
"""The PP stage loader's missing-shard gate for per-expert fusion tasks. Those keys skip the plan list, so a
wrong-topology resume dies rank-local in ``safe_open`` between collectives, stranding other stages."""

import pytest

from halo.model_loading.parallel.expert_fuser import FusionTask
from halo.model_loading.parallel.pp import _missing_fusion_shards_reason


def _task(shard_file: str, expert_idx: int = 0):
    return FusionTask(
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
    raise SystemExit(pytest.main([__file__, "-v"]))
