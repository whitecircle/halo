#!/usr/bin/env python
"""The order of ``ParallelismConfig``'s gates: each row breaks two adjacent gates and pins the earlier
one's message, the one a user acts on, and breaks the later gate alone to prove the overlap is real.

Run: python tests/cpu/parallelism/test_parallelism_config_gate_order.py
"""

import pytest

from src.distributed.parallelism_config import ParallelismConfig
from tests.common.parallelism import create_config, simulated_world

# (shape breaking both gates, the earlier gate's message head, shape breaking only the later gate,
# the later gate's message head) on 8-GPU nodes, world 8 unless a row says otherwise.
GATE_OVERLAPS = (
    pytest.param(
        {"ep_scope": "rack", "tp_size": 2, "cp_size": 2},
        "ep_scope must be one of",
        {"tp_size": 2, "cp_size": 2},
        "TP + CP is not a supported parallelism combination",
        id="field-values-before-the-axis-allowlist",
    ),
    pytest.param(
        {"ep_size": 0, "tp_size": 2, "cp_size": 2},
        "expert_parallel_size must be >= 1",
        {"tp_size": 2, "cp_size": 2},
        "TP + CP is not a supported parallelism combination",
        id="axis-sizes-before-the-axis-allowlist",
    ),
    pytest.param(
        {"world_size": 12, "tp_size": 2, "cp_size": 2},
        "TP + CP is not a supported parallelism combination",
        {"world_size": 12},
        "world_size (12) must be divisible by gpus_per_node (8).",
        id="axis-allowlist-before-the-node-width",
    ),
    pytest.param(
        {"world_size": 12, "nvlink_domain_size": -8},
        "world_size (12) must be divisible by gpus_per_node (8).",
        {"nvlink_domain_size": -8},
        "nvlink_domain_size must be positive",
        id="node-width-before-the-domain-sign",
    ),
    pytest.param(
        {"gpus_per_node": -2, "nvlink_domain_size": -8},
        "gpus_per_node resolved to -2",
        {"nvlink_domain_size": -8},
        "nvlink_domain_size must be positive",
        id="node-width-sign-before-the-domain-sign",
    ),
    pytest.param(
        {"nvlink_domain_size": -12},
        "nvlink_domain_size must be positive",
        {"world_size": 16, "nvlink_domain_size": 12},
        "nvlink_domain_size (12) must be a multiple of gpus_per_node (8).",
        id="domain-sign-before-whole-node-domains",
    ),
    pytest.param(
        {"world_size": 16, "nvlink_domain_size": 20},
        "NVLINK_DOMAIN_SIZE (20) exceeds this job's world_size (16) on a multi-node job.",
        {"world_size": 16, "nvlink_domain_size": 12},
        "nvlink_domain_size (12) must be a multiple of gpus_per_node (8).",
        id="domain-past-a-multi-node-world-before-whole-node-domains",
    ),
    pytest.param(
        {"world_size": 16, "nvlink_domain_size": 12},
        "nvlink_domain_size (12) must be a multiple of gpus_per_node (8).",
        {"world_size": 24, "nvlink_domain_size": 16},
        "world_size (24) must be divisible by nvlink_domain_size (16);",
        id="whole-node-domains-before-whole-domain-worlds",
    ),
    pytest.param(
        {"world_size": 24, "nvlink_domain_size": 16, "pp_size": 5},
        "world_size (24) must be divisible by nvlink_domain_size (16);",
        {"world_size": 24, "pp_size": 5},
        "world_size (24) must be divisible by pipeline_parallel_size (5):",
        id="whole-domain-worlds-before-whole-stage-worlds",
    ),
    pytest.param(
        {"world_size": 24, "pp_size": 5, "ep_size": 3},
        "world_size (24) must be divisible by pipeline_parallel_size (5):",
        {"world_size": 24, "pp_size": 3, "ep_size": 3},
        "EP group size (3) must divide the NVLink domain (8).",
        id="whole-stage-worlds-before-the-composition-rules",
    ),
    pytest.param(
        {"pp_split": [1], "ep_size": 3},
        "pipeline_split=[1] is only meaningful with pipeline_parallel_size > 1",
        {"ep_size": 3},
        "EP group size (3) must divide the NVLink domain (8).",
        id="pipeline-rules-before-the-ep-group",
    ),
    pytest.param(
        {"ep_size": 3, "cp_size": 3},
        "EP group size (3) must divide the NVLink domain (8).",
        {"cp_size": 3},
        "context_parallel_size (3) must divide the NVLink domain (8)",
        id="ep-group-before-cp-locality",
    ),
    pytest.param(
        {"ep_size": 3, "tp_size": 3},
        "EP group size (3) must divide the NVLink domain (8).",
        {"tp_size": 3},
        "tensor_parallel_size (3) must divide world size (8)",
        id="ep-group-before-the-tp-rules",
    ),
    pytest.param(
        {"ep_size": 2, "expert_tp_size": 3},
        "EP group size (6) must divide the NVLink domain (8).",
        {"world_size": 48, "ep_size": 2, "expert_tp_size": 3, "ep_scope": "global"},
        "expert_tensor_parallel_size (3) must divide the NVLink domain (8).",
        id="ep-group-before-the-expert-tp-rules",
    ),
    pytest.param(
        {"ep_size": 4, "cp_size": 3},
        "context_parallel_size (3) must divide the NVLink domain (8)",
        {"ep_size": 4, "cp_size": 4},
        "Node-local EP+CP requires ep_group_size=nvlink_domain_size.",
        id="cp-locality-before-node-local-ep-cp",
    ),
    pytest.param(
        {"ep_size": 4, "cp_size": 4},
        "Node-local EP+CP requires ep_group_size=nvlink_domain_size.",
        {"ep_size": 4},
        "expert_parallel_size=4 on a single 8-GPU NVLink domain forms 2 concurrent",
        id="node-local-ep-cp-before-the-racy-dispatch-topology",
    ),
    pytest.param(
        {"ep_size": 4, "tp_size": 8},
        "EP+TP requires expert_parallel_size (4) to be a multiple of tensor_parallel_size (8)",
        {"ep_size": 4},
        "expert_parallel_size=4 on a single 8-GPU NVLink domain forms 2 concurrent",
        id="tp-rules-before-the-racy-dispatch-topology",
    ),
    pytest.param(
        {"world_size": 12, "gpus_per_node": 12, "ep_size": 3, "expert_tp_size": 2, "ep_scope": "global"},
        "Cross-node EP+ETP supports one ETP group per NVLink domain only",
        {"world_size": 12, "gpus_per_node": 12, "ep_size": 3, "expert_tp_size": 2},
        "expert_parallel_size=3 on a single 12-GPU NVLink domain forms 4 concurrent",
        id="expert-tp-rules-before-the-racy-dispatch-topology",
    ),
    pytest.param(
        {"world_size": 6, "gpus_per_node": 6, "ep_size": 3, "ep_buffer_backend": "legacy"},
        "expert_parallel_size=3 on a single 6-GPU NVLink domain forms 2 concurrent",
        {"world_size": 3, "gpus_per_node": 3, "ep_size": 3, "ep_buffer_backend": "legacy"},
        "ep_buffer_backend='legacy' has no tuned Config for a 3-rank dispatch group",
        id="racy-dispatch-topology-before-the-legacy-backend",
    ),
    pytest.param(
        {"world_size": 3, "gpus_per_node": 3, "ep_size": 3, "ep_buffer_backend": "legacy", "use_hsdp": True},
        "ep_buffer_backend='legacy' has no tuned Config for a 3-rank dispatch group",
        {"ep_size": 2, "use_hsdp": True},
        "use_hsdp=True is not supported with EP (expert_parallel_size=2).",
        id="legacy-backend-before-hsdp",
    ),
    pytest.param(
        {"tp_size": 2, "use_hsdp": True, "fsdp_reshard_after_forward": True},
        "use_hsdp=True is not supported with TP (tensor_parallel_size=2)",
        {"tp_size": 2, "fsdp_reshard_after_forward": True},
        "fsdp_reshard_after_forward=True (FULL_SHARD / ZeRO-3) is not supported with Tensor Parallelism",
        id="hsdp-before-the-fsdp-settings",
    ),
    pytest.param(
        {"fsdp_defer_grad_sync": True, "fsdp_reshard_after_forward": True, "lowp_precision": "int4"},
        "fsdp_defer_grad_sync=True contradicts fsdp_reshard_after_forward=True",
        {"lowp_precision": "int4"},
        "lowp_precision must be one of",
        id="fsdp-settings-before-low-precision",
    ),
)


def _refusal(shape: dict) -> str:
    with pytest.raises(ValueError) as excinfo:
        create_config(**shape)
    return str(excinfo.value)


@pytest.mark.parametrize(("shape", "first", "later_shape", "later"), GATE_OVERLAPS)
def test_the_earlier_gate_reports_a_shape_both_refuse(shape, first, later_shape, later):
    assert _refusal(later_shape).startswith(later), _refusal(later_shape)
    assert _refusal(shape).startswith(first), _refusal(shape)


def test_the_node_width_is_resolved_before_the_node_width_rule(monkeypatch):
    """Left at 0, the node width would skip the node-width rule and fall through to the domain rules."""
    monkeypatch.delenv("NVLINK_DOMAIN_SIZE", raising=False)
    with simulated_world(world_size=12, gpus_per_node=8), pytest.raises(ValueError) as excinfo:
        ParallelismConfig()
    assert str(excinfo.value).startswith("world_size (12) must be divisible by gpus_per_node (8)."), excinfo.value


def test_a_sub_node_job_clamps_the_node_width_before_the_node_width_rule(monkeypatch):
    """Four ranks on an 8-GPU node: the resolved width is clamped to the world, not refused."""
    monkeypatch.delenv("NVLINK_DOMAIN_SIZE", raising=False)
    with simulated_world(world_size=4, gpus_per_node=8):
        config = ParallelismConfig()
    assert (config.world_size, config.gpus_per_node, config.nvlink_domain_size, config.num_nodes) == (4, 4, 4, 1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
