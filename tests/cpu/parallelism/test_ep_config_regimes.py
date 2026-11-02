#!/usr/bin/env python
"""The regime properties ``EPConfig`` answers, each pinned to what its readers do with it.

``uses_backward_grad_hooks`` decides whether an EP layer registers in-backward grad hooks,
``keeps_fp32_experts`` whether the expert weights stay fp32 masters, ``grouped_gemm_enabled`` whether
the experts compute with grouped GEMM, and ``spans_domains`` whether the transport dispatches over Gin
across NVLink domains, which ``ParallelismConfig.requires_rdma`` gates the run with at config time. A
property that drifts from its reader syncs experts twice or not at all, or sizes the wrong transport.

    python tests/cpu/parallelism/test_ep_config_regimes.py
"""

import contextlib

import pytest

from halo.distributed.expert_parallel import config as ep_config_mod
from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from tests.common.parallelism import make_parallelism_config, single_process_ep_config, stubbed_process_groups
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

# A family whose layer trains its experts on CPU through the per-expert loop.
_FAMILY = "qwen3_moe"


def _pp_stage_config(*, experts_fsdp_managed: bool) -> EPConfig:
    """Rank 0's ep1 config in the first of two 8-rank pipeline stages, which defers its sync unless FSDP2
    owns the experts."""
    with stubbed_process_groups(16, 0):
        return EPConfig(ep_size=1, stage_world_size=8, nvlink_domain_size=8, experts_fsdp_managed=experts_fsdp_managed)


def _hooked_layers(ep_config: EPConfig) -> list[bool]:
    """Whether each EP layer of a tiny model patched on ``ep_config`` registered in-backward grad hooks."""
    model = tiny_family_model(TINY_MOE_FAMILIES[_FAMILY])
    apply_ep_layers(model, ep_config)
    return [bool(layer._grad_sync_hooks) for _, layer in named_ep_layers(model)]


def _single_group_etp_config() -> EPConfig:
    """Rank 0's config for a 2-rank pure-ETP job: one EP group with peers, its router synced by in-backward
    hooks (the expert divide is the trainer's sweep either way)."""
    return EPConfig(ep_size=1, expert_tp_size=2, stage_world_size=2, nvlink_domain_size=2, experts_fsdp_managed=False)


@pytest.mark.parametrize(
    ("job", "build", "expected"),
    [
        (lambda: stubbed_process_groups(2, 0), _single_group_etp_config, True),
        (contextlib.nullcontext, lambda: single_process_ep_config(experts_fsdp_managed=True), False),
        (contextlib.nullcontext, lambda: _pp_stage_config(experts_fsdp_managed=False), False),
    ],
    ids=["hooks", "fsdp-managed", "deferred"],
)
def test_backward_grad_hooks_register_exactly_where_the_config_says(job, build, expected):
    with job():
        ep_config = build()
        assert ep_config.uses_backward_grad_hooks is expected
        hooked = _hooked_layers(ep_config)
    assert hooked and all(flag is expected for flag in hooked), hooked


@pytest.mark.parametrize("fp32_experts", [True, False])
@pytest.mark.parametrize("managed", [True, False])
def test_fp32_experts_are_kept_only_outside_fsdp_management(fp32_experts, managed):
    """FSDP2's bf16 param policy casts the experts it manages, so fp32 masters there would buy nothing."""
    config = single_process_ep_config(fp32_experts=fp32_experts, experts_fsdp_managed=managed)
    assert config.keeps_fp32_experts is (fp32_experts and not managed)


@pytest.mark.parametrize("hardware", [True, False])
@pytest.mark.parametrize("requested", [True, False])
def test_grouped_gemm_runs_where_requested_and_supported(monkeypatch, requested, hardware):
    monkeypatch.setattr(ep_config_mod, "supports_grouped_mm", lambda: hardware)
    config = single_process_ep_config(use_grouped_gemm=requested)
    model = tiny_family_model(TINY_MOE_FAMILIES[_FAMILY])
    apply_ep_layers(model, config)
    assert config.grouped_gemm_enabled is (requested and hardware)
    assert {layer._use_grouped_mm for _, layer in named_ep_layers(model)} == {config.grouped_gemm_enabled}


# (world, nvlink domain, ep_size, ep_scope) on 8-GPU nodes: node-local, cross-node over two domains, a
# global group spanning nine OS nodes of one NVL72 rack, and one spanning two racks.
_TOPOLOGIES = {
    "node-local": (16, 8, 8, "node"),
    "cross-node": (16, 8, 16, "global"),
    "nvl72-one-rack": (72, 72, 72, "global"),
    "nvl72-two-racks": (144, 72, 144, "global"),
}


@pytest.mark.parametrize(("world", "domain", "ep_size", "scope"), _TOPOLOGIES.values(), ids=_TOPOLOGIES.keys())
def test_the_transport_spans_domains_exactly_where_the_run_requires_rdma(world, domain, ep_size, scope):
    config = make_parallelism_config(
        world_size=world, gpus_per_node=8, nvlink_domain_size=domain, ep_size=ep_size, ep_scope=scope
    )
    assert config.ep_config().spans_domains is config.requires_rdma


def test_the_topologies_reach_both_transports():
    """Anti-vacuity: a sweep where every topology resolves the same way would pass on a constant."""
    verdicts = {
        make_parallelism_config(
            world_size=world, gpus_per_node=8, nvlink_domain_size=domain, ep_size=ep_size, ep_scope=scope
        ).requires_rdma
        for world, domain, ep_size, scope in _TOPOLOGIES.values()
    }
    assert verdicts == {True, False}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
