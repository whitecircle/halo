#!/usr/bin/env python
"""
Tests for ParallelismConfig computed fields, validation, properties, and rank methods.

Run: python tests/cpu/parallelism/test_parallelism_config.py
"""

import datetime
import os
from unittest.mock import patch

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from src.args.distributed_args import DistributedArguments
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.group_layout import cross_node_rank_and_group, node_local_rank_and_group
from tests.common.parallelism import create_config, make_parallelism_config
from tests.common.ports import free_port

# Module path prefix for mocking the src.distributed.runtime imports in parallelism_config
_MOD = "src.distributed.parallelism_config"


# Computed fields


def test_defaults():
    """Default config: single-GPU-like, no parallelism."""
    cfg = create_config(world_size=8, gpus_per_node=8)
    assert cfg.world_size == 8
    assert cfg.gpus_per_node == 8
    assert cfg.ep_group_size == 1  # ep_size=1 * expert_tp_size=1
    assert cfg.data_parallel_size == 8
    assert cfg.num_nodes == 1


def test_ep_only_computed():
    """EP-only: dp stays full world_size (EP is orthogonal to DP)."""
    cfg = create_config(ep_size=8, world_size=8, gpus_per_node=8)
    assert cfg.ep_group_size == 8
    assert cfg.data_parallel_size == 8
    assert cfg.num_nodes == 1


def test_tp_only_computed():
    """TP reduces DP by tp_size."""
    cfg = create_config(tp_size=4, world_size=8, gpus_per_node=8)
    assert cfg.data_parallel_size == 2  # 8 / 4


def test_cp_only_computed():
    """CP reduces DP by cp_size."""
    cfg = create_config(cp_size=4, world_size=8, gpus_per_node=8)
    assert cfg.data_parallel_size == 2  # 8 / 4


def test_expert_tp_only_computed():
    """Pure ETP (ep_size=1) reduces DP by expert_tp_size and shards the expert FFN only.

    ep_size>1 + expert_tp_size>1 (EP+ETP) is also supported — see
    test_validate_ep_plus_expert_tp_accepted.
    """
    cfg = create_config(ep_size=1, expert_tp_size=2, world_size=8, gpus_per_node=8)
    assert cfg.ep_group_size == 2  # ep_size(1) * expert_tp_size(2)
    assert cfg.data_parallel_size == 4  # 8 / 2


def test_validate_ep_plus_expert_tp_accepted():
    """ep_size>1 AND expert_tp_size>1 (EP+ETP) is supported (experimental, node-local).

    The expert-TP reduction runs in token space (outside DeepEP's dispatch->combine span — see
    EPMoELayerBase._dispatch_compute_combine), so the combo does not deadlock the intranode
    combine barrier under FSDP2. ParallelismConfig must ACCEPT it: ep_group_size = ep_size *
    expert_tp_size, and only expert_tp_size reduces DP (EP stays orthogonal).
    """
    for eps in (2, 4):
        cfg = create_config(ep_size=eps, expert_tp_size=2, world_size=8, gpus_per_node=8)
        assert cfg.ep_group_size == eps * 2  # ep_size * expert_tp_size
        assert cfg.data_parallel_size == 4  # world(8) / expert_tp_size(2)


def test_ep_tp_computed():
    """EP+TP: only TP reduces DP. Cross-node EP+TP must be a SINGLE EP group."""
    cfg = create_config(ep_size=16, tp_size=4, world_size=16, gpus_per_node=8, ep_scope="global")
    assert cfg.data_parallel_size == 4  # 16 / 4
    assert cfg.num_nodes == 2


def test_ep_tp_multi_domain_multi_group_rejected():
    """Multi-domain MULTI-GROUP EP+TP is rejected: the deferred cross-replica DP sweep assumes FSDP
    shards over the EP group, but EP+TP shards over the (dp, tp) mesh — averaging replica peers
    would mix different dp shards (silent gradient corruption)."""
    try:
        create_config(ep_size=8, tp_size=4, world_size=16, gpus_per_node=8, ep_scope="node")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Multi-domain multi-group EP+TP" in str(e), e


def test_ep_cp_computed():
    """EP+CP orthogonal node-local: only CP reduces DP."""
    cfg = create_config(ep_size=8, cp_size=8, world_size=16, gpus_per_node=8, ep_scope="node")
    assert cfg.data_parallel_size == 2  # 16 / 8


def test_auto_scope_node():
    """Auto scope resolves to 'node' when ep_group_size <= gpus_per_node."""
    cfg = create_config(ep_size=4, world_size=16, gpus_per_node=8, ep_scope="auto")
    assert cfg.ep_scope == "node"


def test_auto_scope_global():
    """Auto scope resolves to 'global' when ep_group_size > gpus_per_node."""
    cfg = create_config(ep_size=16, world_size=16, gpus_per_node=8, ep_scope="auto")
    assert cfg.ep_scope == "global"


def test_num_nodes():
    """Multi-node: num_nodes = world_size / gpus_per_node."""
    cfg = create_config(world_size=32, gpus_per_node=8)
    assert cfg.num_nodes == 4


def test_rank_identity_is_global_not_node_local():
    """Every rank field derives from the GLOBAL rank, never the launcher's local rank.

    Rank 10 of 16 sits on node 1 with local rank 2. If any identity field were taken from the
    local rank, ranks 2 and 10 would collide and their EP/CP/DP group membership with them.
    """
    cfg = create_config(rank=10, world_size=16, gpus_per_node=8)
    assert cfg.global_rank == 10
    assert cfg.stage_local_rank == 10  # pp_size == 1, so the stage is the world
    assert cfg.num_nodes == 2

    peer = create_config(rank=2, world_size=16, gpus_per_node=8)
    assert peer.global_rank != cfg.global_rank
    assert peer.get_data_parallel_rank() != cfg.get_data_parallel_rank()


# Validation errors


def test_validate_tp_cp_conflict():
    """TP + CP is refused by the capability matrix, naming the double-partition mechanism."""
    try:
        create_config(tp_size=4, cp_size=4, world_size=16, gpus_per_node=8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "TP + CP is not a supported parallelism combination" in str(e), e
        assert "partition the same ranks twice" in str(e), e


def test_validate_ep_tp_node_local():
    """EP+TP: tp_size must fit in the NVLink domain (the ep_tp_locality 'NVLink-local TP' message)."""
    try:
        create_config(ep_size=16, tp_size=16, world_size=16, gpus_per_node=8, ep_scope="global")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "must divide the NVLink domain" in str(e), e


def test_validate_ep_tp_divisibility():
    """EP+TP: tp_size must divide the NVLink domain so the DTensor all-reduce stays NVLink-local."""
    try:
        create_config(ep_size=3, tp_size=3, world_size=24, gpus_per_node=8, ep_scope="global")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        # The cross-domain members-per-domain guard fires first for this shape.
        assert "must be a multiple of tp_size" in str(e) or "must divide the NVLink domain" in str(e)


def test_validate_ep_node_scope_too_large():
    """Node-local EP: ep_group_size cannot exceed the NVLink domain (the node branch must fire)."""
    try:
        create_config(ep_size=16, world_size=16, gpus_per_node=8, ep_scope="node")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        # _validate_ep_group node branch — not the global branch, not _validate_ep_tp_locality.
        assert "Node-local EP group size (16) cannot exceed the NVLink domain (8)" in str(e), e


def test_validate_ep_node_scope_divisibility():
    """Node-local EP: ep_group_size must divide the NVLink domain (gpus_per_node)."""
    # ep_group(3) <= domain(8) but 8 % 3 != 0; world(8) is divisible by the domain so the earlier
    # world/domain guard passes and _validate_ep_group's node-divisibility branch is what fires.
    try:
        create_config(ep_size=3, world_size=8, gpus_per_node=8, ep_scope="node")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "must divide the NVLink domain" in str(e)


def test_validate_ep_global_scope_too_large():
    """Global EP: ep_group_size cannot exceed world_size (global branch, not the node branch)."""
    try:
        create_config(ep_size=32, world_size=16, gpus_per_node=8, ep_scope="global")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "EP group size (32) cannot exceed world size (16)" in str(e), e


def test_validate_ep_global_scope_divisibility():
    """Global EP: world_size must be divisible by ep_group_size (global-branch divisibility check)."""
    try:
        create_config(ep_size=3, world_size=8, gpus_per_node=8, ep_scope="global")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "EP group size (3) must divide world size (8)" in str(e), e


def test_validate_expert_tp_group_tiling_node_scope():
    """ep2×etp3 node-local: the EP GROUP (6) fails to tile the domain — that gate fires first, so
    this shape can never reach the expert-TP divisor check; pin the tiling message, not a fragment
    both gates emit."""
    try:
        create_config(ep_size=2, expert_tp_size=3, world_size=8, gpus_per_node=8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "EP group size (6) must divide the NVLink domain (8)" in str(e), e


def test_validate_expert_tp_divides_domain_global_scope():
    """The expert-TP divisor gate is reachable only at global scope: ep2×etp3 over six 8-GPU domains
    tiles cleanly (members_per_domain=1), then nvlink_domain_size % expert_tp_size fires."""
    try:
        create_config(ep_size=2, expert_tp_size=3, world_size=48, gpus_per_node=8, ep_scope="global")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "expert_tp_size (3) must divide the NVLink domain (8)" in str(e), e
        assert "Expert TP groups must stay on NVLink" in str(e), e


def test_cross_node_ep_etp_valid_shape_accepted():
    """The documented cross-node EP+ETP shape — one ETP group per domain, single global EP group —
    must pass every gate the two rejections above sit behind."""
    cfg = create_config(ep_size=6, expert_tp_size=8, world_size=48, gpus_per_node=8, ep_scope="global")
    assert cfg.ep_group_size == 48
    assert cfg.data_parallel_size == 48 // 8


def test_validate_cp_exceeds_domain_rejected():
    try:
        create_config(cp_size=16, world_size=8, gpus_per_node=8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "CP size (16) cannot exceed the NVLink domain (8)" in str(e), e


def test_validate_cp_domain_divisibility_rejected():
    try:
        create_config(cp_size=3, world_size=8, gpus_per_node=8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "CP size (3) must divide the NVLink domain (8)" in str(e), e


def test_validate_tp_exceeds_stage_world_rejected():
    try:
        create_config(tp_size=16, world_size=8, gpus_per_node=8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "TP size (16) cannot exceed world size (8)" in str(e), e


def test_validate_tp_stage_world_divisibility_rejected():
    try:
        create_config(tp_size=3, world_size=8, gpus_per_node=8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "TP size (3) must divide world size (8)" in str(e), e


def test_validate_pure_tp_domain_divisibility_rejected():
    """Pure TP wider than one NVLink domain: fits and divides the stage world, straddles domains."""
    try:
        create_config(tp_size=16, world_size=16, gpus_per_node=8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "must divide the NVLink domain" in str(e), e


def test_validate_multi_domain_multigroup_ep_etp_rejected():
    """ep2×etp2 on two domains forms four dispatch groups — the deferred-DP-off race the EP+ETP
    composition validator names; its EP+TP twin has its own test, this one pins the ETP arm."""
    try:
        create_config(ep_size=2, expert_tp_size=2, world_size=16, gpus_per_node=8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "multiple dispatch groups across NVLink domains" in str(e), e


def test_validate_shard_ep1_experts_off_under_tp_cp_rejected():
    """fsdp_shard_ep1_experts=False is a silent no-op under TP/CP (their wraps shard experts
    unconditionally) — the config must refuse rather than not honor the flag."""
    for axis_kwargs in ({"tp_size": 2}, {"cp_size": 2}):
        try:
            create_config(fsdp_shard_ep1_experts=False, world_size=8, gpus_per_node=8, **axis_kwargs)
            raise AssertionError(f"Should have raised ValueError for {axis_kwargs}")
        except ValueError as e:
            assert "fsdp_shard_ep1_experts=False is not honored under TP or CP" in str(e), e
    # Pure DP keeps the full replicated-expert copy — the flag's documented purpose.
    cfg = create_config(fsdp_shard_ep1_experts=False, world_size=8, gpus_per_node=8)
    assert cfg.experts_fsdp_managed is False


def test_validate_nonpositive_gpus_per_node_rejected():
    try:
        create_config(world_size=8, gpus_per_node=-2)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "gpus_per_node resolved to -2" in str(e), e


def test_validate_negative_nvlink_domain_size_rejected():
    """Python modulo passes a negative multiple through both domain divisibility gates, so the sign
    must be rejected explicitly before them."""
    try:
        create_config(world_size=8, gpus_per_node=8, nvlink_domain_size=-8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "nvlink_domain_size must be positive" in str(e), e


def test_validate_expert_tp_plus_tp():
    """Attention TP + expert TP is refused by the matrix, quoting the flags the user typed."""
    try:
        create_config(ep_size=2, tp_size=2, expert_tp_size=2, world_size=8, gpus_per_node=8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "is not a supported parallelism combination" in str(e), e
        assert "expert_tensor_parallel_size=2" in str(e) and "tensor_parallel_size=2" in str(e), e


def test_validate_expert_tp_plus_cp():
    """Pure ETP + CP is refused by the matrix (ep_size=1)."""
    try:
        create_config(ep_size=1, expert_tp_size=2, cp_size=2, world_size=8, gpus_per_node=8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "ETP + CP is not a supported parallelism combination" in str(e), e
        assert "expert_tensor_parallel_size=2" in str(e), e


def test_validate_ep_cp_node_scope_mismatch():
    """Node-local EP+CP: ep_group_size must equal the NVLink domain (the _validate_ep_cp message)."""
    try:
        create_config(ep_size=4, cp_size=4, world_size=16, gpus_per_node=8, ep_scope="node")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Node-local EP+CP requires ep_group_size=nvlink_domain_size" in str(e), e


def test_validate_ep_cp_global_scope_rejected():
    """EP+CP with cross-NVLink-domain (global) EP is rejected (must be node-local)."""
    try:
        create_config(ep_size=16, cp_size=2, world_size=16, gpus_per_node=8, ep_scope="global")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "node-local EP" in str(e) or "ep_scope='node'" in str(e)


def test_validate_ep_tp_size_not_multiple_of_tp():
    """EP+TP requires ep_size to be a multiple of tp_size (whole TP groups per EP group)."""
    # ep_size=2, tp_size=4: ep_group(2) divides the node and tp(4) divides the node, so the
    # earlier locality/divisibility guards pass — the ep_size % tp_size == 0 check is what fires
    # (2 % 4 != 0).
    try:
        create_config(ep_size=2, tp_size=4, world_size=8, gpus_per_node=8, ep_scope="node")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "multiple of tp_size" in str(e)


def test_validate_max_concurrent_loading_negative():
    """max_concurrent_loading < 0 is rejected by the basic sanity guard."""
    try:
        create_config(max_concurrent_loading=-1, world_size=8, gpus_per_node=8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "max_concurrent_loading" in str(e)


# Sanity / enum / range guards (sizes, ep_scope, world divisibility)


def _expect_reject(substr=None, **kw):
    try:
        create_config(**kw)
    except ValueError as e:
        if substr:
            assert substr in str(e), f"expected {substr!r} in error, got: {e}"
        return
    raise AssertionError(f"config {kw} should have raised ValueError")


def test_guard_size_below_one():
    for name in ("ep_size", "tp_size", "cp_size", "expert_tp_size"):
        _expect_reject(name, **{name: 0, "world_size": 8, "gpus_per_node": 8})


def test_guard_bad_ep_scope():
    _expect_reject("ep_scope", ep_size=8, ep_scope="rack", world_size=8, gpus_per_node=8)


def test_guard_world_not_multiple_of_gpus_per_node():
    _expect_reject("divisible by", world_size=12, gpus_per_node=8)


def test_guard_nvlink_domain_multiple_of_node():
    # nvlink_domain_size must be a whole number of OS nodes
    _expect_reject("nvlink_domain_size", world_size=16, gpus_per_node=8, nvlink_domain_size=12)


# Low-precision (mixed-precision compute) guards


def test_guard_lowp_bad_precision():
    _expect_reject("lowp_precision", lowp_precision="int4", world_size=8, gpus_per_node=8)


def test_guard_lowp_applied_nowhere():
    _expect_reject(
        "apply to nothing",
        lowp_precision="fp8",
        lowp_apply_dense_mlp=False,
        lowp_apply_moe_experts=False,
        world_size=8,
        gpus_per_node=8,
    )


def test_guard_lowp_negative_keep_blocks():
    _expect_reject(">= 0", lowp_precision="fp8", lowp_keep_first_blocks=-1, world_size=8, gpus_per_node=8)


def test_lowp_valid_combos_accepted():
    # No backend knob: low precision is the fake-quant oracle, with the native DeepGEMM kernel opt-in
    # via env (HALO_DEEPGEMM_NATIVE), not a config field.
    for kw in (
        {"lowp_precision": "fp8"},
        {"lowp_precision": "fp4"},
        {"lowp_precision": "fp8", "lowp_apply_moe_experts": False},
    ):
        cfg = create_config(world_size=8, gpus_per_node=8, **kw)
        assert cfg.lowp_precision in ("fp8", "fp4")


# Properties


def test_mode_string_ep():
    cfg = create_config(ep_size=8, world_size=8, gpus_per_node=8)
    assert cfg.mode_string == "ep"


def test_mode_string_tp():
    cfg = create_config(tp_size=4, world_size=8, gpus_per_node=8)
    assert cfg.mode_string == "tp"


def test_mode_string_cp():
    cfg = create_config(cp_size=4, world_size=8, gpus_per_node=8)
    assert cfg.mode_string == "cp"


def test_mode_string_ep_tp():
    cfg = create_config(ep_size=8, tp_size=4, world_size=8, gpus_per_node=8)
    assert cfg.mode_string == "ep-tp"


def test_mode_string_ep_cp():
    cfg = create_config(ep_size=8, cp_size=8, world_size=16, gpus_per_node=8, ep_scope="node")
    assert cfg.mode_string == "ep-cp"


def test_mode_string_gmm_only():
    """Grouped GEMM without EP: mode_string should be 'gmm'."""
    cfg = create_config(use_grouped_gemm=True, world_size=8, gpus_per_node=8)
    assert cfg.mode_string == "gmm"


def test_mode_string_expert_tp():
    cfg = create_config(ep_size=1, expert_tp_size=2, world_size=8, gpus_per_node=8)
    assert "expert-tp" in cfg.mode_string


def test_mode_string_none():
    """No parallelism and no grouped GEMM: mode_string is None."""
    cfg = create_config(use_grouped_gemm=False, world_size=8, gpus_per_node=8)
    assert cfg.mode_string is None


def test_requires_rdma():
    """RDMA required: global scope, multi-node, ep_group_size > gpus_per_node."""
    cfg = create_config(ep_size=16, world_size=16, gpus_per_node=8, ep_scope="global")
    assert cfg.requires_rdma is True


def test_no_rdma_node_local():
    """No RDMA for node-local EP."""
    cfg = create_config(ep_size=8, world_size=16, gpus_per_node=8, ep_scope="node")
    assert cfg.requires_rdma is False


def test_requires_rdma_multi_domain_global_small_group():
    """Global scope across >1 domain needs RDMA even when the group fits one domain:
    the column-block layout places members of every EP group in every domain."""
    cfg = create_config(ep_size=8, world_size=16, gpus_per_node=8, ep_scope="global")
    assert cfg.requires_rdma is True


def test_needs_ep_wrappers_ep():
    """EP mode needs wrappers."""
    cfg = create_config(ep_size=8, world_size=8, gpus_per_node=8)
    assert cfg.needs_ep_wrappers is True


def test_needs_ep_wrappers_gmm():
    """Grouped GEMM mode (no EP) also needs wrappers."""
    cfg = create_config(use_grouped_gemm=True, world_size=8, gpus_per_node=8)
    assert cfg.needs_ep_wrappers is True


def test_no_ep_wrappers_disabled():
    """Config with use_grouped_gemm=False and no EP does not need EP wrappers."""
    cfg = create_config(use_grouped_gemm=False, world_size=8, gpus_per_node=8)
    assert cfg.needs_ep_wrappers is False


def test_boolean_mode_properties():
    """Test is_ep_mode, is_cp_mode, is_tp_mode, is_ep_tp_mode, is_ep_cp_mode."""
    cfg = create_config(ep_size=8, tp_size=4, world_size=8, gpus_per_node=8)
    assert cfg.is_ep_mode is True
    assert cfg.is_tp_mode is True
    assert cfg.is_ep_tp_mode is True
    assert cfg.is_cp_mode is False
    assert cfg.is_ep_cp_mode is False


def test_num_ep_groups_node():
    """Node-local EP: num_ep_groups = (gpus_per_node / ep_group_size) * num_nodes."""
    cfg = create_config(ep_size=4, world_size=16, gpus_per_node=8, ep_scope="node")
    # 2 EP groups per node * 2 nodes = 4
    assert cfg.num_ep_groups == 4


def test_num_ep_groups_global():
    """Global EP: num_ep_groups = world_size / ep_group_size."""
    cfg = create_config(ep_size=8, world_size=16, gpus_per_node=8, ep_scope="global")
    assert cfg.num_ep_groups == 2  # 16 / 8


@pytest.mark.parametrize(
    "topology",
    [
        {"ep_size": 8, "world_size": 8, "gpus_per_node": 8, "ep_scope": "node"},
        {"ep_size": 2, "world_size": 8, "gpus_per_node": 8, "ep_scope": "node"},
        {"ep_size": 4, "world_size": 16, "gpus_per_node": 8, "ep_scope": "node"},
        {"ep_size": 8, "world_size": 16, "gpus_per_node": 8, "ep_scope": "node"},
        {"ep_size": 1, "expert_tp_size": 2, "world_size": 8, "gpus_per_node": 8, "ep_scope": "node"},
        # ep_group_size == 1: EPConfig tiles the stage with one SINGLETON group per rank.
        {"ep_size": 1, "world_size": 8, "gpus_per_node": 8, "ep_scope": "node"},
        {"ep_size": 8, "world_size": 16, "gpus_per_node": 8, "ep_scope": "global"},
        {"ep_size": 16, "world_size": 16, "gpus_per_node": 8, "ep_scope": "global"},
        {"ep_size": 8, "world_size": 32, "gpus_per_node": 8, "ep_scope": "global"},
    ],
)
def test_num_ep_groups_equals_the_groups_the_layout_actually_builds(topology):
    """``num_ep_groups`` must equal the number of DISTINCT groups ``EPConfig`` constructs.

    ``EPConfig`` places each rank with the ``group_layout`` functions below; ``num_ep_groups`` is
    read by the sharded-EP save guard, the deferred cross-replica divisor and the DP-rank math. A
    count derived independently of the layout is a silent wrong answer at some topology — an
    over-count divides expert grads by too much, an under-count lets a sharded save merge duplicated
    experts — so this walks every rank and compares the two.
    """
    cfg = create_config(**topology)
    ranks = range(cfg.stage_world_size)
    if cfg.ep_scope == "node":
        groups = {node_local_rank_and_group(r, cfg.nvlink_domain_size, cfg.ep_group_size)[1] for r in ranks}
    else:
        groups = {
            cross_node_rank_and_group(r, cfg.stage_world_size, cfg.ep_group_size, cfg.nvlink_domain_size)[1]
            for r in ranks
        }
    assert len(groups) == cfg.num_ep_groups, f"{sorted(groups)} vs num_ep_groups={cfg.num_ep_groups}"
    assert groups == set(range(cfg.num_ep_groups)), sorted(groups)


_GLOO_WORLD = 8
# A construction desync would otherwise sit in the gloo group for the default timeout; the suite
# must fail on it instead of hanging.
_GLOO_TIMEOUT_SEC = 180


def _ep_group_size_one_worker(rank: int, out_dir: str, port: int) -> None:
    """Build the REAL ``EPConfig`` and ``ParallelismConfig`` for ``rank`` on a live gloo group."""
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(_GLOO_WORLD))
    dist.init_process_group(
        "gloo", rank=rank, world_size=_GLOO_WORLD, timeout=datetime.timedelta(seconds=_GLOO_TIMEOUT_SEC)
    )
    try:
        ep = EPConfig(ep_size=1, world_size=_GLOO_WORLD, gpus_per_node=_GLOO_WORLD, use_grouped_gemm=False)
        pc = make_parallelism_config(world_size=_GLOO_WORLD, gpus_per_node=_GLOO_WORLD, rank=rank, ep_size=1)
        failures = []
        if ep.ep_group_size != 1 or not ep.needs_expert_grad_sync:
            failures.append(f"precondition: ep_group_size={ep.ep_group_size} sync={ep.needs_expert_grad_sync}")
        if pc.num_ep_groups != ep.num_ep_groups:
            failures.append(f"num_ep_groups {pc.num_ep_groups} != EPConfig's {ep.num_ep_groups}")
        if sorted(pc.get_expert_replica_ranks()) != sorted(ep.expert_replica_ranks):
            failures.append(f"replicas {pc.get_expert_replica_ranks()} != EPConfig's {ep.expert_replica_ranks}")
        if dist.get_world_size(ep.expert_replica_group) != _GLOO_WORLD:
            failures.append(f"replica group holds {dist.get_world_size(ep.expert_replica_group)} of {_GLOO_WORLD}")
        with open(os.path.join(out_dir, f"result_{rank}.txt"), "w") as fh:
            fh.write("PASS" if not failures else "FAIL: " + "; ".join(failures))
    finally:
        dist.destroy_process_group()


def test_ep_group_size_one_agrees_with_a_real_eight_rank_ep_config(tmp_path):
    """``num_ep_groups`` and ``get_expert_replica_ranks`` at ``ep_group_size == 1``, against the groups
    a REAL 8-rank ``EPConfig`` builds over a live gloo world.

    The shape is the DEFAULT for every dense run and every ``ep_size=1`` MoE run, and it is the one
    the layout helpers cannot express: ``cross_node_layout`` refuses a group of 1, so both spellings
    take their own branch and nothing but this comparison holds them to what ``_create_ep_groups``
    does — one singleton EP group per rank, and a single expert-replica group over the whole block.
    A ``num_ep_groups`` of 1 there under-counts the replica set the deferred sweep divides by.
    """
    out_dir = str(tmp_path)
    mp.start_processes(
        _ep_group_size_one_worker, args=(out_dir, free_port()), nprocs=_GLOO_WORLD, join=True, start_method="spawn"
    )
    for rank in range(_GLOO_WORLD):
        with open(os.path.join(out_dir, f"result_{rank}.txt")) as fh:
            assert fh.read() == "PASS", f"rank {rank}"


def test_is_expert_tp_mode_property():
    """is_expert_tp_mode tracks expert_tp_size > 1; ep_group_size folds in the ETP factor."""
    etp = create_config(ep_size=1, expert_tp_size=2, world_size=8, gpus_per_node=8)
    assert etp.is_expert_tp_mode is True
    assert etp.is_ep_mode is True  # ep_group_size = 1*2 = 2 > 1
    plain = create_config(ep_size=8, world_size=8, gpus_per_node=8)
    assert plain.is_expert_tp_mode is False


def test_requires_rdma_false_when_global_fits_in_domain():
    """Global scope but ep_group_size <= nvlink domain → still no RDMA (intra-domain)."""
    # ep_size=8 == gpus_per_node on a single node: global scope, but num_nvlink_domains == 1.
    cfg = create_config(ep_size=8, world_size=8, gpus_per_node=8, ep_scope="global")
    assert cfg.requires_rdma is False


def test_mode_string_ep_expert_tp_combo():
    """EP+ETP mode_string lists both ep and expert-tp segments."""
    cfg = create_config(ep_size=2, expert_tp_size=2, world_size=8, gpus_per_node=8)
    assert cfg.mode_string == "ep-expert-tp"


def test_data_parallel_rank_expert_tp_path():
    """Pure ETP routes get_data_parallel_rank through the expert-TP branch (not tp/cp divisor).

    ep_size=1, expert_tp_size=2 on 8 GPUs, node-local: ep_group_size=2, num_ep_groups=4.
    For rank 0: ep_rank=0, dispatch_ep_rank = 0 % 1 = 0, ep_group_idx=0 → dp_rank=0.
    For rank 1 (an ETP partner of rank 0's group): ep_rank=1, dispatch_ep_rank = 1 % 1 = 0,
    ep_group_idx=0 → dp_rank 0 (ETP partners share the same DP batch).
    """
    cfg0 = create_config(ep_size=1, expert_tp_size=2, world_size=8, gpus_per_node=8, rank=0)
    cfg1 = create_config(ep_size=1, expert_tp_size=2, world_size=8, gpus_per_node=8, rank=1)
    assert cfg0.get_data_parallel_rank() == cfg1.get_data_parallel_rank()


def test_summary_standard_ddp_string():
    """No parallelism → summary reports DATA PARALLEL (torchrun lands on FSDP2, accelerate on DDP — neither is the axis)."""
    cfg = create_config(world_size=8, gpus_per_node=8)
    s = cfg.summary()
    assert "DATA PARALLEL" in s
    assert "DDP" not in s


# Convenience factory functions


def _DistArgs() -> DistributedArguments:
    """Parsed DistributedArguments as the entry scripts hand them to the builder.

    The real dataclass, not a hand-rolled stand-in: a stub would have to be updated by hand for
    every new field, and the builder reads its fields directly — a missing one is an AttributeError
    at launch, exactly what these tests exist to catch.
    """
    return DistributedArguments(
        expert_parallel_size=8,
        ep_scope="node",
        max_concurrent_loading=1,
        fsdp_shard_ep1_experts=False,
    )


def test_parallelism_config_from_args_basic():
    """parallelism_config_from_args wires DistributedArguments into a ParallelismConfig."""
    with (
        patch(f"{_MOD}.get_global_world_size", return_value=8),
        patch(f"{_MOD}.get_local_world_size", return_value=8),
        patch(f"{_MOD}.get_global_rank", return_value=0),
        patch(f"{_MOD}.is_global_main_process", return_value=True),
    ):
        from src.training.parallelism_args import parallelism_config_from_args

        # Non-default values on the knobs most prone to silent drift, so the asserts below fail if
        # the builder drops them back to ParallelismConfig defaults instead of forwarding.
        args = _DistArgs()
        args.fp32_grad_reduce = True
        args.max_concurrent_loading = 3
        cfg = parallelism_config_from_args(args)
        assert cfg.ep_size == 8
        assert cfg.expert_tp_size == 1
        assert cfg.use_grouped_gemm is True
        assert cfg.lowp_precision == "bf16"
        assert cfg.fp32_grad_reduce is True  # forwarded, not reset to the default
        assert cfg.max_concurrent_loading == 3


def test_parallelism_config_from_args_rejects_pp_when_unsupported():
    """supports_pp=False rejects a requested pipeline_parallel_size>1 at config time — BEFORE the
    model (or a teacher/reference/vLLM probe) loads; the trainer's _supports_pp gate fires far later."""
    args = _DistArgs()
    args.pipeline_parallel_size = 2
    from src.training.parallelism_args import parallelism_config_from_args

    try:
        parallelism_config_from_args(args, supports_pp=False)
        raise AssertionError("supports_pp=False must reject pipeline_parallel_size=2")
    except ValueError as e:
        assert "does not support Pipeline Parallelism" in str(e)


def test_parallelism_config_from_args_rejects_lowp_when_disallowed():
    """A non-bf16 lowp_precision is rejected when allow_low_precision=False (non-SFT trainers)."""
    args = _DistArgs()
    args.lowp_precision = "fp8"
    with (
        patch(f"{_MOD}.get_global_world_size", return_value=8),
        patch(f"{_MOD}.get_local_world_size", return_value=8),
        patch(f"{_MOD}.get_global_rank", return_value=0),
        patch(f"{_MOD}.is_global_main_process", return_value=True),
    ):
        from src.training.parallelism_args import parallelism_config_from_args

        try:
            parallelism_config_from_args(args, allow_low_precision=False)
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "lowp_precision" in str(e)


def test_parallelism_config_from_args_lowp_allowed_for_sft():
    """allow_low_precision=True forwards the lowp_* knobs (SFT path)."""
    args = _DistArgs()
    args.lowp_precision = "fp8"
    with (
        patch(f"{_MOD}.get_global_world_size", return_value=8),
        patch(f"{_MOD}.get_local_world_size", return_value=8),
        patch(f"{_MOD}.get_global_rank", return_value=0),
        patch(f"{_MOD}.is_global_main_process", return_value=True),
    ):
        from src.training.parallelism_args import parallelism_config_from_args

        cfg = parallelism_config_from_args(args, allow_low_precision=True)
        assert cfg.lowp_precision == "fp8"


# Rank methods (mocked rank=0)


def test_ep_rank_node():
    """Node-local EP rank = the rank's NVLink-domain coordinate % ep_group_size."""
    cfg = create_config(ep_size=4, world_size=16, gpus_per_node=8, rank=5)
    assert cfg.get_ep_rank() == 1  # 5 % 4


def test_ep_rank_global():
    """Global (cross-node) EP rank under the column-block layout."""
    cfg = create_config(ep_size=8, world_size=16, gpus_per_node=8, rank=3, ep_scope="global")
    # 2 domains of 8, ep_group_size=8 → members_per_domain m=4, num_ep_groups=2.
    # rank 3: domain=0, pos=3, group_idx=3//4=0, ep_rank=0*4 + 3%4 = 3.
    assert cfg.get_ep_rank() == 3


def test_ep_group_idx_node():
    """Node-local EP group index."""
    cfg = create_config(ep_size=4, world_size=16, gpus_per_node=8, rank=12, ep_scope="node")
    # node_id = 12 // 8 = 1; local_ep_group_idx = 4 // 4 = 1
    # ep_groups_per_node = 8 // 4 = 2; result = 1 * 2 + 1 = 3
    assert cfg.get_ep_group_idx() == 3


def test_ep_group_idx_global():
    """Global (cross-node) EP group index under the column-block layout."""
    cfg = create_config(ep_size=8, world_size=16, gpus_per_node=8, rank=5, ep_scope="global")
    # m=4: rank 5 → pos 5, group_idx = 5 // 4 = 1.
    assert cfg.get_ep_group_idx() == 1


def test_cp_rank():
    """CP rank = the rank's NVLink-domain coordinate % cp_size."""
    cfg = create_config(cp_size=4, world_size=8, gpus_per_node=8, rank=6)
    assert cfg.get_cp_rank() == 2  # 6 % 4


def test_data_parallel_rank_tp():
    """DP rank with TP = global_rank // tp_size."""
    cfg = create_config(tp_size=4, world_size=8, gpus_per_node=8, rank=5)
    assert cfg.get_data_parallel_rank() == 1  # 5 // 4


def test_data_parallel_rank_no_parallelism():
    """DP rank without TP/CP falls through to get_cp_group_idx."""
    cfg = create_config(world_size=8, gpus_per_node=8, rank=3)
    # cp_size=1 => dp_divisor = max(1,1) = 1 => falls to get_cp_group_idx
    # cp_groups_per_node = 8/1 = 8; local_cp_group_idx = 3/1 = 3; node_id=0 => 0*8+3 = 3
    assert cfg.get_data_parallel_rank() == 3


def test_ep_group_ranks_node():
    """EP group ranks for node-local EP."""
    cfg = create_config(ep_size=4, world_size=16, gpus_per_node=8, rank=0, ep_scope="node")
    ranks = cfg.get_ep_group_ranks()
    assert ranks == [0, 1, 2, 3]


def test_ep_group_ranks_global():
    """EP group ranks for global EP — contiguous device block per domain (column-block)."""
    cfg = create_config(ep_size=8, world_size=16, gpus_per_node=8, rank=0, ep_scope="global")
    ranks = cfg.get_ep_group_ranks()
    # m=4, group_idx=0: domain 0 block [0,1,2,3] + domain 1 block [8,9,10,11].
    assert ranks == [0, 1, 2, 3, 8, 9, 10, 11]


def test_cp_group_ranks():
    """CP group ranks are contiguous within a node."""
    cfg = create_config(cp_size=4, world_size=8, gpus_per_node=8, rank=4)
    ranks = cfg.get_cp_group_ranks()
    assert ranks == [4, 5, 6, 7]


def test_expert_replica_ranks_single_group():
    """One EP group spanning the world: no other group holds this rank's experts."""
    cfg = create_config(ep_size=8, world_size=8, gpus_per_node=8, rank=3)
    assert cfg.num_ep_groups == 1
    assert cfg.get_expert_replica_ranks() == [3]


def test_expert_replica_ranks_at_ep_group_size_one_span_the_stage():
    """``ep_group_size == 1``: every rank is a singleton EP group holding the FULL expert set, so the
    whole rank block is ONE replica set — which is exactly the single group ``EPConfig`` builds there
    (``test_ep_group_size_one_agrees_with_a_real_eight_rank_ep_config``). Returning ``[global_rank]``
    said "nothing replicates my experts" for the default MoE shape, contradicting the group the
    deferred sweep actually reduces over."""
    cfg = create_config(world_size=8, gpus_per_node=8, rank=3)
    assert cfg.ep_group_size == 1
    assert cfg.get_expert_replica_ranks() == list(range(8))
    # PP-confined: stage 1 of a 2-stage job replicates within its own block, never across stages.
    staged = create_config(world_size=8, gpus_per_node=4, pp_size=2, rank=6)
    assert staged.get_expert_replica_ranks() == [4, 5, 6, 7]


def test_expert_replica_ranks_node():
    """Node-local EP with multiple groups: replicas across groups."""
    cfg = create_config(ep_size=4, world_size=16, gpus_per_node=8, rank=0, ep_scope="node")
    replicas = cfg.get_expert_replica_ranks()
    # ep_rank = 0 % 4 = 0; ep_groups_per_node = 8/4 = 2; 2 nodes
    # For node 0: group 0 -> rank 0*8 + 0*4 + 0 = 0, group 1 -> 0*8 + 1*4 + 0 = 4
    # For node 1: group 0 -> 1*8 + 0*4 + 0 = 8, group 1 -> 1*8 + 1*4 + 0 = 12
    assert replicas == [0, 4, 8, 12]


# Summary


def test_summary_contains_mode():
    """Summary string should contain the parallelism mode."""
    cfg = create_config(ep_size=8, cp_size=8, world_size=16, gpus_per_node=8, ep_scope="node")
    s = cfg.summary()
    assert "EP=" in s
    assert "CP=" in s


# HSDP (Hybrid Sharded Data Parallel)


def test_hsdp_off_by_default():
    """Without use_hsdp, the DP path is 1D full-shard: shard width == world, replicate == 1."""
    cfg = create_config(world_size=16, gpus_per_node=8)
    assert cfg.is_hsdp is False
    assert cfg.dp_shard_size == cfg.world_size
    assert cfg.dp_replicate_size == 1


def test_hsdp_multi_node_derives_topology():
    """use_hsdp on multi-node shards within the NVLink domain and replicates across domains."""
    cfg = create_config(world_size=16, gpus_per_node=8, use_hsdp=True)
    assert cfg.is_hsdp is True
    assert cfg.dp_shard_size == 8  # the NVLink domain (one node)
    assert cfg.dp_replicate_size == 2  # two domains
    assert "HSDP" in cfg.summary()


def test_hsdp_single_node_is_noop():
    """use_hsdp on a single NVLink domain has nothing to replicate across → not HSDP, 1D full-shard."""
    cfg = create_config(world_size=8, gpus_per_node=8, use_hsdp=True)
    assert cfg.is_hsdp is False
    assert cfg.dp_shard_size == 8
    assert cfg.dp_replicate_size == 1


def test_hsdp_rejects_ep():
    """use_hsdp with EP is rejected — multi-group EP runs deferred EP-group sharding (use_hsdp would be a
    silent no-op), and a single global EP group under HSDP races the DeepEP combine."""
    try:
        create_config(ep_size=8, world_size=16, gpus_per_node=8, use_hsdp=True, ep_scope="node")
        raise AssertionError("Should have raised ValueError for use_hsdp + EP")
    except ValueError as e:
        assert "use_hsdp" in str(e).lower() and "ep" in str(e).lower()


def test_hsdp_rejects_tp():
    """use_hsdp with TP is rejected — TP builds its own (dp, tp) mesh."""
    try:
        create_config(tp_size=2, world_size=16, gpus_per_node=8, use_hsdp=True)
        raise AssertionError("Should have raised ValueError for use_hsdp + TP")
    except ValueError as e:
        assert "use_hsdp" in str(e).lower() and "tp" in str(e).lower()


def test_hsdp_rejects_expert_tp():
    """use_hsdp with Expert-TP is rejected — ETP builds its own mesh. (Single dispatch group so the
    ETP-specific multi-domain rejection doesn't fire first.)"""
    try:
        create_config(ep_size=2, expert_tp_size=2, world_size=4, gpus_per_node=4, use_hsdp=True)
        raise AssertionError("Should have raised ValueError for use_hsdp + Expert-TP")
    except ValueError as e:
        assert "use_hsdp" in str(e).lower()


# fsdp_reshard_after_forward (ZeRO-3 / FULL_SHARD) — EP=1-only guard


def test_reshard_allowed_on_ep1_paths():
    """FULL_SHARD is allowed on every EP=1 path FSDP2 can plain-all-gather: pure DP, pure TP (dp=1),
    CP, and ep1 MoE. TP *with* data parallelism (dp>1) is rejected separately — the backward re-gather
    hits the TP-sharded DTensor params (see test_reshard_rejects_tp_with_dp)."""
    # pure DP (dense)
    create_config(world_size=8, fsdp_reshard_after_forward=True)
    # pure TP (dp=1: tp_size == world_size, so there is no FSDP DP axis whose re-gather would conflict)
    create_config(tp_size=8, world_size=8, fsdp_reshard_after_forward=True)
    # CP (EP=1)
    create_config(cp_size=2, world_size=8, fsdp_reshard_after_forward=True)
    # ep_size==1 MoE + FSDP-sharded experts → the full ZeRO-3 combo
    create_config(world_size=8, fsdp_reshard_after_forward=True, fsdp_shard_ep1_experts=True)


def test_reshard_after_backward_false_is_gated():
    """fsdp_reshard_after_backward=False keeps params unsharded across a grad-accum window's
    microsteps (drops the per-microstep re-gather — the dominant step cost when NCCL is forced onto
    sockets). It must reject the contradictory FULL_SHARD mode and the unwired TP/PP paths, and pass
    on plain DP."""
    create_config(world_size=8, fsdp_reshard_after_backward=False)
    create_config(cp_size=2, world_size=8, fsdp_reshard_after_backward=False)
    for bad in (
        {"fsdp_reshard_after_forward": True},
        {"tp_size": 2},
        {"pp_size": 2, "world_size": 16},  # 8-rank stages = one NVLink domain each (valid PP shape)
    ):
        try:
            create_config(**{"world_size": 8, "fsdp_reshard_after_backward": False, **bad})
            raise AssertionError(f"Should have raised ValueError for reshard_after_backward=False + {bad}")
        except ValueError as e:
            assert "fsdp_reshard_after_backward" in str(e)


def test_reshard_rejects_ep():
    """FULL_SHARD with real EP (ep_size>1) is rejected — its backward all-gather races DeepEP combine."""
    try:
        create_config(ep_size=8, world_size=8, gpus_per_node=8, fsdp_reshard_after_forward=True)
        raise AssertionError("Should have raised ValueError for fsdp_reshard_after_forward + EP")
    except ValueError as e:
        msg = str(e).lower()
        assert "fsdp_reshard_after_forward" in msg and "ep" in msg


def test_reshard_rejects_pure_etp():
    """FULL_SHARD with pure Expert-TP (ep_size=1, expert_tp_size>1) is rejected — ep_group_size>1."""
    try:
        create_config(expert_tp_size=2, world_size=8, gpus_per_node=8, fsdp_reshard_after_forward=True)
        raise AssertionError("Should have raised ValueError for fsdp_reshard_after_forward + Expert-TP")
    except ValueError as e:
        assert "fsdp_reshard_after_forward" in str(e).lower()


def test_reshard_rejects_tp_with_dp():
    """FULL_SHARD with TP *and* data parallelism (dp>1) is rejected: FSDP2's backward re-gather issues
    a plain c10d all-gather on the TP-sharded DTensor params (no registered DTensor sharding strategy
    → NotImplementedError mid-step). tp_size=2 on world=8 leaves data_parallel_size=4."""
    try:
        create_config(tp_size=2, world_size=8, fsdp_reshard_after_forward=True)
        raise AssertionError("Should have raised ValueError for fsdp_reshard_after_forward + TP + DP")
    except ValueError as e:
        msg = str(e).lower()
        assert "tensor parallelism" in msg and "data parallelism" in msg


# Cross-node EP column-block layout


def _all_global_group_ranks(ep_size, world_size, gpus_per_node):
    """Collect every EP group's rank list for a global-scope config, indexed by group idx."""
    groups = {}
    for rank in range(world_size):
        cfg = create_config(
            ep_size=ep_size,
            world_size=world_size,
            gpus_per_node=gpus_per_node,
            rank=rank,
            ep_scope="global",
        )
        groups[cfg.get_ep_group_idx()] = cfg.get_ep_group_ranks()
    return groups


def test_cross_node_groups_partition_the_world():
    """Every rank lands in exactly one EP group and the groups tile the world."""
    groups = _all_global_group_ranks(ep_size=8, world_size=16, gpus_per_node=8)
    seen = sorted(r for ranks in groups.values() for r in ranks)
    assert seen == list(range(16))  # exact partition, no overlap, no gap
    assert all(len(ranks) == 8 for ranks in groups.values())


def test_cross_node_group_members_contiguous_per_domain():
    """Each EP group's members within an NVLink domain are a CONTIGUOUS device block (DeepEP IPC)."""
    groups = _all_global_group_ranks(ep_size=8, world_size=16, gpus_per_node=8)
    for ranks in groups.values():
        for domain_start in (0, 8):  # two domains of 8
            in_domain = sorted(r - domain_start for r in ranks if domain_start <= r < domain_start + 8)
            # contiguous block: max - min + 1 == len
            assert in_domain == list(range(in_domain[0], in_domain[0] + len(in_domain)))


def test_cross_node_single_global_group_is_contiguous():
    """ep_group_size == world_size → one contiguous group spanning the whole job."""
    cfg = create_config(ep_size=16, world_size=16, gpus_per_node=8, rank=0, ep_scope="global")
    assert cfg.get_ep_group_ranks() == list(range(16))
    assert cfg.num_ep_groups == 1


def test_cross_node_single_member_per_domain_is_strided():
    """Column-block degenerate where each EP group has ONE member per domain (m == 1).

    ``ep_group_size == num_domains`` → members_per_domain = 1 → the groups are pure-RDMA and stride
    by the domain size, ``[g, g+D, ...]``. This is the column-block sub-shape distinct from the
    contiguous-per-domain block and the single-global-group cases the other tests cover.
    """
    # world=16, 2 domains of 8, ep_group_size=2 → 8 EP groups, one member per domain each.
    cfg = create_config(ep_size=2, world_size=16, gpus_per_node=8, rank=9, ep_scope="global")
    assert cfg.num_ep_groups == 8
    # rank 9 → domain 1, position 1 → EP group 1, which strides across domains as [1, 9].
    assert cfg.get_ep_group_idx() == 1
    assert cfg.get_ep_group_ranks() == [1, 9]


def test_cross_node_replica_ranks_share_group_rank():
    """Expert-replica ranks all share the same group-rank across the EP groups."""
    cfg = create_config(ep_size=8, world_size=16, gpus_per_node=8, rank=0, ep_scope="global")
    # rank 0 → ep_rank 0; the other group's matching block-position rank is 4 (group 1 domain-0 block).
    replicas = cfg.get_expert_replica_ranks()
    for r in replicas:
        rcfg = create_config(ep_size=8, world_size=16, gpus_per_node=8, rank=r, ep_scope="global")
        assert rcfg.get_ep_rank() == cfg.get_ep_rank()
    assert sorted(replicas) == [0, 4]


def test_cross_node_layout_rejects_indivisible_by_domains():
    """A cross-node EP group that can't split into equal per-domain blocks is rejected at config time."""
    # world=24 (3 domains of 8), ep_group_size=4 → 4 % 3 != 0 → cannot place equal members per domain.
    try:
        create_config(ep_size=4, world_size=24, gpus_per_node=8, ep_scope="global")
        raise AssertionError("Should have raised ValueError for non-divisible cross-node EP")
    except ValueError as e:
        assert "domain" in str(e).lower()


def test_cross_node_layout_matches_layout_module():
    """ParallelismConfig resolves cross-node EP membership to the column-block layout's CONCRETE ranks.

    ``get_ep_group_ranks`` delegates to the shared column-block helper in
    ``src.distributed.group_layout``, so this pins the END-TO-END result to hard-coded ranks. That
    catches both a regression in the layout math and a wrong argument wired into the helper — a
    self-comparison against the same helper the method calls could not fail on either.
    """
    # world=16, 2 domains of 8, ep_group_size=8 → group 0 = [0,1,2,3, 8,9,10,11],
    # group 1 = [4,5,6,7, 12,13,14,15]. rank 10 sits in domain 1, block 0 → EP group 0.
    cfg = create_config(ep_size=8, world_size=16, gpus_per_node=8, rank=10, ep_scope="global")
    assert cfg.get_ep_group_idx() == 0
    assert cfg.get_ep_group_ranks() == [0, 1, 2, 3, 8, 9, 10, 11]
    assert cfg.global_rank in cfg.get_ep_group_ranks()


def test_node_local_layout_matches_group_layout_module():
    """ParallelismConfig / CPConfig resolve node-local EP and CP membership to CONCRETE ranks.

    Both delegate to the shared node-local helper in ``src.distributed.group_layout``, so pinning
    the END-TO-END result to hard-coded ranks catches a regression in the helper AND a wrong-arg
    delegation (a live self-comparison against that helper could not). Includes the NVL72 case
    where the NVLink domain != the OS node (nvlink_domain_size=16 over two 8-GPU OS nodes), exactly
    where an open-coded copy would drift.
    """
    # EP: 2 NVLink domains of 8, ep_group_size=4 → 4 groups; rank 10 → domain 1, block 0 → group 2.
    cfg = create_config(ep_size=4, world_size=16, gpus_per_node=8, rank=10, ep_scope="node")
    assert cfg.get_ep_group_idx() == 2
    assert cfg.get_ep_group_ranks() == [8, 9, 10, 11]
    # ep_rank 2's replicas: the same block-position rank across all four groups.
    assert cfg.get_ep_rank() == 2
    assert cfg.get_expert_replica_ranks() == [2, 6, 10, 14]

    # CP uses the identical helper (group_size=cp_size): rank 10 → CP group [8, 9, 10, 11].
    cpcfg = create_config(cp_size=4, world_size=16, gpus_per_node=8, rank=10)
    assert cpcfg.get_cp_group_ranks() == [8, 9, 10, 11]

    # NVL72: one 16-GPU NVLink domain across two OS nodes → a single contiguous EP group [0..15].
    nvl = create_config(ep_size=16, world_size=16, gpus_per_node=8, nvlink_domain_size=16, rank=3, ep_scope="node")
    assert nvl.get_ep_group_ranks() == list(range(16))


def test_node_local_groups_partition_the_world():
    """Node-local EP groups tile the world exactly, contiguous within each domain."""
    groups = {}
    for rank in range(16):
        cfg = create_config(ep_size=4, world_size=16, gpus_per_node=8, rank=rank, ep_scope="node")
        groups[cfg.get_ep_group_idx()] = cfg.get_ep_group_ranks()
    assert sorted(r for ranks in groups.values() for r in ranks) == list(range(16))
    # Each group is a contiguous block within one domain.
    assert groups[0] == [0, 1, 2, 3] and groups[1] == [4, 5, 6, 7]
    assert groups[2] == [8, 9, 10, 11] and groups[3] == [12, 13, 14, 15]


# Run all


def test_expert_lora_reaches_validation_through_the_builder():
    """``expert_lora`` must be a CONSTRUCTOR argument, not assigned onto a built config.

    The PP rejection keys on this field, so an assignment made after ``__post_init__`` sails past
    every validator: the run then trains to completion and its PP save writes only frozen base
    weights, because the EP export skips the adapter keys and ``PeftAdapterSaver`` never engages
    (expert-only LoRA leaves no attention ``PeftModel`` to find). Nothing errors, at any point.
    """
    from src.distributed.expert_parallel.config import ExpertLoraSpec
    from src.training.parallelism_args import parallelism_config_from_args

    spec = ExpertLoraSpec(r=8, alpha=16.0)
    with (
        patch(f"{_MOD}.get_global_world_size", return_value=8),
        patch(f"{_MOD}.get_local_world_size", return_value=8),
        patch(f"{_MOD}.get_global_rank", return_value=0),
        patch(f"{_MOD}.is_global_main_process", return_value=True),
    ):
        args = _DistArgs()
        cfg = parallelism_config_from_args(args, expert_lora=spec)
        assert cfg.expert_lora is spec, "the builder must forward expert_lora into the constructor"
        # The same spec under PP must be REJECTED, which only happens if it reached __post_init__.
        # Built directly: the from_args builder refuses pipeline_parallel_size > 1 outright in this
        # release (the schedule engine is not shipped), before the constructor's validators run.
        from src.distributed.parallelism_config import ParallelismConfig

        try:
            ParallelismConfig(ep_size=4, pp_size=2, expert_lora=spec)
            raise AssertionError("PP + expert LoRA must be rejected at config time, but was accepted")
        except ValueError as e:
            assert "Expert LoRA" in str(e), f"wrong validator fired: {e}"

        # Anti-vacuity: the field is not populated by some default, and expert LoRA under expert TP
        # is refused at CONFIG time — before any checkpoint download. (EPConfig repeats the check as
        # defense for hand-built configs that skip ParallelismConfig.)
        args_etp = _DistArgs()
        args_etp.expert_parallel_size = 1
        args_etp.expert_tensor_parallel_size = 2
        assert parallelism_config_from_args(args_etp).expert_lora is None
        try:
            parallelism_config_from_args(args_etp, expert_lora=spec)
            raise AssertionError("expert LoRA under expert_tp_size > 1 must be rejected at config time")
        except ValueError as e:
            assert "Expert LoRA is not supported with expert_tp_size" in str(e), f"wrong validator fired: {e}"


def test_epconfig_second_timing_rejects_expert_lora_with_etp():
    """A HAND-BUILT ``EPConfig`` (bypassing ``ParallelismConfig``) must still refuse expert LoRA
    under ETP — the shared raiser keeps the two timings' messages identical; this pins that the
    second timing actually fires."""
    from src.distributed.expert_parallel.config import EPConfig, ExpertLoraSpec

    try:
        EPConfig(
            ep_size=1,
            world_size=2,
            gpus_per_node=2,
            expert_tp_size=2,
            use_grouped_gemm=False,
            expert_lora=ExpertLoraSpec(r=8, alpha=16.0),
        )
        raise AssertionError("hand-built EPConfig accepted expert LoRA under expert_tp_size > 1")
    except ValueError as e:
        assert "Expert LoRA is not supported with expert_tp_size" in str(e)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
