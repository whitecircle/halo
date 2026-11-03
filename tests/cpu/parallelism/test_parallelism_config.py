#!/usr/bin/env python
import datetime
import os

import pytest
import torch.distributed as dist

from halo.configs.parallelism_arguments import ParallelismArguments, parallelism_config_from_args
from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.group_layout import NodeLocalLayout
from halo.kernels.lowp.spec import LowpSpec
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from halo.trainers.distillation.self_distillation import DistributedSelfDistillationTrainer
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.preference.dpo import DistributedDPOTrainer
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.gloo import run_gloo_ranks
from tests.common.parallelism import create_config, make_parallelism_config, simulated_world


def test_defaults():
    cfg = create_config(world_size=8, gpus_per_node=8)
    assert cfg.world_size == 8
    assert cfg.gpus_per_node == 8
    assert cfg.ep_group_size == 1
    assert cfg.data_parallel_size == 8
    assert cfg.num_nodes == 1


def test_ep_only_computed():
    """EP is orthogonal to DP."""
    cfg = create_config(ep_size=8, world_size=8, gpus_per_node=8)
    assert cfg.ep_group_size == 8
    assert cfg.data_parallel_size == 8
    assert cfg.num_nodes == 1


def test_tp_only_computed():
    cfg = create_config(tp_size=4, world_size=8, gpus_per_node=8)
    assert cfg.data_parallel_size == 2


def test_cp_only_computed():
    cfg = create_config(cp_size=4, world_size=8, gpus_per_node=8)
    assert cfg.data_parallel_size == 2


def test_expert_tp_only_computed():
    cfg = create_config(ep_size=1, expert_tp_size=2, world_size=8, gpus_per_node=8)
    assert cfg.ep_group_size == 2
    assert cfg.data_parallel_size == 4


def test_validate_ep_plus_expert_tp_accepted():
    """The expert-TP reduction runs in token space, outside DeepEP's dispatch->combine span, so it does not deadlock
    the combine barrier."""
    for eps in (2, 4):
        cfg = create_config(ep_size=eps, expert_tp_size=2, world_size=8, gpus_per_node=8)
        assert cfg.ep_group_size == eps * 2
        assert cfg.data_parallel_size == 4


def test_ep_tp_computed():
    """Cross-node EP+TP must be a single EP group."""
    cfg = create_config(ep_size=16, tp_size=4, world_size=16, gpus_per_node=8, ep_scope="global")
    assert cfg.data_parallel_size == 4
    assert cfg.num_nodes == 2


def test_ep_tp_multi_domain_multi_group_rejected():
    """The deferred sweep assumes FSDP shards over the EP group, but EP+TP shards over the (dp, tp) mesh, so
    averaging replica peers corrupts gradients."""
    try:
        create_config(ep_size=8, tp_size=4, world_size=16, gpus_per_node=8, ep_scope="node")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Multi-domain multi-group EP+TP" in str(e), e


def test_ep_cp_computed():
    cfg = create_config(ep_size=8, cp_size=8, world_size=16, gpus_per_node=8, ep_scope="node")
    assert cfg.data_parallel_size == 2


def test_auto_scope_node():
    cfg = create_config(ep_size=4, world_size=16, gpus_per_node=8, ep_scope="auto")
    assert cfg.ep_scope == "node"


def test_auto_scope_global():
    cfg = create_config(ep_size=16, world_size=16, gpus_per_node=8, ep_scope="auto")
    assert cfg.ep_scope == "global"


def test_num_nodes():
    cfg = create_config(world_size=32, gpus_per_node=8)
    assert cfg.num_nodes == 4


def test_rank_identity_is_global_not_node_local():
    """Rank 10 of 16 has local rank 2; a local-rank field would collide ranks 2 and 10 and their group membership."""
    cfg = create_config(rank=10, world_size=16, gpus_per_node=8)
    assert cfg.global_rank == 10
    assert cfg.stage_local_rank == 10  # pp_size == 1, so the stage is the world
    assert cfg.num_nodes == 2

    peer = create_config(rank=2, world_size=16, gpus_per_node=8)
    assert peer.global_rank != cfg.global_rank
    assert peer.get_data_parallel_rank() != cfg.get_data_parallel_rank()


def test_validate_tp_cp_conflict():
    try:
        create_config(tp_size=4, cp_size=4, world_size=16, gpus_per_node=8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "TP + CP is not a supported parallelism combination" in str(e), e
        assert "partition the same ranks twice" in str(e), e


def test_validate_ep_tp_node_local():
    try:
        create_config(ep_size=16, tp_size=16, world_size=16, gpus_per_node=8, ep_scope="global")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "must divide the NVLink domain" in str(e), e


def test_validate_ep_tp_divisibility():
    """Keeps the DTensor all-reduce NVLink-local."""
    try:
        create_config(ep_size=3, tp_size=3, world_size=24, gpus_per_node=8, ep_scope="global")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        # The cross-domain members-per-domain guard fires first for this shape.
        assert "must be a multiple of tensor_parallel_size" in str(e) or "must divide the NVLink domain" in str(e)


def test_validate_ep_node_scope_too_large():
    try:
        create_config(ep_size=16, world_size=16, gpus_per_node=8, ep_scope="node")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Node-local EP group size (16) cannot exceed the NVLink domain (8)" in str(e), e


def test_validate_ep_node_scope_divisibility():
    # 8 % 3 != 0 while world/domain passes, so _reject_ep_group's node-divisibility branch is what fires.
    try:
        create_config(ep_size=3, world_size=8, gpus_per_node=8, ep_scope="node")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "must divide the NVLink domain" in str(e)


def test_validate_ep_global_scope_too_large():
    try:
        create_config(ep_size=32, world_size=16, gpus_per_node=8, ep_scope="global")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "EP group size (32) cannot exceed world size (16)" in str(e), e


def test_validate_ep_global_scope_divisibility():
    try:
        create_config(ep_size=3, world_size=8, gpus_per_node=8, ep_scope="global")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "EP group size (3) must divide world size (8)" in str(e), e


def test_validate_expert_tp_group_tiling_node_scope():
    """The EP group (6) fails to tile the domain first, so this shape never reaches the expert-TP divisor check."""
    try:
        create_config(ep_size=2, expert_tp_size=3, world_size=8, gpus_per_node=8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "EP group size (6) must divide the NVLink domain (8)" in str(e), e


def test_validate_expert_tp_divides_domain_global_scope():
    """The expert-TP divisor gate is reachable only at global scope."""
    try:
        create_config(ep_size=2, expert_tp_size=3, world_size=48, gpus_per_node=8, ep_scope="global")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "expert_tensor_parallel_size (3) must divide the NVLink domain (8)" in str(e), e
        assert "Expert TP groups must stay on NVLink" in str(e), e


def test_cross_node_ep_etp_valid_shape_accepted():
    """One ETP group per domain, single global EP group."""
    cfg = create_config(ep_size=6, expert_tp_size=8, world_size=48, gpus_per_node=8, ep_scope="global")
    assert cfg.ep_group_size == 48
    assert cfg.data_parallel_size == 48 // 8


def test_validate_cp_exceeds_domain_rejected():
    try:
        create_config(cp_size=16, world_size=8, gpus_per_node=8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "context_parallel_size (16) cannot exceed the NVLink domain (8)" in str(e), e


def test_validate_cp_domain_divisibility_rejected():
    try:
        create_config(cp_size=3, world_size=8, gpus_per_node=8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "context_parallel_size (3) must divide the NVLink domain (8)" in str(e), e


def test_validate_tp_exceeds_stage_world_rejected():
    try:
        create_config(tp_size=16, world_size=8, gpus_per_node=8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "tensor_parallel_size (16) cannot exceed world size (8)" in str(e), e


def test_validate_tp_stage_world_divisibility_rejected():
    try:
        create_config(tp_size=3, world_size=8, gpus_per_node=8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "tensor_parallel_size (3) must divide world size (8)" in str(e), e


def test_validate_pure_tp_domain_divisibility_rejected():
    """Fits and divides the stage world, straddles domains."""
    try:
        create_config(tp_size=16, world_size=16, gpus_per_node=8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "must divide the NVLink domain" in str(e), e


def test_validate_multi_domain_multigroup_ep_etp_rejected():
    """Four dispatch groups on two domains, the deferred-DP-off race; its EP+TP twin has its own test."""
    try:
        create_config(ep_size=2, expert_tp_size=2, world_size=16, gpus_per_node=8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "multiple dispatch groups across NVLink domains" in str(e), e


def test_validate_shard_ep1_experts_off_under_tp_cp_rejected():
    """The flag is a silent no-op under TP/CP, whose wraps shard experts unconditionally."""
    for axis_kwargs, named_knob in (
        ({"tp_size": 2}, "tensor_parallel_size=2"),
        ({"cp_size": 2}, "context_parallel_size=2"),
    ):
        try:
            create_config(fsdp_shard_ep1_experts=False, world_size=8, gpus_per_node=8, **axis_kwargs)
            raise AssertionError(f"Should have raised ValueError for {axis_kwargs}")
        except ValueError as e:
            assert "fsdp_shard_ep1_experts=False is not honored under TP or CP" in str(e), e
            assert named_knob in str(e), e
    cfg = create_config(fsdp_shard_ep1_experts=False, world_size=8, gpus_per_node=8)
    assert cfg.experts_fsdp_managed is False


def test_validate_nonpositive_gpus_per_node_rejected():
    try:
        create_config(world_size=8, gpus_per_node=-2)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "gpus_per_node resolved to -2" in str(e), e


def test_validate_negative_nvlink_domain_size_rejected():
    """Python modulo passes a negative multiple through both domain divisibility gates."""
    try:
        create_config(world_size=8, gpus_per_node=8, nvlink_domain_size=-8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "nvlink_domain_size must be positive" in str(e), e


def test_validate_expert_tp_plus_tp():
    try:
        create_config(ep_size=2, tp_size=2, expert_tp_size=2, world_size=8, gpus_per_node=8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "is not a supported parallelism combination" in str(e), e
        assert "expert_tensor_parallel_size=2" in str(e) and "tensor_parallel_size=2" in str(e), e


def test_validate_expert_tp_plus_cp():
    try:
        create_config(ep_size=1, expert_tp_size=2, cp_size=2, world_size=8, gpus_per_node=8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "ETP + CP is not a supported parallelism combination" in str(e), e
        assert "expert_tensor_parallel_size=2" in str(e), e


def test_validate_ep_cp_node_scope_mismatch():
    try:
        create_config(ep_size=4, cp_size=4, world_size=16, gpus_per_node=8, ep_scope="node")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Node-local EP+CP requires ep_group_size=nvlink_domain_size" in str(e), e


def test_validate_ep_cp_global_scope_rejected():
    try:
        create_config(ep_size=16, cp_size=2, world_size=16, gpus_per_node=8, ep_scope="global")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "node-local EP" in str(e) or "ep_scope='node'" in str(e)


def test_validate_ep_tp_size_not_multiple_of_tp():
    # Both sizes divide the node, so the earlier guards pass and ep_size % tp_size == 0 is what fires.
    try:
        create_config(ep_size=2, tp_size=4, world_size=8, gpus_per_node=8, ep_scope="node")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "multiple of tensor_parallel_size" in str(e)


def test_validate_max_concurrent_loading_negative():
    try:
        create_config(max_concurrent_loading=-1, world_size=8, gpus_per_node=8)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "max_concurrent_loading" in str(e)


def _expect_reject(substr=None, **kw):
    try:
        create_config(**kw)
    except ValueError as e:
        if substr:
            assert substr in str(e), f"expected {substr!r} in error, got: {e}"
        return
    raise AssertionError(f"config {kw} should have raised ValueError")


def test_guard_size_below_one():
    for field_name, knob in (
        ("ep_size", "expert_parallel_size"),
        ("tp_size", "tensor_parallel_size"),
        ("cp_size", "context_parallel_size"),
        ("expert_tp_size", "expert_tensor_parallel_size"),
    ):
        _expect_reject(f"{knob} must be >= 1", **{field_name: 0, "world_size": 8, "gpus_per_node": 8})


def test_guard_bad_ep_scope():
    _expect_reject("ep_scope", ep_size=8, ep_scope="rack", world_size=8, gpus_per_node=8)


def test_guard_world_not_multiple_of_gpus_per_node():
    _expect_reject("world_size (12) must be divisible by gpus_per_node (8).", world_size=12, gpus_per_node=8)


def test_guard_nvlink_domain_multiple_of_node():
    # nvlink_domain_size must be a whole number of OS nodes
    _expect_reject("nvlink_domain_size", world_size=16, gpus_per_node=8, nvlink_domain_size=12)


def test_guard_lowp_bad_precision():
    _expect_reject("lowp_precision", lowp=LowpSpec(precision="int4"), world_size=8, gpus_per_node=8)


def test_guard_lowp_applied_nowhere():
    _expect_reject(
        "apply to nothing",
        lowp=LowpSpec(precision="fp8", apply_dense_mlp=False, apply_moe_experts=False),
        world_size=8,
        gpus_per_node=8,
    )


def test_guard_lowp_negative_keep_blocks():
    _expect_reject(">= 0", lowp=LowpSpec(precision="fp8", keep_first_blocks=-1), world_size=8, gpus_per_node=8)


def test_lowp_valid_combos_accepted():
    # No backend knob; the native DeepGEMM kernel is opt-in via HALO_DEEPGEMM_NATIVE.
    for lowp in (
        LowpSpec(precision="fp8"),
        LowpSpec(precision="fp4"),
        LowpSpec(precision="fp8", apply_moe_experts=False),
    ):
        cfg = create_config(world_size=8, gpus_per_node=8, lowp=lowp)
        assert cfg.lowp.precision in ("fp8", "fp4")


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
    cfg = create_config(use_grouped_gemm=True, world_size=8, gpus_per_node=8)
    assert cfg.mode_string == "gmm"


def test_mode_string_expert_tp():
    cfg = create_config(ep_size=1, expert_tp_size=2, world_size=8, gpus_per_node=8)
    assert "expert-tp" in cfg.mode_string


def test_mode_string_none():
    cfg = create_config(use_grouped_gemm=False, world_size=8, gpus_per_node=8)
    assert cfg.mode_string is None


def test_requires_rdma():
    cfg = create_config(ep_size=16, world_size=16, gpus_per_node=8, ep_scope="global")
    assert cfg.requires_rdma is True


def test_no_rdma_node_local():
    cfg = create_config(ep_size=8, world_size=16, gpus_per_node=8, ep_scope="node")
    assert cfg.requires_rdma is False


def test_requires_rdma_multi_domain_global_small_group():
    """The column-block layout places members of every EP group in every domain."""
    cfg = create_config(ep_size=8, world_size=16, gpus_per_node=8, ep_scope="global")
    assert cfg.requires_rdma is True


def test_needs_ep_layers_ep():
    cfg = create_config(ep_size=8, world_size=8, gpus_per_node=8)
    assert cfg.needs_ep_layers is True


def test_needs_ep_layers_gmm():
    cfg = create_config(use_grouped_gemm=True, world_size=8, gpus_per_node=8)
    assert cfg.needs_ep_layers is True


def test_no_ep_layers_disabled():
    cfg = create_config(use_grouped_gemm=False, world_size=8, gpus_per_node=8)
    assert cfg.needs_ep_layers is False


def test_boolean_mode_properties():
    cfg = create_config(ep_size=8, tp_size=4, world_size=8, gpus_per_node=8)
    assert cfg.has_ep_group is True
    assert cfg.is_tp_mode is True
    assert (cfg.has_ep_group and cfg.is_tp_mode) is True
    assert cfg.is_cp_mode is False
    assert (cfg.has_ep_group and cfg.is_cp_mode) is False


def test_default_sizes_and_mode_flags():
    cfg = create_config(world_size=8, gpus_per_node=8)
    assert (cfg.ep_size, cfg.cp_size, cfg.tp_size, cfg.expert_tp_size) == (1, 1, 1, 1)
    assert not (cfg.has_ep_group or cfg.is_cp_mode or cfg.is_tp_mode)
    assert cfg.use_grouped_gemm is True
    assert cfg.needs_ep_layers is True


@pytest.mark.parametrize(
    ("axis", "ep_cp_tp_flags"),
    [
        ({"ep_size": 8}, (True, False, False)),
        ({"cp_size": 4}, (False, True, False)),
        ({"tp_size": 4}, (False, False, True)),
    ],
)
def test_a_single_axis_sets_only_its_own_mode_flag(axis, ep_cp_tp_flags):
    cfg = create_config(world_size=8, gpus_per_node=8, **axis)
    assert (cfg.has_ep_group, cfg.is_cp_mode, cfg.is_tp_mode) == ep_cp_tp_flags


def test_ep_cp_mode_flags():
    cfg = create_config(ep_size=8, cp_size=8, world_size=16, gpus_per_node=8, ep_scope="node")
    assert cfg.has_ep_group and cfg.is_cp_mode and (cfg.has_ep_group and cfg.is_cp_mode)
    assert not (cfg.is_tp_mode or (cfg.has_ep_group and cfg.is_tp_mode))


def test_num_ep_groups_node():
    cfg = create_config(ep_size=4, world_size=16, gpus_per_node=8, ep_scope="node")
    assert cfg.num_ep_groups == 4


def test_num_ep_groups_global():
    cfg = create_config(ep_size=8, world_size=16, gpus_per_node=8, ep_scope="global")
    assert cfg.num_ep_groups == 2


@pytest.mark.parametrize(
    "topology",
    [
        {"ep_size": 8, "world_size": 8, "gpus_per_node": 8, "ep_scope": "node"},
        {"ep_size": 2, "world_size": 8, "gpus_per_node": 8, "ep_scope": "node"},
        {"ep_size": 4, "world_size": 16, "gpus_per_node": 8, "ep_scope": "node"},
        {"ep_size": 8, "world_size": 16, "gpus_per_node": 8, "ep_scope": "node"},
        {"ep_size": 1, "expert_tp_size": 2, "world_size": 8, "gpus_per_node": 8, "ep_scope": "node"},
        # EPConfig tiles the stage with one singleton group per rank at ep_group_size == 1.
        {"ep_size": 1, "world_size": 8, "gpus_per_node": 8, "ep_scope": "node"},
        {"ep_size": 8, "world_size": 16, "gpus_per_node": 8, "ep_scope": "global"},
        {"ep_size": 16, "world_size": 16, "gpus_per_node": 8, "ep_scope": "global"},
        {"ep_size": 8, "world_size": 32, "gpus_per_node": 8, "ep_scope": "global"},
    ],
)
def test_num_ep_groups_equals_the_groups_the_layout_actually_builds(topology):
    """A wrong count divides expert grads by too much or lets a sharded save merge duplicated experts."""
    cfg = create_config(**topology)
    groups = {cfg.ep_layout.rank_and_group(r)[1] for r in range(cfg.stage_world_size)}
    assert len(groups) == cfg.num_ep_groups, f"{sorted(groups)} vs num_ep_groups={cfg.num_ep_groups}"
    assert groups == set(range(cfg.num_ep_groups)), sorted(groups)


_GLOO_WORLD = 8
# A construction desync would otherwise hang in the gloo group for the default timeout.
_GLOO_TIMEOUT_SEC = 180


def _ep_group_size_one_worker(rank: int, out_dir: str) -> None:
    ep = EPConfig(
        ep_size=1,
        stage_world_size=_GLOO_WORLD,
        nvlink_domain_size=_GLOO_WORLD,
        use_grouped_gemm=False,
        experts_fsdp_managed=True,
    )
    pc = make_parallelism_config(world_size=_GLOO_WORLD, gpus_per_node=_GLOO_WORLD, rank=rank, ep_size=1)
    failures = []
    if ep.ep_group_size != 1 or not ep.needs_expert_grad_sync:
        failures.append(f"precondition: ep_group_size={ep.ep_group_size} sync={ep.needs_expert_grad_sync}")
    if pc.num_ep_groups != ep.num_ep_groups:
        failures.append(f"num_ep_groups {pc.num_ep_groups} != EPConfig's {ep.num_ep_groups}")
    if ep.expert_replica_ranks != list(range(_GLOO_WORLD)):
        failures.append(f"replicas {ep.expert_replica_ranks} != the whole block")
    if dist.get_world_size(ep.expert_replica_group) != _GLOO_WORLD:
        failures.append(f"replica group holds {dist.get_world_size(ep.expert_replica_group)} of {_GLOO_WORLD}")
    with open(os.path.join(out_dir, f"result_{rank}.txt"), "w") as fh:
        fh.write("PASS" if not failures else "FAIL: " + "; ".join(failures))


def test_ep_group_size_one_agrees_with_a_real_eight_rank_ep_config(tmp_path):
    """The default shape of every dense and ep_size=1 run; a ``num_ep_groups`` of 1 under-counts the replica set the
    deferred sweep divides by."""
    out_dir = str(tmp_path)
    run_gloo_ranks(
        _ep_group_size_one_worker, _GLOO_WORLD, out_dir, pg_timeout=datetime.timedelta(seconds=_GLOO_TIMEOUT_SEC)
    )
    for rank in range(_GLOO_WORLD):
        with open(os.path.join(out_dir, f"result_{rank}.txt")) as fh:
            assert fh.read() == "PASS", f"rank {rank}"


def test_is_expert_tp_mode_property():
    etp = create_config(ep_size=1, expert_tp_size=2, world_size=8, gpus_per_node=8)
    assert etp.is_expert_tp_mode is True
    assert etp.has_ep_group is True
    plain = create_config(ep_size=8, world_size=8, gpus_per_node=8)
    assert plain.is_expert_tp_mode is False


def test_requires_rdma_false_when_global_fits_in_domain():
    cfg = create_config(ep_size=8, world_size=8, gpus_per_node=8, ep_scope="global")
    assert cfg.requires_rdma is False


def test_mode_string_ep_expert_tp_combo():
    cfg = create_config(ep_size=2, expert_tp_size=2, world_size=8, gpus_per_node=8)
    assert cfg.mode_string == "ep-expert-tp"


def test_data_parallel_rank_expert_tp_path():
    """ETP partners share the same DP batch, so ranks 0 and 1 both get dp_rank 0."""
    cfg0 = create_config(ep_size=1, expert_tp_size=2, world_size=8, gpus_per_node=8, rank=0)
    cfg1 = create_config(ep_size=1, expert_tp_size=2, world_size=8, gpus_per_node=8, rank=1)
    assert cfg0.get_data_parallel_rank() == cfg1.get_data_parallel_rank()


def test_summary_standard_ddp_string():
    """torchrun lands on FSDP2 and accelerate on DDP, so neither is named as the axis."""
    cfg = create_config(world_size=8, gpus_per_node=8)
    s = cfg.summary()
    assert "DATA PARALLEL" in s
    assert "DDP" not in s


def _DistArgs() -> ParallelismArguments:
    """The real dataclass, since the builder reads its fields directly and a missing one is an AttributeError at
    launch."""
    return ParallelismArguments(
        expert_parallel_size=8,
        ep_scope="node",
        max_concurrent_loading=1,
        fsdp_shard_ep1_experts=False,
    )


def test_parallelism_config_from_args_basic():
    with simulated_world(world_size=8, gpus_per_node=8):
        # Non-default values, so the asserts fail if the builder drops them back to ParallelismConfig defaults.
        args = _DistArgs()
        args.fp32_grad_reduce = True
        args.max_concurrent_loading = 3
        cfg = parallelism_config_from_args(args, trainer_cls=DistributedTrainerMixin)
        assert cfg.ep_size == 8
        assert cfg.expert_tp_size == 1
        assert cfg.use_grouped_gemm is True
        assert cfg.lowp.precision == "bf16"
        assert cfg.fp32_grad_reduce is True
        assert cfg.max_concurrent_loading == 3


def test_parallelism_config_from_args_builds_pp_only_for_trainers_that_support_it():
    """A trainer keeping a refusal is refused before any load; PP+TP stays refused by the axis allowlist."""
    args = ParallelismArguments(pipeline_parallel_size=2, nvlink_domain_size=8)
    with simulated_world(world_size=16, gpus_per_node=8):
        with pytest.raises(ValueError, match="DistributedTrainerMixin does not support Pipeline Parallelism"):
            parallelism_config_from_args(args, trainer_cls=DistributedTrainerMixin)

        cfg = parallelism_config_from_args(args, trainer_cls=DistributedSFTTrainer)
        assert cfg.pp_size == 2
        assert cfg.stage_world_size == 8
        assert cfg.data_parallel_size == 8

        pp_tp = ParallelismArguments(pipeline_parallel_size=2, tensor_parallel_size=2, nvlink_domain_size=8)
        with pytest.raises(ValueError, match="is not a supported parallelism combination") as excinfo:
            parallelism_config_from_args(pp_tp, trainer_cls=DistributedSFTTrainer)
        assert "pipeline_parallel_size=2" in str(excinfo.value)
        assert "tensor_parallel_size=2" in str(excinfo.value)


@pytest.mark.parametrize(
    ("pp_split", "match"),
    [
        ([4], r"pipeline_split has 1 entries but pipeline_parallel_size=2"),
        ([0, 4], r"pipeline_split entries must be >= 1 decoder layer, got \[0, 4\]"),
        ([3, -1], r"pipeline_split entries must be >= 1 decoder layer, got \[3, -1\]"),
    ],
    ids=["wrong-length", "empty-stage", "negative-stage"],
)
def test_a_pipeline_split_must_give_every_stage_a_decoder_layer(pp_split, match):
    """An empty stage has nothing to run between its P2P edges."""
    assert create_config(world_size=16, pp_size=2, pp_split=[3, 1]).pp_split == [3, 1]
    with pytest.raises(ValueError, match=match):
        create_config(world_size=16, pp_size=2, pp_split=pp_split)


def test_parallelism_config_from_args_names_the_trainers_own_pp_refusal():
    args = ParallelismArguments(pipeline_parallel_size=2, nvlink_domain_size=8)
    with simulated_world(world_size=16, gpus_per_node=8):
        with pytest.raises(ValueError, match="DistributedSelfDistillationTrainer does not support Pipeline") as err:
            parallelism_config_from_args(args, trainer_cls=DistributedSelfDistillationTrainer)
    assert DistributedSelfDistillationTrainer.PP_REFUSAL in str(err.value)


def test_parallelism_config_from_args_rejects_cp_when_unsupported():
    args = ParallelismArguments(context_parallel_size=2)
    with simulated_world(world_size=8, gpus_per_node=8):
        with pytest.raises(ValueError, match="DistributedDPOTrainer does not support Context Parallelism"):
            parallelism_config_from_args(args, trainer_cls=DistributedDPOTrainer)
        assert parallelism_config_from_args(args, trainer_cls=DistributedSFTTrainer).cp_size == 2


def test_parallelism_config_from_args_rejects_lowp_when_disallowed():
    args = _DistArgs()
    args.lowp_precision = "fp8"
    with simulated_world(world_size=8, gpus_per_node=8):
        try:
            parallelism_config_from_args(args, trainer_cls=DistributedTrainerMixin)
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "lowp_precision" in str(e)


def test_parallelism_config_from_args_lowp_allowed_for_sft():
    args = _DistArgs()
    args.lowp_precision = "fp8"
    with simulated_world(world_size=8, gpus_per_node=8):
        cfg = parallelism_config_from_args(args, trainer_cls=DistributedSFTTrainer)
        assert cfg.lowp.precision == "fp8"


def _ep_coords(cfg) -> tuple[int, int]:
    return cfg.ep_layout.rank_and_group(cfg.stage_local_rank)


def test_ep_rank_node():
    cfg = create_config(ep_size=4, world_size=16, gpus_per_node=8, rank=5)
    assert _ep_coords(cfg)[0] == 1


def test_ep_rank_global():
    cfg = create_config(ep_size=8, world_size=16, gpus_per_node=8, rank=3, ep_scope="global")
    # m=4, rank 3: domain 0, pos 3, group_idx 0, ep_rank 0*4 + 3%4 = 3.
    assert _ep_coords(cfg)[0] == 3


def test_ep_group_idx_node():
    cfg = create_config(ep_size=4, world_size=16, gpus_per_node=8, rank=12, ep_scope="node")
    # node 1, local group 1, 2 groups per node: 1*2 + 1 = 3.
    assert _ep_coords(cfg)[1] == 3


def test_ep_group_idx_global():
    cfg = create_config(ep_size=8, world_size=16, gpus_per_node=8, rank=5, ep_scope="global")
    # m=4: rank 5 → pos 5, group_idx = 5 // 4 = 1.
    assert _ep_coords(cfg)[1] == 1


def test_cp_rank():
    assert NodeLocalLayout(world_size=8, group_size=4, domain_size=8).rank_and_group(6)[0] == 2


def test_data_parallel_rank_tp():
    cfg = create_config(tp_size=4, world_size=8, gpus_per_node=8, rank=5)
    assert cfg.get_data_parallel_rank() == 1


def test_data_parallel_rank_no_parallelism():
    cfg = create_config(world_size=8, gpus_per_node=8, rank=3)
    assert cfg.get_data_parallel_rank() == 3


def test_ep_group_ranks_node():
    cfg = create_config(ep_size=4, world_size=16, gpus_per_node=8, rank=0, ep_scope="node")
    assert cfg.ep_layout.group_ranks(_ep_coords(cfg)[1]) == [0, 1, 2, 3]


def test_ep_group_ranks_global():
    cfg = create_config(ep_size=8, world_size=16, gpus_per_node=8, rank=0, ep_scope="global")
    # m=4, group_idx=0: domain 0 block [0,1,2,3] + domain 1 block [8,9,10,11].
    assert cfg.ep_layout.group_ranks(_ep_coords(cfg)[1]) == [0, 1, 2, 3, 8, 9, 10, 11]


def test_cp_group_ranks():
    layout = NodeLocalLayout(world_size=8, group_size=4, domain_size=8)
    assert layout.group_ranks(layout.rank_and_group(4)[1]) == [4, 5, 6, 7]


def test_expert_replica_ranks_single_group():
    cfg = create_config(ep_size=8, world_size=8, gpus_per_node=8, rank=3)
    assert cfg.num_ep_groups == 1
    assert cfg.ep_layout.replica_ranks(_ep_coords(cfg)[0]) == [3]


def test_expert_replica_ranks_at_ep_group_size_one_span_the_stage():
    """Every rank holds the full expert set, so the whole rank block is one replica set, the single group
    ``EPConfig`` builds."""
    cfg = create_config(world_size=8, gpus_per_node=8, rank=3)
    assert cfg.ep_group_size == 1
    assert cfg.ep_layout.replica_ranks(0) == list(range(8))
    # Stage-local: stage 1 of a 2-stage job replicates within its own block, never across stages.
    staged = create_config(world_size=8, gpus_per_node=4, pp_size=2, rank=6)
    assert [staged.stage_base_rank + r for r in staged.ep_layout.replica_ranks(0)] == [4, 5, 6, 7]


def test_expert_replica_ranks_node():
    cfg = create_config(ep_size=4, world_size=16, gpus_per_node=8, rank=0, ep_scope="node")
    replicas = cfg.ep_layout.replica_ranks(_ep_coords(cfg)[0])
    # ep_rank 0 across 2 groups on each of 2 nodes, stride 4 within a node.
    assert replicas == [0, 4, 8, 12]


def test_summary_contains_mode():
    cfg = create_config(ep_size=8, cp_size=8, world_size=16, gpus_per_node=8, ep_scope="node")
    s = cfg.summary()
    assert "EP=" in s
    assert "CP=" in s


def test_hsdp_off_by_default():
    cfg = create_config(world_size=16, gpus_per_node=8)
    assert cfg.is_hsdp is False
    assert cfg.dp_shard_size == cfg.world_size
    assert cfg.dp_replicate_size == 1


def test_hsdp_multi_node_derives_topology():
    cfg = create_config(world_size=16, gpus_per_node=8, use_hsdp=True)
    assert cfg.is_hsdp is True
    assert cfg.dp_shard_size == 8
    assert cfg.dp_replicate_size == 2
    assert "HSDP" in cfg.summary()


def test_hsdp_single_node_is_noop():
    cfg = create_config(world_size=8, gpus_per_node=8, use_hsdp=True)
    assert cfg.is_hsdp is False
    assert cfg.dp_shard_size == 8
    assert cfg.dp_replicate_size == 1


def test_hsdp_rejects_ep():
    """Multi-group EP runs deferred EP-group sharding (use_hsdp would be a silent no-op); a single global EP group
    under HSDP races the DeepEP combine."""
    try:
        create_config(ep_size=8, world_size=16, gpus_per_node=8, use_hsdp=True, ep_scope="node")
        raise AssertionError("Should have raised ValueError for use_hsdp + EP")
    except ValueError as e:
        assert "use_hsdp" in str(e).lower() and "ep" in str(e).lower()


def test_hsdp_rejects_tp():
    """TP builds its own (dp, tp) mesh."""
    try:
        create_config(tp_size=2, world_size=16, gpus_per_node=8, use_hsdp=True)
        raise AssertionError("Should have raised ValueError for use_hsdp + TP")
    except ValueError as e:
        assert "use_hsdp" in str(e).lower() and "tp" in str(e).lower()


def test_hsdp_rejects_expert_tp():
    """ETP builds its own mesh. A single dispatch group keeps the ETP multi-domain rejection from firing first."""
    try:
        create_config(ep_size=2, expert_tp_size=2, world_size=4, gpus_per_node=4, use_hsdp=True)
        raise AssertionError("Should have raised ValueError for use_hsdp + Expert-TP")
    except ValueError as e:
        assert "use_hsdp" in str(e).lower()


def test_reshard_allowed_on_ep1_paths():
    """TP with dp>1 is rejected separately (test_reshard_rejects_tp_with_dp)."""
    create_config(world_size=8, fsdp_reshard_after_forward=True)
    # dp=1, so there is no FSDP DP axis whose re-gather would conflict.
    create_config(tp_size=8, world_size=8, fsdp_reshard_after_forward=True)
    create_config(cp_size=2, world_size=8, fsdp_reshard_after_forward=True)
    create_config(world_size=8, fsdp_reshard_after_forward=True, fsdp_shard_ep1_experts=True)


def test_reshard_after_backward_false_is_gated():
    """Keeps params unsharded across a grad-accum window; rejects contradictory FULL_SHARD and the unwired TP/PP
    paths."""
    create_config(world_size=8, fsdp_reshard_after_backward=False)
    create_config(cp_size=2, world_size=8, fsdp_reshard_after_backward=False)
    for bad, named_knob in (
        ({"fsdp_reshard_after_forward": True}, "fsdp_reshard_after_forward"),
        ({"tp_size": 2}, "tensor_parallel_size=2"),
        # 8-rank stages = one NVLink domain each (valid PP shape)
        ({"pp_size": 2, "world_size": 16}, "pipeline_parallel_size=2"),
    ):
        try:
            create_config(**{"world_size": 8, "fsdp_reshard_after_backward": False, **bad})
            raise AssertionError(f"Should have raised ValueError for reshard_after_backward=False + {bad}")
        except ValueError as e:
            assert "fsdp_reshard_after_backward" in str(e)
            assert named_knob in str(e), e


@pytest.mark.parametrize(
    "shape",
    [
        {"world_size": 1, "gpus_per_node": 1},  # no peers: nothing to reduce, the toggle stays inert
        {},
        {"cp_size": 2},
        {"tp_size": 2},  # TP+DP: FSDP2 reduces over the mesh's dp dimension
        {"ep_size": 8},  # one EP group: expert/router hooks gate on the same window-end signal
        {"ep_size": 2},  # multi-group EP: experts synced by the post-backward sweep, once per step
        {"world_size": 16, "use_hsdp": True},
        {"fsdp_reshard_after_backward": False},
        {"fp32_grad_reduce": True},
    ],
)
def test_defer_grad_sync_accepted_wherever_fsdp2_reduces_per_microstep(shape):
    create_config(fsdp_defer_grad_sync=True, **shape)


@pytest.mark.parametrize(
    ("shape", "named_knob"),
    [
        # the schedule already reduces once per optimizer step
        ({"world_size": 16, "pp_size": 2}, "pipeline_parallel_size=2"),
        # pure TP: data_parallel_size=1 applies no FSDP2 wrap
        ({"tp_size": 8}, "tensor_parallel_size=8"),
    ],
)
def test_defer_grad_sync_rejected_without_a_per_microstep_reduce(shape, named_knob):
    """The refusal names the knobs a user sets."""
    with pytest.raises(ValueError, match="fsdp_defer_grad_sync") as excinfo:
        create_config(fsdp_defer_grad_sync=True, **shape)
    assert named_knob in str(excinfo.value)


@pytest.mark.parametrize("fp32_grad_reduce", [False, True])
def test_defer_grad_sync_rejected_under_full_shard(fp32_grad_reduce):
    """Deferring holds every module's full unsharded gradient across the window, the state ZeRO-3 shards."""
    with pytest.raises(
        ValueError, match="fsdp_defer_grad_sync=True contradicts fsdp_reshard_after_forward=True"
    ) as excinfo:
        create_config(fsdp_defer_grad_sync=True, fsdp_reshard_after_forward=True, fp32_grad_reduce=fp32_grad_reduce)
    assert "fsdp_reshard_after_forward=False" in str(excinfo.value)
    create_config(fsdp_defer_grad_sync=False, fsdp_reshard_after_forward=True, fp32_grad_reduce=fp32_grad_reduce)


def test_reshard_rejects_ep():
    """Its backward all-gather races DeepEP combine."""
    try:
        create_config(ep_size=8, world_size=8, gpus_per_node=8, fsdp_reshard_after_forward=True)
        raise AssertionError("Should have raised ValueError for fsdp_reshard_after_forward + EP")
    except ValueError as e:
        msg = str(e).lower()
        assert "fsdp_reshard_after_forward" in msg and "ep" in msg


def test_reshard_rejects_pure_etp():
    try:
        create_config(expert_tp_size=2, world_size=8, gpus_per_node=8, fsdp_reshard_after_forward=True)
        raise AssertionError("Should have raised ValueError for fsdp_reshard_after_forward + Expert-TP")
    except ValueError as e:
        assert "fsdp_reshard_after_forward" in str(e).lower()


def test_reshard_rejects_tp_with_dp():
    """FSDP2's backward re-gather issues a plain c10d all-gather on TP-sharded DTensor params, a NotImplementedError
    mid-step."""
    try:
        create_config(tp_size=2, world_size=8, fsdp_reshard_after_forward=True)
        raise AssertionError("Should have raised ValueError for fsdp_reshard_after_forward + TP + DP")
    except ValueError as e:
        msg = str(e).lower()
        assert "tensor parallelism" in msg and "data parallelism" in msg


def _all_global_group_ranks(ep_size, world_size, gpus_per_node):
    groups = {}
    for rank in range(world_size):
        cfg = create_config(
            ep_size=ep_size,
            world_size=world_size,
            gpus_per_node=gpus_per_node,
            rank=rank,
            ep_scope="global",
        )
        groups[_ep_coords(cfg)[1]] = cfg.ep_layout.group_ranks(_ep_coords(cfg)[1])
    return groups


def test_cross_node_groups_partition_the_world():
    groups = _all_global_group_ranks(ep_size=8, world_size=16, gpus_per_node=8)
    seen = sorted(r for ranks in groups.values() for r in ranks)
    assert seen == list(range(16))
    assert all(len(ranks) == 8 for ranks in groups.values())


def test_cross_node_group_members_contiguous_per_domain():
    """DeepEP IPC needs each group's members in a domain to form a contiguous device block."""
    groups = _all_global_group_ranks(ep_size=8, world_size=16, gpus_per_node=8)
    for ranks in groups.values():
        for domain_start in (0, 8):
            in_domain = sorted(r - domain_start for r in ranks if domain_start <= r < domain_start + 8)
            assert in_domain == list(range(in_domain[0], in_domain[0] + len(in_domain)))


def test_cross_node_single_global_group_is_contiguous():
    cfg = create_config(ep_size=16, world_size=16, gpus_per_node=8, rank=0, ep_scope="global")
    assert cfg.ep_layout.group_ranks(0) == list(range(16))
    assert cfg.num_ep_groups == 1


def test_cross_node_single_member_per_domain_is_strided():
    """m == 1: pure-RDMA groups stride by the domain size."""
    cfg = create_config(ep_size=2, world_size=16, gpus_per_node=8, rank=9, ep_scope="global")
    assert cfg.num_ep_groups == 8
    # rank 9 → domain 1, position 1 → EP group 1, which strides across domains as [1, 9].
    assert _ep_coords(cfg)[1] == 1
    assert cfg.ep_layout.group_ranks(1) == [1, 9]


def test_cross_node_replica_ranks_share_group_rank():
    cfg = create_config(ep_size=8, world_size=16, gpus_per_node=8, rank=0, ep_scope="global")
    # rank 0 → ep_rank 0; the other group's matching block-position rank is 4 (group 1 domain-0 block).
    replicas = cfg.ep_layout.replica_ranks(_ep_coords(cfg)[0])
    for r in replicas:
        assert cfg.ep_layout.rank_and_group(r)[0] == _ep_coords(cfg)[0]
    assert sorted(replicas) == [0, 4]


def test_cross_node_layout_rejects_indivisible_by_domains():
    # world=24 (3 domains of 8), ep_group_size=4 → 4 % 3 != 0 → cannot place equal members per domain.
    try:
        create_config(ep_size=4, world_size=24, gpus_per_node=8, ep_scope="global")
        raise AssertionError("Should have raised ValueError for non-divisible cross-node EP")
    except ValueError as e:
        assert "domain" in str(e).lower()


def test_cross_node_layout_matches_layout_module():
    """Pinned to hard-coded ranks so a layout-math regression or a wrong argument to ``ParallelismConfig.ep_layout``
    fails."""
    # world=16, 2 domains of 8, ep_group_size=8 → group 0 = [0,1,2,3, 8,9,10,11],
    # group 1 = [4,5,6,7, 12,13,14,15]. rank 10 sits in domain 1, block 0 → EP group 0.
    cfg = create_config(ep_size=8, world_size=16, gpus_per_node=8, rank=10, ep_scope="global")
    assert _ep_coords(cfg)[1] == 0
    assert cfg.ep_layout.group_ranks(0) == [0, 1, 2, 3, 8, 9, 10, 11]


def test_node_local_layout_matches_group_layout_module():
    """Includes the NVL72 case where the NVLink domain is not the OS node."""
    # EP: 2 NVLink domains of 8, ep_group_size=4 → 4 groups; rank 10 → domain 1, block 0 → group 2.
    cfg = create_config(ep_size=4, world_size=16, gpus_per_node=8, rank=10, ep_scope="node")
    assert _ep_coords(cfg) == (2, 2)
    assert cfg.ep_layout.group_ranks(2) == [8, 9, 10, 11]
    assert cfg.ep_layout.replica_ranks(2) == [2, 6, 10, 14]

    cp = NodeLocalLayout(world_size=16, group_size=4, domain_size=8)
    assert cp.group_ranks(cp.rank_and_group(10)[1]) == [8, 9, 10, 11]

    # NVL72: one 16-GPU NVLink domain across two OS nodes → a single contiguous EP group [0..15].
    nvl = create_config(ep_size=16, world_size=16, gpus_per_node=8, nvlink_domain_size=16, rank=3, ep_scope="node")
    assert nvl.ep_layout.group_ranks(0) == list(range(16))


def test_node_local_groups_partition_the_world():
    groups = {}
    for rank in range(16):
        cfg = create_config(ep_size=4, world_size=16, gpus_per_node=8, rank=rank, ep_scope="node")
        groups[_ep_coords(cfg)[1]] = cfg.ep_layout.group_ranks(_ep_coords(cfg)[1])
    assert sorted(r for ranks in groups.values() for r in ranks) == list(range(16))
    assert groups[0] == [0, 1, 2, 3] and groups[1] == [4, 5, 6, 7]
    assert groups[2] == [8, 9, 10, 11] and groups[3] == [12, 13, 14, 15]


def test_expert_lora_reaches_validation_through_the_builder():
    """The PP rejection keys on this field; assigned after ``__post_init__`` it passes every validator and the PP
    save writes only frozen base weights."""
    spec = ExpertLoraSpec(r=8, alpha=16.0)
    with simulated_world(world_size=8, gpus_per_node=8):
        args = _DistArgs()
        cfg = parallelism_config_from_args(args, trainer_cls=DistributedTrainerMixin, expert_lora=spec)
        assert cfg.expert_lora is spec, "the builder must forward expert_lora into the constructor"

        # Anti-vacuity: no default populates the field, and expert LoRA under expert TP is refused at config time.
        args_etp = _DistArgs()
        args_etp.expert_parallel_size = 1
        args_etp.expert_tensor_parallel_size = 2
        assert parallelism_config_from_args(args_etp, trainer_cls=DistributedTrainerMixin).expert_lora is None
        try:
            parallelism_config_from_args(args_etp, trainer_cls=DistributedTrainerMixin, expert_lora=spec)
            raise AssertionError("expert LoRA under expert_tp_size > 1 must be rejected at config time")
        except ValueError as e:
            assert "Expert LoRA is not supported with expert_tensor_parallel_size" in str(e), (
                f"wrong validator fired: {e}"
            )

    # This shape builds without the spec, so expert LoRA is the only reason left for the refusal.
    with simulated_world(world_size=16, gpus_per_node=8):
        args_pp = _DistArgs()
        args_pp.pipeline_parallel_size = 2
        args_pp.nvlink_domain_size = 8
        assert parallelism_config_from_args(args_pp, trainer_cls=DistributedSFTTrainer).pp_size == 2
        with pytest.raises(ValueError, match="Expert LoRA"):
            parallelism_config_from_args(args_pp, trainer_cls=DistributedSFTTrainer, expert_lora=spec)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
