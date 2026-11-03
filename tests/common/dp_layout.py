"""The data-parallel side of a run's layout, read off the model, since a loader or wrap that ignored a field would
still train."""

from collections.abc import Callable, Mapping

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard

from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.distributed.fsdp import fsdp2_param_groups, reshard_fsdp2_modules
from halo.distributed.mesh import MeshDim, has_tp_dim, mesh_dim_names
from halo.runtime.dtensor import to_local
from tests.common.distributed import group_max_abs_diff, max_abs_diff_across
from tests.common.ep_reference import ep_layers
from tests.common.numerics import max_or_nan
from tests.common.rank_log import log

HSDP_MESH_DIMS = ("dp_replicate", "dp_shard")
PROBE_FORWARD_TOKENS = 8


def probe_forward(model, device: torch.device) -> None:
    """One tiny no-grad forward of the run's module, taking every FSDP2 group through unshard and post-forward reshard
    (or leaving it gathered). Enter at the module the run forwards: the root's pre-forward unshards its group, so
    entering below meets its params as sharded DTensors."""
    with torch.no_grad():
        model(input_ids=torch.ones(1, PROBE_FORWARD_TOKENS, dtype=torch.long, device=device))


def probe_pipeline_forward(pipeline, device: torch.device) -> None:
    input_ids = torch.ones(1, PROBE_FORWARD_TOKENS, dtype=torch.long, device=device)
    with torch.no_grad():
        pipeline({"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)})


def fsdp2_groups_sharded(model) -> bool:
    groups = fsdp2_param_groups(model)
    return bool(groups) and all(group.is_sharded for group in groups)


def full_shard_checks(model, forward) -> dict[str, bool]:
    """FULL_SHARD engaged: after ``forward`` every FSDP2 group, the root's included, is back on its shards.
    SHARD_GRAD_OP leaves groups gathered until the backward, so a run that lost ``fsdp_reshard_after_forward`` fails
    here. No collective past the forward's."""
    reshard_fsdp2_modules(model)
    forward()
    groups = fsdp2_param_groups(model)
    gathered = sum(1 for group in groups if not group.is_sharded)
    ok = bool(groups) and gathered == 0
    log(f"  FULL_SHARD: {len(groups) - gathered}/{len(groups)} FSDP2 groups resharded after a forward")
    reshard_fsdp2_modules(model)
    return {"fsdp_groups_resharded_after_forward": ok}


def _expert_params(model) -> list[tuple[EPMoELayerBase, torch.nn.Parameter]]:
    """``(layer, param)`` for expert weights and native adapters in ``parameters()`` order: adapter accessors iterate a
    frozenset, whose order varies by process."""
    experts = []
    for layer in ep_layers(model):
        expert_ids = {id(param) for _, param in layer.expert_named_params()}
        experts += [(layer, param) for param in layer.parameters() if id(param) in expert_ids]
    return experts


def ep1_expert_checks(model, *, sharded: bool) -> dict[str, bool]:
    """Where ep1 experts live: DTensors when ``sharded``, else whole plain tensors, bit-identical on every rank.
    COLLECTIVE when not ``sharded`` (one world gather per expert tensor)."""
    reshard_fsdp2_modules(model)
    experts = _expert_params(model)
    if sharded:
        name = "ep1_experts_fsdp_sharded"
        ok = bool(experts) and all(isinstance(param.data, DTensor) for _, param in experts)
        log(f"  {name}: {'PASS' if ok else 'FAIL'} ({len(experts)} expert tensors)")
        return {name: ok}
    name = "ep1_experts_whole_plain_tensors"
    ok = bool(experts) and all(
        not isinstance(param.data, DTensor) and param.shape[0] == layer.num_experts for layer, param in experts
    )
    worst = max_abs_diff_across(param.data for _, param in experts)
    log(f"  {name}: {'PASS' if ok else 'FAIL'} ({len(experts)} expert tensors), replica max|Δ| {worst:.2e}")
    return {name: ok, "ep1_whole_expert_replicas_identical": worst == 0.0}


def hsdp_params(model) -> list[tuple[str, DTensor]]:
    return [
        (name, param.data)
        for name, param in model.named_parameters()
        if isinstance(param.data, DTensor) and param.data.device_mesh.ndim == 2
    ]


def hsdp_layout(world_size: int) -> dict[str, object]:
    """``ParallelismConfig`` fields of HSDP over two simulated NVLink domains on one host, the path a two-node job
    takes."""
    return {"use_hsdp": True, "gpus_per_node": world_size // 2}


def hsdp_checks(model) -> dict[str, bool]:
    """HSDP engaged and its replicas in step: FSDP2 params on a 2-D mesh, each local shard bit-identical across
    ``dp_replicate``. COLLECTIVE."""
    reshard_fsdp2_modules(model)
    params = hsdp_params(model)
    placed = bool(params) and all(
        tuple(param.device_mesh.mesh_dim_names or ()) == HSDP_MESH_DIMS
        and isinstance(param.placements[0], Replicate)
        and isinstance(param.placements[1], Shard)
        for _, param in params
    )
    worst = max_or_nan(
        (
            group_max_abs_diff(param.to_local(), param.device_mesh[HSDP_MESH_DIMS[0]].get_group())
            for _, param in params
        ),
        default=float("nan"),
    )
    log(f"  HSDP: {len(params)} params on the 2-D mesh, placed {placed}, replica max|Δ| {worst:.2e}")
    return {"hsdp_mesh_engaged": placed, "hsdp_replicas_identical": worst == 0.0}


def expert_replica_checks(model, ep_config) -> dict[str, bool]:
    """Several EP groups engaged and expert replicas bit-identical; a deferred sweep that never ran leaves them apart.
    COLLECTIVE over each replica group."""
    groups = 0 if ep_config is None else ep_config.num_ep_groups
    engaged = groups > 1 and len(ep_config.expert_replica_ranks) == groups
    experts = [to_local(param) for _, param in _expert_params(model)]
    worst = max_abs_diff_across(experts, ep_config.expert_replica_group) if engaged else float("nan")
    log(f"  EP groups: {groups}, expert replica max|Δ| {worst:.2e}")
    return {"ep_groups_replicate_experts": engaged and bool(experts), "expert_replicas_identical": worst == 0.0}


def tp_dp_checks(model, data_parallel_size: int) -> dict[str, bool]:
    """TP with DP replicas: every TP-sharded param sits on a ``(dp, tp)`` mesh whose ``dp`` dim spans
    ``data_parallel_size``. No collective."""
    reshard_fsdp2_modules(model)
    params = [
        param.data
        for param in model.parameters()
        if isinstance(param.data, DTensor) and has_tp_dim(param.data.device_mesh)
    ]
    ok = bool(params) and all(
        mesh_dim_names(param.device_mesh) == (MeshDim.DP, MeshDim.TP)
        and param.device_mesh[MeshDim.DP].size() == data_parallel_size
        for param in params
    )
    log(f"  TP x DP: {len(params)} TP params on a (dp={data_parallel_size}, tp) mesh: {'PASS' if ok else 'FAIL'}")
    return {"tp_params_sharded_over_dp": ok}


def dp_layout_checks(
    model, requested: Mapping[str, object], *, world_size: int, ep_config, forward: Callable[[], object]
) -> dict[str, bool]:
    """The data-parallel half of the ``ParallelismConfig`` fields ``requested``, each read off ``model``. COLLECTIVE.
    The expectation comes from ``requested``, so a run whose loader or wrap dropped a field fails here."""
    ep_size = requested.get("ep_size", 1)
    expert_group = ep_size * requested.get("expert_tp_size", 1)
    tp_size = requested.get("tp_size", 1)
    domain = requested.get("gpus_per_node", 0)
    checks: dict[str, bool] = {}
    if requested.get("fsdp_reshard_after_forward"):
        checks |= full_shard_checks(model, forward)
    if expert_group == 1 and tp_size == 1 and world_size > 1 and ep_layers(model):
        checks |= ep1_expert_checks(model, sharded=requested.get("fsdp_shard_ep1_experts", True))
    if requested.get("use_hsdp"):
        checks |= hsdp_checks(model)
    if ep_size > 1 and world_size > expert_group:
        checks |= expert_replica_checks(model, ep_config)
    if ep_size > 1 and domain:
        # The EP groups tile the simulated NVLink domains: one node-local group set per domain.
        domains = 0 if ep_config is None else ep_config.num_domains
        checks["ep_groups_tile_the_simulated_domains"] = domains == world_size // domain
        log(f"  EP NVLink domains: {domains} (requested {world_size // domain})")
    if tp_size > 1 and world_size > tp_size:
        checks |= tp_dp_checks(model, world_size // tp_size)
    return checks
