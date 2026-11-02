"""The data-parallel side of a run's layout, read off the model rather than the config it was handed.

``fsdp_reshard_after_forward``, ``fsdp_shard_ep1_experts``, ``use_hsdp``, several EP groups and TP with DP
replicas each change what the FSDP2 wrap and the EP groups leave on the model, and a loader or wrap that ignored
one would still train. These probes read those effects: the groups FULL_SHARD reshards after a forward, where
ep1 experts live, the HSDP mesh, the expert replicas the deferred sweep keeps identical, the (dp, tp) mesh.
"""

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
# Tokens in the throwaway forward a probe runs the FSDP2 groups through.
PROBE_FORWARD_TOKENS = 8


def probe_forward(model, device: torch.device) -> None:
    """One tiny no-grad forward of ``model``, which takes every FSDP2 group through its unshard and post-forward
    reshard (FULL_SHARD) or leaves it gathered (SHARD_GRAD_OP), the state a training step's forward ends in.
    ``model`` is the module the run forwards: the root's own pre-forward unshards the root's group, so a forward
    entering below the root meets its parameters (an embedding the root holds) as sharded DTensors."""
    with torch.no_grad():
        model(input_ids=torch.ones(1, PROBE_FORWARD_TOKENS, dtype=torch.long, device=device))


def probe_pipeline_forward(pipeline, device: torch.device) -> None:
    """:func:`probe_forward` through a SentenceTransformer ``pipeline``, whose forward takes the tokenized features."""
    input_ids = torch.ones(1, PROBE_FORWARD_TOKENS, dtype=torch.long, device=device)
    with torch.no_grad():
        pipeline({"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)})


def fsdp2_groups_sharded(model) -> bool:
    """Every FSDP2 parameter group under ``model`` holds its sharded parameters (and there is at least one)."""
    groups = fsdp2_param_groups(model)
    return bool(groups) and all(group.is_sharded for group in groups)


def full_shard_checks(model, forward) -> dict[str, bool]:
    """FULL_SHARD (ZeRO-3) engaged: after ``forward`` every FSDP2 group, the root's included, is back on its shards.

    The wrap passes ``reshard_after_forward`` explicitly, so FSDP2 reshards the root after its forward too (only an
    unset value keeps the root gathered for the backward); a wrap whose one group is the root's (a SentenceTransformer)
    is held to it like any other. SHARD_GRAD_OP leaves every group gathered after a forward until its backward, so a
    run whose config lost ``fsdp_reshard_after_forward`` fails here. Starts from the sharded state; ``forward`` is any
    no-grad forward of the module the run forwards. No collective beyond the forward's own.
    """
    reshard_fsdp2_modules(model)
    forward()
    groups = fsdp2_param_groups(model)
    gathered = sum(1 for group in groups if not group.is_sharded)
    ok = bool(groups) and gathered == 0
    log(f"  FULL_SHARD: {len(groups) - gathered}/{len(groups)} FSDP2 groups resharded after a forward")
    reshard_fsdp2_modules(model)
    return {"fsdp_groups_resharded_after_forward": ok}


def _expert_params(model) -> list[tuple[EPMoELayerBase, torch.nn.Parameter]]:
    """``(layer, param)`` for every expert weight and native expert adapter of ``model``'s EP layers, in
    ``parameters()`` order, the same on every rank: the adapter accessors iterate a frozenset of names, whose order
    differs across processes, so a collective per tensor in that order would pair different tensors."""
    experts = []
    for layer in ep_layers(model):
        expert_ids = {id(param) for _, param in layer.expert_named_params()}
        experts += [(layer, param) for param in layer.parameters() if id(param) in expert_ids]
    return experts


def ep1_expert_checks(model, *, sharded: bool) -> dict[str, bool]:
    """Where ep1 experts live, against what the run asked for: FSDP2 shards (DTensors) when ``sharded``
    (``fsdp_shard_ep1_experts``), so FSDP's reduce-scatter is their sync; whole plain tensors holding every
    expert when not, synced by the EP layer's own all-reduce, and then bit-identical on every rank, native
    expert adapters included. Read with the groups resharded. COLLECTIVE when not ``sharded`` (one gather per
    expert tensor over the world, the DP scope ep1 replicates them across)."""
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
    """Every parameter held on a 2-D FSDP2 mesh, in ``named_parameters`` order (the same on every rank)."""
    return [
        (name, param.data)
        for name, param in model.named_parameters()
        if isinstance(param.data, DTensor) and param.data.device_mesh.ndim == 2
    ]


def hsdp_layout(world_size: int) -> dict[str, object]:
    """The ``ParallelismConfig`` fields of HSDP over two simulated NVLink domains of ``world_size // 2`` ranks on
    one host: a ``(dp_replicate=2, dp_shard=world_size // 2)`` mesh, the path a two-node job takes."""
    return {"use_hsdp": True, "gpus_per_node": world_size // 2}


def hsdp_checks(model) -> dict[str, bool]:
    """HSDP engaged and its replicas in step: FSDP2 parameters on a ``(dp_replicate, dp_shard)`` mesh,
    replicated across the first dim and sharded along the second, each local shard bit-identical across its
    ``dp_replicate`` group. COLLECTIVE (one gather per parameter over its replicate group)."""
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
    """Several EP groups engaged and their expert replicas in step: ``num_ep_groups`` groups, each rank's expert
    tensors bit-identical to those its replicas in the other groups hold. A deferred cross-replica sweep that
    never ran leaves replicas that trained on different batches apart. ``ep_config`` is ``None`` where the run
    built no EP groups. COLLECTIVE over each replica group."""
    groups = 0 if ep_config is None else ep_config.num_ep_groups
    engaged = groups > 1 and len(ep_config.expert_replica_ranks) == groups
    experts = [to_local(param) for _, param in _expert_params(model)]
    worst = max_abs_diff_across(experts, ep_config.expert_replica_group) if engaged else float("nan")
    log(f"  EP groups: {groups}, expert replica max|Δ| {worst:.2e}")
    return {"ep_groups_replicate_experts": engaged and bool(experts), "expert_replicas_identical": worst == 0.0}


def tp_dp_checks(model, data_parallel_size: int) -> dict[str, bool]:
    """TP with DP replicas: every TP-sharded parameter sits on a ``(dp, tp)`` mesh whose ``dp`` dim spans
    ``data_parallel_size`` ranks, the FSDP2 shard of each TP group's replica. No collective."""
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
    """The data-parallel half of the ``ParallelismConfig`` fields ``requested``, each read off ``model``.

    FULL_SHARD under ``fsdp_reshard_after_forward`` (``forward`` is a no-grad forward of the run's model), ep1
    experts on the EP layers sharded or (``fsdp_shard_ep1_experts=False``) whole, where several ranks replicate
    them without TP, the HSDP mesh and its replicas under ``use_hsdp``, the
    expert replicas of several EP groups where ``world_size`` exceeds the expert group (tiling the simulated NVLink
    domains of ``gpus_per_node`` ranks where that is set), and the (dp, tp) mesh where it exceeds the TP group. The expectation comes from ``requested``, so a run whose loader or wrap
    dropped a field fails here. ``ep_config`` is the trainer's (``None`` without EP groups). COLLECTIVE.
    """
    ep_size = requested.get("ep_size", 1)
    expert_group = ep_size * requested.get("expert_tp_size", 1)
    tp_size = requested.get("tp_size", 1)
    domain = requested.get("gpus_per_node", 0)
    checks: dict[str, bool] = {}
    if requested.get("fsdp_reshard_after_forward"):
        checks |= full_shard_checks(model, forward)
    if expert_group == 1 and tp_size == 1 and world_size > 1 and ep_layers(model):
        # Replicated experts on the grouped-GEMM EP layers: in the FSDP2 shard unless the run kept them whole.
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
