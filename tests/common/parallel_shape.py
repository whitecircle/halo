"""Model-side evidence that each parallel axis took effect; ``has_ep_group`` and friends only echo the config a run was
handed."""

from collections.abc import Sequence
from typing import NamedTuple

import torch.distributed as dist
from torch.distributed.tensor import DTensor

from halo.distributed.context_parallel.ulysses_attention import UlyssesAttention
from halo.distributed.mesh import has_tp_dim
from halo.models.moe.family_specs import wraps_experts
from tests.common.ep_reference import ep_layers
from tests.common.rank_log import log


class ExpertTPMembership(NamedTuple):
    """One rank's view of its expert-TP group: global ranks (None without a group), ``expert_tp_rank``, and the DP rank
    whose batch it reads."""

    members: tuple[int, ...] | None
    expert_tp_rank: int
    dp_rank: int


def tp_sharded_param_names(model) -> list[str]:
    return [
        name
        for name, param in model.named_parameters()
        if isinstance(param.data, DTensor) and has_tp_dim(param.data.device_mesh)
    ]


def ulysses_attention_layers(model) -> list[UlyssesAttention]:
    return [module for module in model.modules() if isinstance(module, UlyssesAttention)]


def _bank_split_ep_way(layer) -> bool:
    """The layer owns an ``ep_size``-th of the bank; EP-sharded experts are FSDP-ignored plain tensors, so dim 0 is the
    local count."""
    weights = layer.expert_named_params()
    return (
        layer.experts_per_rank * layer.ep_size == layer.num_experts
        and layer.experts_per_rank < layer.num_experts
        and layer.expert_end - layer.expert_start == layer.experts_per_rank
        and bool(weights)
        and all(param.shape[0] == layer.experts_per_rank for _, param in weights)
    )


def parallel_shape_checks(model, parallelism_config) -> dict[str, bool]:
    """One named check per parallel axis ``parallelism_config`` enables, each read off ``model``. Call after the FSDP2
    wrap; no collective."""
    pc = parallelism_config
    checks: dict[str, bool] = {}
    layers = ep_layers(model)
    if wraps_experts(model.config, ep_layers_requested=pc.needs_ep_layers):
        checks["ep_layers_wrapped"] = bool(layers)
        log(f"  EP layers wrapped: {'PASS' if layers else 'FAIL'} ({len(layers)})")
    if pc.ep_size > 1:
        checks["expert_bank_split_ep_way"] = bool(layers) and all(_bank_split_ep_way(layer) for layer in layers)
        log(f"  Expert bank split {pc.ep_size}-way: {'PASS' if checks['expert_bank_split_ep_way'] else 'FAIL'}")
    if pc.is_expert_tp_mode:
        checks["expert_ffn_sharded_etp_way"] = bool(layers) and all(
            layer.expert_tp_size == pc.expert_tp_size for layer in layers
        )
        log(
            f"  Expert FFN sharded {pc.expert_tp_size}-way: "
            f"{'PASS' if checks['expert_ffn_sharded_etp_way'] else 'FAIL'}"
        )
    if pc.is_tp_mode:
        sharded = tp_sharded_param_names(model)
        checks["tp_sharded_params"] = bool(sharded)
        log(f"  TP-sharded params: {'PASS' if sharded else 'FAIL'} ({len(sharded)})")
    if pc.is_cp_mode:
        attention = ulysses_attention_layers(model)
        checks["cp_attention_wrapped"] = bool(attention) and all(layer.cp_size == pc.cp_size for layer in attention)
        log(f"  Ulysses attention layers: {'PASS' if checks['cp_attention_wrapped'] else 'FAIL'} ({len(attention)})")
    return checks


def expert_tp_group_problems(views: Sequence[ExpertTPMembership], expert_tp_size: int) -> list[str]:
    """What is wrong with the expert-TP groups ``views[r]`` reports; empty when every rank is in one exact-size group.
    Members must agree, hold each shard index once and read one DP rank's batch, since partners sum their shards'
    outputs token by token."""
    problems = []
    for rank, view in enumerate(views):
        if view.members is None:
            problems.append(f"rank {rank} reports no expert-TP group")
            continue
        members = tuple(sorted(view.members))
        if len(members) != expert_tp_size:
            problems.append(f"rank {rank}'s expert-TP group {members} has {len(members)} ranks, not {expert_tp_size}")
        if rank not in members:
            problems.append(f"rank {rank} is not in its own expert-TP group {members}")
        if any(not 0 <= member < len(views) for member in members):
            problems.append(f"rank {rank}'s expert-TP group {members} names a rank outside the world")
            continue
        disagreeing = [
            member
            for member in members
            if views[member].members is None or tuple(sorted(views[member].members)) != members
        ]
        if disagreeing:
            problems.append(f"ranks {disagreeing} do not report rank {rank}'s expert-TP group {members}")
        shard_ranks = sorted(views[member].expert_tp_rank for member in members)
        if shard_ranks != list(range(len(members))):
            problems.append(f"rank {rank}'s expert-TP group {members} holds expert-TP ranks {shard_ranks}")
        dp_ranks = sorted({views[member].dp_rank for member in members})
        if len(dp_ranks) != 1:
            problems.append(f"rank {rank}'s expert-TP group {members} reads the batches of DP ranks {dp_ranks}")
    return problems


def expert_tp_group_views(layer, dp_rank: int) -> list[ExpertTPMembership]:
    """COLLECTIVE (the world): every rank's :class:`ExpertTPMembership` in global-rank order; a rank with no group
    joins the gather too."""
    group = layer.expert_tp_group
    mine = ExpertTPMembership(
        None if group is None else tuple(dist.get_process_group_ranks(group)), layer.expert_tp_rank, dp_rank
    )
    views: list[ExpertTPMembership | None] = [None] * dist.get_world_size()
    dist.all_gather_object(views, mine)
    return views
