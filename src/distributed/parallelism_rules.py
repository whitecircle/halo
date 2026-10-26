"""The parallelism axis vocabulary, the allowlist of axis combinations, and every config-time refusal.

Each ``reject_*`` reads a :class:`~src.distributed.parallelism_config.ParallelismConfig` and raises
``ValueError``. ``reject_unsupported_axes`` runs before topology resolution and reads only the axis sizes;
the composition refusals read the resolved topology. The config applies both at construction; the
accelerate-launch and determinism refusals are applied by the trainers and entry scripts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from src.distributed.expert_parallel.deepep.capacity import reject_legacy_backend_topology
from src.distributed.group_layout import (
    CrossNodeLayout,
    reject_cross_node_ep_group,
    reject_cross_node_etp_shape,
    reject_node_local_ep_group,
)
from src.env import env_str, is_accelerate_launch
from src.models.moe.expert_lora_spec import reject_expert_lora_with_expert_tp
from src.runtime.ranks import is_global_main_process

if TYPE_CHECKING:
    from src.distributed.parallelism_config import ParallelismConfig

logger = logging.getLogger(__name__)

# Each Literal doubles as the validation table of its field.
PPSchedule = Literal["1f1b", "gpipe"]
PP_DEFAULT_SCHEDULE: PPSchedule = "1f1b"
EPScope = Literal["auto", "node", "global"]


@dataclass(frozen=True)
class Axis:
    """A model-sharding axis: its allowlist name, the CLI flag a refusal quotes, the field sizing it."""

    name: str
    flag: str
    size_field: str


AXES = (
    Axis("pp", "pipeline_parallel_size", "pp_size"),
    Axis("ep", "expert_parallel_size", "ep_size"),
    Axis("etp", "expert_tensor_parallel_size", "expert_tp_size"),
    Axis("tp", "tensor_parallel_size", "tp_size"),
    Axis("cp", "context_parallel_size", "cp_size"),
)

# ALLOWLIST of combinable model-sharding axes (FSDP is not an axis — it shards the leftover DP
# width). Unlisted combinations are rejected rather than running unvalidated, as a denylist allows.
SUPPORTED_AXIS_SETS: frozenset[frozenset[str]] = frozenset(
    map(
        frozenset,
        (
            (),  # plain data parallelism (FSDP2 / DDP)
            ("ep",),
            ("etp",),  # pure ETP (ep_size == 1): expert FFN sharded, experts replicated
            ("tp",),
            ("cp",),
            ("pp",),
            ("ep", "tp"),
            ("ep", "cp"),
            ("ep", "etp"),
            ("pp", "ep"),
            ("pp", "etp"),
        ),
    )
)

# Why a rejected combination is rejected; a missing entry degrades the explanation, not the rejection.
AXIS_SET_MECHANISMS: dict[frozenset[str], str] = {
    frozenset({"tp", "cp"}): (
        "TP and CP would partition the same ranks twice: both groups are contiguous rank blocks, so "
        "at tensor_parallel_size == context_parallel_size a rank's TP partners ARE its CP partners, "
        "required to hold the same tokens (TP) and different sequence chunks (CP) at once. Ulysses "
        "also redistributes attention over HEADS, of which TP has already left this rank only "
        "num_attention_heads / tensor_parallel_size, while reject_unsupported_ulysses_model checks "
        "divisibility against the config's full head count; and data_parallel_size takes "
        "max(tensor_parallel_size, context_parallel_size), which counts one of the two axes only. Use "
        "EP+TP or EP+CP."
    ),
    frozenset({"etp", "cp"}): (
        "expert-TP partners hold shards of one expert and must see the SAME tokens — ReduceFromExpertTP "
        "sums their outputs element-wise in token space — but CP hands each rank a different sequence "
        "chunk, so that sum would add unrelated tokens. get_data_parallel_rank keys on the expert-TP "
        "layout alone, so it also stops agreeing with data_parallel_size once context_parallel_size "
        "exceeds expert_tensor_parallel_size. Drop one of the two."
    ),
    frozenset({"tp", "etp"}): (
        "attention TP and expert TP would shard the same ranks along two different axes. Use EP+TP "
        "(attention sharded, experts distributed) or pure ETP (expert FFN sharded)."
    ),
    frozenset({"pp", "tp"}): (
        "PP+TP needs at least 2 nodes to launch at all (a stage must be whole NVLink domains), and "
        "the composition was never exercised on real multi-node hardware. It also excludes the "
        "dense families most likely to want it: transformers reconciles a "
        "'replicated_with_grad_allreduce' plan entry (Qwen3 q_norm/k_norm) with a per-backward "
        "all-reduce hook, which the pipeline schedule disables on non-final microbatches, so the "
        "hook re-reduces accumulated history. Use PP+EP, or TP inside a single stage-less job."
    ),
    frozenset({"pp", "cp"}): (
        "the pipeline loss normalizer is already stage-wide and carries no cancelling x "
        "context_parallel_size factor, so every gradient would come out context_parallel_size x too "
        "small with no error raised. Use PP without CP, or CP without PP."
    ),
    frozenset({"pp", "ep", "etp"}): (
        "ReduceFromExpertTP.backward is model math, not gradient sync, so it CANNOT be deferred the "
        "way the DP sweep defers its all-reduce: it runs inside every microbatch backward. At "
        "expert_parallel_size > 1 that strided expert-TP all-reduce interleaves with the DeepEP "
        "combine the same backward is inside, and the two orderings are not guaranteed to agree "
        "across ranks. At expert_parallel_size == 1 (PP+ETP) the dispatch group is width 1, so "
        "there is no combine to interleave with and the composition is safe. Use PP+EP or PP+ETP, "
        "not all three."
    ),
    frozenset({"pp", "ep", "tp"}): (
        "the deferred cross-replica expert sweep needs FSDP to shard non-expert params over the EP "
        "group (a 1-D ep-sized mesh), while EP+TP shards them over the 2-D (dp, tp) mesh — the two "
        "contracts cannot both hold. EP already shards the experts, so TP would add only attention. "
        "Use PP+EP."
    ),
    frozenset({"pp", "ep", "cp"}): (
        "under PP the expert gradients are synced by the deferred post-backward sweep, whose divisor "
        "counts every rank of the stage as a distinct DP replica. CP ranks are not — they hold "
        "sequence shards of the SAME batch — and the pipeline loss carries no cancelling x "
        "context_parallel_size factor, so every expert gradient would come out context_parallel_size "
        "x too small. Use PP+EP without CP."
    ),
    frozenset({"ep", "tp", "etp"}): (
        "attention TP and expert TP cannot both shard the EP group. Use EP+TP or EP+ETP."
    ),
}


def render_axis_set(axes: frozenset[str]) -> str:
    """Axis set as the user's mental model: 'PP + EP', or 'plain data parallelism' when empty."""
    ordered = [a.name for a in AXES if a.name in axes]
    return " + ".join(a.upper() for a in ordered) if ordered else "plain data parallelism"


def reject_unsupported_axes(pc: ParallelismConfig) -> None:
    """Reject any axis combination outside :data:`SUPPORTED_AXIS_SETS`, naming why it is unsupported."""
    axes = pc.active_axes
    if axes in SUPPORTED_AXIS_SETS:
        return
    mechanism = AXIS_SET_MECHANISMS.get(axes) or (
        "this combination has no validated composition — no equivalence gate has ever compared "
        "its gradients against an unsplit reference, so enabling it would train silently "
        "different gradients."
    )
    typed = ", ".join(f"{a.flag}={getattr(pc, a.size_field)}" for a in AXES if a.name in axes)
    supported = ", ".join(sorted(render_axis_set(s) for s in SUPPORTED_AXIS_SETS))
    raise ValueError(
        f"{render_axis_set(axes)} is not a supported parallelism combination ({typed}): "
        f"{mechanism} Supported combinations: {supported} — each composes with FSDP data "
        f"parallelism over whatever world width is left."
    )


def _reject_pipeline_parallel(pc: ParallelismConfig) -> None:
    """PP stages must be whole NVLink domains, keeping every domain-local group inside one stage, and PP
    refuses the settings whose collectives or loss aggregation it would break."""
    if pc.pp_size == 1:
        # Nothing reads the PP-only knobs at pp_size == 1, so a set one does not do what it says.
        for name, value, default in (
            ("pipeline_split", pc.pp_split, None),
            ("pipeline_microbatches", pc.pp_microbatches, 0),
            ("pipeline_schedule", pc.pp_schedule, PP_DEFAULT_SCHEDULE),
        ):
            if value != default:
                raise ValueError(
                    f"{name}={value!r} is only meaningful with pipeline_parallel_size > 1; at 1 nothing reads it. "
                    f"Remove it or set pipeline_parallel_size."
                )
        return

    if pc.expert_lora is not None:
        raise ValueError(
            "Expert LoRA (native grouped adapters) is not supported under pipeline parallelism: "
            "the adapter save/merge paths (save_ep_lora_adapters, merge_expert_lora_on_save) "
            "write flat global expert names with no stage→global remap, so a PP save would "
            "silently record stage-local layer indices. Train expert LoRA without PP."
        )

    if pc.pp_split is not None:
        if len(pc.pp_split) != pc.pp_size:
            raise ValueError(
                f"pipeline_split has {len(pc.pp_split)} entries but pipeline_parallel_size={pc.pp_size}: "
                f"one decoder-layer count per stage is required."
            )
        if min(pc.pp_split) < 1:
            raise ValueError(f"pipeline_split entries must be >= 1 decoder layer, got {pc.pp_split}.")

    if pc.stage_world_size % pc.nvlink_domain_size != 0:
        raise ValueError(
            f"Each pipeline stage owns {pc.stage_world_size} ranks (world_size "
            f"{pc.world_size} / pipeline_parallel_size {pc.pp_size}), which is not a "
            f"multiple of the NVLink domain ({pc.nvlink_domain_size}). Stage boundaries "
            f"must fall on NVLink-domain boundaries so EP/TP/ETP/CP groups stay inside one "
            f"stage and only PP's point-to-point activations cross RDMA. Choose a "
            f"pipeline_parallel_size dividing {pc.world_size // pc.nvlink_domain_size} "
            f"(the domain count), or set NVLINK_DOMAIN_SIZE to the real NVLink partition size."
        )

    if pc.stage_world_size == 1:
        # setup_fsdp2_for_dp skips wrapping at dp<=1, so the runtime FSDP contract assert would
        # blame a wrap bug that doesn't exist. Reject honestly instead.
        raise ValueError(
            f"pipeline_parallel_size={pc.pp_size} leaves each pipeline stage with a single "
            f"rank (world_size={pc.world_size}): unsharded one-rank stages are not "
            f"supported yet (the stage FSDP wrap and its grad-reduction contract assume dp >= "
            f"2 per stage). Use a smaller pipeline_parallel_size, or add data-parallel width "
            f"per stage."
        )

    if pc.needs_ep_wrappers and pc.ep_group_size == 1 and not pc.experts_fsdp_managed:
        raise ValueError(
            f"PP with fsdp_shard_ep1_experts=False (pipeline_parallel_size={pc.pp_size}, "
            f"expert_parallel_size=1) is not supported: a MoE stage would then hold plain "
            f"replicated expert tensors while a fully-dense stage of the same pipeline runs the "
            f"plain FSDP clip — the two stages would issue different collective programs and "
            f"deadlock at the gradient norm. Keep fsdp_shard_ep1_experts=True (the default) so "
            f"experts are stage-mesh DTensors."
        )

    if pc.use_hsdp:
        # The shape is coherent; what is missing is a stage-restricted 2-D mesh.
        raise ValueError(
            f"PP + HSDP is not supported (pipeline_parallel_size={pc.pp_size}, "
            f"stage_world_size={pc.stage_world_size}, {pc.num_nvlink_domains} NVLink domain(s) "
            f"per stage). HSDP's 2-D (dp_replicate, dp_shard) mesh is built by init_device_mesh "
            f"over the whole world, and cannot be restricted to a stage's rank block "
            f"(create_dp_mesh takes a process group on the 1-D path only) — a stage-sized 2-D mesh "
            f"would silently be made of the FIRST stage's ranks on every rank. Enabling it needs "
            f"the pipeline as an outer mesh dimension: init_device_mesh((pp, dp_replicate, "
            f"dp_shard))[dp_replicate, dp_shard]. At one domain per stage HSDP is a no-op anyway "
            f"(dp_replicate_size counts a stage's domains). Use plain FSDP inside each stage "
            f"(use_hsdp=False)."
        )

    if (pc.fp32_grad_reduce or pc.fp32_non_ep_params) and is_global_main_process():
        logger.warning(
            "fp32 gradient reduction under pipeline parallelism doubles the PERSISTENT gradient "
            "buffer: the schedule defers FSDP's reduce-scatter to the optimizer step, so each "
            "stage holds a full unsharded fp32 grad (4 B/param) for the whole microbatch loop "
            "instead of a bf16 one. Budget ~2xP_stage (params) + 4xP_stage (grads) per rank, or "
            "leave fp32_grad_reduce/fp32_non_ep_params off under PP."
        )

    if pc.fsdp_reshard_after_forward:
        # Numerically equal, but unproven through the trainer and an all-gather the schedule does not want.
        raise ValueError(
            f"PP + fsdp_reshard_after_forward=True (FULL_SHARD / ZeRO-3) is not enabled "
            f"(pipeline_parallel_size={pc.pp_size}): torch's pipeline schedule already pins "
            f"each stage unsharded across the backward (set_reshard_after_backward(False)) and "
            f"drives FSDP's post-backward itself, so resharding after every FORWARD only "
            f"re-fetches parameters the next microbatch immediately needs — and no equivalence "
            f"gate covers the composition through the trainer. Use "
            f"fsdp_reshard_after_forward=False (SHARD_GRAD_OP / ZeRO-2), with activation "
            f"checkpointing for the memory."
        )

    if pc.lowp.precision != "bf16":
        raise ValueError(
            f"PP + lowp_precision={pc.lowp.precision!r} is not supported "
            f"(pipeline_parallel_size={pc.pp_size}). No equivalence gate has ever compared PP "
            f"low-precision gradients against an unsplit reference, and the composition has a "
            f"known silent failure: apply_lowp_compute derives each block's index from "
            f"its MODULE NAME and keeps lowp_keep_first_blocks / lowp_keep_last_blocks at the ends "
            f"of that numbering, but a stage's layers are re-based to 0 — so every stage would "
            f"keep its OWN first and last blocks in bf16 while the network's true ends get low "
            f"precision, with nothing raised. bf16 is the production default; train the "
            f"low-precision recipe without PP."
        )


def _reject_ep_group(pc: ParallelismConfig) -> None:
    """The full EP group must fit in and divide its scope: the NVLink domain ('node') or the stage ('global')."""
    if pc.ep_group_size > 1:
        if pc.ep_scope == "node":
            reject_node_local_ep_group(
                pc.ep_group_size,
                pc.nvlink_domain_size,
                ep_size=pc.ep_size,
                expert_tp_size=pc.expert_tp_size,
                remedy="Use ep_scope='global' for EP spanning NVLink domains (RDMA).",
            )
        else:  # global scope
            # Named after the unit, or an ep16+pp2 job reads "cannot exceed world size (8)" on 16 GPUs.
            scope_desc = (
                "world size"
                if pc.pp_size == 1
                else f"stage world size (world size / pipeline_parallel_size {pc.pp_size})"
            )
            reject_cross_node_ep_group(
                pc.ep_group_size,
                pc.stage_world_size,
                scope_desc=scope_desc,
                ep_size=pc.ep_size,
                expert_tp_size=pc.expert_tp_size,
            )
            # Raises unless the groups tile the stage as equal contiguous per-domain blocks (DeepEP IPC peers).
            CrossNodeLayout(
                world_size=pc.stage_world_size, group_size=pc.ep_group_size, domain_size=pc.nvlink_domain_size
            )


def _reject_cp_locality(pc: ParallelismConfig) -> None:
    """CP must divide the NVLink domain: the Ulysses all-to-all is bandwidth-heavy."""
    if pc.cp_size > 1:
        if pc.cp_size > pc.nvlink_domain_size:
            raise ValueError(
                f"context_parallel_size ({pc.cp_size}) cannot exceed the NVLink domain "
                f"({pc.nvlink_domain_size}). CP must stay on NVLink for efficient Ulysses attention."
            )
        if pc.nvlink_domain_size % pc.cp_size != 0:
            raise ValueError(
                f"context_parallel_size ({pc.cp_size}) must divide the NVLink domain ({pc.nvlink_domain_size})"
            )


def _is_multi_domain_multi_group_ep(pc: ParallelismConfig) -> bool:
    """Several EP groups across several NVLink domains: refused under TP and ETP, each for its grad sync."""
    return pc.ep_size > 1 and pc.num_nvlink_domains > 1 and pc.stage_world_size // pc.ep_group_size > 1


def _reject_tp(pc: ParallelismConfig) -> None:
    """TP must divide the stage and the NVLink domain; under EP+TP each EP group spans whole TP groups."""
    if pc.tp_size > 1:
        if pc.tp_size > pc.stage_world_size:
            raise ValueError(f"tensor_parallel_size ({pc.tp_size}) cannot exceed world size ({pc.stage_world_size})")
        if pc.stage_world_size % pc.tp_size != 0:
            raise ValueError(f"tensor_parallel_size ({pc.tp_size}) must divide world size ({pc.stage_world_size})")
        if pc.ep_size > 1 and pc.ep_size % pc.tp_size != 0:
            raise ValueError(
                f"EP+TP requires expert_parallel_size ({pc.ep_size}) to be a multiple of "
                f"tensor_parallel_size ({pc.tp_size}) so each EP group spans whole TP groups."
            )
        if pc.nvlink_domain_size % pc.tp_size != 0:
            raise ValueError(
                f"tensor_parallel_size ({pc.tp_size}) must divide the NVLink domain "
                f"({pc.nvlink_domain_size}): TP groups are contiguous rank blocks, so a "
                f"non-dividing tensor_parallel_size makes some TP groups straddle a domain boundary "
                f"and every attention all-reduce crosses RDMA. Use a tensor_parallel_size that "
                f"divides {pc.nvlink_domain_size}, with DP across domains."
            )
        if _is_multi_domain_multi_group_ep(pc):
            raise ValueError(
                f"Multi-domain multi-group EP+TP is not supported: expert_parallel_size={pc.ep_size} "
                f"with tensor_parallel_size={pc.tp_size} on {pc.stage_world_size} ranks forms "
                f"{pc.stage_world_size // pc.ep_group_size} EP groups across "
                f"{pc.num_nvlink_domains} NVLink domains, and the cross-replica gradient "
                "average is incompatible with the EP+TP (dp, tp) FSDP mesh. Use a SINGLE EP "
                f"group (expert_parallel_size={pc.stage_world_size} with ep_scope='global'), drop TP (node-local "
                "EP uses the deferred cross-replica sync), or use EP+ETP."
            )


def _reject_expert_tp(pc: ParallelismConfig) -> None:
    """Expert-TP groups stay inside one NVLink domain, under cross-node EP too."""
    if pc.expert_tp_size > 1:
        # EPConfig re-checks it, but only after the checkpoint has downloaded.
        if pc.expert_lora is not None:
            reject_expert_lora_with_expert_tp()
        if pc.nvlink_domain_size % pc.expert_tp_size != 0:
            raise ValueError(
                f"expert_tensor_parallel_size ({pc.expert_tp_size}) must divide the NVLink domain "
                f"({pc.nvlink_domain_size}). Expert TP groups must stay on NVLink for efficient "
                f"all-reduce."
            )
        if _is_multi_domain_multi_group_ep(pc):
            raise ValueError(
                f"EP+Expert-TP with multiple dispatch groups across NVLink domains is not "
                f"supported (ep_group_size={pc.ep_group_size}, world={pc.stage_world_size}): "
                f"expert-TP keeps the deferred cross-replica DP path off, so FSDP2's DP-wide "
                f"reduce-scatter would race the narrower DeepEP combine across domains. Use a "
                f"single EP group (expert_parallel_size * expert_tensor_parallel_size == world size) or set "
                f"expert_tensor_parallel_size to 1."
            )
        # One ETP group per NVLink domain; on one domain too, where EPConfig would raise after the load.
        if pc.ep_scope == "global":
            layout = CrossNodeLayout(
                world_size=pc.stage_world_size, group_size=pc.ep_group_size, domain_size=pc.nvlink_domain_size
            )
            reject_cross_node_etp_shape(
                pc.expert_tp_size,
                layout.members_per_domain,
                pc.ep_size,
                layout.num_domains,
                remedy=(
                    f"Set expert_tensor_parallel_size={layout.members_per_domain} with "
                    f"expert_parallel_size={layout.num_domains}, or use NVLink-local EP (ep_scope='node')."
                ),
            )


def _reject_ep_cp(pc: ParallelismConfig) -> None:
    """EP+CP needs one node-local EP group per domain, keeping the dispatch orthogonal to CP's partition."""
    if pc.ep_group_size > 1 and pc.cp_size > 1:
        if pc.ep_scope == "global":
            raise ValueError(
                f"EP+CP requires node-local EP (ep_scope='node' with ep_group_size == nvlink_domain_size); "
                f"cross-NVLink-domain EP (ep_scope='global') is incompatible with CP. Got "
                f"ep_group_size={pc.ep_group_size}, nvlink_domain_size={pc.nvlink_domain_size}, "
                f"context_parallel_size={pc.cp_size}."
            )
        if pc.ep_scope == "node" and pc.ep_group_size != pc.nvlink_domain_size:
            raise ValueError(
                f"Node-local EP+CP requires ep_group_size=nvlink_domain_size.\n"
                f"Got ep_group_size={pc.ep_group_size} (expert_parallel_size={pc.ep_size} * "
                f"expert_tensor_parallel_size={pc.expert_tp_size}), nvlink_domain_size={pc.nvlink_domain_size}"
            )


def _reject_single_domain_multigroup_ep(pc: ParallelismConfig) -> None:
    """Refuse ``is_racy_single_domain_multigroup_ep`` topologies before any model loading."""
    if pc.is_racy_single_domain_multigroup_ep:
        raise ValueError(
            f"expert_parallel_size={pc.ep_size} on a single {pc.nvlink_domain_size}-GPU NVLink "
            f"domain forms {pc.nvlink_domain_size // pc.ep_size} concurrent >2-rank DeepEP dispatch "
            f"groups (ep_group_size={pc.ep_group_size}), whose combine barriers race FSDP2's "
            f"DP-wide collectives. Measured on 8xB300: the legacy buffer deadlocks, the elastic "
            f"default faults with 'Invalid access of peer GPU memory over nvlink' — both with and "
            f"without gradient checkpointing. Use a SINGLE dispatch group per domain: expert_parallel_size=2, "
            f"or raise expert_parallel_size * expert_tensor_parallel_size to the domain "
            f"({pc.nvlink_domain_size}), or shrink the job to {pc.ep_group_size} GPUs. Sizing "
            f"expert_parallel_size itself to the domain works only where the model has that many "
            f"experts per rank to give (rarely on a {pc.nvlink_domain_size}-wide rack); attention TP "
            f"leaves ep_group_size unchanged, so EP+TP lands back on this same rejection."
        )


def _reject_ep_buffer_backend(pc: ParallelismConfig) -> None:
    """Refuse a ``legacy`` (DeepEP V1) backend this topology cannot drive, before the model load."""
    if pc.ep_buffer_backend != "legacy" or pc.ep_size <= 1:
        return
    # The DISPATCH group is what rides the buffer; under expert-TP it is narrower than the EP group.
    reject_legacy_backend_topology(
        pc.ep_size,
        is_cross_node=pc.num_nodes > 1 and pc.ep_scope != "node",
        gpus_per_node=pc.gpus_per_node,
        scope=f"ep_scope={pc.ep_scope!r} over {pc.num_nodes} nodes",
    )


def _reject_hsdp(pc: ParallelismConfig) -> None:
    """HSDP takes pure DP or CP only: TP/expert-TP build their own mesh, and EP's grad sync must share
    the combine's membership."""
    if not pc.use_hsdp:
        return
    if pc.tp_size > 1 or pc.expert_tp_size > 1:
        raise ValueError(
            f"use_hsdp=True is not supported with TP (tensor_parallel_size={pc.tp_size}) or Expert-TP "
            f"(expert_tensor_parallel_size={pc.expert_tp_size}); those modes build their own (dp, tp) "
            f"device mesh. Use HSDP on the standard DP path (pure DP or CP)."
        )
    if pc.ep_size > 1:
        raise ValueError(
            f"use_hsdp=True is not supported with EP (expert_parallel_size={pc.ep_size}). "
            f"Multi-group EP already shards over the EP group (deferred cross-replica sync); "
            f"single-group EP must use 1D FSDP so backward collectives share the combine's "
            f"membership."
        )
    if pc.num_nvlink_domains <= 1 and is_global_main_process():
        logger.warning(
            "use_hsdp=True but there is only one NVLink domain (single-node job) — nothing to "
            "replicate across, so FSDP full-shards over the node exactly as 1D. HSDP engages "
            "automatically once the job spans multiple domains."
        )


def _reject_fsdp_settings(pc: ParallelismConfig) -> None:
    """Refuse contradictory FSDP knobs and the ones a mode would silently ignore."""
    if not pc.fsdp_reshard_after_backward and pc.fsdp_reshard_after_forward:
        raise ValueError(
            "fsdp_reshard_after_backward=False contradicts fsdp_reshard_after_forward=True: "
            "FULL_SHARD exists to reshard for memory, keeping params unsharded across the "
            "grad-accum window defeats it. Use SHARD_GRAD_OP (fsdp_reshard_after_forward=False) "
            "with the backward reshard off."
        )
    if pc.fsdp_defer_grad_sync and pc.fsdp_reshard_after_forward:
        raise ValueError(
            "fsdp_defer_grad_sync=True contradicts fsdp_reshard_after_forward=True: deferring the "
            "reduce holds every module's full unsharded gradient across the grad-accum window (2 B/param "
            "at a bf16 reduce dtype, 4 B/param under fp32_grad_reduce), the state FULL_SHARD exists to "
            "shard. Use SHARD_GRAD_OP (fsdp_reshard_after_forward=False) with the deferred reduce, or "
            "remove fsdp_defer_grad_sync."
        )
    if not pc.fsdp_reshard_after_backward and (pc.tp_size > 1 or pc.pp_size > 1):
        raise ValueError(
            f"fsdp_reshard_after_backward=False is only wired through the plain-DP/CP/EP torchrun "
            f"path (tensor_parallel_size={pc.tp_size}, pipeline_parallel_size={pc.pp_size}): "
            f"the TP setup shards through its own fully_shard calls and PP already pins params "
            f"unsharded per stage. Remove the flag for those modes."
        )
    if pc.fsdp_defer_grad_sync and (pc.pp_size > 1 or (pc.tp_size > 1 and pc.data_parallel_size == 1)):
        raise ValueError(
            f"fsdp_defer_grad_sync=True has no per-microstep FSDP2 gradient reduce to defer "
            f"(pipeline_parallel_size={pc.pp_size}, tensor_parallel_size={pc.tp_size}, "
            f"data_parallel_size={pc.data_parallel_size}): the PP schedule already reduces once "
            f"per optimizer step, and TP at data_parallel_size=1 applies no FSDP2 wrap. Remove the flag."
        )
    if not pc.fsdp_shard_ep1_experts and (pc.tp_size > 1 or pc.cp_size > 1):
        raise ValueError(
            f"fsdp_shard_ep1_experts=False is not honored under TP or CP "
            f"(tensor_parallel_size={pc.tp_size}, context_parallel_size={pc.cp_size}): those paths "
            f"FSDP-shard the replicated experts unconditionally. Remove the flag (sharded experts are "
            f"grad-equivalent; the memory/throughput trade is in "
            f"agent-docs/parallelism/data-parallelism.md#ep1-expert-sharding), or use pure DP for "
            f"the full replicated expert copy."
        )
    if not pc.fsdp_reshard_after_forward:
        return
    if pc.has_ep_group:
        raise ValueError(
            f"fsdp_reshard_after_forward=True (FULL_SHARD / ZeRO-3) is not supported where an "
            f"expert-distribution group exists (expert_parallel_size={pc.ep_size}, "
            f"expert_tensor_parallel_size={pc.expert_tp_size}, ep_group_size={pc.ep_group_size}): "
            f"its backward-pass all-gather can race the DeepEP combine, and pure ETP shares that "
            f"path. Full-shard is supported where ep_group_size==1 (pure DP, CP, and "
            f"expert_parallel_size=1 MoE without expert TP). Otherwise reduce peak memory with "
            f"activation checkpointing instead, or set fsdp_reshard_after_forward=False "
            f"(SHARD_GRAD_OP / ZeRO-2)."
        )
    if pc.is_tp_mode and pc.data_parallel_size > 1:
        raise ValueError(
            f"fsdp_reshard_after_forward=True (FULL_SHARD / ZeRO-3) is not supported with Tensor "
            f"Parallelism + data parallelism (tensor_parallel_size={pc.tp_size}, "
            f"data_parallel_size={pc.data_parallel_size}): FSDP2's backward re-gather issues a plain "
            f"c10d all-gather on the TP-sharded DTensor params, which has no registered DTensor sharding "
            f"strategy (NotImplementedError mid-step). Use fsdp_reshard_after_forward=False (SHARD_GRAD_OP / "
            f"ZeRO-2), which keeps params gathered between forward and backward, or use_hsdp for replica "
            f"memory savings."
        )


def reject_invalid_composition(pc: ParallelismConfig) -> None:
    """Apply every composition refusal, in a fixed order."""
    _reject_pipeline_parallel(pc)
    _reject_ep_group(pc)
    _reject_cp_locality(pc)
    _reject_tp(pc)
    _reject_expert_tp(pc)
    _reject_ep_cp(pc)
    _reject_single_domain_multigroup_ep(pc)
    _reject_ep_buffer_backend(pc)
    _reject_hsdp(pc)
    _reject_fsdp_settings(pc)


def reject_accelerate_launch(pc: ParallelismConfig) -> None:
    """Refuse the custom EP/CP/TP/PP meshes under ``accelerate launch``, which owns the wrapping for every
    ``distributed_type``; the trainer and the loader both ask, with this one message."""
    if not pc.has_custom_parallelism or not is_accelerate_launch():
        return

    raise ValueError(
        "EP/CP/TP/PP modes are not compatible with 'accelerate launch' "
        "(any distributed_type: FSDP and MULTI_GPU/DDP alike).\n"
        "\n"
        "Detected accelerate launcher env: "
        f"ACCELERATE_MIXED_PRECISION={env_str('ACCELERATE_MIXED_PRECISION')}, "
        f"ACCELERATE_USE_FSDP={env_str('ACCELERATE_USE_FSDP')}\n"
        f"Active parallelism: EP={pc.ep_size}, CP={pc.cp_size}, TP={pc.tp_size}, PP={pc.pp_size}\n"
        "\n"
        "For EP/CP/TP/PP training, use 'torchrun' instead:\n"
        "\n"
        "  # Single node\n"
        "  torchrun --nproc_per_node=8 scripts/training/sft.py <config> \\\n"
        "      --expert_parallel_size=8\n"
        "\n"
        "  # Multi-node\n"
        "  torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \\\n"
        "      --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \\\n"
        "      scripts/training/sft.py <config> \\\n"
        "      --expert_parallel_size=8\n"
        "\n"
        "For standard data parallelism (no EP/CP/TP/PP), use 'torchrun' too:\n"
        "\n"
        "  torchrun --nproc_per_node=<N> scripts/training/sft.py <config>\n"
        "\n"
        "See agent-docs/parallelism/multi-node.md and agent-docs/parallelism/data-parallelism.md for details."
    )


def reject_cross_domain_determinism(pc: ParallelismConfig, full_determinism: bool) -> None:
    """Refuse ``full_determinism`` on an EP group spanning NVLink domains: DeepEP's hybrid RDMA kernels
    have no deterministic mode."""
    if full_determinism and pc.requires_rdma:
        raise ValueError(
            f"full_determinism cannot run over an EP group spanning NVLink domains (ep_scope=global "
            f"over {pc.num_nvlink_domains} domains of {pc.nvlink_domain_size} GPUs): DeepEP's "
            f"cross-domain (hybrid RDMA) dispatch has no deterministic mode and places received tokens "
            f"in atomic claim order, so every expert weight gradient is summed in a different order on "
            f"every step. Keep each EP group inside one NVLink domain (ep_scope=node with data "
            f"parallelism across domains; on a multi-node NVLink fabric, NVLINK_DOMAIN_SIZE set to the "
            f"fabric's size), or drop full_determinism."
        )
