"""Pipeline-parallel checkpoints: one complete-tensor shard per stage under the hub's names plus a merged
index, and the resume that loads only this stage's tensors back."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from functools import partial

import torch
import torch.nn as nn

from src.checkpoint.config_export import save_model_config
from src.checkpoint.format import PP_WRAPPER_PART_PREFIX, pp_stage_part_prefix
from src.checkpoint.hub_layout import ModuleLayoutView, revert_conversions
from src.checkpoint.parallel.context import CheckpointContext, CheckpointLoadContext
from src.checkpoint.parallel.reading import joined_streaming_reader, reject_ep_sharded_resume
from src.checkpoint.parallel.writing import chunked_saveable_tensors
from src.checkpoint.reading import PER_NODE_PLACEMENT_REMEDY, resolve_checkpoint_weights
from src.checkpoint.weights_source import (
    LOADED_WEIGHTS_FROM_ATTR,
    built_from_checkpoint,
    describe_weights_source,
    weights_read_from,
)
from src.checkpoint.writing import StreamingShardWriter, merge_shard_index, save_dtype_caster, write_merged_index
from src.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from src.distributed.fsdp import reshard_fsdp2_modules
from src.log import KEY_PREVIEW_COUNT
from src.models.unwrapping import unwrap_toolkit_wrappers
from src.runtime.collectives import barrier, barrier_on_exit, combine_across_ranks
from src.runtime.consensus import DeferredRankFailure, rank_consensus, reject_across_ranks
from src.runtime.dtensor import copy_full_tensor
from src.runtime.filesystem import is_output_shared_filesystem
from src.runtime.ranks import is_global_main_process, is_local_main_process

logger = logging.getLogger(__name__)


def reject_unhandled_pp_axes(config, phase: str) -> None:
    """Refuse a PP checkpoint ``phase`` ("save"/"resume") combined with an axis it cannot express.

    The PP shard writer/reader un-shards exactly one thing: FSDP2's dp DTensors. TP shards the
    planned projections along a second mesh dimension and CP renames every projection one level
    deeper, so either needs a second inverse; ``SUPPORTED_AXIS_SETS`` rejects both at config time, and
    this keeps the shortcut honest if that allowlist ever widens.
    """
    unhandled = [name for name, size in (("tp", config.tp_size), ("cp", config.cp_size)) if size > 1]
    if unhandled:
        raise NotImplementedError(
            f"Pipeline-parallel checkpoint {phase} does not handle {'+'.join(unhandled).upper()} "
            f"sharding: the stage shards carry COMPLETE tensors of the unsplit model, and only FSDP2's "
            f"dp DTensors are reconstructed. PP+TP / PP+CP are rejected by SUPPORTED_AXIS_SETS; if "
            f"that changes, this path needs the matching inverse (a TP-dimension unfold / a CP key "
            f"remap) before it can be trusted."
        )


def is_pp_shard_writer(config, shared_fs: bool) -> bool:
    """Whether this rank writes its pipeline stage's checkpoint shard.

    Shared FS: one rank per stage (all shards land in one directory). Per-node storage: one rank per
    stage per node — the node's first rank and the first rank of each stage that starts on the node —
    so each node's directory holds the shard of every stage it runs and resumes on its own. A node of
    the launch can run several stages where ``gpus_per_node`` is set below the launcher's node size.
    """
    if shared_fs:
        return config.stage_local_rank == 0
    return is_local_main_process() or config.stage_local_rank == 0


def _write_hub_group(
    writer: StreamingShardWriter, stage: nn.Module, tensors: Iterable[tuple[str, torch.Tensor]]
) -> None:
    """Write one conversion-closed group of the stage's tensors, named in the unsplit model, under the
    hub's names: the save-side revert of the conversions the stage snapshotted, as the EP-gathered save
    applies it. ``tensors`` is consumed here, so a cast it carries runs inside the caller's guard.
    """
    for key, tensor in revert_conversions(stage, dict(tensors), list(stage.export_conversions)).items():
        writer.add(key, tensor)


def _write_stage_tensors(
    ctx: CheckpointContext, output_dir: str, guard: DeferredRankFailure, *, is_writer: bool
) -> tuple[dict[str, str], int]:
    """COLLECTIVE: stream this stage's tensors into its shard; ``(weight-map fragment, bytes written)``.

    Every stage rank enters the DTensor and EP gathers; only the writer retains and writes. Closing
    writes the last part, which must precede the index exchange: that exchange is the sync
    guaranteeing the index never names a file that is not on disk yet.
    """
    config, stage = ctx.parallelism_config, ctx.model
    writer = StreamingShardWriter(
        output_dir,
        ctx.max_shard_size,
        enabled=is_writer,
        filename_prefix=pp_stage_part_prefix(config.pp_rank, config.pp_size),
    )
    # The map the resume path reads by, so the two cannot diverge.
    name_map = stage.checkpoint_name_map()
    # Keyed by the live (stage-local) spelling the caster's keep-sets use, so it runs before the hub respell.
    cast = save_dtype_caster(stage, keep_live_dtype=ctx.training_checkpoint)
    state_dict = stage.state_dict()
    chunks = chunked_saveable_tensors(
        stage, retain=is_writer, items=((local, state_dict[local]) for local in name_map.values())
    )
    for chunk in chunks:
        hub = ((stage.global_parameter_name(local), cast(local, tensor)) for local, tensor in chunk.items())
        guard.run(partial(_write_hub_group, writer, stage, hub))
    for layer_name, module in named_ep_layers(stage):
        # One gathered layer at a time bounds the writer's peak to a layer, not the whole stage.
        ep_weights = module.gather_layer_state_dict(layer_name, retain=is_writer)
        hub = ((stage.global_parameter_name(key), cast(key, tensor)) for key, tensor in ep_weights.items())
        guard.run(partial(_write_hub_group, writer, stage, hub))
        del ep_weights
    return guard.run(writer.close) or ({}, 0)


def _write_wrapper_state(
    ctx: CheckpointContext, output_dir: str, guard: DeferredRankFailure
) -> tuple[dict[str, str], int]:
    """Write the tensors no stage holds (a multimodal wrapper's vision tower) under their own part name.

    Untouched by training and already at the artifact's save dtype, they are re-emitted from the save
    rank so the checkpoint reloads and serves as the wrapper class and a resume finds every tensor it
    plans for. Per-node storage has one save rank per node, so every node's directory carries them.
    """
    writer = StreamingShardWriter(output_dir, ctx.max_shard_size, enabled=True, filename_prefix=PP_WRAPPER_PART_PREFIX)
    guard.run(partial(_write_hub_group, writer, ctx.model, ctx.pp_wrapper_state.items()))
    return guard.run(writer.close) or ({}, 0)


def save_pp_checkpoint(ctx: CheckpointContext, output_dir: str) -> None:
    """COLLECTIVE: one safetensors shard per stage under the hub's names, plus a merged index.

    ``ctx.model`` is this rank's ``PipelineStageModule``, whose ``global_parameter_name`` maps
    stage-local names back to the unsplit model's, so the merged checkpoint loads through plain
    ``from_pretrained``. The writers (:func:`is_pp_shard_writer`) write their stage's shard, and the
    index and config follow a world-wide exchange of the per-stage key maps. The index carries standard
    HF metadata only, no repo "format" marker: that marker means per-rank partial tensors (ep_sharded).
    """
    config = ctx.parallelism_config
    stage = ctx.model
    reject_unhandled_pp_axes(config, "save")
    # An evaluation right before the save leaves the stage's FSDP2 modules holding their transient
    # unsharded params, and ``state_dict()`` would then hand the walk plain tensors instead of the dp
    # DTensors it gathers.
    reshard_fsdp2_modules(stage)
    shared_fs = is_output_shared_filesystem()
    is_writer = is_pp_shard_writer(config, shared_fs)

    # The writer's disk writes are interleaved with the gathers below, so one must not raise here:
    # it would strand every other stage rank in the next collective until the watchdog fires.
    guard = DeferredRankFailure(f"PP checkpoint write to {output_dir}")
    stage_weight_map, shard_bytes = _write_stage_tensors(ctx, output_dir, guard, is_writer=is_writer)
    if ctx.is_save_rank and ctx.pp_wrapper_state:
        wrapper_map, wrapper_bytes = _write_wrapper_state(ctx, output_dir, guard)
        stage_weight_map = {**stage_weight_map, **wrapper_map}
        shard_bytes += wrapper_bytes
    # A failed write stops every rank here, before an index is built over a missing or truncated shard.
    guard.reject()

    # A global parameter lives on exactly one stage, so the merge's collision check is a real gate here.
    weight_map, total_size = combine_across_ranks(
        (stage_weight_map, shard_bytes) if is_writer else None, merge_shard_index, what="the checkpoint index"
    )

    with barrier_on_exit():
        if ctx.is_save_rank:
            # Every stage's parts are on disk and ``weight_map`` is the merged map, so the sweep inside
            # deletes exactly what no stage claimed.
            write_merged_index(output_dir, weight_map, {"total_size": total_size})
            save_model_config(stage, output_dir)
            logger.info(f"Saved PP model to {output_dir} ({len(weight_map)} keys, {config.pp_size} stage shards)")
            if not shared_fs:
                # The index is deliberately the GLOBAL key map, so resume works on every node.
                nodes_per_stage = config.stage_world_size // config.gpus_per_node
                logger.warning(
                    "Non-shared output filesystem: writing one copy of stage %d's shard per node "
                    "(%d nodes in this stage, so %dx duplication of this stage's bytes across the "
                    "job). Each node's directory resumes on its own; to export, merge every node's "
                    "directory with scripts/after_training/merge_pp_stages.py before from_pretrained.",
                    config.pp_rank,
                    nodes_per_stage,
                    nodes_per_stage,
                )


def restore_pp_stage_weights(
    ctx: CheckpointLoadContext, checkpoint: str, model, *, for_best_model: bool = False
) -> None:
    """COLLECTIVE. Load a PP checkpoint's tensors into this rank's pipeline stage.

    Skipped when ``load_distributed_model`` built every rank's stage from this checkpoint (it read the
    same values, FP32 masters at FP32), unless for a best-model load. The hub-namespaced keys map onto
    the unsplit model's names (:class:`ModuleLayoutView`) and back to the stage's local ones, and only
    this rank's are read; the copies (``distribute_tensor``) stay inside the stage's own DP mesh. The
    weights are keyed globally, so they resume onto another ``pp_size``; the stage-local optimizer
    shards do not.
    """
    stage = unwrap_toolkit_wrappers(model)
    reject_unhandled_pp_axes(ctx.parallelism_config, "resume")
    reject_ep_sharded_resume(checkpoint)

    # EP experts only load at construction. Both inputs are rank-local (a stage may hold no MoE layer,
    # realpath resolves per node), so every rank joins the verdict.
    live_source = weights_read_from(stage)
    reason = None
    if named_ep_layers(stage) and not built_from_checkpoint(live_source, checkpoint):
        whence = describe_weights_source(live_source, "stage")
        reason = (
            "a stage's EP expert weights only load at construction, so resume requires the model to be constructed "
            f"FROM the checkpoint: {whence}, not '{checkpoint}' — its experts would resume from BASE weights. "
            "Launch via the training scripts (prepare_distributed_resume repoints model_name_or_path at the "
            "checkpoint), or pass a model loaded from the checkpoint directory."
        )
    reject_across_ranks(reason, "PP+EP resume construction identity", exc_type=ValueError)

    # Only load_distributed_model's own stamp vouches for an exact read (FP32 masters unrounded).
    constructed_everywhere = rank_consensus(
        built_from_checkpoint(getattr(stage, LOADED_WEIGHTS_FROM_ATTR, None), checkpoint)
    ).all_ok
    if constructed_everywhere and not for_best_model:
        if is_global_main_process():
            logger.info(f"PP resume: every stage was constructed from {checkpoint}; skipping the weight reload.")
        return

    # An evaluation before a best-model load can leave the stage's FSDP2 modules unsharded.
    reshard_fsdp2_modules(stage)

    # The map the save walked; the index is read per rank, so a torn one is joined.
    local_by_global = stage.checkpoint_name_map()
    layout, checkpoint_keys, reason = None, set(), None
    try:
        layout = resolve_checkpoint_weights(checkpoint)
        checkpoint_keys = layout.key_set()
    except Exception as e:
        reason = f"unreadable checkpoint index at {checkpoint} ({type(e).__name__}: {e}), a torn save"
    reject_across_ranks(reason, "PP resume checkpoint index")
    view = ModuleLayoutView(
        stage, checkpoint_keys, live_names=local_by_global.keys(), namespace=stage.checkpoint_namespace
    )
    # Index only: a single file's header would be opened rank-locally here; the joined reader opens it below.
    absent = layout.absent_shard_files(view.sources) if layout.index is not None else []
    reject_across_ranks(
        f"PP stage {ctx.parallelism_config.pp_rank}: {checkpoint} indexes this stage's tensors in shard file(s) "
        f"absent from this node's copy ({absent[:KEY_PREVIEW_COUNT]}). {PER_NODE_PLACEMENT_REMEDY}"
        if absent
        else None,
        "PP resume shard placement",
    )

    with joined_streaming_reader(checkpoint, view.sources, what="PP checkpoint") as reader:
        # A stage tensor the checkpoint does not supply would silently keep BASE weights.
        missing = sorted(set(local_by_global) - view.keys)
        if not rank_consensus(not missing).all_ok:
            raise RuntimeError(
                f"PP resume from {checkpoint}: {len(missing)} of this stage's tensors are absent "
                f"from the checkpoint on at least one rank — resume would silently continue from "
                f"base weights for them. First few missing: {missing[:KEY_PREVIEW_COUNT]}. Resume from a complete "
                f"checkpoint saved under the identical pipeline topology."
            )

        live: dict[str, torch.Tensor] = dict(stage.named_parameters())
        live.update(dict(stage.named_buffers()))
        with torch.no_grad():
            # Natural key order, so every stage rank issues the mesh collectives in the same order.
            for global_name, value in view.items(reader.get):
                local_name = local_by_global[global_name]
                target = live.get(local_name)
                if target is None:
                    raise RuntimeError(
                        f"PP resume: stage state_dict key '{local_name}' has no live parameter/buffer to load into "
                        "— a state-dict hook the PP loader does not understand."
                    )
                copy_full_tensor(target, value)

    if is_global_main_process():
        logger.info(f"✓ PP stage checkpoint loaded from {checkpoint} ({len(local_by_global)} tensors)")
    barrier()
