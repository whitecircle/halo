"""Pipeline-parallel checkpoints: one complete-tensor shard per stage under hub names plus a merged index, and the
resume that loads only this stage's tensors."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from functools import partial

import torch
import torch.nn as nn

from halo.checkpoint.config_export import save_model_config
from halo.checkpoint.external_tables import write_external_tables
from halo.checkpoint.format import PP_WRAPPER_PART_PREFIX, pp_stage_part_prefix
from halo.checkpoint.hub_layout import ModuleLayoutView, revert_conversions
from halo.checkpoint.parallel.context import CheckpointContext, CheckpointLoadContext
from halo.checkpoint.parallel.reading import joined_streaming_reader, reject_ep_sharded_resume
from halo.checkpoint.parallel.writing import chunked_saveable_tensors
from halo.checkpoint.reading import PER_NODE_PLACEMENT_REMEDY, resolve_checkpoint_weights
from halo.checkpoint.weights_source import (
    LOADED_WEIGHTS_FROM_ATTR,
    built_from_checkpoint,
    describe_weights_source,
    weights_read_from,
)
from halo.checkpoint.writing import StreamingShardWriter, merge_shard_index, save_dtype_caster, write_merged_index
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.log import KEY_PREVIEW_COUNT
from halo.models.introspection.unwrapping import unwrap_toolkit_wrappers
from halo.runtime.collectives import barrier, combine_across_ranks
from halo.runtime.consensus import DeferredRankFailure, barrier_on_exit, rank_consensus, reject_across_ranks
from halo.runtime.dtensor import copy_full_tensor
from halo.runtime.filesystem import is_output_shared_filesystem
from halo.runtime.ranks import is_global_main_process, is_local_main_process

logger = logging.getLogger(__name__)


def reject_unhandled_pp_axes(config, phase: str) -> None:
    """Refuse a PP checkpoint ``phase`` with an axis it cannot express; only FSDP2's dp DTensors get un-sharded.
    ``SUPPORTED_AXIS_SETS`` rejects TP and CP at config time, and this holds if that allowlist widens."""
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
    """Shared FS: one rank per stage. Per-node storage: the node's first rank and each stage's first rank on it, so
    each node's directory holds the shard of every stage it runs and resumes on its own (``gpus_per_node`` below the
    launcher's node size can put several stages on one node)."""
    if shared_fs:
        return config.stage_local_rank == 0
    return is_local_main_process() or config.stage_local_rank == 0


def _write_hub_group(
    writer: StreamingShardWriter, stage: nn.Module, tensors: Iterable[tuple[str, torch.Tensor]]
) -> None:
    """Write one conversion-closed group under hub names, reverting the stage's snapshotted conversions. ``tensors`` is
    consumed here, so a cast it carries runs inside the caller's guard."""
    conversions = list(stage.checkpoint_namespace.export_conversions)
    for key, tensor in revert_conversions(stage, dict(tensors), conversions).items():
        writer.add(key, tensor)


def _write_stage_tensors(
    ctx: CheckpointContext, output_dir: str, guard: DeferredRankFailure, *, is_writer: bool
) -> tuple[dict[str, str], int]:
    """COLLECTIVE: stream this stage's tensors and external tables into its shard; ``(weight-map fragment, bytes)``.
    Every stage rank enters the gathers, only the writer retains. Closing writes the last part, which must precede the
    index exchange, the sync guaranteeing the index never names a file not yet on disk."""
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
    # A stage's part prefix keeps its table files apart from another stage's; the writer indexes them with its parts.
    guard.run(
        partial(
            write_external_tables,
            stage,
            writer,
            checkpoint_name=stage.global_parameter_name,
            file_prefix=writer.filename_prefix,
        )
    )
    return guard.run(writer.close) or ({}, 0)


def _write_wrapper_state(
    ctx: CheckpointContext, output_dir: str, guard: DeferredRankFailure
) -> tuple[dict[str, str], int]:
    """Write the tensors no stage holds (a multimodal vision tower), already at the save dtype, so the checkpoint
    reloads as the wrapper class. Per-node storage has one save rank per node, so every node's directory carries
    them."""
    writer = StreamingShardWriter(output_dir, ctx.max_shard_size, enabled=True, filename_prefix=PP_WRAPPER_PART_PREFIX)
    guard.run(partial(_write_hub_group, writer, ctx.model, ctx.pp_wrapper_state.items()))
    return guard.run(writer.close) or ({}, 0)


def _merge_writer_fragments(gathered: list[list[tuple[dict[str, str], int]] | None]) -> tuple[dict[str, str], int]:
    """Each writer's stage and wrapper fragments stay apart, as every node's save rank writes the same-named wrapper
    parts beside its own stage's and :func:`merge_shard_index` counts a fragment only when none of its files were
    seen."""
    return merge_shard_index([fragment for fragments in gathered if fragments for fragment in fragments])


def save_pp_checkpoint(ctx: CheckpointContext, output_dir: str) -> None:
    """COLLECTIVE: one safetensors shard per stage under hub names plus a merged index, loadable by plain
    ``from_pretrained``. The index has HF metadata only, since the repo ``format`` marker means per-rank partial
    tensors."""
    config = ctx.parallelism_config
    stage = ctx.model
    reject_unhandled_pp_axes(config, "save")
    # An evaluation right before the save leaves transient unsharded params, and ``state_dict()`` would then hand the
    # walk plain tensors instead of the dp DTensors it gathers.
    reshard_fsdp2_modules(stage)
    shared_fs = is_output_shared_filesystem()
    is_writer = is_pp_shard_writer(config, shared_fs)

    # Writes interleave with the gathers below, so a raise would strand the other stage ranks until the watchdog fires.
    guard = DeferredRankFailure(f"PP checkpoint write to {output_dir}")
    fragments = [_write_stage_tensors(ctx, output_dir, guard, is_writer=is_writer)]
    if ctx.is_save_rank and ctx.pp_wrapper_state:
        fragments.append(_write_wrapper_state(ctx, output_dir, guard))
    # A failed write stops every rank here, before an index is built over a missing or truncated shard; over the
    # store, since the peers wait out each stage writer's last part.
    guard.store_reject("checkpoint/pp_stages")

    # A global parameter lives on exactly one stage, so the merge's collision check is a real gate here.
    weight_map, total_size = combine_across_ranks(
        fragments if is_writer else None, _merge_writer_fragments, what="the checkpoint index"
    )

    with barrier_on_exit(f"Writing the PP checkpoint index to {output_dir}"):
        if ctx.is_save_rank:
            # Every stage's parts are on disk, so the sweep inside deletes exactly what no stage claimed.
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
    """COLLECTIVE. Load this rank's tensors into its stage; skipped when ``load_distributed_model`` built every stage
    from this checkpoint, unless for a best-model load. Keyed globally, so weights resume onto another ``pp_size``; the
    stage-local optimizer shards do not."""
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
