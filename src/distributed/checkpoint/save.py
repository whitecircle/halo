"""Model checkpoint saving: the mode ladder and the per-mode savers it dispatches to.

:func:`save_checkpoint` is the one entry point every trainer calls. :func:`select_checkpoint_saver`
walks an ordered ladder — PP first (a stage is a partial model, so every other saver would write it
as if complete), then EP before CP (so EP+CP gathers experts), CP before TP, TP before plain FSDP2 —
and returns ``None`` when no mode owns the save, which is the trainer's signal to fall through to
``Trainer.save_model``. Every predicate is rank-uniform, so all ranks pick the identical saver. PEFT
adapters are dispatched separately via :class:`~src.distributed.checkpoint.peft.PeftAdapterSaver`.

FSDP2, CP and TP differ only in where their chunks come from: all three stream through
:func:`~src.distributed.checkpoint.write.stream_gathered_checkpoint`. EP and PP own genuinely
different artifacts — a family-specific expert gather, and one shard per stage under global names.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import replace
from functools import partial

import torch
import torch.nn as nn
from transformers.trainer import TRAINER_STATE_NAME

from src.checkpoint.adapters import EXPERT_LORA_PEFT_TYPE
from src.checkpoint.atomic import publish_staged_file, withheld_file_name
from src.checkpoint.config_export import save_model_config
from src.checkpoint.format import (
    OPTIMIZER_META_FILE,
    RESUME_ADAPTER_DIR,
    RESUME_ADAPTER_MARKER_FILE,
    save_dtype_caster,
    write_merged_index,
    write_resume_adapter_marker,
)
from src.checkpoint.shard_writer import StageShardWriter
from src.distributed.checkpoint.context import CheckpointContext
from src.distributed.checkpoint.ep_save import save_ep_lora_adapters, save_ep_model
from src.distributed.checkpoint.peft import PeftAdapterSaver, expert_lora_config_fields, find_peft_model
from src.distributed.checkpoint.tp_save import save_tp_model
from src.distributed.checkpoint.write import (
    chunked_saveable_tensors,
    exchange_shard_index,
    resolve_retained,
    stream_gathered_checkpoint,
)
from src.distributed.expert_parallel.expert_weights import gather_ep_layer_weights
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.runtime import (
    DeferredRankFailure,
    barrier_on_exit,
    fs_aware_save_rank,
    is_local_main_process,
    is_output_shared_filesystem,
)
from src.models.patches.gpt_oss_sinks import neutralized_gpt_oss_sinks
from src.models.structure import lora_fold_targets, unwrap_model

logger = logging.getLogger(__name__)

# A saver always handles the save it is given; "nobody owns this" is the ladder's ``None``.
CheckpointSaver = Callable[[CheckpointContext, str], None]
# Where the base save's trainer_state.json waits until every other file of the checkpoint is on disk:
# resume detection takes a checkpoint whose trainer state every rank holds as complete.
UNCOMMITTED_TRAINER_STATE = withheld_file_name(TRAINER_STATE_NAME)
# The files that vouch for a checkpoint, each written last in its own half: the trainer state (and its
# uncommitted spelling), the meta that vouches for the optimizer shard set, and the resume adapter marker.
CHECKPOINT_COMPLETION_MARKERS = (
    TRAINER_STATE_NAME,
    UNCOMMITTED_TRAINER_STATE,
    OPTIMIZER_META_FILE,
    RESUME_ADAPTER_MARKER_FILE,
)


def save_checkpoint(ctx: CheckpointContext, output_dir: str) -> bool:
    """Save the model for the active parallelism mode. False = fall through to ``Trainer.save_model``."""
    saver = select_checkpoint_saver(ctx)
    if saver is None:
        return False
    saver(ctx, output_dir)
    return True


def select_checkpoint_saver(ctx: CheckpointContext) -> CheckpointSaver | None:
    """The saver for the active mode, or ``None`` when no mode owns the save (see module docstring)."""
    if ctx.is_pp_mode:
        return save_pp_checkpoint
    if ctx.is_ep_tp_mode or ctx.has_ep_layers:
        return save_ep_checkpoint
    if ctx.is_cp_mode:
        # The wrapper owns the CP key remap; without it there is no CP save to make, only the base one.
        return save_cp_checkpoint if ctx.cp_wrapper is not None else None
    if ctx.is_tp_mode:
        return save_tp_checkpoint
    # Mixin-managed FSDP2 only; _fsdp_wrapped is False for accelerate-managed FSDP.
    if ctx.fsdp_wrapped and not ctx.accelerate_manages_fsdp:
        return save_fsdp2_checkpoint
    return None


def _save_streamed(
    ctx: CheckpointContext,
    output_dir: str,
    model: nn.Module,
    chunks: Iterable[dict[str, torch.Tensor]],
    label: str,
) -> None:
    """The body FSDP2 and CP share: stream the gathered chunks, then the tokenizer, under one fence.

    Gathered on all ranks (the chunk source's resolves are collective) but written one chunk at a
    time, so the save rank never holds the whole model in host RAM. Fenced because one rank writes
    while all must reach the trailing barrier — an ENOSPC would otherwise strand the peers.
    """
    with barrier_on_exit():
        stream_gathered_checkpoint(
            model,
            chunks,
            output_dir,
            is_save_rank=ctx.is_save_rank,
            max_shard_size=ctx.max_shard_size,
            keep_live_dtype=ctx.training_checkpoint,
        )
        if ctx.is_save_rank:
            if ctx.tokenizer is not None:
                ctx.tokenizer.save_pretrained(output_dir)
            logger.info(f"Saved {label} model to {output_dir}")


def save_fsdp2_checkpoint(ctx: CheckpointContext, output_dir: str) -> None:
    """Mixin-managed FSDP2 (torchrun standard DP). Params are DTensors — gathered via full_tensor()."""
    _save_streamed(ctx, output_dir, ctx.model, chunked_saveable_tensors(ctx.model, retain=ctx.is_save_rank), "FSDP2")


def save_cp_checkpoint(ctx: CheckpointContext, output_dir: str) -> None:
    """CP-only (no EP). The CP wrapper's ``state_dict()`` filters/remaps attention keys.

    That dict is the item source rather than the module walk — it already carries the persistent
    buffers under the remapped keys — and it holds references only: the streamed gather is what
    resolves them, chunk by chunk. The sinks still come off the unwrapped model.
    """
    cp_wrapper = ctx.cp_wrapper
    inner = unwrap_model(cp_wrapper)
    items = cp_wrapper.state_dict().items()
    _save_streamed(ctx, output_dir, inner, chunked_saveable_tensors(inner, retain=ctx.is_save_rank, items=items), "CP")


def save_tp_checkpoint(ctx: CheckpointContext, output_dir: str) -> None:
    """TP-only / TP+DP. ``save_tp_model`` runs the second gather its hand-sliced params need."""
    save_tp_model(
        ctx.model,
        output_dir,
        tokenizer=ctx.tokenizer,
        max_shard_size=ctx.max_shard_size,
        keep_live_dtype=ctx.training_checkpoint,
    )


def _expert_lora_adapter_config(ctx: CheckpointContext) -> dict | None:
    """A PEFT-style adapter_config dict from the run's ExpertLoraSpec (for adapter-only EP saves).

    The expert fields come from :func:`expert_lora_config_fields`, shared with the mixed
    attention+expert save so both artifacts describe the expert half identically.
    ``base_model_name_or_path`` keeps the adapter directory self-describing, as PEFT's own
    ``adapter_config.json`` does.
    """
    spec = getattr(ctx.parallelism_config, "expert_lora", None)
    if spec is None:
        return None
    return {
        "peft_type": EXPERT_LORA_PEFT_TYPE,
        "base_model_name_or_path": getattr(ctx.model.config, "_name_or_path", None),
        **expert_lora_config_fields(spec),
    }


def save_ep_checkpoint(ctx: CheckpointContext, output_dir: str) -> None:
    """EP / EP+TP / EP+CP / EP+ETP. Gathers distributed expert weights via ``save_ep_model``.

    With native grouped-LoRA on experts, writes a standalone adapter unless
    ``merge_expert_lora_on_save`` requests a merged checkpoint. That merge covers BOTH halves of a
    mixed run: the expert deltas fold inside each family's gather, and any attention adapters fold
    into each base tensor as it is written, out of place (:func:`~src.models.structure.lora_folded`,
    the weight sync's fold), so the save never writes the trainer's own weights. A training
    checkpoint also carries the unmerged adapters it resumes from (:func:`save_resume_adapter`).
    """
    if ctx.has_expert_lora and not ctx.merge_expert_lora_on_save:
        save_ep_lora_adapters(
            ctx.model,
            output_dir,
            adapter_config=_expert_lora_adapter_config(ctx),
            tokenizer=ctx.tokenizer,
            keep_live_dtype=ctx.training_checkpoint,
        )
        return
    # Structural, so a layer the fold cannot reproduce raises on every rank before any gather.
    peft_model = find_peft_model(ctx.model)
    save_ep_model(
        ctx.model,
        output_dir,
        tokenizer=ctx.tokenizer,
        sharded=ctx.save_sharded_ep,
        cp_key_remap=ctx.is_cp_mode,
        max_shard_size=ctx.max_shard_size,
        merge_lora=ctx.has_expert_lora and ctx.merge_expert_lora_on_save,
        lora_folds=lora_fold_targets(peft_model) if peft_model is not None else None,
        keep_live_dtype=ctx.training_checkpoint,
    )


def save_resume_adapter(ctx: CheckpointContext, checkpoint_dir: str) -> None:
    """Write a ``merge_expert_lora_on_save`` checkpoint's resume state beside its merged weights.

    The merged weights serve but cannot resume: the bf16 fold loses part of the delta, and the
    optimizer state belongs to the adapters, not to the fold. The unmerged adapters go to
    :data:`~src.checkpoint.format.RESUME_ADAPTER_DIR` through the writer the non-merged save uses
    (:class:`PeftAdapterSaver` when a PeftModel carries an attention half,
    :func:`save_ep_lora_adapters` for expert-only), at their live dtype as any training checkpoint
    writes them, so the adapter restore reads them unchanged. Each save rank then writes the marker
    the resume classifies on, after its own copy is complete. With any older marker removed before
    the save began (:func:`remove_stale_completion_markers`), a failed adapter write leaves no marker.
    Collective: every rank enters the adapter gathers.
    """
    # Rank-uniform, and a no-op when sharded: a forward's transient unsharded params predate the
    # last optimizer step, like every writer's.
    reshard_fsdp2_modules(ctx.model)
    adapter_dir = os.path.join(checkpoint_dir, RESUME_ADAPTER_DIR)
    peft_model = find_peft_model(ctx.model)
    if peft_model is not None:
        # No tokenizer: the checkpoint root carries it, and nothing loads this directory standalone.
        PeftAdapterSaver().save(replace(ctx, tokenizer=None, training_checkpoint=True), peft_model, adapter_dir)
    else:
        save_ep_lora_adapters(
            ctx.model, adapter_dir, adapter_config=_expert_lora_adapter_config(ctx), keep_live_dtype=True
        )
    mark_resume_adapter_complete(checkpoint_dir, is_save_rank=ctx.is_save_rank)


def mark_resume_adapter_complete(checkpoint_dir: str, *, is_save_rank: bool) -> None:
    """Write the marker a resume classifies ``checkpoint_dir`` on, once its resume adapter is on disk.

    Shared by every writer of a resume adapter, the embedding trainer's included. Fenced: every rank
    enters, and each save rank marks its own copy only after its own adapter write completed.
    """
    with barrier_on_exit():
        if is_save_rank:
            write_resume_adapter_marker(checkpoint_dir)
            logger.info(f"Saved the resume adapter of merged checkpoint {checkpoint_dir}")


def remove_stale_completion_markers(checkpoint_dir: str) -> None:
    """Remove the :data:`CHECKPOINT_COMPLETION_MARKERS` an earlier save left, before this save rewrites it.

    A run resumed from an earlier checkpoint saves the same ``checkpoint-N`` again, in place, and
    whatever markers that directory holds describe the abandoned save: until new ones land they would
    vouch for its trainer state, optimizer shards and adapter beside this save's partly written files.
    Removed first, a save torn before its own markers is refused on resume instead. Fenced: every rank
    enters, and each FS-aware save rank clears its own copy.
    """
    with barrier_on_exit():
        if fs_aware_save_rank():
            for name in CHECKPOINT_COMPLETION_MARKERS:
                with suppress(FileNotFoundError):
                    os.remove(os.path.join(checkpoint_dir, name))


@contextmanager
def trainer_state_withheld(state) -> Iterator[None]:
    """Have the base save write ``state`` as :data:`UNCOMMITTED_TRAINER_STATE` instead of ``trainer_state.json``.

    The base Trainer writes the trainer state before the toolkit's sidecars and optimizer shards, so
    under its own name it would vouch for a checkpoint those have not reached yet. Written under the
    withheld name from the start, ``trainer_state.json`` appears only when :func:`commit_trainer_state`
    publishes it, and a save stopped at any point leaves a directory resume detection passes over.
    """
    write = state.save_to_json

    def withheld(json_path: str) -> None:
        directory, name = os.path.split(json_path)
        write(os.path.join(directory, UNCOMMITTED_TRAINER_STATE) if name == TRAINER_STATE_NAME else json_path)

    state.save_to_json = withheld
    try:
        yield
    finally:
        del state.save_to_json


def commit_trainer_state(checkpoint_dir: str) -> None:
    """Publish the withheld trainer state as the checkpoint's last file, synced with its directory.

    Collective: each FS-aware save rank publishes its own copy, and a failure raises on every rank, a
    save rank holding no withheld state included: its checkpoint would never resume, and rotation, which
    orders by mtime or step rather than completeness, could remove the last one that does. The caller
    runs it once every other file of the checkpoint is on disk and before rotation can remove an older
    one.
    """
    guard = DeferredRankFailure(f"Committing {TRAINER_STATE_NAME} in {checkpoint_dir}")
    if fs_aware_save_rank():
        guard.run(partial(_publish_withheld_trainer_state, checkpoint_dir))
    guard.reject()


def _publish_withheld_trainer_state(checkpoint_dir: str) -> None:
    staged = os.path.join(checkpoint_dir, UNCOMMITTED_TRAINER_STATE)
    if not os.path.isfile(staged):
        raise FileNotFoundError(
            f"{staged} is missing: the base save wrote no trainer state on this save rank. Its writers "
            f"(args.should_save, set by save_on_each_node) must be the output filesystem's save ranks "
            f"(DIST_OUTPUT_SHARED_FILESYSTEM)."
        )
    publish_staged_file(staged, os.path.join(checkpoint_dir, TRAINER_STATE_NAME))


def reject_unhandled_pp_axes(config, phase: str) -> None:
    """Refuse a PP checkpoint ``phase`` ("save"/"resume") combined with an axis it cannot express.

    The PP shard writer/reader un-shards exactly one thing: FSDP2's dp DTensors. TP shards the
    planned projections along a second mesh dimension (2-D ``(dp, tp)`` DTensors) and CP renames
    every projection one level deeper, so either needs a second inverse. ``SUPPORTED_AXIS_SETS`` rejects PP+TP and PP+CP at config
    time; this keeps the shortcut honest locally if that allowlist ever widens.
    """
    unhandled = [name for name, size in (("tp", config.tp_size), ("cp", config.cp_size)) if size > 1]
    if unhandled:
        raise NotImplementedError(
            f"Pipeline-parallel checkpoint {phase} does not handle {'+'.join(unhandled).upper()} "
            f"sharding: the stage shards carry COMPLETE tensors under global names, and only FSDP2's "
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


def save_pp_checkpoint(ctx: CheckpointContext, output_dir: str) -> None:
    """Pipeline parallelism: one safetensors shard per stage under GLOBAL names + a merged index.

    ``ctx.model`` is this rank's ``PipelineStageModule``; its ``global_parameter_name`` maps stage-local
    FQNs back to the unsplit model's names, so the merged checkpoint loads via plain ``from_pretrained``.
    EP MoE layers export through each family's ``gather_ep_layer_weights`` instead of their
    internal-shard state_dict entries. Every stage rank enters the DTensor and EP gathers; the writers
    (see :func:`is_pp_shard_writer`) retain the result and write their stage's shard, and the
    index/config/tokenizer follow a world-wide exchange of per-stage key maps.

    The index carries standard HF metadata only, deliberately NOT a repo "format" marker: that means
    per-rank PARTIAL tensors (ep_sharded), while these shards hold complete tensors.

    Only FSDP2's dp DTensors are un-sharded here, which is why :func:`reject_unhandled_pp_axes`
    guards the axes that would need a second inverse.
    """
    config = ctx.parallelism_config
    stage = ctx.model
    reject_unhandled_pp_axes(config, "save")
    # HF evaluates immediately before the end-of-training save, and a forward-only drive leaves
    # the stage's FSDP2 modules holding their transient UNSHARDED params: ``state_dict()`` would
    # then hand the walk below plain full tensors on every rank of the stage instead of the dp
    # DTensors it gathers, so the gathers this save is built on would not run. Per-rank, and a
    # no-op when already sharded. The optimizer half of the checkpoint does the same.
    reshard_fsdp2_modules(stage)
    shared_fs = is_output_shared_filesystem()
    is_writer = is_pp_shard_writer(config, shared_fs)

    # One prefix per stage, chosen without coordinating with the other writers: a global HF
    # "k-of-n" counter is not derivable locally once a stage emits a variable number of parts.
    writer = StageShardWriter(
        output_dir,
        f"model-pp{config.pp_rank:05d}-of-{config.pp_size:05d}",
        ctx.max_shard_size,
        enabled=is_writer,
    )
    # The writer's disk writes are interleaved with the gathers below, so one must not raise here:
    # it would strand every other stage rank in the next collective until the watchdog fires.
    guard = DeferredRankFailure(f"PP checkpoint write to {output_dir}")
    # The map the resume path reads by, so the two cannot diverge. Walked on EVERY rank: the
    # gathers below are collective, and the writer is a no-op on non-writers.
    name_map = stage.checkpoint_name_map()
    # Same artifact contract as every other writer: save-dtype cast with the norm / balancing /
    # fp32-pin keep-sets held at trained dtype (nothing cast in a training checkpoint). Keyed by the
    # live (stage-local / gather) spelling, which is what the caster's tree-derived keep-sets use.
    cast = save_dtype_caster(stage, keep_live_dtype=ctx.training_checkpoint)
    state_dict = stage.state_dict()
    for key, local_name in name_map.items():
        # Collective on every stage rank; only the writer pays the host copy, as in the EP gather below.
        for full in resolve_retained(((local_name, state_dict[local_name]),), retain=is_writer).values():
            guard.run(partial(writer.add, key, cast(local_name, full)))
    del state_dict
    for layer_name, module in stage.ep_moe_layers():
        # Collective on every stage rank; only the writer retains (and pays the host copy). The
        # per-layer loop is the flush unit that bounds the writer's peak: one gathered layer,
        # not the whole stage.
        ep_weights = gather_ep_layer_weights(layer_name, module, merge_lora=False, retain=is_writer)
        for key, tensor in ep_weights.items():
            guard.run(partial(writer.add, stage.global_parameter_name(key), cast(key, tensor)))
        del ep_weights
    if is_writer:
        # GptOss FA2 sets sinks to None (leaves state_dict); re-emit neutralized. The helper names
        # them by position in the SLICED layer list, so raw names would collide between stages.
        for sink_name, sink_tensor in neutralized_gpt_oss_sinks(stage).items():
            global_sink = stage.global_parameter_name(sink_name)
            if global_sink not in name_map:
                guard.run(partial(writer.add, global_sink, cast(sink_name, sink_tensor)))

    # Closing writes the last part; it MUST precede the exchange below, which is the sync
    # guaranteeing the index never names a file that is not on disk yet.
    stage_weight_map, shard_bytes = guard.run(writer.close) or ({}, 0)

    # The tensors no stage holds (a multimodal wrapper's vision tower), untouched by training,
    # already at the artifact's save dtype, re-emitted from the save rank under their own part
    # name — the checkpoint then reloads and serves as the wrapper class, and a resume finds
    # every tensor it plans for. Per-node storage has one save rank per node, so every node's
    # directory carries them.
    if ctx.is_save_rank and ctx.pp_wrapper_state:
        wrapper_writer = StageShardWriter(output_dir, "model-wrapper", ctx.max_shard_size, enabled=True)
        for key, tensor in ctx.pp_wrapper_state.items():
            guard.run(partial(wrapper_writer.add, key, tensor))
        wrapper_map, wrapper_bytes = guard.run(wrapper_writer.close) or ({}, 0)
        stage_weight_map = {**stage_weight_map, **wrapper_map}
        shard_bytes += wrapper_bytes
    # Collective. A failed write stops every rank here rather than letting the exchange below
    # build an index over a shard that is missing or truncated on one stage.
    guard.reject()

    # World-wide exchange (non-writers contribute nothing) so rank 0 can index keys it does not
    # hold. A global parameter lives on exactly one stage, so the merge's collision check is a
    # real gate here; it raises on every rank, which is what keeps a detected collision from
    # stranding the peers in the barrier below.
    weight_map, total_size = exchange_shard_index(stage_weight_map, shard_bytes, contribute=is_writer)

    # Shared FS → rank 0 alone writes the index; per-node storage → one index per node.
    with barrier_on_exit():
        if ctx.is_save_rank:
            # The stage shards are already on disk (every writer closed above) and `weight_map` is the
            # merged map, so the sweep inside deletes exactly what no stage claimed.
            write_merged_index(output_dir, weight_map, {"total_size": total_size})
            save_model_config(stage, output_dir)
            if ctx.tokenizer is not None:
                ctx.tokenizer.save_pretrained(output_dir)
            logger.info(f"Saved PP model to {output_dir} ({len(weight_map)} keys, {config.pp_size} stage shards)")
            if not shared_fs:
                # The index is deliberately the GLOBAL key map, so resume works on every node.
                nodes_per_stage = config.stage_world_size // config.gpus_per_node
                logger.warning(
                    "Non-shared output filesystem: writing one copy of stage %d's shard per node "
                    "(%d nodes in this stage, so %dx duplication of this stage's bytes across the "
                    "job). Each node's directory resumes on its own; to export, gather every node's "
                    "directory into one before from_pretrained.",
                    config.pp_rank,
                    nodes_per_stage,
                    nodes_per_stage,
                )
