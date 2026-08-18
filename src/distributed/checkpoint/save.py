"""Model checkpoint saving: the mode ladder and the per-mode savers it dispatches to.

:func:`save_checkpoint` is the entry point every trainer calls. :func:`select_checkpoint_saver` walks
an ordered ladder — PP first (a stage is a partial model, so every other saver would write it as if
complete), then EP before CP (so EP+CP gathers experts), CP before TP, TP before plain FSDP2 — and
returns ``None`` when no mode handles the save, which tells the trainer to fall through to
``Trainer.save_model``. Every predicate is rank-uniform, so all ranks pick the same saver. PEFT
adapters are dispatched separately via :class:`~src.distributed.checkpoint.peft.PeftAdapterSaver`.

FSDP2, CP and TP differ only in where their chunks come from and all stream through
:func:`~src.distributed.checkpoint.write.stream_gathered_checkpoint`. EP and PP write different
artifacts: a family-specific expert gather, and one shard per stage under global names.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from functools import partial

import torch
import torch.nn as nn

from src.checkpoint.adapters import EXPERT_LORA_PEFT_TYPE
from src.checkpoint.config_export import save_model_config
from src.checkpoint.format import save_dtype_caster, write_merged_index
from src.checkpoint.shard_writer import StageShardWriter
from src.distributed.checkpoint.context import CheckpointContext
from src.distributed.checkpoint.peft import expert_lora_config_fields, find_peft_model
from src.distributed.checkpoint.write import (
    chunked_saveable_tensors,
    exchange_shard_index,
    stream_gathered_checkpoint,
)
from src.distributed.expert_parallel.expert_weights import gather_ep_layer_weights
from src.distributed.expert_parallel.saving import save_ep_lora_adapters, save_ep_model
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.runtime import (
    DeferredRankFailure,
    barrier_on_exit,
    is_local_main_process,
    is_output_shared_filesystem,
    resolve_param_tensor,
)
from src.distributed.tensor_parallel.checkpoint import save_tp_model
from src.models.patches.gpt_oss_sinks import neutralized_gpt_oss_sinks
from src.models.structure import merged_adapters, unwrap_model

logger = logging.getLogger(__name__)

# A saver always handles the save it is given; the ladder returns ``None`` when no mode applies.
CheckpointSaver = Callable[[CheckpointContext, str], None]


def save_checkpoint(ctx: CheckpointContext, output_dir: str) -> bool:
    """Save the model for the active parallelism mode. False = fall through to ``Trainer.save_model``."""
    saver = select_checkpoint_saver(ctx)
    if saver is None:
        return False
    saver(ctx, output_dir)
    return True


def select_checkpoint_saver(ctx: CheckpointContext) -> CheckpointSaver | None:
    """The saver for the active mode, or ``None`` when no mode handles it (see module docstring)."""
    if ctx.is_pp_mode:
        return save_pp_checkpoint
    if ctx.is_ep_tp_mode or ctx.has_ep_layers:
        return save_ep_checkpoint
    if ctx.is_cp_mode:
        # The wrapper performs the CP key remap; without it there is no CP save to make, only the base one.
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
    """Shared FSDP2/CP body: stream the gathered chunks, then the tokenizer, under one fence.

    Gathered on all ranks (the chunk source's resolves are collective) but written one chunk at a
    time, so the save rank never holds the whole model in host RAM. Fenced because one rank writes
    while all must reach the trailing barrier, and an ENOSPC would otherwise leave the peers there.
    """
    with barrier_on_exit():
        stream_gathered_checkpoint(
            model, chunks, output_dir, is_save_rank=ctx.is_save_rank, max_shard_size=ctx.max_shard_size
        )
        if ctx.is_save_rank:
            if ctx.tokenizer is not None:
                ctx.tokenizer.save_pretrained(output_dir)
            logger.info(f"Saved {label} model to {output_dir}")


def save_fsdp2_checkpoint(ctx: CheckpointContext, output_dir: str) -> None:
    """Mixin-managed FSDP2 (torchrun standard DP). Params are DTensors — gathered via full_tensor()."""
    _save_streamed(ctx, output_dir, ctx.model, chunked_saveable_tensors(ctx.model, retain=ctx.is_save_rank), "FSDP2")


def save_cp_checkpoint(ctx: CheckpointContext, output_dir: str) -> None:
    """CP-only (no EP). The CP wrapper's ``state_dict()`` filters and remaps attention keys.

    That dict is the item source rather than the module walk, since it already carries the persistent
    buffers under the remapped keys; it holds references only, and the streamed gather resolves them
    chunk by chunk. The sinks still come off the unwrapped model.
    """
    cp_wrapper = ctx.cp_wrapper
    inner = unwrap_model(cp_wrapper)
    items = cp_wrapper.state_dict().items()
    _save_streamed(ctx, output_dir, inner, chunked_saveable_tensors(inner, retain=ctx.is_save_rank, items=items), "CP")


def save_tp_checkpoint(ctx: CheckpointContext, output_dir: str) -> None:
    """TP-only / TP+DP. ``save_tp_model`` runs the second gather its hand-sliced params need."""
    save_tp_model(ctx.model, output_dir, tokenizer=ctx.tokenizer, max_shard_size=ctx.max_shard_size)


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
    ``merge_expert_lora_on_save`` requests a merged checkpoint. That merge covers both halves of a
    mixed run: the expert deltas fold inside each family's gather, and any attention adapters fold
    into their base weights for the duration of the write.
    """
    if ctx.has_expert_lora and not ctx.merge_expert_lora_on_save:
        save_ep_lora_adapters(
            ctx.model, output_dir, adapter_config=_expert_lora_adapter_config(ctx), tokenizer=ctx.tokenizer
        )
        return
    # Collective and rank-uniform: merge_adapter is an in-place DTensor op under FSDP2, and the
    # unmerge on exit leaves the adapters trainable after an intermediate merged save.
    with merged_adapters(find_peft_model(ctx.model)) as adapters_merged:
        save_ep_model(
            ctx.model,
            output_dir,
            tokenizer=ctx.tokenizer,
            sharded=ctx.save_sharded_ep,
            cp_key_remap=ctx.is_cp_mode,
            max_shard_size=ctx.max_shard_size,
            merge_lora=ctx.has_expert_lora and ctx.merge_expert_lora_on_save,
            adapters_merged=adapters_merged,
        )


def reject_unhandled_pp_axes(config, phase: str) -> None:
    """Reject a PP checkpoint ``phase`` ("save"/"resume") combined with an axis it cannot express.

    The PP shard writer/reader un-shards one thing: FSDP2's dp DTensors. TP shards the planned
    projections along a second mesh dimension (2-D ``(dp, tp)`` DTensors) and CP renames every
    projection one level deeper, so either needs a second inverse. ``SUPPORTED_AXIS_SETS`` rejects
    PP+TP and PP+CP at config time; this is the local check should that allowlist widen.
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
    node, so each node holds a directory it can resume from. The per-node widening is valid only
    because a node never straddles a stage (stage world size is a multiple of ``gpus_per_node``).
    """
    return config.stage_local_rank == 0 if shared_fs else is_local_main_process()


def save_pp_checkpoint(ctx: CheckpointContext, output_dir: str) -> None:
    """Pipeline parallelism: one safetensors shard per stage under global names, plus a merged index.

    ``ctx.model`` is this rank's ``PipelineStageModule``; its ``global_parameter_name`` maps stage-local
    FQNs back to the unsplit model's names, so the merged checkpoint loads via plain ``from_pretrained``.
    EP MoE layers export through each family's ``gather_ep_layer_weights`` instead of their
    internal-shard state_dict entries. Every stage rank enters the DTensor and EP gathers; the writers
    (see :func:`is_pp_shard_writer`) retain the result and write their stage's shard, and the
    index/config/tokenizer follow a world-wide exchange of per-stage key maps.

    The index carries standard HF metadata only and no repo "format" marker, which would mean
    per-rank partial tensors (ep_sharded) rather than the complete tensors these shards hold.

    Only FSDP2's dp DTensors are un-sharded here, which is what :func:`reject_unhandled_pp_axes`
    guards.
    """
    config = ctx.parallelism_config
    stage = ctx.model
    reject_unhandled_pp_axes(config, "save")
    # HF evaluates immediately before the end-of-training save, and a forward-only drive leaves the
    # stage's FSDP2 modules holding transient unsharded params: ``state_dict()`` would then hand the
    # walk below plain full tensors instead of the dp DTensors it gathers, so the gathers this save
    # is built on would not run. Per-rank, and a no-op when already sharded; the optimizer half of
    # the checkpoint does the same.
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
    # The writer's disk writes are interleaved with the gathers below, so a raise here would leave
    # every other stage rank in the next collective until the watchdog fires.
    guard = DeferredRankFailure(f"PP checkpoint write to {output_dir}")
    # The map the resume path reads by, so the two cannot diverge. Walked on every rank, since the
    # gathers below are collective and the writer is a no-op on non-writers.
    name_map = stage.checkpoint_name_map()
    # Same artifact contract as every other writer: save-dtype cast with the norm / balancing /
    # fp32-pin keep-sets held at trained dtype. Keyed by the live (stage-local / gather) spelling,
    # which is what the caster's tree-derived keep-sets use.
    cast = save_dtype_caster(stage)
    state_dict = stage.state_dict()
    for key, local_name in name_map.items():
        full = resolve_param_tensor(state_dict[local_name])  # collective: every stage rank participates
        guard.run(partial(writer.add, key, cast(local_name, full)))
        del full
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
        # them by position in the sliced layer list, so raw names would collide between stages.
        for sink_name, sink_tensor in neutralized_gpt_oss_sinks(stage).items():
            global_sink = stage.global_parameter_name(sink_name)
            if global_sink not in name_map:
                guard.run(partial(writer.add, global_sink, cast(sink_name, sink_tensor)))

    # Closing writes the last part, and must precede the exchange below, the sync that keeps the
    # index from naming a file not yet on disk.
    stage_weight_map, shard_bytes = guard.run(writer.close) or ({}, 0)

    # Tensors no stage holds (a multimodal wrapper's vision tower) are untouched by training and
    # already at the artifact's save dtype; the save rank re-emits them under their own part name so
    # the checkpoint reloads and serves as the wrapper class and a resume finds every tensor it plans
    # for. Per-node storage has one save rank per node, so every node's directory carries them.
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
    # hold. A global parameter lives on exactly one stage, so the merge's collision check is a real
    # gate here; it raises on every rank, keeping a collision from leaving peers in the barrier below.
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
                # The index is the global key map, so resume works on every node.
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
