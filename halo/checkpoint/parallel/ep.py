"""EP checkpoint saves: experts gathered into one HF checkpoint, or per-rank shards ``merge_ep_shards.py`` merges."""

import gc
import itertools
import logging
import os
from collections.abc import Callable, Iterable, Iterator
from functools import partial

import torch
from accelerate.utils import is_peft_model
from transformers.core_model_loading import rename_source_key

from halo.checkpoint.atomic import save_safetensors_file
from halo.checkpoint.config_export import save_model_config
from halo.checkpoint.format import EP_SHARD_KEY_INFIX, ep_shard_filename, ep_sharded_index_metadata
from halo.checkpoint.hub_layout import export_conversions, reversed_export_transforms, revert_conversions
from halo.checkpoint.parallel.context import CheckpointContext
from halo.checkpoint.parallel.writing import (
    WriterStripe,
    gather_hand_sliced_params,
    resolve_writer_stripe,
    stream_gathered_checkpoint,
)
from halo.checkpoint.writing import (
    merge_shard_index,
    reconcile_tie_word_embeddings,
    save_dtype_caster,
    write_merged_index,
)
from halo.distributed.context_parallel.ulysses_attention import UlyssesAttention
from halo.distributed.expert_parallel.moe_layer.expert_lora import model_has_expert_lora
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.lora_fold import lora_folded_data
from halo.models.attention.gpt_oss_sinks import neutralized_gpt_oss_sinks
from halo.models.external_tables import named_external_tables
from halo.models.moe.family_specs import find_moe_family_spec, moe_model_types
from halo.models.param_roles import persistent_buffers
from halo.models.peft_adapters import LoraFolds, lora_fold_targets
from halo.models.unwrapping import (
    base_transformers_model,
    plain_peft_param_name,
    strip_cp_attention_level,
    unwrap_toolkit_wrappers,
)
from halo.runtime.collectives import combine_across_ranks
from halo.runtime.consensus import DeferredRankFailure, barrier_on_exit
from halo.runtime.dtensor import gather_full_tensors
from halo.runtime.filesystem import fs_aware_makedirs, is_output_shared_filesystem
from halo.runtime.ranks import get_global_rank, get_global_world_size, get_local_world_size, is_global_main_process

logger = logging.getLogger(__name__)


def _save_key_remap(*, cp_key_remap: bool, peft_prefix: str | None) -> Callable[[str], str | None]:
    """Live name → checkpoint name, or ``None`` for a dropped tensor; name-based, as the drop gates a collective."""

    def remap(key: str) -> str | None:
        if peft_prefix is not None:
            normalized = plain_peft_param_name(key, peft_prefix)
            if normalized is None:
                return None
            key = normalized
        return strip_cp_attention_level(key) if cp_key_remap else key

    return remap


def _resolve_and_cast(
    items: Iterable[tuple[str, torch.Tensor]], cast: Callable[[str, torch.Tensor], torch.Tensor], *, retain: bool
) -> dict[str, torch.Tensor]:
    """COLLECTIVE: :func:`gather_full_tensors`, casting each as it lands, so the retaining rank never holds the
    replicated half at an fp32 master's width."""
    return {key: cast(key, t) for item in items for key, t in gather_full_tensors([item], retain=retain).items()}


def _ep_gathered_chunks(
    model: torch.nn.Module,
    ep_layers: list[tuple[str, torch.nn.Module]],
    remap: Callable[[str], str | None],
    lora_folds: LoraFolds | None,
    cast: Callable[[str, torch.Tensor], torch.Tensor],
    *,
    stripe: WriterStripe,
    merge_lora: bool,
) -> Iterator[dict[str, torch.Tensor]]:
    """COLLECTIVE: the non-expert tensors, then one chunk per EP layer's gather, live-named. Only the stripe's owner of
    a chunk keeps its host copies (every rank holds the layer after its all-gather); the walk is name-based, so
    dropping an adapter skips its collective on every rank alike."""
    retain = stripe.owns(0)
    # Anchored prefixes: an unanchored substring test could over-match a sibling param and drop it.
    ep_prefixes = tuple(f"{path}." for path, _ in ep_layers)
    # One tensor at a time; a folded DTensor's delta is a collective every rank enters.
    params = (
        (name, lora_folded_data(param, lora_folds))
        for name, param in model.named_parameters()
        if not name.startswith(ep_prefixes) and remap(name) is not None
    )
    state = _resolve_and_cast(params, cast, retain=retain)
    # A modules_to_save copy duplicates its module's buffers, and the frozen `original_module` side must not be saved.
    buffers = (
        (name, buf) for name, buf in persistent_buffers(model, exclude_prefixes=ep_prefixes) if remap(name) is not None
    )
    for name, buf in gather_full_tensors(buffers, retain=retain).items():
        state.setdefault(name, buf)
    # EP+TP: attention sinks are TP-sharded as plain tensors, needing this explicit mesh gather.
    state.update(gather_hand_sliced_params(model, retain=retain))
    if retain:
        # GptOss FA2 fine-tuning sets self_attn.sinks=None; the checkpoint still needs them.
        for name, sink in neutralized_gpt_oss_sinks(model).items():
            state.setdefault(name, sink)
    yield state
    del state
    for chunk, (path, layer) in enumerate(ep_layers, start=1):
        yield layer.gather_layer_state_dict(path, merge_lora=merge_lora, retain=stripe.owns(chunk))


def _ep_chunk_normalizer(
    model: torch.nn.Module,
    remap: Callable[[str], str | None],
    cast: Callable[[str, torch.Tensor], torch.Tensor],
) -> Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Per-chunk ``normalize``: save-dtype cast on the live name (a no-op on tensors the walk cast), LoRA/CP respell,
    hub revert. A ``None`` from ``remap`` is a folded adapter or an ``original_module`` half, never an expert
    weight."""
    export = partial(revert_conversions, model, conversions=export_conversions(model))

    def normalize(chunk: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return export({key: cast(live, tensor) for live, tensor in chunk.items() if (key := remap(live)) is not None})

    return normalize


def save_ep_gathered_checkpoint(ctx: CheckpointContext, output_dir: str) -> None:
    """COLLECTIVE: EP / EP+TP / EP+CP / EP+ETP, every expert gathered into one HF checkpoint, streamed a layer at a
    time. A merged save folds expert deltas inside each family's gather and attention adapters out of place as written.
    The CP wrapper is unwrapped first, since its ``named_modules()`` paths would classify every expert param as
    non-EP."""
    model = unwrap_toolkit_wrappers(ctx.model)
    peft_prefix = lora_folds = None
    if is_peft_model(model):
        peft_prefix = model.prefix
        # Structural, so a layer the fold cannot reproduce raises on every rank before any gather.
        lora_folds = lora_fold_targets(model)
        # A PeftModel wraps outside the CP wrapper and unwrap_toolkit_wrappers stops at it, so under EP+CP every key
        # would carry an extra `model.` the CP remap does not strip; peft_prefix normalizes the LoRA infixes.
        model = base_transformers_model(model)
    fs_aware_makedirs(output_dir)
    ep_layers = named_ep_layers(model)
    if ctx.is_save_rank and is_global_main_process():
        logger.info(f"Saving EP model to {output_dir} (gathered mode)")
        logger.info(f"Found {len(ep_layers)} EP layers" if ep_layers else "No EP layers found, saving resolved params")

    remap = _save_key_remap(cp_key_remap=ctx.parallelism_config.is_cp_mode, peft_prefix=peft_prefix)
    cast = save_dtype_caster(model, keep_live_dtype=ctx.training_checkpoint)
    stripe = resolve_writer_stripe(is_save_rank=ctx.is_save_rank)
    chunks = _ep_gathered_chunks(
        model, ep_layers, remap, lora_folds, cast, stripe=stripe, merge_lora=ctx.merges_expert_lora
    )
    written, total_bytes = stream_gathered_checkpoint(
        model,
        chunks,
        output_dir,
        is_save_rank=ctx.is_save_rank,
        max_shard_size=ctx.max_shard_size,
        normalize=_ep_chunk_normalizer(model, remap, cast),
        stripe=stripe,
    )
    if ctx.is_save_rank and is_global_main_process():
        logger.info(f"✓ Model saved to {output_dir} ({written} parameters, {total_bytes / 1e9:.2f}GB streamed)")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _reject_unmergeable_hub_keys(model: torch.nn.Module) -> None:
    """Refuse a family or load whose conversions the merge's key-by-key stream cannot revert to the hub layout."""
    model_type = getattr(getattr(model, "config", None), "model_type", "") or ""
    spec = find_moe_family_spec(model_type)
    if spec is None:
        raise ValueError(
            f"Sharded EP save (save_sharded_ep=True) is not supported for model_type "
            f"{model_type!r}: no MoE family spec claims it, so "
            f"scripts/after_training/merge_ep_shards.py cannot resolve an HF-layout transform and the "
            f"per-rank shards could never be merged into a loadable checkpoint. Supported model types: "
            f"{', '.join(sorted(moe_model_types()))}. Use save_sharded_ep=False (gathered) instead."
        )
    # A renamed namespace (SigLIP's ``vision_model`` level, a vendor namespace) or a tensor converted outside the
    # per-expert merges would merge into a layout no hub or engine uses. Name-only, so rank-uniform.
    base = base_transformers_model(model)
    renamings, converters = reversed_export_transforms(export_conversions(base))
    stuck = []
    if renamings or converters:
        for name, _ in itertools.chain(base.named_parameters(), persistent_buffers(base)):
            merged = spec.to_hub_key(name)
            hub, converted_by = rename_source_key(merged, renamings, converters, reverse=True)
            if converted_by is not None or hub != merged:
                stuck.append(merged)
    if stuck:
        raise ValueError(
            f"Sharded EP save (save_sharded_ep=True) is not supported for this {model_type!r} model: its "
            f"load converted {len(stuck)} tensor name(s) from the hub namespace (first few: {stuck[:4]}), "
            f"which the gathered save reverts through transformers' save-side conversion but "
            f"merge_ep_shards.py's key-by-key stream cannot — the merged checkpoint would carry the "
            f"module-tree spelling the hub and the serving engines do not use. Use "
            f"save_sharded_ep=False (gathered) instead."
        )


def _reject_unmergeable_ep_topology(ep_cfg, world_size: int) -> None:
    """Refuse expert TP, several EP groups, or shards scattered over nodes' local disks."""
    expert_tp_size = ep_cfg.expert_tp_size
    if expert_tp_size > 1:
        raise ValueError(
            f"Sharded EP save (save_sharded_ep=True) is not supported with "
            f"expert_tensor_parallel_size={expert_tp_size}. Expert TP shards expert FFN weights "
            f"across ranks, which the merge script cannot reconstruct. "
            f"Use save_sharded_ep=False (default) for gathered save instead."
        )

    ep_group_size = ep_cfg.ep_group_size
    if ep_group_size != world_size:
        raise ValueError(
            f"Sharded EP save (save_sharded_ep=True) requires a single EP group spanning all "
            f"ranks (ep_group_size == world_size), got ep_group_size={ep_group_size}, "
            f"world_size={world_size} ({ep_cfg.num_ep_groups} EP groups / DP "
            f"replicas). The per-rank shards would merge with duplicated experts. "
            f"Use save_sharded_ep=False (gathered) instead."
        )

    num_nodes = world_size // max(get_local_world_size(), 1)
    if not is_output_shared_filesystem() and num_nodes > 1:
        raise ValueError(
            f"Sharded EP save (save_sharded_ep=True) requires a shared output filesystem on a "
            f"multi-node job (non-shared output FS with {num_nodes} nodes; see "
            f"DIST_OUTPUT_SHARED_FILESYSTEM / DIST_SHARED_FILESYSTEM). Per-rank expert shards are "
            f"keyed by global rank and must all be co-located for merge_ep_shards.py, but a "
            f"non-shared filesystem scatters them across nodes' local disks with no gather path. "
            f"Use save_sharded_ep=False (gathered) — it writes a complete checkpoint on each node — "
            f"or run on a shared filesystem."
        )


def reject_unmergeable_ep_sharded_save(
    model: torch.nn.Module, world_size: int | None = None, merge_expert_lora_on_save: bool = False
) -> None:
    """Refuse a ``save_sharded_ep=True`` run whose per-rank shards ``merge_ep_shards.py`` could not merge; runs at
    construction and again on save. The merge streams key by key with no model behind it, applying only the family's
    ``export_key_renames``, and needs every shard co-located."""
    if merge_expert_lora_on_save:
        raise ValueError(
            "merge_expert_lora_on_save=True is incompatible with save_sharded_ep=True: "
            "merge_lora is only supported with gathered EP save (sharded=False). "
            "Use save_sharded_ep=False, or disable merge_expert_lora_on_save."
        )
    ep_layers = named_ep_layers(model)
    if not ep_layers:
        # Accepting the flag would write gathered checkpoints while a planned merge waits for shards that never exist.
        raise ValueError(
            "save_sharded_ep=True, but this run has no EP MoE layers (dense model, or a MoE without "
            "EP layers — e.g. accelerate launch with use_grouped_gemm=false): every save would be "
            "an ordinary gathered checkpoint, not the sharded format this flag names. Remove "
            "save_sharded_ep, or run an EP mode."
        )
    if any(isinstance(m, UlyssesAttention) for m in model.modules()):
        raise ValueError(
            "Sharded EP save (save_sharded_ep=True) is not supported with Context Parallelism: "
            "the per-rank shards would carry CP-patched '.original_attention.' keys that "
            "merge_ep_shards.py cannot remap to the HF layout. Use save_sharded_ep=False "
            "(gathered) — it remaps CP attention keys on save."
        )

    if model_has_expert_lora(model):
        raise ValueError(
            "Sharded EP save (save_sharded_ep=True) is not supported with native expert LoRA: the "
            "shard key set comes from expert_named_params(), so the grouped '<attr>_lora_A/_lora_B' "
            "adapters are written as '.shard_N' keys that merge_ep_shards.py's base-root pattern "
            "never matches — they pass through dead while the merged experts are the FROZEN base. "
            "Use save_sharded_ep=False (gathered), which folds the adapters via "
            "merge_expert_lora_on_save or writes them alongside the base."
        )
    tables = [name for name, _ in named_external_tables(model)]
    if tables:
        raise ValueError(
            f"Sharded EP save (save_sharded_ep=True) is not supported for a model holding external tables "
            f"({', '.join(tables[:2])}): merge_ep_shards.py merges the per-rank shards key by key with no model "
            f"to read a table from, so the merged checkpoint would lack it. Use save_sharded_ep=False "
            f"(gathered), which writes the tables."
        )
    _reject_unmergeable_hub_keys(model)
    # Attribute access, since on anything but a finalized EPConfig a default would pass the guards and the sharded save
    # would merge to duplicated or missing experts.
    _reject_unmergeable_ep_topology(
        ep_layers[0][1].ep_config, get_global_world_size() if world_size is None else world_size
    )


def _rank_shard(model: torch.nn.Module, ep_layers, rank: int, *, keep_live_dtype: bool) -> dict[str, torch.Tensor]:
    """COLLECTIVE: this rank's experts under a ``.shard_<rank>`` suffix and, on rank 0, every replicated tensor; EP+TP
    DTensors resolve on every rank. Rank 0 and not the FS-aware save rank, since a non-shared multi-node save is
    refused."""
    cast = save_dtype_caster(model, keep_live_dtype=keep_live_dtype)
    # From ``expert_named_params()``, since a substring match misclassifies whole families.
    expert_names = {f"{path}.{name}" for path, layer in ep_layers for name, _ in layer.expert_named_params()}
    shard = {
        f"{name}{EP_SHARD_KEY_INFIX}{rank}": cast(name, param.data.cpu())
        for name, param in model.named_parameters()
        if name in expert_names
    }
    replicated = ((name, param.data) for name, param in model.named_parameters() if name not in expert_names)
    shard.update(_resolve_and_cast(replicated, cast, retain=rank == 0))
    # The walk above wrote TP-sharded plain tensors (GptOss sinks) as rank 0's head slice under the
    # full-tensor key, a truncated sinks the merge cannot complete.
    for name, full_tensor in gather_hand_sliced_params(model, retain=rank == 0).items():
        shard[name] = cast(name, full_tensor)
    if rank == 0:
        # Persistent buffers only, so rotary caches stay out; the sinks GptOss FA2 dropped, neutralized.
        buffers = ((name, buf.detach().cpu()) for name, buf in persistent_buffers(model))
        for name, tensor in itertools.chain(buffers, neutralized_gpt_oss_sinks(model).items()):
            if name not in shard:
                shard[name] = cast(name, tensor)
    return shard


def save_ep_sharded_checkpoint(ctx: CheckpointContext, output_dir: str) -> None:
    """COLLECTIVE: per-rank expert shards plus a weight-map index, made loadable by ``merge_ep_shards.py``, whose
    shards are the ones ``max_shard_size`` bounds."""
    model = unwrap_toolkit_wrappers(ctx.model)
    fs_aware_makedirs(output_dir)
    rank, world_size = get_global_rank(), get_global_world_size()
    ep_layers = named_ep_layers(model)
    if ep_layers:
        reject_unmergeable_ep_sharded_save(model, world_size, ctx.parallelism_config.merge_expert_lora_on_save)

    if rank == 0:
        logger.info(f"[Rank {rank}] Saving EP model sharded to {output_dir}")
        logger.warning(
            "Sharded EP save is NOT directly loadable — run "
            "scripts/after_training/merge_ep_shards.py to produce a HuggingFace "
            "checkpoint before load_ep_model()/from_pretrained(). Choose this mode for write "
            "throughput (every rank writes its own shard instead of funnelling the model through "
            "one); the gathered save streams, so host memory is not a reason to prefer it."
        )
        logger.info(
            f"save_max_shard_size ({ctx.max_shard_size}) does not bound these shards — a per-rank EP save "
            f"writes one file per rank by design. It bounds the shards merge_ep_shards.py writes."
        )

    shard = _rank_shard(model, ep_layers, rank, keep_live_dtype=ctx.training_checkpoint)
    # Counted off what is written: an export's cast folds fp32 masters to bf16.
    total_size = sum(t.nbytes for t in shard.values())
    shard_filename = ep_shard_filename(rank, world_size)
    # Every rank writes, so a local raise here would leave the peers in the collectives below.
    guard = DeferredRankFailure(f"EP sharded checkpoint write to {output_dir}")
    if shard:
        guard.run(partial(save_safetensors_file, shard, os.path.join(output_dir, shard_filename)))
        if guard.reason is None and is_global_main_process():
            logger.info(
                f"[Rank {rank}] Saved shard: {shard_filename} ({len(shard)} tensors, {total_size / 1e9:.2f}GB)"
            )
    # Over the store: every rank writes at once, so the slowest shard waits on shared storage.
    guard.store_reject("checkpoint/ep_shards")

    # Key-disjoint by construction (per-rank expert suffixes, rank-0-only replicated keys).
    merged_map, merged_bytes = combine_across_ranks(
        (dict.fromkeys(shard, shard_filename), total_size), merge_shard_index, what="the checkpoint index"
    )
    with barrier_on_exit(f"Writing the EP sharded checkpoint index to {output_dir}"):
        if rank == 0:
            # These shards share HF's model-{i}-of-{n} namespace, so the merge's regex cannot tell a leftover gathered
            # save from a rank shard unless this sweep removes it.
            write_merged_index(output_dir, merged_map, ep_sharded_index_metadata(merged_bytes, world_size))
            # A tie broken in training must clear tie_word_embeddings, else the merge drops the lm_head.
            reconcile_tie_word_embeddings(model, shard)
            save_model_config(model, output_dir)
            logger.info(f"[Rank 0] Sharded save complete: {world_size} shards, {merged_bytes / 1e9:.2f}GB")
            logger.info("[Rank 0] To load: use merge_ep_shards.py first, then load_ep_model()")
    gc.collect()
