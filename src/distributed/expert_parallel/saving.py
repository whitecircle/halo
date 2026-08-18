"""Save EP-distributed MoE models to HuggingFace-compatible checkpoints.

:func:`save_ep_model` gathers each EP layer's distributed experts (via :func:`gather_ep_layer_weights`),
collects replicated non-expert params, and writes a single safetensors file (gathered) or per-rank
shards.
"""

import gc
import json
import os
from collections.abc import Callable
from functools import partial

import torch
from accelerate.logging import get_logger
from accelerate.utils import is_peft_model
from safetensors.torch import save_file

from src.checkpoint.config_export import save_model_config
from src.checkpoint.format import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_SAFETENSORS_FILE,
    DEFAULT_MAX_SHARD_SIZE,
    EP_SHARD_KEY_INFIX,
    EP_SHARDED_FORMAT,
    HF_STREAM_PART_PREFIX,
    cast_state_dict_to_save_dtype,
    ep_shard_filename,
    reconcile_tie_word_embeddings,
    revert_load_conversions,
    save_dtype_caster,
    write_merged_index,
)
from src.checkpoint.shard_writer import StageShardWriter
from src.distributed.checkpoint.write import exchange_shard_index, resolve_retained
from src.distributed.context_parallel.base_layer import UlyssesAttentionBase
from src.distributed.context_parallel.key_mapping import strip_cp_attention_prefix
from src.distributed.expert_parallel.base_layer import find_ep_layers
from src.distributed.expert_parallel.expert_weights import (
    gather_ep_layer_weights,
    gather_ep_lora_adapters,
    has_ep_lora,
    resolve_ep_merge_layer_class,
    supported_ep_merge_model_types,
)
from src.distributed.runtime import (
    DeferredRankFailure,
    barrier_on_exit,
    fs_aware_makedirs,
    fs_aware_save_rank,
    get_global_rank,
    get_global_world_size,
    get_local_world_size,
    is_output_shared_filesystem,
    materialize_dtensor,
    resolve_param_tensor,
)
from src.distributed.tensor_parallel.state_dict import gather_tp_sharded_non_dtensor_params
from src.models.patches.gpt_oss_sinks import neutralized_gpt_oss_sinks
from src.models.structure import normalize_peft_param_name, persistent_buffers, unwrap_model

logger = get_logger(__name__)


def _save_key_remap(*, cp_key_remap: bool, peft_prefix: str | None) -> Callable[[str], str | None]:
    """Live parameter name → the name the HF checkpoint uses, or ``None`` to drop the tensor.

    Composes the CP attention-prefix strip with the PEFT base-name normalization, so a checkpoint
    written off a CP-wrapped and/or PEFT-wrapped module tree still loads with ``from_pretrained``.
    ``peft_prefix`` is set only on the merged-adapter save path: the caller holds
    :func:`~src.models.structure.merged_adapters` open, so every adapter param's delta
    is already inside its base weight and the adapter tensors themselves are dropped.

    Name-based and therefore rank-uniform — the drop decision gates a ``full_tensor()`` collective.
    """

    def remap(key: str) -> str | None:
        if peft_prefix is not None:
            normalized = normalize_peft_param_name(key, peft_prefix)
            if normalized is None:
                return None
            key = normalized
        return strip_cp_attention_prefix(key) if cp_key_remap else key

    return remap


def _persistent_non_ep_buffers(model: torch.nn.Module, ep_layer_names):
    """Persistent buffers outside EP layers (EP-layer buffers come from the expert gather instead)."""
    return persistent_buffers(model, exclude_prefixes=tuple(f"{ep_name}." for ep_name in ep_layer_names))


def _hub_namespace_export(model: torch.nn.Module, ep_layers) -> Callable[[dict], dict]:
    """The checkpoint-name transform the gathered save applies to each streamed chunk.

    Identity for every family whose module-tree spelling IS its checkpoint spelling (or has a repair
    tool). For a family declaring ``_EXPORTS_HUB_NAMESPACE`` it is transformers' own save-side
    conversion revert — the same one ``save_pretrained`` runs — so the write lands in the hub
    namespace the serving engines read (prefix renames, the ``moe.*`` respellings, the fused
    ``gate_up_proj`` split back into the hub's two tensors). Applied per chunk: the non-expert params
    as one (a reverse fusion such as a vision tower's q/k/v → in_proj needs all of its sources
    together), each gathered EP layer as its own — the declaring family guarantees no reverse entry
    straddles that boundary.
    """
    if not any(type(module)._EXPORTS_HUB_NAMESPACE for _name, module in ep_layers):
        return lambda state: state
    return partial(revert_load_conversions, model)


def _save_config_and_tokenizer(model, output_dir: str, tokenizer=None):
    """Save model config and optionally tokenizer."""
    save_model_config(model, output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
        logger.info(f"✓ Tokenizer saved to {output_dir}")


def save_ep_model(
    model: torch.nn.Module,
    output_dir: str,
    tokenizer=None,
    sharded: bool = False,
    cp_key_remap: bool = False,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
    merge_lora: bool = False,
    adapters_merged: bool = False,
):
    """Save an EP model to a HuggingFace-compatible safetensors checkpoint.

    ``model`` must be unwrapped (a :class:`UlyssesCPModelWrapper` is unwrapped here — its
    ``named_modules()`` walk yields wrapper-prefixed EP-layer paths that never prefix-match its
    inner-named parameters, silently misclassifying every expert param as non-EP). ``sharded=True``
    writes one shard/rank (requires merge_ep_shards.py before reload; ``max_shard_size`` then bounds
    the shards that merge writes, not these — a per-rank shard is one file by design);
    ``sharded=False`` (default) gathers an HF-standard checkpoint. ``cp_key_remap`` rewrites CP
    attention keys (required for EP+CP). ``merge_lora=True`` folds native grouped-LoRA deltas into
    the gathered base.

    ``adapters_merged`` asserts the caller holds
    :func:`~src.models.structure.merged_adapters` open over this call, which is what
    puts the attention deltas into the base weights being written. The PEFT key spelling is then
    derived from the live model rather than passed in, so the two can never disagree.
    """
    model = unwrap_model(model)
    peft_prefix = None
    if is_peft_model(model):
        if not adapters_merged:
            raise ValueError(
                "save_ep_model received a PeftModel without adapters_merged=True. The gathered base "
                "weights would not carry the LoRA delta, so the checkpoint would be base-quality "
                "while looking trained. Wrap the call in merged_adapters(peft_model), or route "
                "adapter-only saves through PeftAdapterSaver."
            )
        peft_prefix = model.prefix
        # A PeftModel wraps OUTSIDE the CP wrapper and unwrap_model deliberately stops at it, so
        # under EP+CP every key would carry an extra `model.` that cp_key_remap does not strip.
        # Peeling PEFT and re-unwrapping puts the walk back on the base model, where the CP remap is
        # the whole CP correction and the helpers below read the real model rather than PEFT's
        # forwarded attributes. LoRA layers stay in the tree; peft_prefix normalizes their infixes.
        model = unwrap_model(model.get_base_model())
    world_size = get_global_world_size()
    fs_aware_makedirs(output_dir)

    if sharded and world_size > 1:
        if merge_lora or peft_prefix is not None:
            raise ValueError("Adapter merging is only supported with a gathered EP save (sharded=False).")
        _save_ep_sharded(model, output_dir, tokenizer, max_shard_size=max_shard_size)
    else:
        _save_ep_gathered(
            model,
            output_dir,
            tokenizer,
            cp_key_remap=cp_key_remap,
            max_shard_size=max_shard_size,
            merge_lora=merge_lora,
            peft_prefix=peft_prefix,
        )


def _save_ep_gathered(
    model: torch.nn.Module,
    output_dir: str,
    tokenizer=None,
    cp_key_remap: bool = False,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
    merge_lora: bool = False,
    peft_prefix: str | None = None,
):
    """Save EP model by gathering expert weights, in HuggingFace-standard layout.

    The expert all-gathers are collectives every rank enters, but only the save rank retains the
    gathered tensors (``retain=`` on the layer gather, so non-savers never make the host copy).

    The gathered experts are STREAMED to disk one layer at a time rather than accumulated: a
    fine-grained MoE puts almost all of its parameters in those layers (794 of 800 GB at
    Qwen3.5-397B-A17B), so buffering the whole state dict would need the entire model resident in one
    host's RAM. The save rank instead peaks at the replicated non-expert params plus one shard's worth
    of pending experts. On non-shared filesystems each node's local rank 0 saves.
    """
    is_save_rank = fs_aware_save_rank()

    if is_save_rank:
        logger.info(f"Saving EP model to {output_dir} (gathered mode)")

    ep_layers = find_ep_layers(model)
    ep_layer_names = {name for name, _ in ep_layers}

    cast = save_dtype_caster(model)  # norm params keep trained dtype (module-tree derived)

    if is_save_rank:
        logger.info(f"Found {len(ep_layers)} EP layers" if ep_layers else "No EP layers found, saving resolved params")

    # The replicated non-expert params are held whole: they are what
    # :func:`reconcile_tie_word_embeddings` compares (both sides must be resident at once), and they
    # are the small side of a fine-grained MoE. Experts stream past them below.
    state_dict = {}

    remap = _save_key_remap(cp_key_remap=cp_key_remap, peft_prefix=peft_prefix)

    # EP+TP: non-EP DTensors need a full_tensor() collective on every rank; only the save rank retains.
    for name, param in model.named_parameters():
        # Anchored match: an unanchored substring test could over-match a sibling param and drop it.
        is_ep_param = any(name.startswith(f"{ep_name}.") for ep_name in ep_layer_names)
        if is_ep_param:
            continue
        # Rank-uniform (name-based): dropping an adapter param must skip its full_tensor() on EVERY
        # rank, or the savers and non-savers disagree on the collective schedule and the save hangs.
        if remap(name) is None:
            continue
        if is_save_rank:
            state_dict[name] = cast(name, resolve_param_tensor(param.data))
        else:
            materialize_dtensor(param.data)

    # Same filter as the params above: a modules_to_save copy duplicates its module's buffers, and
    # the frozen `original_module` side of that pair must not reach the checkpoint. Name-based, so
    # every rank drops the same ones and walks the same list — the resolve below is the shared
    # retain-gated seam (a buffer registered as a DTensor resolves through a collective every rank
    # must enter, exactly as the params do).
    buffers = (
        (name, buf) for name, buf in _persistent_non_ep_buffers(model, ep_layer_names) if remap(name) is not None
    )
    for name, buf in resolve_retained(buffers, retain=is_save_rank).items():
        state_dict.setdefault(name, cast(name, buf))

    # EP+TP: attention sinks are TP-sharded as PLAIN tensors, needing this explicit mesh gather.
    # A no-op without TP (nothing to iterate), so it is unconditional.
    gather_tp_sharded_non_dtensor_params(model, state_dict, retain=is_save_rank)

    # Live names → checkpoint names, once per chunk: the LoRA/CP normalization (``remap``) first,
    # then the family's hub-namespace revert (identity for most families). A ``None`` from ``remap``
    # is always a tensor the checkpoint must not carry, never a lost weight: a folded LoRA adapter,
    # or the FROZEN ``original_module`` half of a ``modules_to_save`` pair whose trained twin maps to
    # the same base name. Expert weights never reach either case — ``expert_weight_attrs`` excludes
    # their attrs, adapters included, from the walk.
    export = _hub_namespace_export(model, ep_layers)

    def to_checkpoint_names(chunk: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return export({hf_key: tensor for key, tensor in chunk.items() if (hf_key := remap(key)) is not None})

    if is_save_rank:
        # GptOss FA2 fine-tuning sets self_attn.sinks=None; the checkpoint still needs them.
        for sink_name, sink_tensor in neutralized_gpt_oss_sinks(model).items():
            state_dict.setdefault(sink_name, cast(sink_name, sink_tensor))
        state_dict = to_checkpoint_names(state_dict)
        # Must precede the config write below, which persists the flag this may clear. Fed the
        # CHECKPOINT key space: the probe is a `lm_head.weight` / `embed_tokens.weight` suffix match,
        # and a LoRA-wrapped or modules_to_save head lives at `lm_head.base_layer.weight` /
        # `lm_head.modules_to_save.default.weight`, which that probe misses — leaving `config.json`
        # claiming tied embeddings over a head that training pulled apart, so `from_pretrained` would
        # re-tie and discard the trained head.
        reconcile_tie_word_embeddings(model, state_dict)

    # One writer (this rank), so the parts can be renamed to HF's own names once the count is known.
    writer = StageShardWriter(output_dir, HF_STREAM_PART_PREFIX, max_shard_size, enabled=is_save_rank)
    # Streaming puts the save rank's disk writes BETWEEN the per-layer expert gathers below, so a
    # failing write must not raise here: it would strand every peer in the next layer's all-gather.
    guard = DeferredRankFailure(f"EP checkpoint write to {output_dir}")

    def stage_expert_layer(gathered: dict[str, torch.Tensor]) -> None:
        """Cast, respell and stage one gathered EP layer. Runs INSIDE the guard: a host-OOM in the
        cast strands the peers exactly as a failing write would, so both belong on the same side of
        it. The cast keys on the LIVE name (the caster's keep-sets speak that spelling); the respell
        follows."""
        chunk = to_checkpoint_names({key: cast(key, tensor) for key, tensor in gathered.items()})
        for key, tensor in chunk.items():
            writer.add(key, tensor)

    for key, tensor in state_dict.items():
        guard.run(partial(writer.add, key, tensor))
    non_expert_keys = len(state_dict)
    del state_dict

    # ``merge_lora`` folds the grouped expert-LoRA delta inside each family's gather, BEFORE any
    # unfuse/re-interleave, so the fold is layout-correct for every family.
    expert_keys = 0
    for layer_name, module in ep_layers:
        gathered = gather_ep_layer_weights(layer_name, module, merge_lora=merge_lora, retain=is_save_rank)
        if gathered:  # non-savers gather onto the device and keep nothing
            guard.run(partial(stage_expert_layer, gathered))
        expert_keys += len(gathered)
        del gathered

    # Collective. Must precede the finalize: a stream that failed anywhere stops the job here with
    # its real cause, instead of finalizing an index over tensors that were never written.
    guard.reject()

    # The finalize is the other ENOSPC-prone half — the last flush, the renames, the destructive
    # stale sweep and the index write — so it gets the same treatment.
    finalize = DeferredRankFailure(f"EP checkpoint finalize in {output_dir}")
    if is_save_rank:
        # Config/tokenizer first: the weights are only a valid checkpoint once the index names
        # them, and the index is the last thing close_as_hf_checkpoint writes.
        finalize.run(partial(_save_config_and_tokenizer, model, output_dir, tokenizer))
        total_bytes = finalize.run(writer.close_as_hf_checkpoint) or 0
        if finalize.reason is None:
            logger.info(
                f"✓ Model saved to {output_dir} "
                f"({non_expert_keys + expert_keys} parameters, {total_bytes / 1e9:.2f}GB streamed)"
            )
    finalize.reject()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_ep_lora_adapters(model: torch.nn.Module, output_dir: str, *, adapter_config=None, tokenizer=None):
    """Gather EP grouped-LoRA adapters and write a standalone adapter checkpoint.

    Writes ``adapter_model.safetensors`` (the gathered grouped ``A``/``B``) and an optional
    ``adapter_config.json``. The frozen base is not written; reload pairs this with the base
    via :func:`apply_ep_lora_adapters`. The gather is a collective — every rank must call together.
    """
    model = unwrap_model(model)  # a wrapper walk would prefix every adapter key (see save_ep_model)
    is_save_rank = fs_aware_save_rank()
    fs_aware_makedirs(output_dir)
    # Collective on every rank; only the writer keeps the result (else one host holds
    # local_ranks copies of the whole adapter set).
    adapters = gather_ep_lora_adapters(model, retain=is_save_rank)
    with barrier_on_exit():
        if is_save_rank:
            state = cast_state_dict_to_save_dtype(adapters)
            save_file(state, os.path.join(output_dir, ADAPTER_SAFETENSORS_FILE))
            if adapter_config is not None:
                with open(os.path.join(output_dir, ADAPTER_CONFIG_FILE), "w") as f:
                    json.dump(adapter_config, f, indent=2)
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)
            logger.info(f"✓ EP LoRA adapters saved to {output_dir} ({len(state)} tensors)")


def _check_ep_sharded_save_supported(
    model: torch.nn.Module, ep_layers: list[tuple[str, torch.nn.Module]], world_size: int
) -> None:
    """Raise if the EP topology / filesystem / model family can't produce a mergeable sharded EP
    checkpoint.

    Shared by the save path and the construction-time validator. Sharded EP keys each shard by global
    rank and ``merge_ep_shards.py`` needs every shard co-located, so supported shapes are narrow.
    """
    if not ep_layers:
        return

    if any(isinstance(m, UlyssesAttentionBase) for m in model.modules()):
        raise ValueError(
            "Sharded EP save (save_sharded_ep=True) is not supported with Context Parallelism: "
            "the per-rank shards would carry CP-patched '.original_attention.' keys that "
            "merge_ep_shards.py cannot remap to the HF layout. Use save_sharded_ep=False "
            "(gathered) — it remaps CP attention keys on save."
        )

    if has_ep_lora(model):
        raise ValueError(
            "Sharded EP save (save_sharded_ep=True) is not supported with native expert LoRA: the "
            "shard key set comes from expert_named_params(), so the grouped '<attr>_lora_A/_lora_B' "
            "adapters are written as '.shard_N' keys that merge_ep_shards.py's base-root pattern "
            "never matches — they pass through dead while the merged experts are the FROZEN base. "
            "Use save_sharded_ep=False (gathered), which folds the adapters via "
            "merge_expert_lora_on_save or writes them alongside the base."
        )

    _check_ep_merge_family_supported(model)
    ep_cfg = ep_layers[0][1].ep_config

    # Read as attributes (AttributeError = fail loud): a getattr default would make the two guards
    # below pass VACUOUSLY on any object that is not a finalized EPConfig, and a sharded save whose
    # topology was never checked merges into a checkpoint with duplicated or missing experts.
    expert_tp_size = ep_cfg.expert_tp_size
    if expert_tp_size > 1:
        raise ValueError(
            f"Sharded EP save (save_sharded_ep=True) is not supported with "
            f"expert_tp_size={expert_tp_size}. Expert TP shards expert FFN weights "
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


def _check_ep_merge_family_supported(model: torch.nn.Module) -> None:
    """Raise if this model's ``config.model_type`` resolves to no registered EP layer class.

    The per-rank shards are only usable after ``merge_ep_shards.py``, which picks the HF-layout
    transform by resolving ``model_type`` to the EP layer class that owns it — so an unclaimed
    ``model_type`` could never be merged. Reject the save up front rather than after training produced
    unmergeable shards; both sides go through the same resolver, so they can never disagree.
    """
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", "") or ""
    layer_cls = resolve_ep_merge_layer_class(model_type)
    if layer_cls is None:
        raise ValueError(
            f"Sharded EP save (save_sharded_ep=True) is not supported for model_type "
            f"{model_type!r}: no registered EP layer class claims it, so "
            f"scripts/after_training/merge_ep_shards.py cannot resolve an HF-layout transform and the "
            f"per-rank shards could never be merged into a loadable checkpoint. Supported model types: "
            f"{', '.join(supported_ep_merge_model_types())}. Use save_sharded_ep=False (gathered) instead."
        )
    if layer_cls._EXPORTS_HUB_NAMESPACE:
        raise ValueError(
            f"Sharded EP save (save_sharded_ep=True) is not supported for model_type {model_type!r}: "
            f"{layer_cls.__name__} writes the hub checkpoint namespace through transformers' save-side "
            f"conversion revert, which merge_ep_shards.py's key-by-key stream cannot apply — the merged "
            f"checkpoint would carry the module-tree spelling no serving engine reads. Use "
            f"save_sharded_ep=False (gathered) instead."
        )


def validate_ep_sharded_save(
    model: torch.nn.Module, world_size: int | None = None, merge_expert_lora_on_save: bool = False
) -> None:
    """Construction-time fail-fast for ``save_sharded_ep=True``: validate EP topology, filesystem, and
    merge support up front so an unsupported shape raises immediately (rather than at the first save,
    or worse — after training wrote unmergeable shards)."""
    if merge_expert_lora_on_save:
        raise ValueError(
            "merge_expert_lora_on_save=True is incompatible with save_sharded_ep=True: "
            "merge_lora is only supported with gathered EP save (sharded=False). "
            "Use save_sharded_ep=False, or disable merge_expert_lora_on_save."
        )
    if world_size is None:
        world_size = get_global_world_size()
    ep_layers = find_ep_layers(model)
    if not ep_layers:
        # Accepting the flag on a run that can never shard would silently write ordinary gathered
        # checkpoints while a planned merge_ep_shards.py step waits for shards that never exist.
        raise ValueError(
            "save_sharded_ep=True, but this run has no EP MoE layers (dense model, or a MoE without "
            "EP wrappers — e.g. accelerate launch with use_grouped_gemm=false): every save would be "
            "an ordinary gathered checkpoint, not the sharded format this flag names. Remove "
            "save_sharded_ep, or run an EP mode."
        )
    _check_ep_sharded_save_supported(model, ep_layers, world_size)


def _save_ep_sharded(
    model: torch.nn.Module,
    output_dir: str,
    tokenizer=None,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
):
    """Save an EP model as per-rank expert shards + a weight-map index.

    Requires merge_ep_shards.py before loading. Constraints enforced by
    :func:`_check_ep_sharded_save_supported`.
    """
    rank, world_size = get_global_rank(), get_global_world_size()

    ep_layers = find_ep_layers(model)
    _check_ep_sharded_save_supported(model, ep_layers, world_size)

    logger.info(f"[Rank {rank}] Saving EP model sharded to {output_dir}")
    if rank == 0:
        logger.warning(
            "Sharded EP save is NOT directly loadable — run "
            "scripts/after_training/merge_ep_shards.py to produce a HuggingFace "
            "checkpoint before load_ep_model()/from_pretrained(). Choose this mode for write "
            "throughput (every rank writes its own shard instead of funnelling the model through "
            "one); the gathered save streams, so host memory is not a reason to prefer it."
        )
        # Reported rather than silently dropped: a per-rank shard is one file by construction (the
        # merge reads exactly one per rank), so the run's cap describes the artifact of the merge.
        logger.info(
            f"save_max_shard_size ({max_shard_size}) does not bound these shards — a per-rank EP save "
            f"writes one file per rank by design. It bounds the shards merge_ep_shards.py writes."
        )

    ep_layer_names = [name for name, _ in ep_layers]
    cast = save_dtype_caster(model)  # norm params keep trained dtype (module-tree derived)

    # From ``expert_named_params()``, never a substring match — that misclassifies whole families.
    ep_expert_param_names = {
        f"{layer_name}.{pname}" for layer_name, module in ep_layers for pname, _ in module.expert_named_params()
    }

    shard = {}

    # EP+TP: non-EP DTensors need a full_tensor() collective on ALL ranks even though only rank 0
    # writes — rank==0 (not fs_aware_save_rank) since non-shared multi-node is blocked upstream.
    for name, param in model.named_parameters():
        if name in ep_expert_param_names:
            shard[f"{name}{EP_SHARD_KEY_INFIX}{rank}"] = cast(name, param.data.cpu())
        else:
            resolved = resolve_param_tensor(param.data)
            if rank == 0:
                shard[name] = cast(name, resolved)

    # The loop above wrote TP-sharded PLAIN tensors (GptOss sinks) as rank 0's head slice under the
    # full-tensor key — a truncated sinks the merge cannot complete.
    gather_tp_sharded_non_dtensor_params(model, shard, retain=(rank == 0))

    # Anchored prefixes so a sibling can't over-match; persistent-only so rotary caches stay out.
    if rank == 0:
        for name, buf in persistent_buffers(model):
            is_in_ep_layer = any(name.startswith(f"{ep_name}.") for ep_name in ep_layer_names)
            if is_in_ep_layer and name not in shard:
                shard[name] = cast(name, buf.data.cpu())

    if rank == 0:
        for name, buf in _persistent_non_ep_buffers(model, ep_layer_names):
            if name not in shard:
                shard[name] = cast(name, buf.detach().cpu())

    if rank == 0:
        for sink_name, sink_tensor in neutralized_gpt_oss_sinks(model).items():
            if sink_name not in shard:
                shard[sink_name] = cast(sink_name, sink_tensor)

    # Counted off the tensors this rank actually WRITES, never the live params: the cast above folds
    # fp32 masters to bf16, so a pre-cast count would put an index total_size on the artifact that
    # overstates it by up to 2x — and the gathered writer counts what it writes.
    total_size = sum(t.numel() * t.element_size() for t in shard.values())

    shard_filename = ep_shard_filename(rank, world_size)
    shard_path = os.path.join(output_dir, shard_filename)

    # Every rank writes its own shard, so the odds that some rank's write fails scale with world
    # size. Raising here would leave the survivors in the collectives below until the watchdog fires.
    guard = DeferredRankFailure(f"EP sharded checkpoint write to {output_dir}")
    if shard:
        guard.run(partial(save_file, shard, shard_path))
        if guard.reason is None:
            logger.info(
                f"[Rank {rank}] Saved shard: {shard_filename} ({len(shard)} tensors, {total_size / 1e9:.2f}GB)"
            )

    weight_map = {key: shard_filename for key in shard}

    guard.reject()

    # EP shards are key-disjoint (experts carry a per-rank ``.shard_N`` suffix and every replicated
    # key is rank-0-only), so the merge's collision check is a real gate here — and it raises on every
    # rank, because a raise confined to the rank-0 gate below would strand the peers in the barrier.
    merged_map, merged_bytes = exchange_shard_index(weight_map, total_size)

    with barrier_on_exit():
        if rank == 0:
            # Through the shared writer, so the sweep and the metadata shape match every other
            # sharded save. The sweep matters here specifically: these shards live in HF's own
            # model-{i}-of-{n} namespace, so a leftover gathered save (or a different-world rerun) is
            # indistinguishable from a rank shard to merge_ep_shards' regex — whose ``ep_size`` gate
            # reads the index written here.
            write_merged_index(
                output_dir,
                merged_map,
                {"total_size": merged_bytes, "ep_size": world_size, "format": EP_SHARDED_FORMAT},
            )

            # A tie broken in training must clear tie_word_embeddings, else the merge drops the lm_head.
            reconcile_tie_word_embeddings(model, shard)
            _save_config_and_tokenizer(model, output_dir, tokenizer)
            logger.info(f"[Rank 0] Sharded save complete: {world_size} shards, {merged_bytes / 1e9:.2f}GB")
            logger.info("[Rank 0] To load: use merge_ep_shards.py first, then load_ep_model()")
    gc.collect()
