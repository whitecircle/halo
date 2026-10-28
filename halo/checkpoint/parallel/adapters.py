"""PEFT and native expert-LoRA adapters: their saves, and the restore that resumes them.

EP and CP rebuild a resumed model with zero-init adapters, so the trained ones reload through :func:`restore_adapters`,
which writes DTensors through :func:`~halo.runtime.dtensor.copy_full_tensor` and maps a CP save's keys, respelled for a
non-CP reload (:func:`~halo.models.unwrapping.plain_adapter_key`), back onto the live CP-wrapped names."""

from __future__ import annotations

import copy
import itertools
import logging
import os
from dataclasses import replace

import torch
from peft import PeftModel
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torch.distributed.tensor import DTensor

from halo.checkpoint.adapters import (
    adapter_weight_paths,
    expert_lora_adapter_config,
    expert_lora_config_fields,
    is_expert_lora_key,
    lora_scaling_mismatch,
    read_adapter_file,
    write_adapter_config,
    write_adapter_file,
    write_mixed_adapter_config,
)
from halo.checkpoint.atomic import apply_umask_mode, save_safetensors_file
from halo.checkpoint.format import ADAPTER_SAFETENSORS_FILE, RESUME_ADAPTER_DIR
from halo.checkpoint.model_card import tag_model_card
from halo.checkpoint.parallel.completeness import mark_resume_adapter_complete
from halo.checkpoint.parallel.context import CheckpointContext
from halo.checkpoint.parallel.reading import consensus_read
from halo.checkpoint.sidecars import write_training_provenance
from halo.checkpoint.writing import adapter_save_dtype_caster, cast_to_save_dtype
from halo.distributed.expert_parallel.moe_layer.expert_lora import (
    gather_expert_lora_adapters,
    load_expert_lora_adapters,
)
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.log import KEY_PREVIEW_COUNT
from halo.models.param_roles import persistent_buffers
from halo.models.unwrapping import (
    PEFT_BASE_MODEL_PREFIX,
    base_transformers_model,
    find_peft_model,
    plain_adapter_key,
    strip_peft_adapter_segment,
    unwrap_framework_wrappers,
    unwrap_toolkit_wrappers,
)
from halo.runtime.consensus import barrier_on_exit, reject_across_ranks
from halo.runtime.dtensor import copy_full_tensor, full_cpu_tensor
from halo.runtime.filesystem import fs_aware_makedirs
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


def _is_adapter_relevant(name: str) -> bool:
    # Superset of every param name PEFT may serialize into an adapter file: LoRA tensors,
    # modules_to_save clones, and the embed/lm_head pair save_embedding_layers can add.
    return any(marker in name for marker in ("lora_", "modules_to_save", "embed", "lm_head"))


def gather_adapter_state(peft_model: PeftModel) -> dict:
    """COLLECTIVE: the PEFT adapter state, resolving only the adapter-relevant params and persistent buffers.

    Resolving every parameter would retain the frozen base on every rank. The buffers ride along because
    a ``modules_to_save`` clone serializes its whole state dict: a router carrying a balancing buffer
    would otherwise KeyError PEFT's key lookup at the first save.
    """
    full_state = {
        name: full_cpu_tensor(t.data)
        for name, t in itertools.chain(peft_model.named_parameters(), persistent_buffers(peft_model))
        if _is_adapter_relevant(name)
    }
    return get_peft_model_state_dict(peft_model, state_dict=full_state)


def _cp_normalized_config(peft_config):
    """The adapter config as a non-CP model reads it: module paths respelled like the keys.

    ``exclude_modules``, and ``target_modules`` once ``all-linear`` made PEFT rewrite it to full paths,
    spell the CP-wrapped tree; a plain reload would match none of them, injecting into the very module
    an exclusion named and finding no target. A copy is returned, so the live config keeps the wrapped
    spelling; suffix names (``q_proj``) pass through unchanged.
    """
    config = None
    for field in ("exclude_modules", "target_modules"):
        paths = getattr(peft_config, field, None)
        if not isinstance(paths, (list, tuple, set)):
            continue
        # A module path takes the key rule under the PEFT prefix an adapter key carries.
        plain = sorted(
            plain_adapter_key(PEFT_BASE_MODEL_PREFIX + path).removeprefix(PEFT_BASE_MODEL_PREFIX) for path in paths
        )
        if plain == sorted(paths):
            continue
        config = config or copy.deepcopy(peft_config)
        setattr(config, field, plain)
    return config or peft_config


def _backfill_base_model_name(peft_model: PeftModel) -> None:
    """Recover ``base_model_name_or_path`` from the inner PreTrainedModel: ``get_peft_model`` reads it off
    the wrapped module's own ``__dict__``, which under CP is the wrapper's and has none."""
    name_or_path = getattr(base_transformers_model(peft_model), "name_or_path", None)
    if not name_or_path:
        return
    for peft_config in peft_model.peft_config.values():
        if peft_config.base_model_name_or_path is None:
            peft_config.base_model_name_or_path = name_or_path


def _adapter_file_state(ctx: CheckpointContext, state: dict) -> dict:
    """The adapter tensors a hand-written adapter file holds: at the save dtype for an export, so an
    fp32-master run (``fp32_experts`` / ``fp32_non_ep_params``) does not write a 2x adapter; at the
    live dtype for a training checkpoint, whose fp32 adapters must resume unrounded."""
    if ctx.training_checkpoint:
        return state
    cast = adapter_save_dtype_caster()
    return {key: cast(key, t) for key, t in state.items()}


def save_peft_adapters(ctx: CheckpointContext, output_dir: str) -> None:
    """COLLECTIVE: write the PeftModel's adapters, then their training provenance and model card.

    PEFT's own ``save_pretrained`` writes them unless it cannot: FSDP2-sharded DTensor adapters are
    resolved first (a collective every rank enters), a CP run's keys and config are respelled for a
    non-CP reload, and a mixed run's native expert adapters join the attention ones in one file whose
    config stock PEFT refuses. Only the save rank writes.
    """
    peft_model = find_peft_model(ctx.model)
    _backfill_base_model_name(peft_model)
    peft_config = peft_model.peft_config["default"]
    is_cp_mode = ctx.parallelism_config.is_cp_mode
    mixed = ctx.has_expert_lora
    gathered = any(
        "lora_" in name and isinstance(param.data, DTensor) for name, param in peft_model.named_parameters()
    )
    state, expert_state = None, {}
    if mixed or gathered:
        state = gather_adapter_state(peft_model)
    elif is_cp_mode:
        # CP without DTensors (QLoRA+CP) still needs the manual write for its changed key paths.
        state = get_peft_model_state_dict(peft_model)
    if state is not None and is_cp_mode:
        state = {plain_adapter_key(key): tensor for key, tensor in state.items()}
        peft_config = _cp_normalized_config(peft_config)
    if mixed:
        # unwrap_toolkit_wrappers stops at the PeftModel, so under CP the expert keys keep the wrapper's
        # 'model.' level; resume applies them through the same unwrap of the same root shape
        # (restore_adapters), so the keys round-trip.
        expert_state = gather_expert_lora_adapters(unwrap_toolkit_wrappers(ctx.model), retain=ctx.is_save_rank)
        saved = (
            f"PEFT + EP expert adapters to {output_dir} ({len(state)} attention + {len(expert_state)} expert tensors)"
        )
    elif state is not None:
        saved = (
            f"PEFT adapters ({'with DTensor reconstruction' if gathered else 'CP-normalized keys'}) to {output_dir}"
        )
    else:
        saved = f"PEFT adapters to {output_dir}"

    fs_aware_makedirs(output_dir)
    with barrier_on_exit(f"Writing the PEFT adapters to {output_dir}"):
        if ctx.is_save_rank:
            if state is None:
                peft_model.save_pretrained(output_dir)
                apply_umask_mode(os.path.join(output_dir, ADAPTER_SAFETENSORS_FILE))
            elif mixed:
                write_mixed_adapter_config(
                    peft_config, getattr(ctx.parallelism_config, "expert_lora", None), output_dir
                )
            else:
                peft_config.save_pretrained(output_dir)
            if state is not None:
                write_adapter_file(_adapter_file_state(ctx, {**state, **expert_state}), output_dir)
            logger.info(f"Saved {saved}")
            write_training_provenance(ctx.model, output_dir)
            # The hand-written branches write no card, and PEFT's own save_pretrained rebuilds the card's
            # tags from the base model's model_tags, which only an apply_run_model_policies load carries.
            tag_model_card(output_dir)
        # Released inside the fence, so the barrier is not held under the state's footprint.
        del state, expert_state


def save_expert_lora_adapters(ctx: CheckpointContext, output_dir: str) -> None:
    """COLLECTIVE: gather the native EP grouped-LoRA adapters and write them as a standalone adapter
    checkpoint, without the frozen base, which a reload pairs them with (:func:`load_expert_lora_adapters`).

    The adapters go to the save dtype, or keep their live one in a training checkpoint, whose fp32
    adapters must resume unrounded.
    """
    model = unwrap_toolkit_wrappers(ctx.model)  # a wrapper walk would prefix every adapter key
    fs_aware_makedirs(output_dir)
    # Collective on every rank; only the writer keeps the result (else one host holds
    # local_ranks copies of the whole adapter set).
    adapters = gather_expert_lora_adapters(model, retain=ctx.is_save_rank)
    spec = getattr(ctx.parallelism_config, "expert_lora", None)
    base_name = getattr(ctx.model.config, "_name_or_path", None)
    adapter_config = None if spec is None else expert_lora_adapter_config(spec, base_name)
    with barrier_on_exit(f"Writing the EP LoRA adapters to {output_dir}"):
        if ctx.is_save_rank:
            state = (
                adapters if ctx.training_checkpoint else {key: cast_to_save_dtype(t) for key, t in adapters.items()}
            )
            save_safetensors_file(state, os.path.join(output_dir, ADAPTER_SAFETENSORS_FILE))
            if adapter_config is not None:
                write_adapter_config(adapter_config, output_dir)
            tag_model_card(output_dir)
            logger.info(f"✓ EP LoRA adapters saved to {output_dir} ({len(state)} tensors)")


def save_resume_adapter(ctx: CheckpointContext, checkpoint_dir: str) -> None:
    """COLLECTIVE: write a ``merge_expert_lora_on_save`` checkpoint's unmerged adapters beside its merged weights.

    The merged weights serve but cannot resume: the bf16 fold loses part of the delta, and the
    optimizer state belongs to the adapters. They go to :data:`~halo.checkpoint.format.RESUME_ADAPTER_DIR`
    through the non-merged save's adapter writer at their live dtype, then each save rank marks its
    own complete copy; with older markers removed before the save began
    (:func:`~halo.checkpoint.parallel.completeness.prepare_checkpoint_rewrite`), a failed adapter write
    leaves no marker.
    """
    # Rank-uniform, and a no-op when sharded: a forward's transient unsharded params predate the
    # last optimizer step, like every writer's.
    reshard_fsdp2_modules(ctx.model)
    adapter_dir = os.path.join(checkpoint_dir, RESUME_ADAPTER_DIR)
    save_adapters = save_peft_adapters if find_peft_model(ctx.model) is not None else save_expert_lora_adapters
    save_adapters(replace(ctx, training_checkpoint=True), adapter_dir)
    mark_resume_adapter_complete(checkpoint_dir, is_save_rank=ctx.is_save_rank)


def read_adapter_across_ranks(
    adapter_dir: str, *, checkpoint: str, what: str, attention: dict | None, expert: dict | None = None
) -> tuple[dict | None, str | None]:
    """COLLECTIVE. The adapter tensors under ``adapter_dir`` and the file read, read on every rank, or
    ``(None, None)`` when no rank holds one; refused on every rank, under the read's ``what``, when any
    rank's live ``attention`` / ``expert`` config holds another LoRA scaling (:func:`lora_scaling_mismatch`).
    """
    state, path = consensus_read(
        adapter_weight_paths(adapter_dir), read_adapter_file, what=what, checkpoint=checkpoint
    )
    if path is not None:
        reject_across_ranks(lora_scaling_mismatch(adapter_dir, attention, expert), what, ValueError)
    return state, path


def _load_peft_adapter_state(peft_model, attn_state: dict) -> list[str]:
    """Load PEFT attention-adapter tensors into the live model; returns unmatched saved keys.

    Under FSDP2 the adapter params are DTensors, and PEFT's ``set_peft_model_state_dict`` (plain
    ``copy_``) raises "mixed torch.Tensor and DTensor" — distribute each full tensor to the
    param's placements instead. Plain params keep the PEFT loader (which handles modules_to_save).
    Persistent buffers join the live map because the adapter save carries them (a
    ``modules_to_save`` router's balancing buffer) — parameters-only here would drop the saved
    buffer as "unexpected" and resume it at init.
    """
    live = dict(itertools.chain(peft_model.named_parameters(), persistent_buffers(peft_model)))
    has_dtensor = any(isinstance(p.data, DTensor) for n, p in live.items() if ".lora_" in n)
    if not has_dtensor:
        load_result = set_peft_model_state_dict(peft_model, attn_state)
        return list(getattr(load_result, "unexpected_keys", None) or [])

    # Saved keys omit the adapter segment: ...lora_A.weight ↔ live ...lora_A.default.weight.
    normalized_live = {strip_peft_adapter_segment(name): name for name in live}

    unexpected = []
    with torch.no_grad():
        # Sorted: distribute_tensor issues mesh collectives — same key order on every rank, exactly
        # as the sibling weight loaders enforce. A dict's own order is the file's, and the adapter
        # file can differ per rank on a non-shared filesystem.
        for key, value in sorted(attn_state.items()):
            param = live.get(key)
            if param is None and key in normalized_live:
                param = live[normalized_live[key]]
            if param is None:
                unexpected.append(key)
                continue
            copy_full_tensor(param, value)
    return unexpected


def remap_cp_adapter_keys_to_live(state: dict, peft_model: PeftModel) -> dict:
    """Map CP-normalized adapter keys back onto the live CP-wrapped PeftModel's keys on resume.

    Saver normalizes CP keys for non-CP portability; on resume the model is CP-wrapped again, so live
    keys carry ``.original_attention.`` / extra ``model.`` paths. Without remapping, every saved key
    is unexpected and adapters stay zero-init. The live map comes from ``named_parameters()`` plus
    ``persistent_buffers()`` — the same population the saver serializes (a ``modules_to_save`` router
    carries its balancing buffer) — never from a ``state_dict()``-derived view, which a wrapper
    override can respell. Saved keys carry no adapter-name segment, so ``.default`` is
    dropped here exactly as ``_load_peft_adapter_state`` re-inserts it.
    """
    normalized_to_live = {}
    for name, _ in itertools.chain(peft_model.named_parameters(), persistent_buffers(peft_model)):
        if not _is_adapter_relevant(name):
            continue
        live_key = strip_peft_adapter_segment(name)
        normalized_to_live[plain_adapter_key(live_key)] = live_key
    return {normalized_to_live.get(key, key): value for key, value in state.items()}


def restore_adapters(checkpoint: str, model, *, is_cp_mode: bool) -> str | None:
    """Restore LoRA adapters when base-weight reload is skipped (EP/CP).

    EP/CP rebuild the model with zero-init adapters, so trained weights must reload here or resume
    continues from the untrained adapter. Handles native EP expert adapters (sliced per rank by
    :func:`load_expert_lora_adapters`) and PEFT attention adapters. Refuses, on every rank, an adapter
    whose recorded LoRA scaling differs from the live run's (:func:`lora_scaling_mismatch`). Returns
    the adapter file restored from, or ``None`` when ``checkpoint`` holds none on any rank (a no-op);
    rank-uniform either way.
    """
    unwrapped = unwrap_framework_wrappers(model)
    peft_model = find_peft_model(unwrapped)
    expert_spec = next(
        (layer.ep_config.expert_lora for _, layer in named_ep_layers(unwrapped) if layer.has_expert_lora), None
    )
    state, loaded_path = read_adapter_across_ranks(
        checkpoint,
        checkpoint=checkpoint,
        what="Adapter checkpoint",
        attention=peft_model.peft_config["default"].to_dict() if peft_model is not None else None,
        expert=expert_lora_config_fields(expert_spec) if expert_spec is not None else None,
    )
    if loaded_path is None:
        return None  # no adapters anywhere: full fine-tuning

    expert_state = {k: v for k, v in state.items() if is_expert_lora_key(k)}
    # Unconditional: the apply is collective at ep_size==1 and decides "nothing to load" rank-uniformly.
    # Inner model, matching the saver: the CP wrapper's extra 'model.' level makes keys miss.
    load_expert_lora_adapters(unwrap_toolkit_wrappers(unwrapped), expert_state)

    attn_state = {k: v for k, v in state.items() if k not in expert_state}
    if attn_state:
        if peft_model is None:
            # Attention tensors with no PeftModel would be dropped silently — base-weight resume.
            raise RuntimeError(
                f"Adapter checkpoint at {checkpoint} contains {len(attn_state)} attention/PEFT "
                f"adapter tensors but the model has no PEFT adapters — resume would silently "
                f"continue from base weights. Enable use_peft (with the same lora_target_modules "
                f"as at save) or resume from a full checkpoint. First few: {sorted(attn_state)[:KEY_PREVIEW_COUNT]}"
            )
        # Saver wrote CP-normalized keys; remap to live CP-wrapped names or all keys miss.
        if is_cp_mode:
            attn_state = remap_cp_adapter_keys_to_live(attn_state, peft_model)
        unexpected = _load_peft_adapter_state(peft_model, attn_state)
        if unexpected and len(unexpected) == len(attn_state):
            raise RuntimeError(
                f"PEFT adapter resume from {checkpoint}: all {len(attn_state)} saved adapter "
                f"keys were unexpected — the adapter would resume from zero-init, not the "
                f"trained weights. Likely a key mismatch (CP-remap regression, or different "
                f"lora_target_modules than at save). First few: {unexpected[:KEY_PREVIEW_COUNT]}"
            )
        if unexpected and is_global_main_process():
            logger.warning(
                f"PEFT adapter resume: {len(unexpected)} of {len(attn_state)} saved adapter "
                f"keys were unexpected and dropped (those layers stay at init): {unexpected[:KEY_PREVIEW_COUNT]}"
            )

    if is_global_main_process():
        logger.info(
            f"Restored adapters from {loaded_path} ({len(expert_state)} expert + {len(attn_state)} attention tensors)"
        )
    return loaded_path
