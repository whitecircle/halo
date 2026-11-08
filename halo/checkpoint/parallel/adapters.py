"""PEFT and native expert-LoRA adapters: their saves and the restore that resumes them. EP and CP rebuild a resumed
model with zero-init adapters, so the trained ones reload through :func:`restore_adapters`."""

from __future__ import annotations

import copy
import itertools
import logging
import os
from collections.abc import Iterable
from dataclasses import replace
from functools import partial

import torch
from peft import PeftModel
from peft.utils import get_peft_model_state_dict
from torch.distributed.tensor import DTensor

from halo.checkpoint.adapters import (
    adapter_layout_mismatch,
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
from halo.checkpoint.writing import adapter_save_dtype_caster
from halo.distributed.expert_parallel.moe_layer.expert_lora import (
    gather_expert_lora_adapters,
    load_expert_lora_adapters,
)
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.log import KEY_PREVIEW_COUNT
from halo.model_loading.tokenizer_setup import vocabulary_grown
from halo.models.introspection.param_roles import persistent_buffers
from halo.models.introspection.unwrapping import (
    PEFT_BASE_MODEL_PREFIX,
    base_transformers_model,
    find_peft_model,
    plain_adapter_key,
    unwrap_framework_wrappers,
    unwrap_toolkit_wrappers,
)
from halo.runtime.consensus import DeferredRankFailure, agree_across_ranks, barrier_on_exit, reject_across_ranks
from halo.runtime.dtensor import copy_full_tensor, gather_full_tensors
from halo.runtime.filesystem import fs_aware_makedirs
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


def _is_adapter_relevant(name: str) -> bool:
    # Superset of what PEFT may serialize: LoRA tensors, modules_to_save clones, the embed/lm_head pair.
    return any(marker in name for marker in ("lora_", "modules_to_save", "embed", "lm_head"))


def _saves_embedding_layers(peft_model: PeftModel) -> bool | str:
    """``True`` once the run grew its vocabulary, whose added rows no base checkpoint holds; else PEFT's ``"auto"``,
    which tells a grown vocabulary only by re-reading the base config, skipped offline and defeated by a composite
    config."""
    return True if vocabulary_grown(peft_model) else "auto"


def gather_adapter_state(peft_model: PeftModel, *, retain: bool = True) -> dict:
    """COLLECTIVE: the PEFT adapter state, resolving only adapter-relevant params, since all of them would retain the
    frozen base on every rank; empty off ``retain``, which joins every gather and keeps nothing. Buffers ride along
    because a ``modules_to_save`` clone with a balancing buffer would otherwise KeyError PEFT's key lookup at the first
    save. Only ``retain`` runs that lookup, so its failure is joined."""
    full_state = gather_full_tensors(
        (
            (name, t.data)
            for name, t in itertools.chain(peft_model.named_parameters(), persistent_buffers(peft_model))
            if _is_adapter_relevant(name)
        ),
        retain=retain,
    )
    guard = DeferredRankFailure("Building the PEFT adapter state")
    state = {}
    if retain:
        state = guard.run(
            partial(
                get_peft_model_state_dict,
                peft_model,
                state_dict=full_state,
                save_embedding_layers=_saves_embedding_layers(peft_model),
            )
        )
    guard.reject()
    return state


def _cp_normalized_config(peft_config):
    """The adapter config with module paths respelled like the keys; ``exclude_modules`` and a rewritten
    ``target_modules`` spell the CP-wrapped tree, which a plain reload would never match. A copy, so the live config
    keeps the wrapped spelling."""
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
    """Recover ``base_model_name_or_path``, which ``get_peft_model`` reads off the wrapped module's ``__dict__``; under
    CP that is the wrapper's and has none."""
    name_or_path = getattr(base_transformers_model(peft_model), "name_or_path", None)
    if not name_or_path:
        return
    for peft_config in peft_model.peft_config.values():
        if peft_config.base_model_name_or_path is None:
            peft_config.base_model_name_or_path = name_or_path


def _adapter_file_state(ctx: CheckpointContext, state: dict) -> dict:
    """The save dtype for an export, so an fp32-master run does not write a 2x adapter; the live dtype for a training
    checkpoint, whose fp32 adapters must resume unrounded."""
    if ctx.training_checkpoint:
        return state
    cast = adapter_save_dtype_caster()
    return {key: cast(key, t) for key, t in state.items()}


def save_peft_adapters(ctx: CheckpointContext, output_dir: str) -> None:
    """COLLECTIVE: write the PeftModel's adapters, provenance and model card; only the save rank writes. PEFT's own
    ``save_pretrained`` does it unless DTensor adapters need resolving, a CP run's keys need respelling for a non-CP
    reload, or a mixed run's expert adapters join the attention ones in one file stock PEFT refuses."""
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
        state = gather_adapter_state(peft_model, retain=ctx.is_save_rank)
    elif is_cp_mode:
        # CP without DTensors (QLoRA+CP, which skips FSDP2) still needs the manual write for its changed key paths.
        state = get_peft_model_state_dict(peft_model, save_embedding_layers=_saves_embedding_layers(peft_model))
    if state is not None and is_cp_mode:
        state = {plain_adapter_key(key): tensor for key, tensor in state.items()}
        peft_config = _cp_normalized_config(peft_config)
    if mixed:
        # unwrap_toolkit_wrappers stops at the PeftModel, so under CP the expert keys keep the wrapper's 'model.'
        # level; restore_adapters applies them through the same unwrap, so the keys round-trip.
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
                peft_model.save_pretrained(output_dir, save_embedding_layers=_saves_embedding_layers(peft_model))
                apply_umask_mode(os.path.join(output_dir, ADAPTER_SAFETENSORS_FILE))
            elif mixed:
                write_mixed_adapter_config(peft_config, ctx.parallelism_config.expert_lora, output_dir)
            else:
                peft_config.save_pretrained(output_dir)
            if state is not None:
                write_adapter_file(_adapter_file_state(ctx, {**state, **expert_state}), output_dir)
            logger.info(f"Saved {saved}")
            write_training_provenance(ctx.model, output_dir)
            # The hand-written branches write no card, and PEFT rebuilds the tags from ``model_tags``, which only an
            # apply_run_model_policies load carries.
            tag_model_card(output_dir)
        # Released inside the fence, so the barrier is not held under the state's footprint.
        del state, expert_state


def save_expert_lora_adapters(ctx: CheckpointContext, output_dir: str) -> None:
    """COLLECTIVE: write the native EP grouped-LoRA adapters as a standalone checkpoint without the frozen base, at the
    save dtype or, in a training checkpoint whose fp32 adapters must resume unrounded, the live one."""
    model = unwrap_toolkit_wrappers(ctx.model)  # a wrapper walk would prefix every adapter key
    fs_aware_makedirs(output_dir)
    # Collective on every rank; only the writer keeps the result, else one host holds local_ranks copies.
    adapters = gather_expert_lora_adapters(model, retain=ctx.is_save_rank)
    base_name = getattr(ctx.model.config, "_name_or_path", None)
    with barrier_on_exit(f"Writing the EP LoRA adapters to {output_dir}"):
        if ctx.is_save_rank:
            save_safetensors_file(
                _adapter_file_state(ctx, adapters), os.path.join(output_dir, ADAPTER_SAFETENSORS_FILE)
            )
            write_adapter_config(expert_lora_adapter_config(ctx.parallelism_config.expert_lora, base_name), output_dir)
            tag_model_card(output_dir)
            logger.info(f"✓ EP LoRA adapters saved to {output_dir} ({len(adapters)} tensors)")


def save_resume_adapter(ctx: CheckpointContext, checkpoint_dir: str) -> None:
    """COLLECTIVE: write a ``merge_expert_lora_on_save`` checkpoint's unmerged adapters beside its merged weights,
    which serve but cannot resume, since the bf16 fold loses part of the delta and the optimizer state belongs to the
    adapters. Older markers are removed before the save begins, so a failed adapter write leaves no marker."""
    # Rank-uniform, and a no-op when sharded; a forward's transient unsharded params predate the last optimizer step.
    reshard_fsdp2_modules(ctx.model)
    adapter_dir = os.path.join(checkpoint_dir, RESUME_ADAPTER_DIR)
    save_adapters = save_peft_adapters if find_peft_model(ctx.model) is not None else save_expert_lora_adapters
    save_adapters(replace(ctx, training_checkpoint=True), adapter_dir)
    mark_resume_adapter_complete(checkpoint_dir, is_save_rank=ctx.is_save_rank)


def read_adapter_across_ranks(
    adapter_dir: str, *, checkpoint: str, what: str, attention: dict | None, expert: dict | None = None
) -> tuple[dict | None, str | None]:
    """COLLECTIVE. The adapter tensors and file read under ``adapter_dir``, or ``(None, None)`` when no rank holds one;
    refused on every rank when any rank's live config holds another LoRA scaling."""
    state, path = consensus_read(
        adapter_weight_paths(adapter_dir), read_adapter_file, what=what, checkpoint=checkpoint
    )
    if path is not None:
        reject_across_ranks(lora_scaling_mismatch(adapter_dir, attention, expert), what, ValueError)
    return state, path


def live_adapter_tensors(peft_model: PeftModel) -> dict[str, torch.Tensor]:
    """The live tensor behind every key a PEFT save writes (a tie under each of its names); an FSDP2 DTensor stays as
    it is, so its ``shape`` is the saved full one."""
    live = dict(itertools.chain(peft_model.named_parameters(remove_duplicate=False), persistent_buffers(peft_model)))
    return get_peft_model_state_dict(
        peft_model, state_dict=live, save_embedding_layers=_saves_embedding_layers(peft_model)
    )


def _with_tied_aliases(live_tensors: dict[str, torch.Tensor], attn_state: dict) -> dict:
    """``attn_state`` with each live key it lacks taken from a key it holds for the same tensor, since a gathered save
    writes one name per tied tensor (a resized tied embedding's ``lm_head``)."""
    held: dict[int, str] = {}
    for key in sorted(attn_state):
        if key in live_tensors:
            held.setdefault(id(live_tensors[key]), key)
    aliases = {
        key: attn_state[held[id(tensor)]]
        for key, tensor in live_tensors.items()
        if key not in attn_state and id(tensor) in held
    }
    return {**attn_state, **aliases}


def _without_unsaved_grown_rows(
    peft_model: PeftModel, live_tensors: dict[str, torch.Tensor], attn_state: dict, checkpoint: str
) -> dict[str, torch.Tensor]:
    """COLLECTIVE when the run grew its vocabulary. ``live_tensors`` without the frozen embedding and head tensors no
    key of ``attn_state`` holds, from an adapter saved before saves carried them; their live values stand, with a
    warning. The keys left out are agreed across ranks, since each kept key is a copy and a DTensor copy a mesh
    collective."""
    if not vocabulary_grown(peft_model):
        return live_tensors
    model = base_transformers_model(peft_model)
    frozen = {
        id(param)
        for module in (model.get_input_embeddings(), model.get_output_embeddings())
        if module is not None
        for param in module.parameters()
        if not param.requires_grad
    }
    held = {id(live_tensors[key]) for key in attn_state if key in live_tensors}
    unsaved = sorted(
        key
        for key, tensor in live_tensors.items()
        if key not in attn_state and id(tensor) in frozen and id(tensor) not in held
    )
    if not agree_across_ranks(unsaved):
        raise RuntimeError(
            f"The PEFT adapter at {checkpoint} holds the grown vocabulary's embedding rows on some ranks only: the "
            f"ranks read different adapter files. Resume from a complete checkpoint."
        )
    if unsaved and is_global_main_process():
        logger.warning(
            f"The PEFT adapter at {checkpoint} holds no tensor for the grown vocabulary's frozen embedding rows "
            f"{unsaved[:KEY_PREVIEW_COUNT]}: it was saved before adapter saves carried them. Resuming with the rows "
            f"this run's grow initialized; the saved run kept its own frozen at that mean-resize init, so they differ "
            f"only by its random draw. Later saves carry them."
        )
    return {key: tensor for key, tensor in live_tensors.items() if key not in unsaved}


def _load_peft_adapter_state(live_tensors: dict[str, torch.Tensor], attn_state: dict) -> None:
    """Copy each saved attention tensor into its live tensor, a tied tensor once. Sorted, because a DTensor target's
    ``distribute_tensor`` is a mesh collective and the file's own order can differ per rank on a non-shared
    filesystem."""
    copied: set[int] = set()
    with torch.no_grad():
        for key, value in sorted(attn_state.items()):
            target = live_tensors[key]
            if id(target) not in copied:
                copied.add(id(target))
                copy_full_tensor(target, value)


def _attention_adapter_mismatch(live_tensors: dict | None, attn_state: dict, checkpoint: str) -> str | None:
    """Why ``attn_state`` does not restore ``live_tensors`` (None without a PeftModel) one for one, or None."""
    if live_tensors is None:
        if not attn_state:
            return None
        return (
            f"Adapter checkpoint at {checkpoint} contains {len(attn_state)} attention/PEFT adapter tensors but the "
            f"model has no PEFT adapters — resume would silently continue from base weights. Enable use_peft (with "
            f"the same lora_target_modules as at save) or resume from a full checkpoint. First few: "
            f"{sorted(attn_state)[:KEY_PREVIEW_COUNT]}"
        )
    mismatch = adapter_layout_mismatch(attn_state, {key: tuple(tensor.shape) for key, tensor in live_tensors.items()})
    return mismatch and (
        f"The PEFT adapter at {checkpoint} does not match the adapters this model builds: {mismatch}. Resume with "
        f"the lora_target_modules, lora_r, modules_to_save and added_special_tokens it was saved with, from a "
        f"complete checkpoint."
    )


def remap_cp_adapter_keys_to_live(state: dict, live_keys: Iterable[str]) -> dict:
    """Map CP-normalized adapter keys back onto ``live_keys``, which carry ``.original_attention.`` / extra ``model.``
    paths on a CP-wrapped resume. An unknown key passes through for the restore's layout check to name."""
    normalized_to_live = {plain_adapter_key(key): key for key in live_keys}
    return {normalized_to_live.get(key, key): value for key, value in state.items()}


def restore_adapters(checkpoint: str, model, *, is_cp_mode: bool) -> str | None:
    """COLLECTIVE. Restore LoRA adapters onto a model whose base weights stay as built (an EP/CP rebuild, or a PEFT
    run's adapter-only checkpoint). Refuses on every rank, before any copy, another LoRA scaling and a key or shape
    mismatch. Returns the adapter file restored from, or ``None`` when no rank holds one; rank-uniform either way."""
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
    attn_state = {k: v for k, v in state.items() if k not in expert_state}
    live_tensors = live_adapter_tensors(peft_model) if peft_model is not None else None
    # Saver wrote CP-normalized keys; remap to live CP-wrapped names or all keys miss.
    if is_cp_mode and live_tensors is not None:
        attn_state = remap_cp_adapter_keys_to_live(attn_state, live_tensors)
    if live_tensors is not None:
        attn_state = _with_tied_aliases(live_tensors, attn_state)
        live_tensors = _without_unsaved_grown_rows(peft_model, live_tensors, attn_state, checkpoint)
    # Joined before the expert load, whose own join precedes its first copy, so neither population writes until both
    # have matched on every rank.
    reject_across_ranks(
        _attention_adapter_mismatch(live_tensors, attn_state, checkpoint), f"PEFT adapter resume from {checkpoint}"
    )
    # Inner model, matching the saver: the CP wrapper's extra 'model.' level makes keys miss.
    load_expert_lora_adapters(unwrap_toolkit_wrappers(unwrapped), expert_state)
    if attn_state:
        _load_peft_adapter_state(live_tensors, attn_state)

    if is_global_main_process():
        logger.info(
            f"Restored adapters from {loaded_path} ({len(expert_state)} expert + {len(attn_state)} attention tensors)"
        )
    return loaded_path
