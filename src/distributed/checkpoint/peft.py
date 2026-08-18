"""PEFT (LoRA) adapters: the save and its resume counterpart.

:class:`PeftAdapterSaver` is invoked explicitly by ``save_model`` before the mode ladder — DTensor-LoRA
gather, CP key normalization, the standard ``save_pretrained`` path. :func:`restore_adapters` is the
other half: EP/CP rebuild the model with zero-init adapters, so the loader hands the reload back here,
where the CP key remap both directions share already lives.
"""

from __future__ import annotations

import itertools
import json
import logging
import os

import torch
from peft import PeftModel
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import save_file as safetensors_save_file
from torch.distributed.tensor import DTensor, distribute_tensor

from src.checkpoint.adapters import (
    MIXED_EXPERT_LORA_PEFT_TYPE,
    adapter_weight_paths,
    cast_adapter_state_to_save_dtype,
    is_expert_lora_key,
    read_adapter_file,
)
from src.checkpoint.format import (
    ADAPTER_BIN_FILE,
    ADAPTER_CONFIG_FILE,
    ADAPTER_SAFETENSORS_FILE,
    PROVENANCE_GPT_OSS_SINKS,
    TRAINING_PROVENANCE_FILE,
)
from src.distributed.checkpoint.context import CheckpointContext
from src.distributed.checkpoint.coordination import KEY_PREVIEW_COUNT, consensus_read
from src.distributed.context_parallel.key_mapping import strip_cp_attention_prefix
from src.distributed.expert_parallel.config import ExpertLoraSpec
from src.distributed.expert_parallel.expert_weights import (
    apply_ep_lora_adapters,
    gather_ep_lora_adapters,
)
from src.distributed.runtime import (
    barrier_on_exit,
    fs_aware_makedirs,
    is_global_main_process,
    resolve_param_tensor,
)
from src.models.patches.gpt_oss_sinks import stamped_sinks_policy
from src.models.structure import (
    persistent_buffers,
    strip_peft_adapter_segment,
    unwrap_framework_wrappers,
    unwrap_model,
)

logger = logging.getLogger(__name__)

# Superset of every param name PEFT may serialize into an adapter file: LoRA tensors,
# modules_to_save clones, and the embed/lm_head pair save_embedding_layers can add.
_ADAPTER_PARAM_MARKERS = ("lora_", "modules_to_save", "embed", "lm_head")


def _is_adapter_relevant(name: str) -> bool:
    return any(marker in name for marker in _ADAPTER_PARAM_MARKERS)


def find_peft_model(model) -> PeftModel | None:
    """Find a PeftModel in the (already-unwrapped) model hierarchy."""
    if isinstance(model, PeftModel):
        return model
    if hasattr(model, "model"):
        return find_peft_model(model.model)
    return None


def expert_lora_config_fields(spec: ExpertLoraSpec) -> dict:
    """The ``adapter_config.json`` fields describing native EP grouped expert adapters.

    One home for "what describes the expert half", shared by the expert-only save
    (:func:`~src.distributed.checkpoint.save._expert_lora_adapter_config`, flat) and the mixed
    attention+expert save here
    (nested under ``ep_expert_lora``, so it cannot collide with the attention ``LoraConfig``'s
    identically-named fields). Records every field that changes what the adapter *means* —
    ``scaling`` is derived from ``lora_alpha``/``r``/``use_rslora``, so omitting one would let a
    reload rescale every expert delta silently.
    """
    return {
        "r": spec.r,
        "lora_alpha": spec.alpha,
        "lora_dropout": spec.dropout,
        "use_rslora": spec.use_rslora,
        "expert_projections": sorted(spec.projections),
    }


class PeftAdapterSaver:
    """Saves PEFT adapters with parallelism-aware DTensor gathering / CP key normalization."""

    def save(self, ctx: CheckpointContext, peft_model: PeftModel, output_dir: str) -> bool:
        saved = self._save_adapter_files(ctx, peft_model, output_dir)
        if saved and ctx.is_save_rank:
            self._write_training_provenance(ctx.model, output_dir)
        return saved

    def _save_adapter_files(self, ctx: CheckpointContext, peft_model: PeftModel, output_dir: str) -> bool:
        should_save = ctx.is_save_rank
        self._backfill_base_model_name(peft_model)

        if ctx.has_expert_lora:
            return self._save_with_expert_lora(ctx, peft_model, output_dir, should_save)

        # DTensor LoRA params require collective gathering (FSDP2 wraps the adapter params); CP
        # without DTensors (e.g. QLoRA+CP) needs the same manual write for its changed key paths.
        gathered = self._has_dtensor_lora_params(peft_model)
        if gathered or ctx.is_cp_mode:
            self._save_reconstructed_adapters(ctx, peft_model, output_dir, should_save, gathered=gathered)
            return True

        fs_aware_makedirs(output_dir)
        with barrier_on_exit():
            if should_save:
                peft_model.save_pretrained(output_dir)
                if ctx.tokenizer is not None:
                    ctx.tokenizer.save_pretrained(output_dir)
                logger.info(f"Saved PEFT adapters to {output_dir}")
        return True

    @staticmethod
    def _write_training_provenance(model, output_dir: str) -> None:
        """Record training-time model state a merge cannot recover from the adapter artifacts.

        A GptOss adapter trains against live OR neutralized attention sinks depending on the run's
        ``reset_sinks``, but a merge rebuilds the base from the hub, whose sinks are always live —
        without this record a reset-sinks run's merge silently serves attention the adapter never
        trained under. Only STAMPED state is recorded: the stamp exists exactly on models the sinks
        policy ran on (the family signal derives from it — no model-type list here), and an absent
        stamp means "nothing to record", never "neutralized". Rank-local file write on the save
        rank, after the adapter barriers.
        """
        provenance = {}
        policy = stamped_sinks_policy(model)
        if policy is not None:
            provenance[PROVENANCE_GPT_OSS_SINKS] = policy
        if not provenance:
            return
        with open(os.path.join(output_dir, TRAINING_PROVENANCE_FILE), "w") as fh:
            json.dump(provenance, fh, indent=2)

    def _save_with_expert_lora(self, ctx, peft_model: PeftModel, output_dir: str, should_save: bool) -> bool:
        """Combine attention PEFT adapters with native EP expert adapters in one adapter file.

        Attention adapters resolved DTensor-aware + CP-key-normalized; EP expert adapters from a
        collective gather over every EP layer. All collectives on every rank; only the save rank writes.
        """
        attn_state = self._resolve_adapter_state(peft_model)
        if ctx.is_cp_mode:
            attn_state = self._normalize_cp_adapter_keys(attn_state)

        # Collective over the EP/expert-TP groups; every rank. unwrap_model stops at the PeftModel,
        # so under CP the expert keys keep the wrapper's 'model.' level; resume applies them through
        # the same unwrap of the same root shape (restore_adapters), so the keys round-trip.
        expert_state = gather_ep_lora_adapters(unwrap_model(ctx.model), retain=should_save)

        fs_aware_makedirs(output_dir)
        with barrier_on_exit():
            if should_save:
                self._write_mixed_adapter_config(
                    peft_model.peft_config["default"],
                    getattr(ctx.parallelism_config, "expert_lora", None),
                    output_dir,
                )
                # Cast to save dtype so fp32_experts / fp32_non_ep_params don't write a 2x adapter file.
                self._write_adapter_state_dict(
                    cast_adapter_state_to_save_dtype({**attn_state, **expert_state}), output_dir
                )
                if ctx.tokenizer is not None:
                    ctx.tokenizer.save_pretrained(output_dir)
                logger.info(
                    f"Saved PEFT + EP expert adapters to {output_dir} "
                    f"({len(attn_state)} attention + {len(expert_state)} expert tensors)"
                )
            del attn_state, expert_state
        return True

    @staticmethod
    def _backfill_base_model_name(peft_model: PeftModel) -> None:
        """``get_peft_model`` reads ``name_or_path`` from the wrapped module's own ``__dict__``; under
        CP that module is the wrapper, which has none, so ``adapter_config.json`` would record
        ``base_model_name_or_path: null`` and the adapter loses its base-model pointer. Recover it
        from the inner PreTrainedModel. Local attribute write — rank-uniform, no collectives.
        """
        name_or_path = getattr(unwrap_model(peft_model.get_base_model()), "name_or_path", None)
        if not name_or_path:
            return
        for peft_config in peft_model.peft_config.values():
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = name_or_path

    @staticmethod
    def _write_mixed_adapter_config(peft_config, expert_lora_spec: ExpertLoraSpec | None, output_dir: str) -> None:
        """Write the attention ``LoraConfig``, re-typed so stock PEFT refuses this hybrid adapter.

        The file it labels holds native EP grouped expert deltas next to the attention tensors.
        ``PeftModel.from_pretrained`` cannot wrap grouped experts, and it drops unexpected
        state-dict keys without a warning, so a ``peft_type: LORA`` label would hand back an
        attention-only model that looks fully adapted. :data:`MIXED_EXPERT_LORA_PEFT_TYPE` is absent
        from PEFT's config mapping, turning that load into a raise naming the marker.

        Every LoRA field is preserved (PEFT serializes its own config, so a new field cannot be
        missed here), and the expert half is described under ``ep_expert_lora``. This toolkit's
        resume reads the tensors directly and never parses this file.
        """
        peft_config.save_pretrained(output_dir)
        config_path = os.path.join(output_dir, ADAPTER_CONFIG_FILE)
        with open(config_path) as fh:
            config = json.load(fh)
        config["peft_type"] = MIXED_EXPERT_LORA_PEFT_TYPE
        if expert_lora_spec is not None:
            config["ep_expert_lora"] = expert_lora_config_fields(expert_lora_spec)
        with open(config_path, "w") as fh:
            json.dump(config, fh, indent=2)

    @staticmethod
    def _resolve_adapter_state(peft_model: PeftModel) -> dict:
        """DTensor-resolve ONLY adapter-relevant params and buffers, then apply the PEFT key transform.

        Resolving all of ``named_parameters()`` would ``full_tensor()`` and retain the frozen base on
        every rank. The filter stays a superset of every key PEFT may read — ``lm_head`` covers the
        resized-untied-vocab case, where dropping it saves the input embedding but not the output
        head. Buffers ride along because a ``modules_to_save`` clone serializes its WHOLE state dict:
        a router/gate carrying a balancing buffer (``e_score_correction_bias``, an adopted
        ``router.bias``) would otherwise KeyError PEFT's key lookup at the first save. Persistent
        only, like every other checkpoint writer — a non-persistent cache (rotary, attention mask)
        recomputes on load, and serializing one would reload it stale on resume. Rank-uniform,
        keeping the collectives aligned.
        """
        full_state = {
            name: resolve_param_tensor(t.data)
            for name, t in itertools.chain(peft_model.named_parameters(), persistent_buffers(peft_model))
            if _is_adapter_relevant(name)
        }
        return get_peft_model_state_dict(peft_model, state_dict=full_state)

    @staticmethod
    def _has_dtensor_lora_params(peft_model: PeftModel) -> bool:
        for name, param in peft_model.named_parameters():
            if "lora_" in name and isinstance(param.data, DTensor):
                return True
        return False

    def _save_reconstructed_adapters(self, ctx, peft_model, output_dir, should_save, *, gathered: bool) -> None:
        """Write the adapter files PEFT's own ``save_pretrained`` cannot produce.

        ``gathered``: the adapter params are FSDP2-sharded DTensors, reconstructed to full tensors
        (a collective every rank enters). Otherwise the state is read plainly (e.g. QLoRA+CP). CP
        key normalization applies to both — the non-gathered branch is only reached in CP mode.
        The state is released inside the ``with`` so the barrier is not held under its footprint.
        """
        if gathered:
            adapter_state_dict = self._resolve_adapter_state(peft_model)
        else:
            adapter_state_dict = get_peft_model_state_dict(peft_model)

        if ctx.is_cp_mode:
            adapter_state_dict = self._normalize_cp_adapter_keys(adapter_state_dict)

        reconstruction = "with DTensor reconstruction" if gathered else "CP-normalized keys"
        fs_aware_makedirs(output_dir)
        with barrier_on_exit():
            if should_save:
                peft_model.peft_config["default"].save_pretrained(output_dir)
                self._write_adapter_state_dict(cast_adapter_state_to_save_dtype(adapter_state_dict), output_dir)
                if ctx.tokenizer is not None:
                    ctx.tokenizer.save_pretrained(output_dir)
                logger.info(f"Saved PEFT adapters ({reconstruction}) to {output_dir}")
            del adapter_state_dict

    @staticmethod
    def _write_adapter_state_dict(adapter_state_dict: dict, output_dir: str) -> None:
        st_path = os.path.join(output_dir, ADAPTER_SAFETENSORS_FILE)
        # safetensors refuses a non-contiguous tensor, and a gathered DTensor adapter or a
        # ``modules_to_save`` clone can be a view — one such tensor would send the WHOLE adapter to
        # the .bin fallback. The same pass the full-checkpoint writer makes (``save_sharded_state_dict``).
        adapter_state_dict = {key: tensor.contiguous() for key, tensor in adapter_state_dict.items()}
        try:
            safetensors_save_file(adapter_state_dict, st_path)
        except Exception as e:
            logger.warning(f"safetensors adapter save failed: {e}, using pytorch format")
            # The .bin goes on disk FIRST, then the safetensors leftover (this attempt's partial
            # file, or a previous save's complete one) is removed: every reader prefers the
            # safetensors name, so leaving it would silently resume/merge the OLD adapter.
            torch.save(adapter_state_dict, os.path.join(output_dir, ADAPTER_BIN_FILE))
            try:
                if os.path.exists(st_path):
                    os.remove(st_path)
            except OSError as remove_error:
                logger.warning(
                    f"could not remove stale {ADAPTER_SAFETENSORS_FILE} after the .bin fallback: "
                    f"{remove_error} — readers prefer it over the fresh {ADAPTER_BIN_FILE}"
                )

    @staticmethod
    def _normalize_cp_adapter_key(key: str) -> str:
        """Strip CP-wrapper key-path artifacts (extra ``model.`` level, ``.original_attention.``) from
        a single adapter key so it loads onto a non-CP model.

        Not ``structure.unwrapped_module_name``: that maps MODULE paths to the plain hub tree and
        drops the PEFT prefix entirely, while adapter keys must KEEP it and additionally carry the
        attention-wrapper artifact and the shallower ``modules_to_save`` lm_head depth handled below.
        """
        new_key = strip_cp_attention_prefix(key)
        if new_key.startswith("model.model."):
            new_key = new_key[len("model.") :]
        elif "base_model.model.model.model." in new_key:
            new_key = new_key.replace("base_model.model.model.model.", "base_model.model.model.")
        elif new_key.startswith("base_model.model.model.lm_head."):
            # A ``modules_to_save`` lm_head sits ABOVE the backbone — one fewer ``model.`` level
            # than the backbone keys the rule above collapses. A plain key can never spell lm_head
            # at this depth (plain is ``base_model.model.lm_head.``), so the collapse is unambiguous
            # and the normalization stays idempotent.
            new_key = new_key.replace("base_model.model.model.lm_head.", "base_model.model.lm_head.", 1)
        return new_key

    @staticmethod
    def _normalize_cp_adapter_keys(state_dict: dict) -> dict:
        """Apply :meth:`_normalize_cp_adapter_key` across an adapter state dict."""
        return {PeftAdapterSaver._normalize_cp_adapter_key(k): v for k, v in state_dict.items()}


def restore_adapters(checkpoint: str, model, *, is_cp_mode: bool) -> None:
    """Restore LoRA adapters when base-weight reload is skipped (EP/CP).

    EP/CP rebuild the model with zero-init adapters, so trained weights must reload here or resume
    continues from the untrained adapter. Handles native EP expert adapters (sliced per rank by
    :func:`apply_ep_lora_adapters`) and PEFT attention adapters. No-op when no adapter file.
    """
    state, loaded_path = consensus_read(
        adapter_weight_paths(checkpoint),
        read_adapter_file,
        what="Adapter checkpoint",
        checkpoint=checkpoint,
    )
    if loaded_path is None:
        return  # no adapters anywhere: full fine-tuning
    unwrapped = unwrap_framework_wrappers(model)

    expert_state = {k: v for k, v in state.items() if is_expert_lora_key(k)}
    # Unconditional: the apply is collective at ep_size==1 and decides "nothing to load" rank-uniformly.
    # Inner model, matching the saver: the CP wrapper's extra 'model.' level makes keys miss.
    apply_ep_lora_adapters(unwrap_model(unwrapped), expert_state)

    attn_state = {k: v for k, v in state.items() if k not in expert_state}
    if attn_state:
        peft_model = find_peft_model(unwrapped)
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
            value = value.to(param.dtype)
            if isinstance(param.data, DTensor):
                # Default ``src_data_rank``: mesh rank 0's read is broadcast, so every DP replica
                # holds one node's adapter bytes (``_load_tp`` slices per rank instead — its
                # ranks each need the full tensor anyway, and it joins the key set explicitly).
                value = distribute_tensor(value, param.data.device_mesh, param.data.placements)
            param.data.copy_(value)
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
        normalized_to_live[PeftAdapterSaver._normalize_cp_adapter_key(live_key)] = live_key
    return {normalized_to_live.get(key, key): value for key, value in state.items()}
