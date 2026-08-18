"""Expert-weight classification, roster queries, and whole-layer checkpoint gather for EP.

Per-family layout conversion lives on the layer classes (``EP*MoELayer.gather_expert_state_dict``);
this module holds everything that is not family-specific:

* the live-parameter name predicate the GRPO weight-sync path classifies experts with
  (:func:`is_expert_weight_attr`);
* the registry-derived vocabularies every off-line consumer reads instead of restating a
  ``model_type`` ladder — expert-container attributes, fused/per-expert key layouts, and the
  ``model_type`` roster queries built on :func:`ep_hub_model_types`;
* :func:`gather_ep_layer_weights`, the whole-layer gather including the replicated
  router / shared-expert / buffer tensors, and the grouped expert-LoRA gather/load pair.

The offline balancing-slot seam lives in :mod:`src.models.moe_balancing`, which the layer classes
register into; nothing here is needed to read a checkpoint's balancing state.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import torch

import src.distributed.expert_parallel.layers.roster  # noqa: F401 — registers every family subclass
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.module_registry import build_class_claim_map, iter_subclasses
from src.distributed.runtime import (
    broadcast_from_rank0,
    get_global_rank,
    materialize_dtensor,
    resolve_param_tensor,
)
from src.models.loading.config_levels import config_sources
from src.models.structure import persistent_buffers, unwrap_framework_wrappers


def ep_layer_classes() -> list[type[EPMoELayerBase]]:
    """Every registered EP layer class (the whole subclass tree).

    The EP-side name for the union walk behind every "union what the families declare" helper — a new
    family joins by being listed in :mod:`~src.distributed.expert_parallel.layers`, never by being
    added to a list here. That package is imported above, so the walk is complete for every caller
    regardless of import order (an incomplete walk would silently under-report the union).
    """
    return iter_subclasses(EPMoELayerBase)


def _declared_union(attr: str) -> set[str]:
    """Union of the class attribute ``attr`` over every registered EP layer class.

    The one walk behind every "what does the roster declare" vocabulary below, so a new family joins
    each of them by being imported. Callers cache; this stays uncached because it returns a mutable
    set they shape (sorted longest-first, frozen) before publishing.
    """
    return {value for cls in ep_layer_classes() for value in getattr(cls, attr)}


def _longest_first(values: set[str]) -> tuple[str, ...]:
    """Sorted longest-first — the order a prefix/suffix matcher must try candidates in, so a longer
    spelling is never shadowed by a shorter one it contains."""
    return tuple(sorted(values, key=lambda value: (-len(value), value)))


@lru_cache(maxsize=1)
def expert_weight_roots() -> frozenset[str]:
    """Every expert-weight attribute root the roster declares
    (:attr:`~EPMoELayerBase._EXPERT_WEIGHT_ATTR_ROOTS`).

    Cached: the EP subclass set is fixed once ``layers`` is imported, which precedes any weight sync.
    """
    return frozenset(_declared_union("_EXPERT_WEIGHT_ATTR_ROOTS"))


@lru_cache(maxsize=1)
def experts_container_attrs() -> tuple[str, ...]:
    """Every expert-container attribute name the roster declares
    (:attr:`~EPMoELayerBase._EXPERTS_CONTAINER_ATTRS`), longest-first.

    The one vocabulary — what the live probe walks and what the lazy loader's checkpoint-key regexes
    accept. Stating it twice lets the two drift: a family whose container the regexes miss has every
    fused expert key planned REPLICATE (every rank loading the full tensor), or its per-expert keys
    misread as fused and left on meta.
    """
    return _longest_first(_declared_union("_EXPERTS_CONTAINER_ATTRS"))


@lru_cache(maxsize=1)
def hf_fused_expert_keys() -> tuple[str, ...]:
    """Every fused 3D expert tensor name an HF checkpoint may carry (dim 0 = expert), longest-first —
    :attr:`~EPMoELayerBase._HF_FUSED_EXPERT_KEYS` over the roster, which the lazy loader slices per rank."""
    return _longest_first(_declared_union("_HF_FUSED_EXPERT_KEYS"))


@lru_cache(maxsize=1)
def ep_layer_class_by_model_type() -> dict[str, type[EPMoELayerBase]]:
    """``config.model_type`` → EP layer class, unioned from every family's :attr:`HF_MODEL_TYPES`.

    The off-line counterpart of ``patching.MOE_LAYER_MAP`` (which needs a live module tree): resolves a
    checkpoint's ``config.json`` back to the class that produced it, so class-keyed registries — the
    sharded-merge transforms above all — stay derived instead of restating a ``model_type`` ladder.
    Same builder as the patcher's map, on this registry's own attribute: own declaration only, loud on
    a key two families claim.
    """
    return build_class_claim_map(EPMoELayerBase, "HF_MODEL_TYPES", "model_type")


def resolve_ep_merge_layer_class(model_type: str) -> type[EPMoELayerBase] | None:
    """EP layer class a checkpoint's ``config.model_type`` was trained with, or ``None`` when no
    registered family claims it.

    The sharded-merge entry point: the transform itself lives ON that class
    (:meth:`~EPMoELayerBase.merge_shards_to_hf`, the inverse of the same class's
    ``gather_expert_state_dict``, overridden together or not at all), which is what makes
    ``merged-from-sharded == gathered`` structural — there is no transform table to update and no
    ``model_type`` ladder. Resolution is the class-declared :attr:`HF_MODEL_TYPES` union, never a
    spelling heuristic, which would happily route an unknown future ``model_type`` onto the wrong
    expert layout.
    """
    return ep_layer_class_by_model_type().get(model_type)


def supported_ep_merge_model_types() -> list[str]:
    """Every ``model_type`` spelling a registered EP family claims, sorted — the mergeable set, since
    every family owns a ``merge_shards_to_hf`` matching its own gather."""
    return sorted(ep_layer_class_by_model_type())


def ep_layer_classes_for_config(model_config) -> list[type[EPMoELayerBase]]:
    """EP layer classes resolved from a model CONFIG's ``model_type`` spellings, via the registry.

    The one home for the config → family → EP-wrapper-class walk: gates that run before (or
    without) a live wrapper instance — the fp32-non-EP validation at load, the weight-sync
    contract on a wrapper-less ``use_grouped_gemm: false`` run, the patching predicate below — all
    resolve through here, keyed by ``model_type`` (both the composite wrapper's and its text
    sub-config's spelling, which is where a multimodal checkpoint declares the MoE family).

    The lighter "does this family have a wrapper at all" question is
    :func:`~src.models.moe_balancing.has_ep_wrapper_class`, answered off the same declarations
    without a class in hand.
    """
    registry = ep_layer_class_by_model_type()
    types = {mt for source in config_sources(model_config) if (mt := getattr(source, "model_type", None))}
    return [registry[t] for t in sorted(types) if t in registry]


@lru_cache(maxsize=1)
def per_expert_layouts() -> tuple[tuple[str, str, str], ...]:
    """Every ``(gate, up, down)`` per-expert projection naming the roster declares, deduplicated.

    The vocabulary of hub-side expert layouts, taken from the layer classes rather than restated —
    what a fused→per-expert conversion may emit, and what the lazy loader may fuse back. Keyed on
    :meth:`~EPMoELayerBase.hub_per_expert_keys` so it covers BOTH ways a family declares the layout:
    a family whose own gather already emits per-expert is just as lazy-loadable as one the base
    gather splits, and reading only the base-split attribute would leave its names unknown here.
    """
    return tuple(dict.fromkeys(keys for cls in ep_layer_classes() if (keys := cls.hub_per_expert_keys())))


def ep_hub_model_types(claims: Callable[[type[EPMoELayerBase]], object]) -> tuple[str, ...]:
    """``config.model_type`` spellings, sorted, whose EP layer class satisfies ``claims``.

    The one roster query: it reads :func:`ep_layer_class_by_model_type`, so every caller inherits
    that registry's OWN-declaration semantics (an intermediate base never re-reports its parent's
    families) *and* its duplicate-claim guard. Re-rolling the ``vars(cls)["HF_MODEL_TYPES"]`` walk
    per caller silently drops the guard.
    """
    return tuple(sorted(mt for mt, cls in ep_layer_class_by_model_type().items() if claims(cls)))


@lru_cache(maxsize=1)
def per_expert_hub_model_types() -> tuple[str, ...]:
    """``config.model_type`` spellings whose family's HUB checkpoint stores one tensor per expert.

    Keyed on :meth:`~EPMoELayerBase.hub_per_expert_keys`, so a family joins (or leaves) by changing its
    own declaration — what ``scripts/after_training/unfuse_moe_experts.py`` may rewrite a fused
    checkpoint into, and the set its refusal names.
    """
    return ep_hub_model_types(lambda cls: cls.hub_per_expert_keys())


@lru_cache(maxsize=1)
def per_expert_fusion_map() -> dict[str, tuple[str, int]]:
    """``{"<name>.weight": (fusion_group, position)}`` for every per-expert hub layout on the roster.

    Union of the families' :meth:`~EPMoELayerBase.hub_per_expert_keys` ``(gate, up, down)``
    declarations — the same tuples the gathered save splits back out — so the lazy loader fuses a new
    family's per-expert checkpoint without a second list to update. Halves resolve by POSITION.

    Conflicting claims RAISE, as in :func:`ep_layer_class_by_model_type`: the map is keyed by
    projection NAME across the whole roster, so two families spelling one name at different positions
    cannot coexist — the class the subclass walk happens to visit last would win and silently swap the
    other family's gate/up halves on every lazy load, at identical shapes and with no error.
    """
    fusion: dict[str, tuple[str, int]] = {}
    claimed_by: dict[str, str] = {}
    for cls in ep_layer_classes():
        keys = cls.hub_per_expert_keys()
        if keys is None:
            continue
        gate, up, down = keys
        for name, slot in ((gate, ("gate_up", 0)), (up, ("gate_up", 1)), (down, ("down", 0))):
            key = f"{name}.weight"
            if key in fusion and fusion[key] != slot:
                raise ValueError(
                    f"per-expert projection '{key}' is claimed as {fusion[key]} by "
                    f"{claimed_by[key]} and as {slot} by {cls.__name__}; one of the two families' "
                    f"experts would be fused with its gate/up halves swapped."
                )
            fusion[key] = slot
            claimed_by.setdefault(key, cls.__name__)
    return fusion


def to_hub_layer_key(key: str, layer_cls: type[EPMoELayerBase]) -> str:
    """Rewrite one layer-relative gathered key from the live MODULE spelling to the HUB spelling.

    Identity for every family that declares no :attr:`~EPMoELayerBase._EXPORT_KEY_RENAMES`.
    """
    for module_spelling, hub_spelling in layer_cls._EXPORT_KEY_RENAMES:
        if module_spelling in key:
            return key.replace(module_spelling, hub_spelling, 1)
    return key


def hub_to_module_key_renames(model_type: str) -> tuple[tuple[str, str], ...]:
    """``(hub spelling, module spelling)`` pairs for a checkpoint's family — the load-side inverse of
    :func:`to_hub_layer_key`, for readers that bypass ``from_pretrained``.

    Empty for an unregistered ``model_type`` and for every family whose two spellings agree.
    """
    layer_cls = ep_layer_class_by_model_type().get(model_type)
    if layer_cls is None:
        return ()
    return tuple((hub, module) for module, hub in layer_cls._EXPORT_KEY_RENAMES)


def expert_weight_attrs(module: EPMoELayerBase) -> set[str]:
    """Leading-segment attribute names of this layer's live expert params (base + grouped-LoRA adapters).

    Exact per-instance truth from :meth:`~EPMoELayerBase.expert_named_params`; prefer this to the name-only
    :func:`is_expert_weight_attr` when a layer instance is in hand."""
    return {name.split(".")[0] for name, _ in module.expert_named_params()}


def is_expert_weight_attr(param_name: str) -> bool:
    """Whether a local parameter name belongs to an expert weight.

    Classifies on the leading name segment against :func:`expert_weight_roots`; router/gate, shared
    experts and other replicated weights have a different root → False. Name-only fallback for the GRPO
    weight-sync path (no layer instance); with an instance, prefer :func:`expert_weight_attrs`.

    Native grouped-LoRA adapters (``<base>_lora_A``/``<base>_lora_B``) belong to the expert weight and are
    folded into the base by the EP gather, so classify them as expert params too — else the dense
    weight-sync ships a raw adapter shard vLLM cannot load.
    """
    root = param_name.split(".")[0]
    for suffix in ("_lora_A", "_lora_B"):
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break
    return root in expert_weight_roots()


def gather_ep_layer_weights(
    layer_name: str, module: EPMoELayerBase, merge_lora: bool = False, retain: bool = True
) -> dict[str, torch.Tensor]:
    """Gather a whole EP layer for checkpoint saving.

    Returns gathered experts (keyed ``<layer_name>.experts.*``) plus every persistent replicated
    non-expert param/buffer (router/gate, shared experts, expert-bias buffers, …), all in the family's
    HUB spelling via :func:`to_hub_layer_key`. The replicated pass skips expert params via the layer's
    ``expert_named_params()`` — the exact live expert attrs on this rank. ``merge_lora`` folds the
    grouped expert-LoRA delta into the base experts inside the per-family gather — before any
    family-specific unfuse/re-interleave, so it is correct for every layout.

    Every rank enters the same collectives, but only ``retain=True`` ranks keep the result: the others
    join each gather and return ``{}``, so neither the family's post-gather assembly (per-expert split,
    re-interleave, transpose + ``contiguous``) nor the host copy — per node, ``local_ranks`` x one whole
    gathered layer — ever happens off the writer.
    """
    layer_cls = type(module)
    expert_state = module.gather_expert_state_dict(merge_lora=merge_lora, retain=retain)
    if not retain and expert_state:
        raise RuntimeError(
            f"{layer_cls.__name__}.gather_expert_state_dict ignored retain=False and assembled "
            f"{len(expert_state)} tensors — every non-writing rank would hold a whole gathered layer. "
            f"Thread retain through the family's gather."
        )
    gathered: dict[str, torch.Tensor] = {
        f"{layer_name}.{to_hub_layer_key(key, layer_cls)}": tensor for key, tensor in expert_state.items()
    }
    del expert_state

    expert_roots = expert_weight_attrs(module)

    # Non-expert params/buffers may be FSDP2-sharded DTensors, needing a full_tensor() collective on
    # every rank; a raw .cpu() on a DTensor leaves invalid storage that safetensors rejects.
    for param_name, param in module.named_parameters():
        if param_name.split(".")[0] in expert_roots:
            continue
        if retain:
            gathered.setdefault(
                f"{layer_name}.{to_hub_layer_key(param_name, layer_cls)}", resolve_param_tensor(param.data)
            )
        else:
            materialize_dtensor(param.data)  # same collective, without the host copy

    # Persistent only: a non-persistent buffer (rotary cache) is absent from the SHARDED save, and
    # exporting it here would break the "merged-from-sharded == gathered" invariant the merge relies on.
    for buf_name, buf in persistent_buffers(module):
        if buf_name.split(".")[0] in expert_roots:
            continue
        if retain:
            gathered.setdefault(
                f"{layer_name}.{to_hub_layer_key(buf_name, layer_cls)}", resolve_param_tensor(buf.data)
            )
        else:
            materialize_dtensor(buf.data)

    return gathered


def config_model_types(model: torch.nn.Module) -> set[str]:
    """Every ``model_type`` spelling on the model's config (wrapper + text sub-config).

    The one home for "which family is this model" as a spelling set — the weight-sync gates and the
    sidecar apply both key the EP registry off it.
    """
    cfg = getattr(unwrap_framework_wrappers(model), "config", None)
    if cfg is None:
        return set()
    return {mt for source in config_sources(cfg) if (mt := getattr(source, "model_type", None))}


def has_ep_lora(model: torch.nn.Module) -> bool:
    """Whether any EP layer in ``model`` carries native grouped-LoRA adapters."""
    return any(isinstance(m, EPMoELayerBase) and m.has_expert_lora for _, m in model.named_modules())


def gather_ep_lora_adapters(model: torch.nn.Module, *, retain: bool = True) -> dict[str, torch.Tensor]:
    """Gather every EP layer's grouped LoRA adapters into one CPU checkpoint dict.

    Keys ``<layer_name>.experts.<attr>.lora_{A,B}``. Per-layer gather is a collective over EP (and
    expert-TP) groups — every rank must call together.

    ``retain=False`` enters every gather and keeps nothing, matching
    :func:`gather_ep_layer_weights`. Without it each of a node's ranks built the whole adapter set on
    the host for one writer to use: at a 397B-class MoE with 8 ranks/node that is ~40 GB/node at
    ``lora_r: 16`` and ~160 GB/node at ``lora_r: 64``, on the standard EP LoRA save path.
    """
    adapters: dict[str, torch.Tensor] = {}
    for layer_name, module in model.named_modules():
        if not isinstance(module, EPMoELayerBase) or not module.has_expert_lora:
            continue
        # Same split as gather_ep_layer_weights: a non-retaining rank enters every collective and
        # assembles nothing.
        gathered = module.gather_expert_lora_state_dict(retain=retain)
        for key, tensor in gathered.items():
            adapters[f"{layer_name}.{key}"] = tensor
        del gathered
    return adapters


def apply_ep_lora_adapters(model: torch.nn.Module, adapter_state: dict[str, torch.Tensor]) -> None:
    """Load gathered grouped LoRA adapters back into ``model`` (inverse of
    :func:`gather_ep_lora_adapters`). Each EP layer slices its expert range from the full tensors.

    COLLECTIVE — every rank must call together: at ``ep_size==1`` with ``fsdp_shard_ep1_experts`` the
    adapters are FSDP2 DTensors and ``load_expert_lora_state_dict`` re-shards each one via
    ``distribute_tensor``, a scatter on the param's mesh. The "no adapters" decision is therefore taken
    from rank 0, not per-rank: a rank whose adapter file is missing on a non-shared filesystem would
    otherwise skip collectives its peers enter.

    Raises when ANY EP layer carrying expert LoRA matches none of the saved keys — full and partial
    misses alike (e.g. keys saved under a wrapper-prefixed module path, or a different EP layout):
    resuming would leave exactly those layers' adapters at zero-init. Matching is validated for every
    layer before anything loads, so a miss never leaves the model partially updated.
    """
    if not broadcast_from_rank0(bool(adapter_state)):
        return
    if not adapter_state:
        raise RuntimeError(
            f"apply_ep_lora_adapters: rank {get_global_rank()} received no expert-adapter state while "
            f"global rank 0 did — a torn/partial adapter checkpoint on a non-shared filesystem. This "
            f"rank's adapters would silently resume from zero-init. Resume from a complete checkpoint."
        )
    pending: list[tuple[EPMoELayerBase, dict[str, torch.Tensor]]] = []
    missed: list[str] = []
    for layer_name, module in model.named_modules():
        if not isinstance(module, EPMoELayerBase) or not module.has_expert_lora:
            continue
        prefix = f"{layer_name}.experts."
        layer_state = {
            key[len(f"{layer_name}.") :]: tensor for key, tensor in adapter_state.items() if key.startswith(prefix)
        }
        if layer_state:
            pending.append((module, layer_state))
        else:
            missed.append(layer_name)
    if missed:
        raise RuntimeError(
            f"apply_ep_lora_adapters: {len(missed)} EP layer(s) with expert LoRA matched none of the "
            f"{len(adapter_state)} saved expert-adapter keys: {missed[:5]}"
            f"{' …' if len(missed) > 5 else ''} (first saved key: {next(iter(adapter_state))!r}). "
            f"Their adapters would silently resume from zero-init."
        )
    if not pending:
        # No EP layer carries expert LoRA at all, so `missed` is empty and the walk above cannot speak:
        # the saved expert deltas are simply dropped. Reachable by resuming an expert-LoRA checkpoint
        # into a run that no longer builds the adapters (EP off, use_grouped_gemm off, expert
        # projections removed from lora_target_modules) — the loader would then report a successful
        # adapter restore having loaded none of them.
        raise RuntimeError(
            f"apply_ep_lora_adapters: the checkpoint carries {len(adapter_state)} expert-adapter tensors "
            f"(first: {next(iter(adapter_state))!r}) but this model has no EP layer with expert LoRA to "
            f"receive them, so every saved expert delta would be discarded. Resume with the run's expert "
            f"configuration — EP/grouped-GEMM on and the same expert projections in lora_target_modules."
        )
    for module, layer_state in pending:
        module.load_expert_lora_state_dict(layer_state)
