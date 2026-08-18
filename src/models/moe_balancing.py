"""Router introspection, MoE balancing-mode resolution, and the offline balancing-slot contract.

Defines a family's router-field spellings, which balancing strategy a run gets, and where a trained
routing bias lands in an exported checkpoint; applying that verdict to a live model is
:mod:`src.distributed.expert_parallel.balancing_strategy`. Depends on neither ``src.distributed`` nor
``torch.distributed``, so its consumers (the callbacks package, the checkpoint writers and tools, the
PP split gate) cannot form an import cycle through each other. The EP layer classes push their family
declarations in at subclass definition.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Literal, get_args

import torch

from src.log import warn_once
from src.models.loading.config_levels import (
    config_sources,
    get_config_field,
    set_config_field,
    set_config_field_run_scoped,
)
from src.models.structure import unwrap_framework_wrappers, unwrapped_module_name

logger = logging.getLogger(__name__)

# Per model class, so a resolution repeated by a second consumer (the PP split gate) says it once.
_WARNED_UNSERVABLE_AUTO: set = set()
_WARNED_TRANSIENT_ONLY_AUTO: set = set()

BalancingMode = Literal["auto", "none", "aux_loss", "bias_update", "bias_update_transient"]
_VALID_BALANCING_MODES = get_args(BalancingMode)

# Stamped on model.config when router logits must stay off (EP wrappers that never populate them,
# aux-loss paths that would crash); checked before enabling them again. Run-scoped, so
# :func:`~src.models.loading.config_levels.config_export_ready` restores it away before any
# config serialization.
ROUTER_LOGITS_FORCED_OFF_ATTR = "_toolkit_router_logits_forced_off"

# The attribute name of the hub-native balancing buffer (Zaya) and of the EP base's adoption
# property: the protocol name every balancing consumer keys on.
BALANCING_BIASES_ATTR = "balancing_biases"

# Instance flag the EP base sets when it adopts a family's native balancing slot rather than the
# transient side-buffer. One spelling for the setter and every reader, since a second copy would
# classify adopted state as transient.
NATIVE_BALANCING_BIAS_ADOPTED_ATTR = "_native_balancing_bias_adopted"

# The two spellings of the bias-update strategy. ``bias_update`` requires every balancing router to
# carry the bias in checkpoint-exported state (a native slot the serving engines load); the
# ``_transient`` spelling is the explicit opt-in for families whose architecture has no such slot,
# where the bias steers training-time routing only and every exported checkpoint serves without it.
BIAS_UPDATE_MODES = ("bias_update", "bias_update_transient")

# ``config.model_type`` to ``{flat legacy key: (layer_type, per-layer field)}``, filled by
# ``EPMoELayerBase.__init_subclass__`` from each family's ``_LEGACY_PER_LAYER_CONFIG_KEYS``. Kept here
# so the config-export rewrite reads the roster without importing the EP package, which would close a
# cycle through ``parallelism_config``.
_LEGACY_PER_LAYER_CONFIG_KEYS_BY_MODEL_TYPE: dict[str, dict[str, tuple[str, str]]] = {}

# ``config.model_type`` spellings whose exported ``config.json`` must be the source checkpoint's own
# schema rather than transformers' serialization, filled by ``EPMoELayerBase.__init_subclass__`` from
# each family's ``_EXPORTS_SOURCE_CONFIG_SCHEMA``. Located here for the same reason as the map above.
_SOURCE_CONFIG_SCHEMA_MODEL_TYPES: set[str] = set()

# Every ``config.model_type`` an EP MoE wrapper class claims: the whole roster, since a family
# declares ``HF_MODEL_TYPES`` to exist at all. Also serves as the roster-imported signal every reader
# here checks before answering off an empty registry.
_EP_WRAPPED_MODEL_TYPES: set[str] = set()


@dataclass(frozen=True)
class NativeBalancingSlot:
    """A family's checkpoint-exported balancing slot, as its EP layer class declares it.

    ``attrs`` are the dotted paths the slot answers to (the declared one and its hub respelling, as
    for Laguna); ``config_flag`` is the field an engine reads to decide whether to apply the tensor
    at all (LFM-2's ``use_expert_bias``), ``None`` where the slot is unconditional.
    """

    attrs: tuple[str, ...]
    config_flag: str | None


# ``config.model_type`` to :class:`NativeBalancingSlot`, filled by
# ``EPMoELayerBase.__init_subclass__`` from each family's ``_NATIVE_BALANCING_BIAS_ATTR``. Located
# here so an offline export resolves a checkpoint's balancing slot without importing the EP package.
_NATIVE_BALANCING_SLOTS_BY_MODEL_TYPE: dict[str, NativeBalancingSlot] = {}

# Every spelling a router-balancing tensor's checkpoint key can end with: the hub-native buffer plus
# each registered family's declared slot(s). Grown at registration rather than cached from a walk, so
# a reader is complete the moment the roster is imported and never earlier.
_BALANCING_KEY_SUFFIXES: set[str] = {BALANCING_BIASES_ATTR}

# Router top-k spellings, on a config or a router module: one registry for the load metrics, the EP
# layers and the pipeline split's cost model. First non-zero wins, so the unambiguous fields precede
# bare ``top_k``, which is also the generation sampling parameter.
ROUTER_TOPK_FIELDS: tuple[str, ...] = (
    "num_experts_per_tok",
    "top_k_experts",
    "moe_router_topk",
    "num_active_experts",
    "top_k",
)

# Routed-expert-count spellings, same single-registry rule: two copies could make a family MoE for
# the "is this MoE?" gate and dense for the load metrics.
ROUTER_EXPERT_COUNT_FIELDS: tuple[str, ...] = (
    "num_experts",
    "num_local_experts",
    "num_routed_experts",
    "num_moe_experts",
    "moe_num_experts",
)

# Same registry pattern for the per-expert FFN width, the dimension expert-TP shards. Every supported
# spelling names that dimension directly. The MoE-specific ones come first: a hybrid dense+MoE config
# carries both, and only the MoE one is sharded.
EXPERT_FFN_WIDTH_FIELDS: tuple[str, ...] = (
    "moe_intermediate_size",
    "expert_intermediate_size",
    "moe_ffn_hidden_size",
    "intermediate_size",  # GptOss: the expert FFN is the only FFN, so this is the expert width
)


def register_legacy_per_layer_config_keys(model_types: tuple[str, ...], keys: dict[str, tuple[str, str]]) -> None:
    """Claim ``keys`` for every ``model_type`` in ``model_types`` (a family's own spellings).

    A spelling two families claim with different keys raises, as the EP class-claim maps do;
    otherwise the export would flatten by whichever class was defined last.
    """
    for model_type in model_types:
        existing = _LEGACY_PER_LAYER_CONFIG_KEYS_BY_MODEL_TYPE.setdefault(model_type, dict(keys))
        if existing != keys:
            raise ValueError(
                f"model_type {model_type!r} declares two different _LEGACY_PER_LAYER_CONFIG_KEYS: "
                f"{existing} vs {dict(keys)}"
            )


def legacy_per_layer_config_keys(model_type: str) -> dict[str, tuple[str, str]]:
    """The flat legacy keys a family declares for ``model_type``; ``{}`` when no family claims it or
    the family declares none, so the config-export rewrite is a no-op there."""
    return dict(_LEGACY_PER_LAYER_CONFIG_KEYS_BY_MODEL_TYPE.get(model_type, {}))


def register_source_config_schema(model_types: tuple[str, ...]) -> None:
    """Claim ``model_types`` as families whose exports carry the source repo's config schema."""
    _SOURCE_CONFIG_SCHEMA_MODEL_TYPES.update(model_types)


def register_native_balancing_slot(model_types: tuple[str, ...], slot: NativeBalancingSlot) -> None:
    """Claim ``slot`` for every ``model_type`` in ``model_types`` (a family's own spellings).

    Two families claiming one spelling with different slots raises, as the registrars above do;
    otherwise an offline export would write the bias through whichever class was defined last.
    """
    for model_type in model_types:
        existing = _NATIVE_BALANCING_SLOTS_BY_MODEL_TYPE.setdefault(model_type, slot)
        if existing != slot:
            raise ValueError(
                f"model_type {model_type!r} declares two different native balancing slots: {existing} vs {slot}"
            )
    _BALANCING_KEY_SUFFIXES.update(slot.attrs)


def native_balancing_hub_model_types() -> tuple[str, ...]:
    """``config.model_type`` spellings whose family declares a checkpoint-exported balancing slot.

    The set ``moe_balancing=bias_update`` stays available on, named by the refusal a slot-less family
    raises. A family joins by declaring ``_NATIVE_BALANCING_BIAS_ATTR`` and nothing else.
    """
    return tuple(sorted(_NATIVE_BALANCING_SLOTS_BY_MODEL_TYPE))


def register_ep_wrapped_model_types(model_types: tuple[str, ...]) -> None:
    """Claim ``model_types`` as families an EP MoE wrapper class exists for."""
    _EP_WRAPPED_MODEL_TYPES.update(model_types)


def ep_roster_registered() -> bool:
    """Whether the EP layer package has registered its families into the maps above.

    They fill at EP subclass definition, so a process that reaches a reader here without importing
    ``src.distributed.expert_parallel.layers.roster`` would see an empty roster and every family
    would answer "needs nothing". Readers refuse that state rather than answering from it.
    """
    return bool(_EP_WRAPPED_MODEL_TYPES)


def has_ep_wrapper_class(model_config) -> bool:
    """Whether an EP MoE wrapper class exists for this family, i.e. whether patching wraps anything.

    ``ParallelismConfig.needs_ep_wrappers`` states the run's intent (EP distribution or grouped
    GEMM); only the roster says whether a wrapper is available for this family. Qwen3-Next has none,
    so patching leaves the stock expert loop in place. Both the wrapper's own ``model_type`` and its
    text sub-config's are checked, which is where a multimodal checkpoint declares the MoE family.
    """
    if not ep_roster_registered():
        raise RuntimeError(
            "has_ep_wrapper_class reached with no EP family registered in this process: the roster "
            "fills by importing src.distributed.expert_parallel.layers.roster, and an empty one "
            "reports every family as un-wrapped."
        )
    return any(
        getattr(source, "model_type", None) in _EP_WRAPPED_MODEL_TYPES for source in config_sources(model_config)
    )


def ep_wraps_experts(needs_ep_wrappers: bool, model_config) -> bool:
    """Whether this run will actually replace the family's expert FFN with an EP wrapper.

    ``needs_ep_wrappers`` is :attr:`ParallelismConfig.needs_ep_wrappers`, passed as the flag rather
    than the config because the Liger load-time path is handed the intent as a primitive.
    """
    return needs_ep_wrappers and config_has_experts(model_config) and has_ep_wrapper_class(model_config)


def exports_source_config_schema(model_type: str) -> bool:
    """Whether ``model_type``'s exported ``config.json`` must carry the source repo's own schema.
    False for every family transformers serializes into a schema its serving engines already read."""
    return model_type in _SOURCE_CONFIG_SCHEMA_MODEL_TYPES


def get_first_router_field(cfg, fields: tuple[str, ...], default=None):
    """First set value among ``fields``, searched over the router config sources.

    The registries above list one concept's alternative spellings, so a consumer wants "whichever of
    these this family uses" rather than one named field.
    """
    for field in fields:
        value = get_config_field(cfg, field)
        if value:
            return value
    return default


def resolve_expert_ffn_shard_width(cfg) -> int | None:
    """The per-expert intermediate dimension expert-TP splits, or ``None`` if no spelling is set.

    Shared so the config-time divisibility gate and the layer that does the split cannot disagree on
    which width ETP shards.
    """
    value = get_first_router_field(cfg, EXPERT_FFN_WIDTH_FIELDS)
    return int(value) if value else None


def resolve_router_topk(cfg) -> int:
    """Experts routed per token, from the first :data:`ROUTER_TOPK_FIELDS` spelling any source
    carries (``0`` when none does).

    First source rather than the widest value across sources: that keeps this probe's answer equal to
    the one the expert count and the balancing wiring read, where a maximum could hand one consumer a
    width no other agrees with.
    """
    return int(get_first_router_field(cfg, ROUTER_TOPK_FIELDS, 0))


def detect_moe_experts_topk(model) -> tuple[int, int]:
    """``(num_experts, top_k)`` off ``model.config`` — ``(0, 0)`` when it declares neither.

    The shared "is this MoE, and how wide is its router" probe. Both halves resolve a composite
    config's text sub-config too: ``PreTrainedConfig`` delegates no attribute reads, so a
    wrapper-only read reports a MoE checkpoint as dense.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return 0, 0
    return int(get_first_router_field(cfg, ROUTER_EXPERT_COUNT_FIELDS, 0)), resolve_router_topk(cfg)


def config_has_experts(config) -> bool:
    """Whether a config describes a MoE model (``num_local_experts`` / ``num_experts`` at either level).

    Used by the loader gate and the trainer. The lookup is the same router-field resolver
    ``ParallelismConfig`` asks the question with, so the loader gate and the config-time gate cannot
    answer it differently for one checkpoint.
    """
    if config is None:
        return False
    return bool(get_first_router_field(config, ROUTER_EXPERT_COUNT_FIELDS))


def mark_router_logits_forced_off(config) -> None:
    """Record on the config that ``output_router_logits`` must not be (re-)enabled.

    Stamped on the text sub-config too, so a consumer holding either object sees the same verdict.
    Run-scoped: the stamp configures this run and is restored before config serialization, since an
    exported stamp would suppress ``moe/*`` metrics in every later run loading the artifact.
    """
    set_config_field_run_scoped(config, ROUTER_LOGITS_FORCED_OFF_ATTR, True)


def router_logits_forced_off(config) -> bool:
    """Whether :func:`mark_router_logits_forced_off` stamped this config."""
    return bool(get_config_field(config, ROUTER_LOGITS_FORCED_OFF_ATTR, False))


def resolve_balancing_slot(
    module: torch.nn.Module, path: str | None, *, require_tensor: bool = True
) -> tuple[torch.nn.Module, str] | None:
    """Resolve a dotted native-slot path on ``module`` to ``(owner_module, attr_name)``.

    Shared by the EP layer that adopts the slot, the checkpoint-key derivation and the sidecar apply.
    ``None`` when the path is empty, the owner submodule is absent, or (under ``require_tensor``) the
    slot holds no tensor (LFM-2 with ``use_expert_bias: false``); the materializing caller passes
    ``require_tensor=False`` precisely because the slot is missing.
    """
    if not path:
        return None
    owner_path, _, name = path.rpartition(".")
    try:
        owner = module.get_submodule(owner_path) if owner_path else module
    except AttributeError:  # the owner module itself is absent: same verdict as an absent slot
        return None
    if require_tensor and not isinstance(getattr(owner, name, None), torch.Tensor):
        return None
    return owner, name


def native_balancing_bias_attrs(layer_cls) -> tuple[str, ...]:
    """The declared ``_NATIVE_BALANCING_BIAS_ATTR`` plus its hub respelling (via the class's
    ``_EXPORT_KEY_RENAMES``, Laguna), for one EP layer class; empty when it declares no slot.

    Every consumer naming the slot off a class (the dtype keep-set, the sidecar apply on a hub tree)
    must walk both spellings or miss the one the EP gather emits.
    """
    attr = getattr(layer_cls, "_NATIVE_BALANCING_BIAS_ATTR", None)
    if not attr:
        return ()
    candidates = [attr]
    # Same match semantics as the EP gather's ``to_hub_layer_key`` (first substring hit, replaced
    # once): the two must name the same hub key, or the dtype keep-set and the sidecar apply miss the
    # spelling the gather emits.
    for live, hub in getattr(layer_cls, "_EXPORT_KEY_RENAMES", ()):
        if live in attr:
            candidates.append(attr.replace(live, hub, 1))
            break
    return tuple(candidates)


def balancing_param_keys(model: torch.nn.Module) -> frozenset[str]:
    """State-dict keys of the live router-balancing tensors, in both live and hub spellings.

    Derived from the module tree like ``norm_param_keys`` rather than from a name list: an adopted
    native slot contributes its layer class's :func:`native_balancing_bias_attrs`, a hub-native
    router its registered ``balancing_biases`` buffer. Save paths keep these keys at their trained
    dtype, since a bf16 round-trip would quantize away several 1e-3 sign-steps of trained balancing.
    """
    keys: set[str] = set()
    for name, module in model.named_modules():
        prefix = f"{name}." if name else ""
        if getattr(module, NATIVE_BALANCING_BIAS_ADOPTED_ATTR, False):
            keys.update(f"{prefix}{attr}" for attr in native_balancing_bias_attrs(type(module)))
        elif any(buf_name == BALANCING_BIASES_ATTR for buf_name, _ in module.named_buffers(recurse=False)):
            keys.add(f"{prefix}{BALANCING_BIASES_ATTR}")
    return frozenset(keys)


def is_balancing_state_key(key: str) -> bool:
    """Whether a checkpoint key names a router-balancing tensor: the model-free counterpart of
    :func:`balancing_param_keys` for readers holding only a state dict (the merge scripts), so a
    merged-from-shards save keeps the same tensors at trained dtype as the direct gathered save."""
    return any(key == suffix or key.endswith(f".{suffix}") for suffix in _BALANCING_KEY_SUFFIXES)


def _model_balancing_slot(model: torch.nn.Module) -> NativeBalancingSlot | None:
    """The balancing slot declared for this model's family, or ``None`` when it declares none.

    Resolved off the config's own ``model_type`` levels (wrapper before text sub-config, which is
    where a multimodal checkpoint declares its MoE family), so the answer does not depend on the
    order the roster happened to register in.
    """
    cfg = getattr(unwrap_framework_wrappers(model), "config", None)
    if cfg is None:
        return None
    for source in config_sources(cfg):
        slot = _NATIVE_BALANCING_SLOTS_BY_MODEL_TYPE.get(getattr(source, "model_type", None) or "")
        if slot is not None:
            return slot
    return None


def _native_balancing_tensor(module: torch.nn.Module, slot: NativeBalancingSlot | None) -> torch.Tensor | None:
    """The hub module's own balancing tensor, or None where the architecture has no slot.

    A hub-native ``balancing_biases`` buffer (Zaya) wins; otherwise the family's declared paths,
    including the hub respelling, since Laguna's hub tree differs from the EP wrapper's.
    """
    native = dict(module.named_buffers(recurse=False)).get(BALANCING_BIASES_ATTR)
    if native is not None:
        return native
    for path in slot.attrs if slot else ():
        target = resolve_balancing_slot(module, path)
        if target is not None:
            return getattr(*target)
    return None


def _materialize_hub_balancing_slot(
    model: torch.nn.Module, module: torch.nn.Module, slot: NativeBalancingSlot, bias: torch.Tensor
) -> torch.Tensor | None:
    """Create a config-gated family's native slot on a HUB module assembled without it.

    A PEFT run saves adapters only, with no ``config.json``, so a merge starting from base weights
    rebuilds the tree with the slot's config gate still off (LFM-2 ``use_expert_bias: false``) even
    though the run materialized and trained the slot. Re-materialize it here (zeros, immediately
    overwritten by the sidecar copy), flip the gate on the module tree so live routing consults it,
    and mirror the gate into the merged model's config so the export ships both tensor and flag.
    ``None`` for families without a config-gated slot, whose absence is structural rather than an
    artifact of assembly.
    """
    if not slot.config_flag:
        return None
    for path in slot.attrs:
        target = resolve_balancing_slot(module, path, require_tensor=False)
        if target is None:
            continue
        owner, name = target
        device = next((p.device for p in module.parameters()), torch.device("cpu"))
        owner.register_buffer(name, torch.zeros_like(bias, dtype=torch.float32, device=device))
        for sub in module.modules():
            if hasattr(sub, slot.config_flag):
                setattr(sub, slot.config_flag, True)
        cfg = getattr(unwrap_framework_wrappers(model), "config", None)
        if cfg is not None:
            set_config_field(cfg, slot.config_flag, True, only_declared=False)
        return getattr(owner, name)
    return None


def apply_router_balancing_sidecar(
    model: torch.nn.Module, sidecar: dict[str, torch.Tensor]
) -> tuple[list[str], list[str]]:
    """Copy a run's trained balancing biases (``router_balancing_biases.pt``) into a HUB model's own
    balancing slots, returning ``(applied, skipped)`` module names.

    An offline-assembled artifact (a PEFT merge) starts from base weights, so the bias the run
    trained lives only in the sidecar. Sidecar keys are trainer-tree module names, mapped back to the
    hub tree by :func:`~src.models.structure.unwrapped_module_name` (PEFT and CP each add a level). A
    config-gated slot the base was assembled without is re-materialized (with its config flag
    flipped) before the copy. Entries for families with no native slot at all trained a transient
    bias no artifact can serve, and are returned as skipped.
    """
    slot = _model_balancing_slot(model)
    applied: list[str] = []
    skipped: list[str] = []
    for name, bias in sidecar.items():
        target_name = unwrapped_module_name(name)
        try:
            module = model.get_submodule(target_name)
        except AttributeError:
            raise KeyError(
                f"router_balancing_biases.pt names module {name!r}, which does not exist in "
                f"{type(model).__name__} — the sidecar belongs to a different model tree."
            ) from None
        target = _native_balancing_tensor(module, slot)
        if target is None and slot is not None:
            target = _materialize_hub_balancing_slot(model, module, slot, bias)
        if target is None:
            skipped.append(target_name)
            continue
        if target.shape != bias.shape:
            raise ValueError(
                f"router_balancing_biases.pt entry {name!r} has shape {tuple(bias.shape)} but the "
                f"model's balancing slot is {tuple(target.shape)} — refusing a corrupt pairing."
            )
        target.data.copy_(bias.to(device=target.device, dtype=target.dtype))
        applied.append(target_name)
    return applied, skipped


def iter_balancing_routers(model: torch.nn.Module):
    """Yield every router with both a ``balancing_biases`` buffer and an ``expert_load_counter`` slot
    (set or unset).

    Both are required so the yield stays unique when a wrapper adopts an inner gate's native buffer:
    only the module that also records loads is the balancing router."""
    for module in model.modules():
        if hasattr(module, BALANCING_BIASES_ATTR) and hasattr(module, "expert_load_counter"):
            yield module


def has_discard_expert_slot(model) -> bool:
    """Whether the model's router reserves a trailing discard/null expert slot.

    Declared by the router class (``_has_discard_expert_slot``), the same shape as
    :func:`ep_severs_aux_loss`, so a family opts in where its routing is defined, including the
    hub-native modelings the toolkit patches at load (Zaya). The slot must never receive tokens, so
    the balancing update excludes it and the load metrics drop its column.
    """
    return any(getattr(type(module), "_has_discard_expert_slot", False) for module in model.modules())


def has_balancing_routers(model) -> bool:
    """Whether any router already carries the bias-update state (adopted native slot, side-buffer,
    or a hub-native ``balancing_biases`` the load recording patched in)."""
    return any(True for _ in iter_balancing_routers(model))


def is_transient_balancing_router(module) -> bool:
    """Whether this balancing router holds its bias as the transient side-buffer.

    The side-buffer is a plain instance attribute (set via the EP base's ``balancing_biases``
    property), so it lives in ``__dict__``, invisible to ``state_dict()``, the gathered save and the
    weight sync. A native adoption reads through the property (nothing in ``__dict__``) and a hub
    router's own buffer lives in ``_buffers``, so both report False.
    """
    return BALANCING_BIASES_ATTR in getattr(module, "__dict__", {})


def accepts_bias_balancing(model) -> bool:
    """Whether any EP MoE wrapper on this model can carry a bias-update balancing buffer."""
    return any(getattr(type(module), "_supports_bias_balancing", False) for module in model.modules())


def accepts_native_balancing_bias(model) -> bool:
    """Whether some module would carry the bias-update state in checkpoint-exported tensors.

    Duck-typed on the EP base's ``can_adopt_native_balancing`` (a declared native slot that exists,
    or one the layer can materialize, as for LFM-2 with ``use_expert_bias: false``). A model where
    this is False can only be bias-balanced transiently, and the trained bias would never reach a
    served copy.
    """
    for module in model.modules():
        probe = getattr(module, "can_adopt_native_balancing", None)
        if callable(probe) and probe():
            return True
    return False


def ep_severs_aux_loss(model) -> bool:
    """Whether the model's EP MoE wrappers sever the HF aux-loss path and support bias balancing.

    Duck-typed on the EP layer class attributes (``_ep_severs_aux_loss`` + ``_supports_bias_balancing``,
    see ``EPMoELayerBase``): such wrappers bypass the HF router module, so ``outputs.router_logits``
    stays empty and enabling ``output_router_logits`` would crash the aux-loss reduction (DeepSeek-V4
    under EP)."""
    return any(
        getattr(type(module), "_ep_severs_aux_loss", False)
        and getattr(type(module), "_supports_bias_balancing", False)
        for module in model.modules()
    )


def honors_output_router_logits_config(model) -> bool:
    """Whether setting ``config.output_router_logits`` actually reaches this model's aux-loss term.

    HF resolves a config-backed forward flag on an explicit parameter, so a class that does not
    declare it never consults the config: ``Qwen3_5MoeForConditionalGeneration`` reads it from
    ``kwargs`` only, and the flag then pays a ``[tokens, num_experts]`` plane per MoE layer while the
    aux loss never reaches the loss, so the balancing has no effect.
    """
    # A PEFT wrapper forwards **kwargs into the base model and shares its config, so probing the
    # wrapper's own signature would report False for a base that does honour the flag.
    base = getattr(model, "get_base_model", None)
    if callable(base):
        model = base()
    forward = getattr(type(model), "forward", None)
    if forward is None:
        return False
    try:
        return "output_router_logits" in inspect.signature(forward).parameters
    except (TypeError, ValueError):  # C-implemented or otherwise un-introspectable forward
        return False


def resolve_balancing_mode(requested: str, model, is_moe: bool) -> BalancingMode:
    """Validate ``requested`` and resolve ``auto`` based on the model.

    ``auto`` selects ``bias_update`` on any of three signals: a router exposing a
    ``balancing_biases`` buffer (Zaya); EP wrappers that sever the aux-loss path while supporting the
    bias (DeepSeek-V4); or a forward that never declares ``output_router_logits`` while some wrapper
    accepts the bias. It commits only where the bias lands in checkpoint-exported state
    (:func:`accepts_native_balancing_bias`), since a served copy must route the way the trainer did.
    Where only a transient bias is possible it resolves to ``none``, and the explicit
    ``bias_update_transient`` is the opt-in.

    ``aux_loss`` for other MoE families, but only where the enabled tree can serve it (the forward
    honours the flag this mode sets). With no bias acceptor either, the model has no balancing route
    at all and ``auto`` resolves to ``none`` with the reason; an explicit ``aux_loss`` raises there.
    ``none`` for dense.
    """
    if requested not in _VALID_BALANCING_MODES:
        raise ValueError(f"moe_balancing must be one of {_VALID_BALANCING_MODES}, got {requested!r}")
    if requested != "auto":
        return requested  # type: ignore[return-value]
    if not is_moe:
        return "none"
    has_bias_routers = has_balancing_routers(model)
    honors_aux_loss = honors_output_router_logits_config(model)
    wants_bias_update = (
        has_bias_routers or ep_severs_aux_loss(model) or (not honors_aux_loss and accepts_bias_balancing(model))
    )
    if wants_bias_update:
        if has_bias_routers or accepts_native_balancing_bias(model):
            return "bias_update"
        warn_once(
            logger,
            _WARNED_TRANSIENT_ONLY_AUTO,
            type(model).__name__,
            f"moe_balancing=auto resolves to none on {type(model).__name__}: only the bias update "
            f"could balance it, but this architecture has no checkpoint slot for a routing bias, so "
            f"the trained bias would never reach a served copy — THIS RUN TRAINS UNBALANCED. Set "
            f"moe_balancing=bias_update_transient to balance during training while accepting that "
            f"every exported checkpoint serves WITHOUT the bias (near-tied top-k picks flip between "
            f"trainer and server).",
        )
        return "none"
    if honors_aux_loss:
        return "aux_loss"
    warn_once(
        logger,
        _WARNED_UNSERVABLE_AUTO,
        type(model).__name__,
        f"moe_balancing=auto resolves to none on {type(model).__name__}: its forward does not take "
        f"output_router_logits, so the aux-loss term never reaches the loss, and nothing on this "
        f"tree carries a routing bias (no EP MoE wrapper, no native balancing_biases router) — THIS "
        f"RUN TRAINS UNBALANCED, a real cost at this expert count. Launch under torchrun with "
        f"use_grouped_gemm: true or expert parallelism, where auto resolves to the bias update for "
        f"families whose bias exports, or load the text-only sibling class whose forward honours the "
        f"flag.",
    )
    return "none"
