"""Module-tree introspection for a live model: wrapper peeling, PEFT name normalization,
decoder-layer discovery, persistent buffers, and the name-based embedding/head and
normalization / fp32-pin classification.

Holds the rules that map a wrapped, sharded tree back to plain hub spellings, shared by the
FSDP2/TP/PP wraps, the attention patches and every checkpoint writer. Rank-local, except
:func:`lora_folded`, whose LoRA delta is a DTensor collective every rank must enter.
"""

import re
from collections.abc import Iterable

import torch
from accelerate.utils import extract_model_from_parallel, is_peft_model
from peft.tuners.lora.layer import Conv1d, Conv2d, Conv3d, Embedding, Linear, LoraLayer, ParamWrapper
from peft.tuners.lora.variants import (
    DoraConv1dVariant,
    DoraConv2dVariant,
    DoraConv3dVariant,
    DoraEmbeddingVariant,
    DoraLinearVariant,
)
from peft.tuners.tuners_utils import BaseTunerLayer
from transformers.core_model_loading import build_glob_alternation
from transformers.modeling_utils import PreTrainedModel

# HF per-model RMSNorm classes subclass none of these, so is_normalization_module falls back to name.
_TORCH_NORM_MODULE_BASES = (
    torch.nn.LayerNorm,
    torch.nn.GroupNorm,
    torch.nn.LocalResponseNorm,
    torch.nn.RMSNorm,
    torch.nn.modules.batchnorm._NormBase,
)

# Repo-wide rather than per-family: a backbone spelling missing here is invisible to every consumer.
DECODER_LAYER_LIST_ATTRS: tuple[str, ...] = ("layers", "h")
# ``<attr>.<N>`` at a name boundary inside a dotted module or parameter name: the decoder-layer index
# every consumer numbers blocks by (``xlayers.3`` and ``layers.3x`` do not match).
DECODER_LAYER_INDEX = re.compile(rf"(?<![^.])(?:{'|'.join(map(re.escape, DECODER_LAYER_LIST_ATTRS))})\.(\d+)(?![^.])")

# Parameter-name substrings marking a vocab-indexed embedding or output head (``score`` is the
# classification/reward head), read by Muon (sparsely-updated vocab rows stay on AdamW rather than
# Newton-Schulz) and by the PEFT bf16 cast. Both reach here only when the module accessors are
# unavailable, so a per-consumer copy would misroute a family's head.
EMBEDDING_HEAD_MARKERS: tuple[str, ...] = (
    "embed_tokens",
    "lm_head",
    "embeddings",
    "word_embeddings",
    "wte",
    "wpe",
    "score",
)

# The module level a ``PeftModel`` inserts above the base model, in live names and saved adapter
# keys alike, and the one ``UlyssesCPModelWrapper`` inserts (its ``_toolkit_inner_model_attr``).
# Shared by every consumer that maps such a path back to the hub tree; a second copy drifts into an
# unresolvable key.
PEFT_BASE_MODEL_PREFIX = "base_model.model."
_CP_WRAPPER_MODULE_LEVEL = "model."

# The model-level flags a quantized load sets: bitsandbytes sets the 8/4-bit pair, torchao/quanto only
# ``is_quantized``.
_KBIT_QUANTIZED_FLAGS = ("is_loaded_in_8bit", "is_loaded_in_4bit", "is_quantized")

# The PEFT layers whose merge rewrites one base tensor by ``+= get_delta_weight`` (or DoRA's
# ``merge_safe``), plus a ``lora_bias`` fold, which :func:`lora_folded` reproduces out of place. Exact
# types: a subclass (torchao, Intel Neural Compressor) overrides the merge.
_FOLDABLE_LORA_LAYERS = (Linear, Embedding, Conv1d, Conv2d, Conv3d, ParamWrapper)
_CONV_LORA_LAYERS = (Conv1d, Conv2d, Conv3d)
_FOLDABLE_VARIANTS = (DoraLinearVariant, DoraEmbeddingVariant, DoraConv1dVariant, DoraConv2dVariant, DoraConv3dVariant)

# ``id`` of each base tensor a LoRA merge rewrites → the layers folding into it (:func:`lora_fold_targets`).
LoraFolds = dict[int, list[LoraLayer]]


def unwrap_framework_wrappers(model: torch.nn.Module) -> torch.nn.Module:
    """Peel accelerate/DDP/FSDP **and** ``torch.compile``, leaving toolkit wrappers in place.

    ``keep_torch_compile=False`` is load-bearing: the ``OptimizedModule`` accelerate keeps by default
    names its parameters ``_orig_mod.*``, so a state dict taken through it writes a checkpoint
    ``from_pretrained`` and vLLM read zero tensors from, and an ``isinstance`` check for the CP
    wrapper fails through it. Neither raises.
    """
    return extract_model_from_parallel(model, recursive=True, keep_torch_compile=False)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Peel every framework and toolkit wrapper down to the inner model (a PeftModel deliberately stays).

    Toolkit wrappers declare their inner-model attribute as ``_toolkit_inner_model_attr``, so a new
    wrapper type joins by declaring it. A wrapper may yield inner names from ``named_parameters()``
    while ``named_modules()`` still yields wrapper-prefixed paths, so a consumer cross-referencing
    the two must walk this.
    """
    model = unwrap_framework_wrappers(model)
    # type() lookup, not getattr: a delegating wrapper's __getattr__ would report the inner answer.
    inner_attr = getattr(type(model), "_toolkit_inner_model_attr", None)
    while inner_attr is not None:
        model = unwrap_framework_wrappers(getattr(model, inner_attr))
        inner_attr = getattr(type(model), "_toolkit_inner_model_attr", None)
    return model


def base_transformers_model(model: torch.nn.Module) -> torch.nn.Module:
    """The plain transformers model under every framework, toolkit **and** PEFT wrapper.

    :func:`unwrap_model` deliberately stops at a ``PeftModel``; this goes the one level further, for
    consumers that need the tree as transformers laid it out. Load-bearing for any attribute probe:
    a wrapper answers ``.model`` with what *it* wraps — a ``PeftModel`` through its tuner's
    ``__getattr__``, a toolkit wrapper directly — so the same name read off the wrapper lands one
    level above the plain tree's.
    """
    base = unwrap_model(model)
    return unwrap_model(base.get_base_model()) if is_peft_model(base) else base


def transformers_model_class(model: torch.nn.Module) -> type[PreTrainedModel] | None:
    """The transformers class ``model`` was built as, or None for a non-transformers carrier.

    The first ``PreTrainedModel`` subclass in the MRO that is not torch's FSDP2 in-place class swap
    (``FSDP<Name>``, module ``torch.*``), so a sharded model resolves to its family's class.
    """
    return next(
        (
            cls
            for cls in type(model).__mro__
            if isinstance(cls, type)
            and issubclass(cls, PreTrainedModel)
            and cls is not PreTrainedModel
            and not cls.__module__.startswith("torch")
        ),
        None,
    )


def resolve_tokenizer(processing_class):
    """Unwrap a VLM processor to its tokenizer; a plain tokenizer passes through."""
    return getattr(processing_class, "tokenizer", processing_class)


def model_has_quantized_params(model: torch.nn.Module) -> bool:
    """Whether ``model`` carries quantized (non-floating-point) params, i.e. bnb 4/8-bit packed
    storage (QLoRA). Shared by the FSDP2 routing in the trainer mixin and the GRPO weight-sync
    construction gate."""
    return any(not p.dtype.is_floating_point for p in model.parameters())


def is_kbit_quantized(model: torch.nn.Module) -> bool:
    """Whether ``model`` was loaded quantized (any of :data:`_KBIT_QUANTIZED_FLAGS`), the condition
    ``prepare_peft_model`` runs peft's k-bit preparation on."""
    return any(getattr(model, flag, False) for flag in _KBIT_QUANTIZED_FLAGS)


def unwrapped_module_name(name: str) -> str:
    """Map a module path read off a live wrapped tree to its plain hub-model spelling.

    Two toolkit-visible wrappers deepen every path beneath them: PEFT and the Ulysses CP wrapper,
    which ``unwrap_framework_wrappers`` keeps. Stripping only the PEFT prefix resolves nothing under
    PEFT+CP.
    """
    if name.startswith(f"{_CP_WRAPPER_MODULE_LEVEL}{_CP_WRAPPER_MODULE_LEVEL}"):
        name = name.removeprefix(_CP_WRAPPER_MODULE_LEVEL)
    else:
        cp_wrapped = f"{PEFT_BASE_MODEL_PREFIX}{_CP_WRAPPER_MODULE_LEVEL}{_CP_WRAPPER_MODULE_LEVEL}"
        if name.startswith(cp_wrapped):
            name = name.replace(cp_wrapped, f"{PEFT_BASE_MODEL_PREFIX}{_CP_WRAPPER_MODULE_LEVEL}", 1)
    return name.removeprefix(PEFT_BASE_MODEL_PREFIX)


def normalize_peft_param_name(name: str, peft_prefix: str) -> str | None:
    """Map a ``PeftModel`` param name to its plain base-model name, or ``None`` to drop it.

    Shared by the vLLM weight sync and the merged EP save. Adapter params are dropped rather than
    renamed because both consumers write base weights with the delta already folded in
    (:func:`lora_folded`), so emitting ``lora_A``/``lora_B`` too would apply it a second time on any
    PEFT-aware reload.
    """
    if peft_prefix in name or "original_module" in name:
        return None
    name = strip_base_layer_segment(name.removeprefix(PEFT_BASE_MODEL_PREFIX))
    return name.replace("modules_to_save.default.", "")


def strip_base_layer_segment(name: str) -> str:
    """Drop the level a PEFT tuner layer inserts above the module it wraps: ``q_proj.base_layer.weight``
    becomes ``q_proj.weight``."""
    return name.replace(".base_layer", "")


def tuner_adapter_param_ids(model: torch.nn.Module) -> set[int]:
    """``id`` of every parameter a PEFT tuner layer in ``model`` holds beside the module it wraps: LoRA
    factors, embedding adapters, DoRA magnitudes.

    Structural, so a model's own parameters count as base weights whatever they are named (a
    remote-code backbone's native ``lora_A``), and rank-uniform under FSDP2.
    """
    adapters: set[int] = set()
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            base = {id(param) for param in module.get_base_layer().parameters()}
            adapters.update(id(param) for param in module.parameters() if id(param) not in base)
    return adapters


def strip_peft_adapter_segment(name: str) -> str:
    """Drop the ``default`` adapter-name segment from a live PEFT param name.

    ``...lora_A.default.weight`` becomes ``...lora_A.weight``, ``...modules_to_save.default.weight``
    becomes ``....weight``, and the ParameterDict spelling ``...lora_embedding_A.default`` matches
    what the adapter save writes, since a saved adapter file carries no adapter name. Every resume
    path rebuilding the saved-key to live-key map must strip identically, since unmatched saved keys
    are dropped as unexpected by both adapter loaders.
    """
    name = name.replace(".modules_to_save.default.", ".").replace(".default.", ".")
    return name.removesuffix(".default")


def _lora_weight(layer: LoraLayer) -> torch.nn.Parameter:
    """The base tensor ``layer``'s merge adds its delta to."""
    return layer.get_param() if isinstance(layer, ParamWrapper) else layer.get_base_layer().weight


def _adapters_to_fold(layer: LoraLayer) -> list[str]:
    """The adapters a merge of ``layer`` folds: active ones not merged already (PEFT's
    ``check_adapters_to_merge``), that this layer carries."""
    merged = set(layer.merged_adapters)
    return [
        adapter
        for adapter in layer.active_adapters
        if adapter not in merged and (adapter in layer.lora_A or adapter in layer.lora_embedding_A)
    ]


def _unfoldable(layer: BaseTunerLayer) -> str | None:
    """What makes ``layer``'s merge one :func:`lora_folded` cannot reproduce, or None."""
    if type(layer) not in _FOLDABLE_LORA_LAYERS:
        return f"tuner layer {type(layer).__name__}"
    if isinstance(layer, _CONV_LORA_LAYERS) and layer.get_base_layer().groups > 1:
        return f"grouped {type(layer).__name__}"
    variants = [layer.lora_variant[adapter] for adapter in _adapters_to_fold(layer) if adapter in layer.lora_variant]
    unknown = [variant for variant in variants if not isinstance(variant, _FOLDABLE_VARIANTS)]
    return f"{type(unknown[0]).__name__} adapter" if unknown else None


def lora_fold_targets(model: torch.nn.Module) -> LoraFolds:
    """``id`` of every base tensor PEFT's ``merge_adapter`` would rewrite → the LoRA layers folding
    into it, in merge order.

    Keyed by identity, so a tied weight that ``named_parameters()`` lists under its other name (a
    LoRA'd ``lm_head`` sharing ``embed_tokens``) still gets its delta. Raises on a tuner layer whose
    merge :func:`lora_folded` cannot reproduce (``nn.MultiheadAttention`` LoRA, trainable tokens,
    quantized LoRA, a variant other than DoRA, a grouped conv PEFT cannot merge either), rather than
    send it unfolded.
    """
    targets: LoraFolds = {}
    for name, module in model.named_modules():
        if not isinstance(module, BaseTunerLayer):
            continue
        unfoldable = _unfoldable(module)
        if unfoldable:
            raise ValueError(
                f"{name}: {unfoldable} cannot be folded out of place, which is implemented for plain and DoRA "
                f"adapters on {[cls.__name__ for cls in _FOLDABLE_LORA_LAYERS]} only, so its delta would not reach "
                f"the merged weights. Use a plain LoraConfig (use_dora allowed) on unquantized linear, "
                f"embedding or ungrouped conv modules in lora_target_modules."
            )
        adapters = _adapters_to_fold(module)
        if not adapters:
            continue
        targets.setdefault(id(_lora_weight(module)), []).append(module)
        bias = getattr(module.get_base_layer(), "bias", None)
        if bias is not None and any(module.lora_bias.get(adapter) for adapter in adapters):
            targets.setdefault(id(bias), []).append(module)
    return targets


@torch.no_grad()
def lora_folded(param: torch.nn.Parameter, layers: list[LoraLayer]) -> torch.Tensor:
    """A new tensor holding what PEFT's merge of ``layers`` would write into ``param``; ``param`` itself
    is not touched.

    The same arithmetic as the in-place merge (``+= get_delta_weight``, a variant's ``merge_safe``,
    ``lora_B.bias * scaling`` into a folded bias), on ``param``'s own placement: a DTensor folds shard
    by shard, the delta's redistribution a collective every rank must enter. Costs one copy of
    ``param``'s local shard plus the delta.
    """
    folded = param.detach().clone()
    for layer in layers:
        folds_weight = param is _lora_weight(layer)
        for adapter in _adapters_to_fold(layer):
            if not folds_weight:
                if layer.lora_bias.get(adapter):
                    folded += layer.lora_B[adapter].bias * layer.scaling[adapter]
            elif adapter in layer.lora_variant:
                # A variant caches what its unmerge needs; there is no unmerge here.
                cached = set(layer._caches)
                folded = layer.lora_variant[adapter].merge_safe(layer, adapter, folded)
                for key in set(layer._caches) - cached:
                    layer._cache_pop(key)
            else:
                delta = layer.get_delta_weight(adapter)
                # Linear adds the delta in its own dtype; the other layers cast it first, as their merges do.
                folded += delta if isinstance(layer, Linear) else delta.to(folded.dtype)
    return folded


def lora_folded_data(param: torch.nn.Parameter, folds: LoraFolds | None) -> torch.Tensor:
    """``param``'s data with its LoRA delta folded in out of place (:func:`lora_folded`) when ``folds``
    (:func:`lora_fold_targets`) names it, else the live data. Collective for a folded DTensor."""
    layers = folds.get(id(param)) if folds else None
    return lora_folded(param, layers) if layers else param.data


def decoder_layers(module: torch.nn.Module) -> torch.nn.ModuleList | None:
    """``module``'s own decoder-layer list (``.layers``, or GPT-2-style ``.h``), or None if it has none.

    The single probe for :data:`DECODER_LAYER_LIST_ATTRS`, so the backbone descent below, the
    pipeline splitter and the MFU layer count all resolve the layer attribute the same way.
    """
    for attr in DECODER_LAYER_LIST_ATTRS:
        layers = getattr(module, attr, None)
        if layers is not None:
            return layers
    return None


def decoder_layer_index(name: str) -> int | None:
    """The decoder-layer index in ``name`` (``model.layers.12.mlp`` → 12), or None outside a layer."""
    match = DECODER_LAYER_INDEX.search(name)
    return int(match.group(1)) if match else None


def backbone_with_layers(model: torch.nn.Module) -> torch.nn.Module | None:
    """The module owning the decoder layer list (``.layers`` or ``.h``), or None if none is reachable.

    One rule for FSDP/TP wrapping and the attention patches alike: ``model.model``, then
    ``language_model`` (multimodal wrappers, checked first because the parent also has ``.model``),
    then ``model.transformer``. Framework and toolkit wrappers are not peeled; call
    :func:`unwrap_model` first when the input may be wrapped.
    """
    candidate = model
    seen: set[int] = set()
    while candidate is not None and id(candidate) not in seen:
        seen.add(id(candidate))
        if decoder_layers(candidate) is not None:
            return candidate
        candidate = (
            getattr(candidate, "language_model", None)
            or getattr(candidate, "model", None)
            or getattr(candidate, "transformer", None)
        )
    return None


def input_embedding_backbone(model: torch.nn.Module) -> torch.nn.Module | None:
    """The module whose own forward consumes the input embedding, or None if none is reachable.

    Usually :func:`backbone_with_layers`, which owns ``embed_tokens`` and embeds the ids itself. A
    multimodal composite (``model.model`` holding the decoder-layer backbone below it, as a nested
    ``language_model``) instead builds ``inputs_embeds`` in its own forward and hands them down, so
    there the consumer is the composite and the backbone is entered only afterwards. The test is
    ancestry rather than the one-level ``language_model`` spelling — an ancestor's forward always
    precedes its descendant's — so a text stack nested deeper resolves the same way. Resolved on the
    plain transformers tree (:func:`base_transformers_model`), never through a wrapper's attribute
    forwarding.
    """
    layer_backbone = backbone_with_layers(model)
    if layer_backbone is None:
        return None
    composite = getattr(base_transformers_model(model), "model", None)
    if composite is None or composite is layer_backbone:
        return layer_backbone
    return composite if any(module is layer_backbone for module in composite.modules()) else layer_backbone


def persistent_buffers(model, exclude_prefixes=()):
    """Yield ``(name, buffer)`` for persistent buffers, skipping ``exclude_prefixes``.

    Persistent buffers (Gemma4 ``layer_scalar``) belong in a checkpoint, so a param-only save loop
    drops them and corrupts the reload. Persistence is read from each owning module's
    ``_non_persistent_buffers_set`` rather than by diffing ``model.state_dict()``, which reshards the
    calling rank's parameters under FSDP2; callers gate this on the save rank alone, which would then
    be out of step with its peers in the next collective over those params.
    """
    seen: set[int] = set()  # ``named_buffers()`` dedupes shared buffers by identity; match that
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        for local_name, buf in module.named_buffers(recurse=False):
            if local_name in module._non_persistent_buffers_set or id(buf) in seen:
                continue
            seen.add(id(buf))
            name = f"{prefix}{local_name}"
            if not any(name.startswith(p) for p in exclude_prefixes):
                yield name, buf


def is_normalization_module(module: torch.nn.Module) -> bool:
    """Whether ``module`` is a normalization layer.

    Structural first (isinstance against torch's norm bases). transformers' per-model norm classes
    share no common base, so class name is the remaining signal: every roster family names them
    ``*RMSNorm`` / ``*LayerNorm``, Zaya's EDA norm included.
    """
    if isinstance(module, _TORCH_NORM_MODULE_BASES):
        return True
    return "norm" in type(module).__name__.lower()


def norm_param_keys(model: torch.nn.Module) -> frozenset[str]:
    """State-dict keys of params/buffers owned by the model's normalization modules.

    Derived from the module tree (:func:`is_normalization_module`), not from key spellings, so a
    family with unconventional param paths is still classified correctly. These keys keep their
    trained dtype on save (fp32 norms survive an fp32-master run's checkpoint).
    """
    keys: set[str] = set()
    for mod_name, module in model.named_modules():
        if not is_normalization_module(module):
            continue
        prefix = f"{mod_name}." if mod_name else ""
        keys.update(f"{prefix}{name}" for name, _ in module.named_parameters(recurse=False))
        keys.update(f"{prefix}{name}" for name, _ in module.named_buffers(recurse=False))
    return frozenset(keys)


def _names_matching_fp32_pins(names: Iterable[str], pins: Iterable[str]) -> frozenset[str]:
    """transformers' loader rule, reused rather than restated: each ``_keep_in_fp32_modules[_strict]``
    entry is a glob searched anywhere in the name (its dtype plan's ``build_glob_alternation``)."""
    pins = sorted(pins)
    if not pins:
        return frozenset()
    pattern, _, _ = build_glob_alternation(pins)
    return frozenset(name for name in names if pattern.search(name))


def params_matching_fp32_pins(model: torch.nn.Module, pins: Iterable[str]) -> frozenset[str]:
    """Names of ``model``'s parameters that the fp32 pin entries ``pins`` match."""
    return _names_matching_fp32_pins((name for name, _ in model.named_parameters()), pins)


def _fp32_pins(model: torch.nn.Module) -> set[str]:
    """Both class attributes, read off every class in the tree, so a wrapper (pipeline stage, CP
    wrapper, PEFT model) derives the same pins as the model itself."""
    return {
        pin
        for cls in {type(module) for module in model.modules()}
        for attr in ("_keep_in_fp32_modules", "_keep_in_fp32_modules_strict")
        for pin in (getattr(cls, attr, None) or [])
    }


def fp32_pinned_param_names(model: torch.nn.Module) -> frozenset[str]:
    """Parameter names the model's classes pin in fp32 via ``_keep_in_fp32_modules(_strict)``.

    The training loaders cast these to the run dtype unless the run keeps fp32 masters, and every
    checkpoint writer leaves them at their trained dtype: a reload re-pinning the slot cannot recover
    precision an export already discarded.
    """
    return params_matching_fp32_pins(model, _fp32_pins(model))


def fp32_pinned_state_keys(model: torch.nn.Module) -> frozenset[str]:
    """:func:`fp32_pinned_param_names` plus the buffers the same pins match, the set a writer keeps.

    A pin can name a buffer (GLM-5 Next's ``e_score_correction_bias``): no loader casts it, so it
    stays fp32 live, and an export must not round it either.
    """
    names = [name for name, _ in model.named_parameters()] + [name for name, _ in model.named_buffers()]
    return _names_matching_fp32_pins(names, _fp32_pins(model))
