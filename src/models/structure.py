"""Module-tree introspection for a live model: wrapper peeling, PEFT name normalization,
decoder-layer discovery, persistent buffers, and the name-based embedding/head and
normalization / fp32-pin classification.

Holds the rules that map a wrapped, sharded tree back to plain hub spellings, shared by the
FSDP2/TP/PP wraps, the attention patches and every checkpoint writer. Rank-local, except
:func:`merged_adapters`, whose in-place LoRA merge is a DTensor collective every rank must enter.
"""

import re
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from accelerate.utils import extract_model_from_parallel, is_peft_model

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

# The module level a ``PeftModel`` inserts above the base model, and the one
# ``UlyssesCPModelWrapper`` inserts (its ``_toolkit_inner_model_attr``). Shared by every consumer
# that maps a live path back to the hub tree; a second copy drifts into an unresolvable key.
_PEFT_BASE_MODEL_PREFIX = "base_model.model."
_CP_WRAPPER_MODULE_LEVEL = "model."


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


def resolve_tokenizer(processing_class):
    """Unwrap a VLM processor to its tokenizer; a plain tokenizer passes through."""
    return getattr(processing_class, "tokenizer", processing_class)


def model_has_quantized_params(model: torch.nn.Module) -> bool:
    """Whether ``model`` carries quantized (non-floating-point) params, i.e. bnb 4/8-bit packed
    storage (QLoRA). Shared by the FSDP2 routing in the trainer mixin and the GRPO weight-sync
    construction gate."""
    return any(not p.dtype.is_floating_point for p in model.parameters())


def unwrapped_module_name(name: str) -> str:
    """Map a module path read off a live wrapped tree to its plain hub-model spelling.

    Two toolkit-visible wrappers deepen every path beneath them: PEFT and the Ulysses CP wrapper,
    which ``unwrap_framework_wrappers`` keeps. Stripping only the PEFT prefix resolves nothing under
    PEFT+CP.
    """
    if name.startswith(f"{_CP_WRAPPER_MODULE_LEVEL}{_CP_WRAPPER_MODULE_LEVEL}"):
        name = name.removeprefix(_CP_WRAPPER_MODULE_LEVEL)
    else:
        cp_wrapped = f"{_PEFT_BASE_MODEL_PREFIX}{_CP_WRAPPER_MODULE_LEVEL}{_CP_WRAPPER_MODULE_LEVEL}"
        if name.startswith(cp_wrapped):
            name = name.replace(cp_wrapped, f"{_PEFT_BASE_MODEL_PREFIX}{_CP_WRAPPER_MODULE_LEVEL}", 1)
    return name.removeprefix(_PEFT_BASE_MODEL_PREFIX)


def normalize_peft_param_name(name: str, peft_prefix: str) -> str | None:
    """Map a ``PeftModel`` param name to its plain base-model name, or ``None`` to drop it.

    Shared by the vLLM weight sync and the merged EP save. Adapter params are dropped rather than
    renamed because both consumers run inside :func:`merged_adapters`: the delta is already folded
    into the base weight, so emitting ``lora_A``/``lora_B`` too would apply it a second time on any
    PEFT-aware reload.
    """
    if peft_prefix in name or "original_module" in name:
        return None
    name = name.removeprefix(_PEFT_BASE_MODEL_PREFIX).replace(".base_layer", "")
    return name.replace("modules_to_save.default.", "")


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


@contextmanager
def merged_adapters(model: torch.nn.Module | None) -> Iterator[bool]:
    """Fold LoRA into the base weights for the body, then unfold. Yields whether ``model`` is PEFT.

    ``merge_adapter`` is an in-place DTensor collective under FSDP2, so every rank must enter. The
    unmerge in the ``finally`` is what makes this usable mid-training: an intermediate merged save
    must leave the adapters trainable, unlike ``merge_and_unload``, which dissolves the PeftModel.
    ``None`` (no PEFT model in the tree) is a no-op, so callers can pass a lookup result directly.
    """
    peft = model is not None and is_peft_model(model)
    if peft:
        model.merge_adapter()
    try:
        yield peft
    finally:
        if peft:
            model.unmerge_adapter()


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


def fp32_pinned_param_names(model: torch.nn.Module) -> frozenset[str]:
    """Parameter names the model's classes pin in fp32 via ``_keep_in_fp32_modules(_strict)``.

    transformers' own substring rule against both class attributes, read off every class in the tree,
    so a wrapper (pipeline stage, CP wrapper, PEFT model) derives the same set as the model itself.
    The pin exists because the family's arithmetic breaks in bf16 (DeepSeek-V4's norms, Inkling's
    short convolutions), so every checkpoint writer leaves these at their trained dtype; a reload
    re-pinning the slot cannot recover precision an export already discarded.
    """
    pins = {
        pin
        for cls in {type(module) for module in model.modules()}
        for attr in ("_keep_in_fp32_modules", "_keep_in_fp32_modules_strict")
        for pin in (getattr(cls, attr, None) or [])
    }
    if not pins:
        return frozenset()
    pin_regex = re.compile("|".join(re.escape(pin) for pin in sorted(pins)))
    return frozenset(name for name, _ in model.named_parameters() if pin_regex.search(name))
