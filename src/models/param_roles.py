"""Parameter roles read off a live model: the embedding/head name markers, normalization modules,
persistent buffers and the fp32 pins its classes declare. Rank-local.
"""

from collections.abc import Iterable

import torch
from transformers.core_model_loading import build_glob_alternation

# HF per-model RMSNorm classes subclass none of these, so is_normalization_module falls back to name.
_TORCH_NORM_MODULE_BASES = (
    torch.nn.LayerNorm,
    torch.nn.GroupNorm,
    torch.nn.LocalResponseNorm,
    torch.nn.RMSNorm,
    torch.nn.modules.batchnorm._NormBase,
)

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
