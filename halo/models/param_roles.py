"""Parameter roles read off a live model: the embedding/head name markers, normalization modules,
persistent buffers and the fp32 pins its classes declare. Rank-local.
"""

from collections.abc import Iterable

import torch
from transformers.core_model_loading import build_glob_alternation

# Parameter-name substrings marking a vocab-indexed embedding or output head (``score`` is the
# classification/reward head): the fallback where a model's embedding accessors are unavailable.
EMBEDDING_HEAD_MARKERS: tuple[str, ...] = (
    "embed_tokens",
    "lm_head",
    "embeddings",
    "word_embeddings",
    "wte",
    "wpe",
    "score",
)

# The class attribute a decoder layer sets when it carries its inter-layer residual in fp32.
FP32_INTERLAYER_RESIDUAL_ATTR = "_fp32_interlayer_residual"


def persistent_buffers(model, exclude_prefixes=()):
    """Yield ``(name, buffer)`` for persistent buffers (Gemma4 ``layer_scalar``), skipping
    ``exclude_prefixes``. Read from each module's ``_non_persistent_buffers_set`` rather than by
    diffing ``model.state_dict()``, which reshards the calling rank's parameters under FSDP2 and would
    put a save-rank-only caller out of step with its peers.
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
    """Whether ``module`` is a normalization layer: a torch norm base, else by class name, since
    transformers' per-model norm classes (``*RMSNorm`` / ``*LayerNorm``) share no common base."""
    norm_bases = (
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.LocalResponseNorm,
        torch.nn.RMSNorm,
        torch.nn.modules.batchnorm._NormBase,
    )
    return isinstance(module, norm_bases) or "norm" in type(module).__name__.lower()


def norm_param_keys(model: torch.nn.Module) -> frozenset[str]:
    """State-dict keys of params/buffers owned by the model's normalization modules, read off the
    module tree rather than key spellings; they keep their trained dtype on save."""
    keys: set[str] = set()
    for mod_name, module in model.named_modules():
        if not is_normalization_module(module):
            continue
        prefix = f"{mod_name}." if mod_name else ""
        keys.update(f"{prefix}{name}" for name, _ in module.named_parameters(recurse=False))
        keys.update(f"{prefix}{name}" for name, _ in module.named_buffers(recurse=False))
    return frozenset(keys)


def fp32_pin_matches(names: Iterable[str], pins: Iterable[str]) -> frozenset[str]:
    """The ``names`` the ``_keep_in_fp32_modules[_strict]`` entries ``pins`` match, by transformers' loader
    rule reused rather than restated: each entry is a glob searched anywhere in the name (its dtype
    plan's ``build_glob_alternation``)."""
    pins = sorted(pins)
    if not pins:
        return frozenset()
    pattern, _, _ = build_glob_alternation(pins)
    return frozenset(name for name in names if pattern.search(name))


def fp32_pinned_names(model: torch.nn.Module, *, include_buffers: bool = False) -> frozenset[str]:
    """Parameter names, plus buffer names with ``include_buffers``, that the model's classes pin in fp32
    via ``_keep_in_fp32_modules(_strict)``. The pins are read off every class in the tree, so a wrapper
    derives the same set as the model; a pin can name a buffer (GLM-5 Next's ``e_score_correction_bias``).
    """
    pins = {
        pin
        for cls in {type(module) for module in model.modules()}
        for attr in ("_keep_in_fp32_modules", "_keep_in_fp32_modules_strict")
        for pin in (getattr(cls, attr, None) or [])
    }
    names = [name for name, _ in model.named_parameters()]
    if include_buffers:
        names += [name for name, _ in model.named_buffers()]
    return fp32_pin_matches(names, pins)
