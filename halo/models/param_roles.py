"""Parameter roles read off a live model. Rank-local."""

from collections.abc import Iterable

import torch
from transformers.core_model_loading import build_glob_alternation

# ``score`` is the classification/reward head. The fallback where a model's embedding accessors are unavailable.
EMBEDDING_HEAD_MARKERS: tuple[str, ...] = (
    "embed_tokens",
    "lm_head",
    "embeddings",
    "word_embeddings",
    "wte",
    "wpe",
    "score",
)

FP32_INTERLAYER_RESIDUAL_ATTR = "_fp32_interlayer_residual"


def persistent_buffers(model, exclude_prefixes=()):
    """Read from ``_non_persistent_buffers_set``, as diffing ``model.state_dict()`` reshards the calling rank's
    parameters under FSDP2 and would put a save-rank-only caller out of step with its peers."""
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
    """By class name too, as transformers' per-model norm classes (``*RMSNorm`` / ``*LayerNorm``) share no base."""
    norm_bases = (
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.LocalResponseNorm,
        torch.nn.RMSNorm,
        torch.nn.modules.batchnorm._NormBase,
    )
    return isinstance(module, norm_bases) or "norm" in type(module).__name__.lower()


def norm_param_keys(model: torch.nn.Module) -> frozenset[str]:
    """Read off the module tree rather than key spellings; they keep their trained dtype on save."""
    keys: set[str] = set()
    for mod_name, module in model.named_modules():
        if not is_normalization_module(module):
            continue
        prefix = f"{mod_name}." if mod_name else ""
        keys.update(f"{prefix}{name}" for name, _ in module.named_parameters(recurse=False))
        keys.update(f"{prefix}{name}" for name, _ in module.named_buffers(recurse=False))
    return frozenset(keys)


def fp32_pin_matches(names: Iterable[str], pins: Iterable[str]) -> frozenset[str]:
    """By transformers' loader rule: each entry is a glob searched anywhere in the name."""
    pins = sorted(pins)
    if not pins:
        return frozenset()
    pattern, _, _ = build_glob_alternation(pins)
    return frozenset(name for name in names if pattern.search(name))


def fp32_pinned_names(model: torch.nn.Module, *, include_buffers: bool = False) -> frozenset[str]:
    """Read off every class in the tree, so a wrapper derives the same set as the model; a pin can name a buffer (GLM-5
    Next's ``e_score_correction_bias``)."""
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
