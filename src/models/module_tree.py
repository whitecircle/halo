"""Module-tree lookups on a live model: the decoder-layer list, the layer index and stack prefix a
dotted name carries, the backbone owning the list, the module whose forward consumes the input
embedding, the embedding modules, and module paths found by identity. Rank-local.
"""

import re
from collections.abc import Sequence
from typing import Literal

import torch

from src.models.unwrapping import base_transformers_model

# Repo-wide rather than per-family: a backbone spelling missing here is invisible to every consumer.
DECODER_LAYER_LIST_ATTRS: tuple[str, ...] = ("layers", "h")
# ``<attr>.<N>`` at a name boundary inside a dotted module or parameter name: the decoder-layer index
# every consumer numbers blocks by (``xlayers.3`` and ``layers.3x`` do not match).
DECODER_LAYER_INDEX = re.compile(rf"(?<![^.])(?:{'|'.join(map(re.escape, DECODER_LAYER_LIST_ATTRS))})\.(\d+)(?![^.])")


def find_attr_name(module: torch.nn.Module, candidates: Sequence[str]) -> str | None:
    """The first attribute in ``candidates`` that ``module`` carries (set and not None), or None."""
    return next((name for name in candidates if getattr(module, name, None) is not None), None)


def decoder_layers_attr(backbone: torch.nn.Module) -> str:
    """The attribute holding ``backbone``'s decoder-layer list (:data:`DECODER_LAYER_LIST_ATTRS`)."""
    attr = find_attr_name(backbone, DECODER_LAYER_LIST_ATTRS)
    if attr is None:
        raise ValueError(
            f"{type(backbone).__name__} exposes no decoder-layer list under any of {DECODER_LAYER_LIST_ATTRS}."
        )
    return attr


def decoder_layers(module: torch.nn.Module) -> torch.nn.ModuleList | None:
    """``module``'s own decoder-layer list (``.layers``, or GPT-2-style ``.h``), or None if it has none."""
    attr = find_attr_name(module, DECODER_LAYER_LIST_ATTRS)
    return getattr(module, attr) if attr is not None else None


def decoder_layer_index(name: str, backbone_prefix: str | None = None) -> int | None:
    """The decoder-layer index in ``name`` (``model.layers.12.mlp`` → 12), or None outside a layer or,
    given ``backbone_prefix``, outside that backbone."""
    if backbone_prefix is not None:
        if not name.startswith(f"{backbone_prefix}."):
            return None
        name = name[len(backbone_prefix) + 1 :]
    match = DECODER_LAYER_INDEX.search(name)
    return int(match.group(1)) if match else None


def decoder_layer_stack_prefix(name: str) -> str | None:
    """Everything in ``name`` before its ``layers.<N>``: the stack :func:`decoder_layer_index` counts in.

    Numbering a name is unambiguous only while one stack is in play; a caller with no live module to
    resolve a backbone prefix from checks that its names share one stack prefix instead.
    """
    match = DECODER_LAYER_INDEX.search(name)
    return name[: match.start()] if match else None


def backbone_with_layers(model: torch.nn.Module) -> torch.nn.Module | None:
    """The module owning the decoder layer list (``.layers`` or ``.h``), or None if none is reachable.

    Walks down from ``model`` through ``language_model`` (read first: a multimodal wrapper also has
    ``.model``), else ``.model``, else ``.transformer``. Framework and toolkit wrappers are not peeled;
    call :func:`~src.models.unwrapping.unwrap_toolkit_wrappers` first when the input may be wrapped.
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

    Usually :func:`backbone_with_layers`; a multimodal composite (``model.model`` with the decoder-layer
    backbone nested below it) builds ``inputs_embeds`` in its own forward instead. Tested by ancestry,
    so a text stack nested deeper resolves the same way, on the plain transformers tree.
    """
    layer_backbone = backbone_with_layers(model)
    if layer_backbone is None:
        return None
    composite = getattr(base_transformers_model(model), "model", None)
    if composite is None or composite is layer_backbone:
        return layer_backbone
    return composite if any(module is layer_backbone for module in composite.modules()) else layer_backbone


def find_module_path(model: torch.nn.Module, module: torch.nn.Module) -> str | None:
    """Dotted path of ``module`` inside ``model``, matched by identity, or None outside its tree."""
    return next((name for name, candidate in model.named_modules() if candidate is module), None)


def module_path(model: torch.nn.Module, module: torch.nn.Module) -> str:
    """Dotted path of ``module`` inside ``model``, matched by identity; raises ValueError outside its
    tree, where a lenient miss would surface as a bare ``StopIteration`` or an unfiltered key set."""
    path = find_module_path(model, module)
    if path is None:
        raise ValueError(
            f"{type(module).__name__} is not a module of {type(model).__name__}, so it has no path in that tree."
        )
    return path


def split_layer_path(name: str, layer_root: str) -> tuple[int, str] | None:
    """``(layer index, remainder)`` of a ``<layer_root><index>[.<rest>]`` path, else ``None``."""
    if not name.startswith(layer_root):
        return None
    index, _, rest = name[len(layer_root) :].partition(".")
    return (int(index), rest) if index.isdigit() else None


def embedding_module(model: torch.nn.Module, which: Literal["input", "output"]) -> torch.nn.Module | None:
    """``model.get_{which}_embeddings()`` when the model implements the getter, else None."""
    getter = getattr(model, f"get_{which}_embeddings", None)
    return getter() if callable(getter) else None
