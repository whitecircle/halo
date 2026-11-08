"""Module-tree lookups on a live model. Rank-local."""

import re
from collections.abc import Sequence
from typing import Literal

import torch

from halo.models.introspection.unwrapping import base_transformers_model

# A backbone spelling missing here is invisible to every consumer.
DECODER_LAYER_LIST_ATTRS: tuple[str, ...] = ("layers", "h", "layer")
# ``<attr>.<N>`` at a name boundary (``xlayers.3`` and ``layers.3x`` do not match).
_DECODER_LAYER_INDEX = re.compile(rf"(?<![^.])(?:{'|'.join(map(re.escape, DECODER_LAYER_LIST_ATTRS))})\.(\d+)(?![^.])")
# A composite's towers and the projectors beside them (``model.visual``, ``model.vision_tower``, ``model.embed_vision``,
# ``model.audio_tower``, ``model.multi_modal_projector``), which only an image or audio row reaches.
_MEDIA_MODULE_NAME = re.compile(r"visual|vision_\w+|\w+_vision|audio_\w+|\w+_audio|multi_modal_projector|mm_projector")


def media_module_names(model: torch.nn.Module) -> list[str]:
    """The outermost parameter-bearing modules named as a composite's towers and projectors, by their full names."""
    names: list[str] = []
    for name, module in model.named_modules():
        leaf = name.rpartition(".")[2]
        nested = any(name.startswith(f"{outer}.") for outer in names)
        if not nested and _MEDIA_MODULE_NAME.fullmatch(leaf) and next(module.parameters(), None) is not None:
            names.append(name)
    return names


def find_attr_name(module: torch.nn.Module, candidates: Sequence[str]) -> str | None:
    return next((name for name in candidates if getattr(module, name, None) is not None), None)


def decoder_layers_attr(backbone: torch.nn.Module) -> str:
    attr = find_attr_name(backbone, DECODER_LAYER_LIST_ATTRS)
    if attr is None:
        raise ValueError(
            f"{type(backbone).__name__} exposes no decoder-layer list under any of {DECODER_LAYER_LIST_ATTRS}."
        )
    return attr


def decoder_layers(module: torch.nn.Module) -> torch.nn.ModuleList | None:
    attr = find_attr_name(module, DECODER_LAYER_LIST_ATTRS)
    return getattr(module, attr) if attr is not None else None


def decoder_layer_index(name: str, backbone_prefix: str | None = None) -> int | None:
    """``model.layers.12.mlp`` → 12; None outside a layer or, given ``backbone_prefix``, outside that backbone."""
    if backbone_prefix is not None:
        if not name.startswith(f"{backbone_prefix}."):
            return None
        name = name[len(backbone_prefix) + 1 :]
    match = _DECODER_LAYER_INDEX.search(name)
    return int(match.group(1)) if match else None


def decoder_layer_stack_prefix(name: str) -> str | None:
    """Everything before ``layers.<N>``. Numbering a name is unambiguous only while one stack is in play, so a caller
    with no live module to resolve a backbone prefix from checks that its names share one stack prefix."""
    match = _DECODER_LAYER_INDEX.search(name)
    return name[: match.start()] if match else None


def backbone_with_layers(model: torch.nn.Module) -> torch.nn.Module | None:
    """Walks through ``language_model`` (first, as a multimodal wrapper also has ``.model``), else ``.model``, else
    ``.transformer``. Wrappers are not peeled."""
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
    """Usually :func:`backbone_with_layers`; a multimodal composite builds ``inputs_embeds`` in its own forward. One
    holding ``language_model`` itself answers ``model``, the outermost module: the PEFT tuner above it calls its
    ``.forward()``, which runs no hook of its own."""
    layer_backbone = backbone_with_layers(model)
    if layer_backbone is None:
        return None
    root = base_transformers_model(model)
    composite = getattr(root, "model", None)
    if composite is None:
        return model if getattr(root, "language_model", None) is not None else layer_backbone
    if composite is layer_backbone:
        return layer_backbone
    return composite if any(module is layer_backbone for module in composite.modules()) else layer_backbone


def hidden_state_backbone(model: torch.nn.Module) -> torch.nn.Module | None:
    """The module whose ``last_hidden_state`` a model's head projects, for a path computing logits past the model's
    forward: ``base_model``, or under a composite holding ``language_model`` itself, whose ``base_model`` is the
    composite, the text tower's backbone. Wrappers are peeled."""
    base = base_transformers_model(model)
    backbone = base.base_model
    return backbone if backbone is not base else backbone_with_layers(base)


def find_module_path(model: torch.nn.Module, module: torch.nn.Module) -> str | None:
    return next((name for name, candidate in model.named_modules() if candidate is module), None)


def module_path(model: torch.nn.Module, module: torch.nn.Module) -> str:
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
    getter = getattr(model, f"get_{which}_embeddings", None)
    return getter() if callable(getter) else None
