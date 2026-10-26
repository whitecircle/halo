"""Decoder-layer discovery on a live model: the layer list, the layer index a dotted name carries, the
backbone owning the list and the module whose forward consumes the input embedding. Rank-local.
"""

import re

import torch

from src.models.unwrapping import base_transformers_model

# Repo-wide rather than per-family: a backbone spelling missing here is invisible to every consumer.
DECODER_LAYER_LIST_ATTRS: tuple[str, ...] = ("layers", "h")
# ``<attr>.<N>`` at a name boundary inside a dotted module or parameter name: the decoder-layer index
# every consumer numbers blocks by (``xlayers.3`` and ``layers.3x`` do not match).
DECODER_LAYER_INDEX = re.compile(rf"(?<![^.])(?:{'|'.join(map(re.escape, DECODER_LAYER_LIST_ATTRS))})\.(\d+)(?![^.])")


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
    :func:`~src.models.unwrapping.unwrap_toolkit_wrappers` first when the input may be wrapped.
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
    plain transformers tree (:func:`~src.models.unwrapping.base_transformers_model`), never through a
    wrapper's attribute forwarding.
    """
    layer_backbone = backbone_with_layers(model)
    if layer_backbone is None:
        return None
    composite = getattr(base_transformers_model(model), "model", None)
    if composite is None or composite is layer_backbone:
        return layer_backbone
    return composite if any(module is layer_backbone for module in composite.modules()) else layer_backbone
