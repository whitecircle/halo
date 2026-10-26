"""Attention geometry and the per-layer attention cost layout: which decoder layers score a key matrix,
over how many keys, at what head width.

Keyed on transformers' ``layer_types`` vocabulary, never on a model name, and read off the rank's own
decoder layers, so a pipeline stage describes its slice. Costs are algorithmic key counts.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, replace

import torch

from src.models.config_access import get_config_field, text_config
from src.models.module_tree import backbone_with_layers, decoder_layers
from src.models.unwrapping import unwrap_toolkit_wrappers

# Families interleaving GatedDeltaNet (linear-attention) layers with softmax attention; prefixes, so
# text-tower variants (``qwen3_5_moe_text``) match.
GDN_MODEL_TYPE_PREFIXES = ("qwen3_5", "qwen3_next")

# Score matmul passes per attended key: QK^T and Attn·V forward (2) plus their backward (4). At head
# width ``heads·(d_qk + d_v)`` this is the PaLM/Megatron ``12·S·H`` (``d_qk = d_v = H / heads``).
SCORE_PASSES = 6.0
# An indexer ranks keys with one QK^T forward and no backward (GLM-5's runs under ``no_grad``,
# DeepSeek-V4's yields indices): a single matmul pass.
INDEX_PASSES = 2.0


class _AttentionKind(enum.StrEnum):
    """How many keys a decoder layer's score matrix spans per token."""

    FULL = "full"
    WINDOW = "window"
    CHUNKED = "chunked"
    SPARSE = "sparse"
    COMPRESSED = "compressed"
    NONE = "none"


# ``config.layer_types`` entry -> cost kind ("hybrid" is full attention with a convolution branch). An
# entry outside this table raises rather than costing an unknown mixer as full attention.
LAYER_TYPE_KINDS: dict[str, _AttentionKind] = {
    "full_attention": _AttentionKind.FULL,
    "hybrid": _AttentionKind.FULL,
    "sliding_attention": _AttentionKind.WINDOW,
    "hybrid_sliding": _AttentionKind.WINDOW,
    "chunked_attention": _AttentionKind.CHUNKED,
    "indexed_attention": _AttentionKind.SPARSE,
    "compressed_sparse_attention": _AttentionKind.COMPRESSED,
    "heavily_compressed_attention": _AttentionKind.COMPRESSED,
    "linear_attention": _AttentionKind.NONE,
    "conv": _AttentionKind.NONE,
}

# The kinds whose keys a fixed count bounds: the window, the chunk, the sparse top-k.
_BOUNDED_KINDS = frozenset({_AttentionKind.WINDOW, _AttentionKind.CHUNKED, _AttentionKind.SPARSE})

# The layer types that select their keys through a QK-only indexer bounded by ``index_topk``.
_INDEXED_LAYER_TYPES = frozenset({"indexed_attention", "compressed_sparse_attention"})

# A decoder layer's own declaration of its type, read before ``config.layer_types`` because a sliced
# pipeline stage re-bases list positions.
_LAYER_TYPE_ATTRS = ("layer_type", "attention_type", "block_type")


@dataclass(frozen=True)
class LayerAttention:
    """One decoder layer's score cost: the keys it attends at ``width``, plus a QK-only indexer.

    ``keys`` bounds the attended keys, ``rate`` compresses the key stream, ``local_window`` adds a
    compressed layer's local band, and ``index_width`` / ``index_pool`` describe the indexer.
    """

    kind: _AttentionKind
    width: float
    keys: int | None = None
    rate: int = 1
    local_window: int | None = None
    index_width: float = 0.0
    index_pool: int = 1

    def score_keys(self, lengths: torch.Tensor) -> torch.Tensor:
        """Keys a token of a document of each length attends through the score matrix."""
        if self.kind == _AttentionKind.NONE:
            return torch.zeros_like(lengths)
        if self.kind == _AttentionKind.FULL:
            return lengths
        if self.kind in _BOUNDED_KINDS:
            return lengths.clamp(max=self.keys)
        compressed = lengths / self.rate
        if self.keys is not None:
            compressed = compressed.clamp(max=self.keys)
        if self.local_window:
            compressed = compressed + lengths.clamp(max=self.local_window)
        return compressed


@dataclass(frozen=True)
class AttentionLayout:
    """The score cost of every decoder layer on this rank. ``source`` is ``"layers"`` when read off the
    rank's module tree, ``"config"`` when standing in for the config's full depth."""

    layers: tuple[LayerAttention, ...]
    source: str = "layers"

    def flops_for_documents(self, lengths: torch.Tensor, own_tokens: torch.Tensor | None = None) -> torch.Tensor:
        """Attention-score FLOPs of documents of the given token lengths (a float64 scalar), summed per
        document because the varlen kernels never attend across a packed row. Full attention is costed
        at every key (the PaLM convention), bounded layers at the keys their kernel visits.

        ``own_tokens`` counts only part of each document: a tree node is the tail of its root-to-leaf
        path, so it costs its own tokens at that path's length and its ancestors are costed once, as
        their own nodes.
        """
        lengths = lengths.to(torch.float64)
        own_tokens = lengths if own_tokens is None else own_tokens.to(torch.float64)
        total = torch.zeros((), dtype=torch.float64, device=lengths.device)
        for layer in self.grouped():
            per_token = SCORE_PASSES * layer.width * layer.score_keys(lengths)
            if layer.index_width:
                pooled_keys = lengths / (layer.rate * layer.index_pool)
                per_token = per_token + INDEX_PASSES * layer.index_width * pooled_keys
            total = total + (per_token * own_tokens).sum()
        return total

    def flops_per_token(self, seq_len: int) -> float:
        """Per-token score FLOPs when every token belongs to a document of ``seq_len`` tokens."""
        if seq_len <= 0:
            return 0.0
        return float(self.flops_for_documents(torch.tensor([seq_len], dtype=torch.float64))) / seq_len

    def grouped(self) -> tuple[LayerAttention, ...]:
        """Layers sharing a rule merged, widths summed — one vectorized pass per rule."""
        merged: dict[LayerAttention, LayerAttention] = {}
        for layer in self.layers:
            rule = replace(layer, width=0.0, index_width=0.0)
            head = merged.get(rule, rule)
            merged[rule] = replace(
                head, width=head.width + layer.width, index_width=head.index_width + layer.index_width
            )
        return tuple(merged.values())

    def describe(self) -> str:
        """``18×full + 18×window(128) @ width 8192`` — the banner form."""
        counts: dict[tuple[str, int | None, int], int] = {}
        for layer in self.layers:
            key = (layer.kind, layer.keys if layer.kind != _AttentionKind.COMPRESSED else None, layer.rate)
            counts[key] = counts.get(key, 0) + 1
        parts = []
        for (kind, keys, rate), count in counts.items():
            label = "linear" if kind == _AttentionKind.NONE else str(kind)
            if kind in _BOUNDED_KINDS:
                label += f"({keys})"
            elif kind == _AttentionKind.COMPRESSED:
                label += f"(/{rate})"
            parts.append(f"{count}×{label}")
        widths = sorted({layer.width for layer in self.layers if layer.kind != _AttentionKind.NONE})
        if not widths:
            return " + ".join(parts)
        width = f"{widths[0]:.0f}" if len(widths) == 1 else f"{widths[0]:.0f}–{widths[-1]:.0f}"
        return " + ".join(parts) + f" @ width {width}"


def _layer_field(decoder, index: int, field: str):
    """``field`` for layer ``index``: the per-layer config view, else a ``<field>_per_layer`` list, else
    the global value."""
    view = getattr(decoder, "per_layer_config", None)
    if view is not None:
        try:
            value = getattr(view[index], field, None)
        except (IndexError, TypeError, KeyError):
            value = None
        if value is not None:
            return value
    per_layer = get_config_field(decoder, f"{field}_per_layer")
    if isinstance(per_layer, (list, tuple)) and index < len(per_layer) and per_layer[index] is not None:
        return per_layer[index]
    return get_config_field(decoder, field)


def _positive(decoder, index: int, field: str, layer_type: str) -> int:
    value = _layer_field(decoder, index, field)
    if not value or int(value) <= 0:
        raise ValueError(f"layer_types entry {layer_type!r} (layer {index}) needs a positive {field}")
    return int(value)


def _score_width(decoder, index: int) -> float:
    """``heads·(d_qk + d_v)`` for layer ``index``; ``2·hidden_size`` (the PaLM ``H`` convention)
    when the config declares no head geometry."""
    heads = _layer_field(decoder, index, "num_attention_heads")
    hidden = _layer_field(decoder, index, "hidden_size")
    if not heads:
        if not hidden:
            raise ValueError("the config declares neither num_attention_heads nor hidden_size")
        return 2.0 * float(hidden)
    head_dim = _layer_field(decoder, index, "head_dim") or (int(hidden) // int(heads) if hidden else None)
    qk_dim = _layer_field(decoder, index, "qk_head_dim")
    if not qk_dim:
        # MLA stacks a RoPE slice on a nope slice; with only qk_rope_head_dim, head_dim is the whole width.
        nope = _layer_field(decoder, index, "qk_nope_head_dim")
        qk_dim = nope + (_layer_field(decoder, index, "qk_rope_head_dim") or 0) if nope else head_dim
    v_dim = _layer_field(decoder, index, "v_head_dim") or head_dim
    if not qk_dim or not v_dim:
        raise ValueError("the config declares num_attention_heads but no head dimension to size the scores")
    return float(heads) * (float(qk_dim) + float(v_dim))


def _indexer(decoder, index: int) -> tuple[float, int]:
    """Width and key pooling of a sparse / compressed layer's QK-only indexer (0, 1 without one)."""
    heads = _layer_field(decoder, index, "index_n_heads")
    dim = _layer_field(decoder, index, "index_head_dim")
    if not heads or not dim:
        return 0.0, 1
    return float(heads) * float(dim), int(_layer_field(decoder, index, "index_kpool") or 1)


def _layer_attention(decoder, layer_type: str, index: int) -> LayerAttention:
    kind = LAYER_TYPE_KINDS.get(layer_type)
    if kind is None:
        raise ValueError(
            f"layer_types entry {layer_type!r} (layer {index}) has no attention cost rule; known entries: "
            f"{sorted(LAYER_TYPE_KINDS)}"
        )
    width = _score_width(decoder, index)
    if kind is _AttentionKind.NONE:
        return LayerAttention(kind, 0.0)
    if kind is _AttentionKind.FULL:
        return LayerAttention(kind, width)
    if kind is _AttentionKind.WINDOW:
        return LayerAttention(kind, width, keys=_positive(decoder, index, "sliding_window", layer_type))
    if kind is _AttentionKind.CHUNKED:
        return LayerAttention(kind, width, keys=_positive(decoder, index, "attention_chunk_size", layer_type))
    indexed = layer_type in _INDEXED_LAYER_TYPES
    index_width, index_pool = _indexer(decoder, index) if indexed else (0.0, 1)
    if kind is _AttentionKind.SPARSE:
        return LayerAttention(
            kind,
            width,
            keys=_positive(decoder, index, "index_topk", layer_type),
            index_width=index_width,
            index_pool=index_pool,
        )
    rates = _layer_field(decoder, index, "compress_rates") or {}
    rate = rates.get(layer_type) if isinstance(rates, dict) else None
    if not rate:
        raise ValueError(f"layer_types entry {layer_type!r} (layer {index}) declares no compress_rates entry")
    topk = _layer_field(decoder, index, "index_topk") if indexed else None
    return LayerAttention(
        kind,
        width,
        keys=int(topk) if topk else None,
        rate=int(rate),
        local_window=_layer_field(decoder, index, "sliding_window"),
        index_width=index_width,
        index_pool=index_pool,
    )


def _declared_layer_types(decoder) -> list[str]:
    """``layer_types``, or transformers' derivation where a family declares none: every layer slides when
    ``sliding_window`` is set, else every layer is full attention. Hybrid layer groups raise."""
    layer_types = get_config_field(decoder, "layer_types")
    if layer_types:
        return list(layer_types)
    depth = get_config_field(decoder, "num_hidden_layers")
    if depth is None:
        raise ValueError("the config declares neither layer_types nor num_hidden_layers")
    if get_config_field(decoder, "layer_group_size"):
        raise ValueError(
            "the config declares hybrid layer groups (layer_group_size) without layer_types, so its "
            "linear-attention layers cannot be told from its attention layers"
        )
    window = get_config_field(decoder, "sliding_window")
    return ["sliding_attention" if window else "full_attention"] * int(depth)


def _layer_index(layer: torch.nn.Module, position: int) -> int:
    """The layer's global index: its own ``layer_idx``, its attention's, else the list position."""
    own = getattr(layer, "layer_idx", None)
    if isinstance(own, int):
        return own
    for child in layer.children():
        child_idx = getattr(child, "layer_idx", None)
        if isinstance(child_idx, int):
            return child_idx
    return position


def _layer_type_of(layer: torch.nn.Module, index: int, declared: list[str]) -> str:
    for attr in _LAYER_TYPE_ATTRS:
        value = getattr(layer, attr, None)
        if isinstance(value, str):
            return value
    if index >= len(declared):
        raise ValueError(f"decoder layer {index} lies beyond the {len(declared)} entries of layer_types")
    return declared[index]


def resolve_head_dim(cfg) -> int:
    """The attention head dimension: a declared ``head_dim`` (which may be decoupled from the ratio, as
    Gemma 4's 512 is), else ``hidden_size // num_attention_heads``; the largest across per-layer values,
    since every consumer sizes a buffer from it."""
    declared = get_config_field(cfg, "head_dim", per_layer_reduce=max)
    if declared:
        return int(declared)
    decoder = text_config(cfg)
    return decoder.hidden_size // decoder.num_attention_heads


def declares_attention_heads(cfg) -> bool:
    """Whether the config declares a head width or an attention-head count; Mamba declares neither."""
    return any(get_config_field(cfg, field, per_layer_reduce=max) for field in ("head_dim", "num_attention_heads"))


def resolve_num_key_value_heads(cfg) -> int:
    """The KV head count, falling back to ``num_attention_heads``; the largest across per-layer values."""
    declared = get_config_field(cfg, "num_key_value_heads", per_layer_reduce=max)
    if declared:
        return int(declared)
    return int(get_config_field(cfg, "num_attention_heads", per_layer_reduce=max))


def attention_layout(model: torch.nn.Module) -> AttentionLayout:
    """The layout of the decoder layers this rank holds, the full depth when the tree exposes none.
    Raises ``ValueError`` on a model whose layers cannot be classified."""
    if isinstance(model, torch.nn.Module):
        model = unwrap_toolkit_wrappers(model)
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("the model exposes no config, so its attention layout cannot be read")
    decoder = text_config(config)
    backbone = backbone_with_layers(model)
    layers = decoder_layers(backbone) if backbone is not None else None
    if layers is None:
        return attention_layout_from_config(config)
    declared = _declared_layer_types(decoder)
    return AttentionLayout(
        tuple(
            _layer_attention(decoder, _layer_type_of(layer, index, declared), index)
            for index, layer in ((_layer_index(layer, position), layer) for position, layer in enumerate(layers))
        )
    )


def attention_layout_from_config(config) -> AttentionLayout:
    """The full-depth layout a config describes, for trees whose decoder layers are not reachable."""
    decoder = text_config(config)
    declared = _declared_layer_types(decoder)
    return AttentionLayout(
        tuple(_layer_attention(decoder, kind, index) for index, kind in enumerate(declared)), source="config"
    )


def compressed_layer_types(config) -> list[str]:
    """The declared ``layer_types`` entries of kind ``"compressed"`` (DeepSeek-V4's CSA/HCA): layers that
    pool KV over windows cut at fixed row indices."""
    declared = get_config_field(config, "layer_types") or ()
    return sorted(
        {layer_type for layer_type in declared if LAYER_TYPE_KINDS.get(layer_type) is _AttentionKind.COMPRESSED}
    )
