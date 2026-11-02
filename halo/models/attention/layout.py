"""Attention geometry and the per-layer attention cost layout: which decoder layers score a key matrix,
over how many keys, at what head width.

Keyed on transformers' ``layer_types`` vocabulary, never on a model name, and read off the rank's own
decoder layers, so a pipeline stage describes its slice. Costs are algorithmic key counts.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cached_property, partial
from typing import NamedTuple

import torch

from halo.models.config_access import get_config_field, text_config
from halo.models.module_tree import backbone_with_layers, decoder_layers
from halo.models.unwrapping import unwrap_toolkit_wrappers

# The key/value head split a GatedDeltaNet mixer is built from. Kimi Delta Attention also spells its layers
# ``linear_attention`` but declares a single ``linear_num_heads``.
_GATED_DELTA_NET_HEAD_FIELDS = ("linear_num_key_heads", "linear_num_value_heads")
_LINEAR_ATTENTION_LAYER_TYPE = "linear_attention"
# Families interleaving ShortConv layers with softmax attention, dense, MoE and VL alike.
SHORT_CONV_MODEL_TYPE_PREFIXES = ("lfm2",)

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
    _LINEAR_ATTENTION_LAYER_TYPE: _AttentionKind.NONE,
    "conv": _AttentionKind.NONE,
}

# The kinds whose keys a fixed count bounds: the window, the chunk, the sparse top-k.
_BOUNDED_KINDS = frozenset({_AttentionKind.WINDOW, _AttentionKind.CHUNKED, _AttentionKind.SPARSE})

# The layer types that select their keys through a QK-only indexer bounded by ``index_topk``.
_INDEXED_LAYER_TYPES = frozenset({"indexed_attention", "compressed_sparse_attention"})
# An indexer's (query heads, head width, keys pooled per scored key) in each config spelling: the ``index_*``
# token indexers and the ``indexer_*`` block indexer, which scores pooled blocks of ``indexer_compress_ratio``.
_INDEXER_FIELDS = (
    ("index_n_heads", "index_head_dim", "index_kpool"),
    ("indexer_n_heads", "indexer_head_dim", "indexer_compress_ratio"),
)

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
        for layer in self.grouped:
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

    @cached_property
    def grouped(self) -> tuple[LayerAttention, ...]:
        """Layers sharing a rule merged, widths summed — one vectorized pass per rule; merged once per layout,
        which every batch's costing reads."""
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


def _qk_v_head_dims(field: Callable[[str], int | None], head_dim: int | None) -> tuple[int | None, int | None]:
    """``(d_qk, d_v)`` read through ``field``, a config-field getter: MLA stacks a RoPE slice on a nope slice
    (with only ``qk_rope_head_dim``, ``head_dim`` is the whole width), else both are ``head_dim``."""
    qk_dim = field("qk_head_dim")
    if not qk_dim:
        nope = field("qk_nope_head_dim")
        qk_dim = nope + (field("qk_rope_head_dim") or 0) if nope else head_dim
    return qk_dim, field("v_head_dim") or head_dim


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
    qk_dim, v_dim = _qk_v_head_dims(partial(_layer_field, decoder, index), head_dim)
    if not qk_dim or not v_dim:
        raise ValueError("the config declares num_attention_heads but no head dimension to size the scores")
    return float(heads) * (float(qk_dim) + float(v_dim))


def _indexer(decoder, index: int) -> tuple[float, int]:
    """Width and key pooling of a sparse / compressed layer's QK-only indexer (0, 1 without one), in the first
    spelling of :data:`_INDEXER_FIELDS` the config declares."""
    for heads_field, dim_field, pool_field in _INDEXER_FIELDS:
        heads = _layer_field(decoder, index, heads_field)
        dim = _layer_field(decoder, index, dim_field)
        if heads and dim:
            return float(heads) * float(dim), int(_layer_field(decoder, index, pool_field) or 1)
    return 0.0, 1


def _sparse_keys(decoder, index: int, layer_type: str) -> int:
    """Keys a sparse layer attends: ``index_topk``, or a block indexer's ``indexer_budget`` plus the incomplete
    tail block (up to ``indexer_compress_ratio - 1`` tokens) it always keeps."""
    if _layer_field(decoder, index, "index_topk") is None and _layer_field(decoder, index, "indexer_budget"):
        budget = _positive(decoder, index, "indexer_budget", layer_type)
        return budget + _positive(decoder, index, "indexer_compress_ratio", layer_type) - 1
    return _positive(decoder, index, "index_topk", layer_type)


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
            keys=_sparse_keys(decoder, index, layer_type),
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


def _uniform_layer_type(cfg) -> str:
    """The type of every layer of a family that declares no ``layer_types``, as transformers derives it: sliding
    when ``sliding_window`` is set, else full attention."""
    return "sliding_attention" if get_config_field(cfg, "sliding_window") else "full_attention"


def _declared_layer_types(decoder) -> list[str]:
    """``layer_types``, or :func:`_uniform_layer_type` on every layer where a family declares none. Hybrid layer
    groups raise."""
    layer_types = get_config_field(decoder, "layer_types")
    if layer_types:
        return list(layer_types)
    depth = get_config_field(decoder, "num_hidden_layers")
    if depth is None:
        raise ValueError("the config declares neither layer_types nor num_hidden_layers")
    if declares_linear_layer_groups(decoder):
        raise ValueError(
            "the config declares hybrid layer groups (layer_group_size) without layer_types, so its "
            "linear-attention layers cannot be told from its attention layers"
        )
    return [_uniform_layer_type(decoder)] * int(depth)


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


def resolve_qk_v_head_dims(cfg) -> tuple[int, int]:
    """The query/key and value head widths: MLA's nope + RoPE split and ``v_head_dim`` where declared, else
    :func:`resolve_head_dim` for both."""
    qk_dim, v_dim = _qk_v_head_dims(partial(get_config_field, cfg), resolve_head_dim(cfg))
    return int(qk_dim), int(v_dim)


def has_latent_attention(cfg) -> bool:
    """Whether the decoder runs multi-head latent attention: keys and values expanded from a ``kv_lora_rank`` latent
    (DeepSeek-V3, GLM-4.7-Flash, Mistral4, MiniCPM3)."""
    return bool(get_config_field(cfg, "kv_lora_rank"))


def has_sliding_window_layers(cfg) -> bool:
    """Whether some decoder layer attends a sliding window, by ``layer_types`` or, where a family declares
    none, :func:`_uniform_layer_type`. Unlike the cost layout it does not raise on an entry outside
    :data:`LAYER_TYPE_KINDS`, which it counts as not sliding."""
    declared = get_config_field(cfg, "layer_types") or (_uniform_layer_type(cfg),)
    return any(LAYER_TYPE_KINDS.get(layer_type) is _AttentionKind.WINDOW for layer_type in declared)


def declares_attention_heads(cfg) -> bool:
    """Whether the config declares a head width or an attention-head count; Mamba declares neither."""
    return any(get_config_field(cfg, field, per_layer_reduce=max) for field in ("head_dim", "num_attention_heads"))


def resolve_num_key_value_heads(cfg) -> int:
    """The KV head count, falling back to ``num_attention_heads``; the largest across per-layer values."""
    declared = get_config_field(cfg, "num_key_value_heads", per_layer_reduce=max)
    if declared:
        return int(declared)
    return int(get_config_field(cfg, "num_attention_heads", per_layer_reduce=max))


def has_mrope_rotary(cfg) -> bool:
    """Whether the decoder rotates by M-RoPE (per-section text/temporal/height/width positions), on a text-only
    load too. Read off the text config, whose class lists ``mrope_section`` among the rope keys transformers'
    validation skips: a checkpoint may omit the section and take the rotary's own default."""
    return "mrope_section" in (getattr(text_config(cfg), "ignore_keys_at_rope_validation", None) or ())


class AttentionCall(NamedTuple):
    """The geometry one causal attention module hands its kernel, which a compiled kernel keys on. ``head_dim`` is
    the query/key width (MLA's ``qk_head_dim``) and ``v_head_dim`` the value width, which transformers' flash path
    pads up to ``head_dim`` and SDPA takes as it is; ``window`` is ``None`` on a full-attention layer, ``softcap``
    the logit cap it passes (Gemma 2), whose value FA4 hashes into its kernel key."""

    heads: int
    kv_heads: int
    head_dim: int
    v_head_dim: int
    window: int | None
    scaling: float | None
    softcap: float | None


def _module_heads(module: torch.nn.Module) -> int:
    """The query heads ``module`` attends with: its own ``num_heads`` where it declares one, else its layer's
    config value, since a per-layer config (Step-3.7's 64/96, Laguna's 48/72, Inkling) has no global one."""
    heads = getattr(module, "num_heads", None)
    if isinstance(heads, int):
        return heads
    return int(_layer_field(text_config(module.config), module.layer_idx, "num_attention_heads"))


def _module_window(module: torch.nn.Module) -> int | None:
    """The window ``module`` passes its kernel; a module keeping none passes its config's (Mistral, Phi-3)."""
    if hasattr(module, "sliding_window"):
        return module.sliding_window or None
    return get_config_field(module.config, "sliding_window") or None


def attention_call(module: torch.nn.Module) -> AttentionCall | None:
    """The :class:`AttentionCall` ``module`` makes, or ``None`` for a module that is not a causal attention one
    (no ``is_causal``, no GQA group count)."""
    if not getattr(module, "is_causal", False) or not hasattr(module, "num_key_value_groups"):
        return None
    heads = _module_heads(module)
    head_dim = getattr(module, "qk_head_dim", None) or module.head_dim
    return AttentionCall(
        heads=heads,
        kv_heads=heads // module.num_key_value_groups,
        head_dim=head_dim,
        v_head_dim=getattr(module, "v_head_dim", None) or head_dim,
        window=_module_window(module),
        scaling=getattr(module, "scaling", None),
        softcap=getattr(module, "attn_logit_softcapping", None) or None,
    )


def attention_calls(model: torch.nn.Module) -> set[AttentionCall]:
    """The distinct :func:`attention_call` of every causal attention module ``model`` holds, read per module: a
    pipeline stage reports its own layers, a Ulysses wrapper the module it wraps."""
    return {call for call in map(attention_call, model.modules()) if call is not None}


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


def has_linear_attention_layers(config) -> bool:
    """Whether the decoder (read through a VLM wrapper) declares ``linear_attention`` layer types."""
    return config is not None and _LINEAR_ATTENTION_LAYER_TYPE in (get_config_field(config, "layer_types") or ())


def has_gated_delta_net_layers(config) -> bool:
    """Whether the decoder (read through a VLM wrapper) runs GatedDeltaNet layers, under any ``model_type``:
    it declares linear-attention layers (:func:`has_linear_attention_layers`) and the head split those mixers
    are built from."""
    return has_linear_attention_layers(config) and all(
        get_config_field(config, field) is not None for field in _GATED_DELTA_NET_HEAD_FIELDS
    )


def compressed_layer_types(config) -> list[str]:
    """The declared ``layer_types`` entries of kind ``"compressed"`` (DeepSeek-V4's CSA/HCA): layers that
    pool KV over windows cut at fixed row indices."""
    declared = get_config_field(config, "layer_types") or ()
    return sorted(
        {layer_type for layer_type in declared if LAYER_TYPE_KINDS.get(layer_type) is _AttentionKind.COMPRESSED}
    )


def declares_linear_layer_groups(config) -> bool:
    """Whether the config declares hybrid layer groups (``layer_group_size``) that hold linear-attention
    layers: the last layer of each group is softmax attention, so a group of one holds none."""
    return (get_config_field(config, "layer_group_size") or 0) > 1
