"""Per-layer attention cost layout of a live model: which decoder layers score a key matrix, over
how many keys, at what head width.

A leaf beside ``attention_geometry``: costing a sequence (the MFU accounting) needs one answer to
"what does layer *i* attend over", keyed on transformers' ``layer_types`` vocabulary — full,
sliding, chunked, sparse-indexed, compressed, linear/conv — never on a model name. The layout is read
off the rank's OWN decoder layers, so a pipeline stage describes its slice and a wrapped tree (FSDP2,
EP, TP, CP, PEFT) describes the same layers as the bare one. Costs are the algorithmic key counts: a
family whose kernels run the dense mask (eager / SDPA sparse attention) does more hardware work than
this counts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from src.models.loading.config_levels import get_config_field, text_config
from src.models.structure import backbone_with_layers, decoder_layers, unwrap_model

# Score matmul passes per attended key: QK^T and Attn·V forward (2) plus their backward (4). At head
# width ``heads·(d_qk + d_v)`` this is the PaLM/Megatron ``12·S·H`` (``d_qk = d_v = H / heads``).
SCORE_PASSES = 6.0
# An indexer ranks keys with one QK^T forward and no backward (GLM-5's runs under ``no_grad``,
# DeepSeek-V4's yields indices): a single matmul pass.
INDEX_PASSES = 2.0

# ``config.layer_types`` entry -> cost kind. "hybrid" (Zaya's CCA, Inkling) is full attention with a
# convolution branch; "conv" (LFM-2 ShortConv) and "linear_attention" (GatedDeltaNet, KDA) score no
# key matrix. An entry outside this table raises rather than costing an unknown mixer as full attention.
LAYER_TYPE_KINDS: dict[str, str] = {
    "full_attention": "full",
    "hybrid": "full",
    "sliding_attention": "window",
    "hybrid_sliding": "window",
    "chunked_attention": "chunked",
    "deepseek_sparse_attention": "sparse",
    "compressed_sparse_attention": "compressed",
    "heavily_compressed_attention": "compressed",
    "linear_attention": "none",
    "conv": "none",
}

# The layer types that select their keys through a QK-only indexer bounded by ``index_topk``: GLM-5's
# DSA and DeepSeek-V4's CSA. Its heavily compressed layers attend every compressed key without one.
_INDEXED_LAYER_TYPES = frozenset({"deepseek_sparse_attention", "compressed_sparse_attention"})

# A decoder layer's own declaration of its type, set at construction from the global index. Read
# before ``config.layer_types`` because a sliced pipeline stage re-bases list positions.
_LAYER_TYPE_ATTRS = ("layer_type", "attention_type", "block_type")


@dataclass(frozen=True)
class LayerAttention:
    """One decoder layer's score cost: the keys it attends at ``width``, plus a QK-only indexer.

    ``keys`` bounds the attended keys (window, chunk, or the sparse top-k); ``rate`` compresses the
    key stream and ``local_window`` adds a compressed layer's local band; ``index_width`` /
    ``index_pool`` describe the indexer that ranks (pooled) keys for a sparse or compressed layer.
    """

    kind: str
    width: float
    keys: int | None = None
    rate: int = 1
    local_window: int | None = None
    index_width: float = 0.0
    index_pool: int = 1

    def score_keys(self, lengths: torch.Tensor) -> torch.Tensor:
        """Keys a token of a document of each length attends through the score matrix."""
        if self.kind == "none":
            return torch.zeros_like(lengths)
        if self.kind == "full":
            return lengths
        if self.kind in ("window", "chunked", "sparse"):
            return lengths.clamp(max=self.keys)
        compressed = lengths / self.rate
        if self.keys is not None:
            compressed = compressed.clamp(max=self.keys)
        if self.local_window:
            compressed = compressed + lengths.clamp(max=self.local_window)
        return compressed

    def index_keys(self, lengths: torch.Tensor) -> torch.Tensor:
        """Pooled keys the indexer ranks per token (zero without an indexer)."""
        if not self.index_width:
            return torch.zeros_like(lengths)
        return lengths / (self.rate * self.index_pool)

    def rule(self) -> LayerAttention:
        """The cost rule with the width dropped — layers sharing it sum their widths."""
        return replace(self, width=0.0, index_width=0.0)


@dataclass(frozen=True)
class AttentionLayout:
    """The score cost of every decoder layer on this rank, and its summed per-rule form.

    ``source`` says whether the layers were read off the rank's own module tree (``"layers"``) or
    stand in for the config's full depth (``"config"``) — a pipeline stage owns only a share of the latter.
    """

    layers: tuple[LayerAttention, ...]
    source: str = "layers"

    def flops_for_documents(self, lengths: torch.Tensor) -> torch.Tensor:
        """Attention-score FLOPs of documents of the given token lengths (a float64 scalar).

        Per token the cost is the keys it attends times the layer width, so a document costs
        ``L × per_token(L)``; summing documents rather than the packed row is what keeps a packed
        batch honest — the varlen kernels attend within a document, never across the row. Full
        attention is costed at every key (the PaLM convention, not the causal half), bounded layers
        at the keys their kernel visits.
        """
        lengths = lengths.to(torch.float64)
        total = torch.zeros((), dtype=torch.float64, device=lengths.device)
        for layer in self.grouped():
            per_token = SCORE_PASSES * layer.width * layer.score_keys(lengths)
            if layer.index_width:
                per_token = per_token + INDEX_PASSES * layer.index_width * layer.index_keys(lengths)
            total = total + (per_token * lengths).sum()
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
            rule = layer.rule()
            head = merged.get(rule, rule)
            merged[rule] = replace(
                head, width=head.width + layer.width, index_width=head.index_width + layer.index_width
            )
        return tuple(merged.values())

    def describe(self) -> str:
        """``18×full + 18×window(128) @ width 8192`` — the banner form."""
        counts: dict[tuple[str, int | None, int], int] = {}
        for layer in self.layers:
            key = (layer.kind, layer.keys if layer.kind != "compressed" else None, layer.rate)
            counts[key] = counts.get(key, 0) + 1
        parts = []
        for (kind, keys, rate), count in counts.items():
            label = "linear" if kind == "none" else kind
            if kind in ("window", "chunked", "sparse"):
                label += f"({keys})"
            elif kind == "compressed":
                label += f"(/{rate})"
            parts.append(f"{count}×{label}")
        widths = sorted({layer.width for layer in self.layers if layer.kind != "none"})
        if not widths:
            return " + ".join(parts)
        width = f"{widths[0]:.0f}" if len(widths) == 1 else f"{widths[0]:.0f}–{widths[-1]:.0f}"
        return " + ".join(parts) + f" @ width {width}"


def attention_layout(model: torch.nn.Module) -> AttentionLayout:
    """The layout of the decoder layers this rank holds — the full depth when the tree exposes none.

    Raises ``ValueError`` on a model whose layers cannot be classified; the callers that only
    observe (MFU) catch it and report the estimate as unavailable.
    """
    if isinstance(model, torch.nn.Module):
        model = unwrap_model(model)
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


def _declared_layer_types(decoder) -> list[str]:
    """``layer_types``, or the derivation transformers applies where a family declares none:
    ``sliding_window`` set means every layer slides (Mistral, Qwen3-MoE under ``use_sliding_window``),
    otherwise every layer is full attention. A hybrid family declaring its linear layers only by
    ``layer_group_size`` (the remote Bailing configs) cannot be classified this way and raises."""
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


def _layer_field(decoder, index: int, field: str, default=None):
    """``field`` for layer ``index``: the per-layer config view where transformers registers one
    (Gemma 4's split head_dim, Step-3.7's per-layer heads), else a plain ``<field>_per_layer`` list
    (Laguna's head counts), else the global value."""
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
    return get_config_field(decoder, field, default)


def _layer_attention(decoder, layer_type: str, index: int) -> LayerAttention:
    kind = LAYER_TYPE_KINDS.get(layer_type)
    if kind is None:
        raise ValueError(
            f"layer_types entry {layer_type!r} (layer {index}) has no attention cost rule; known entries: "
            f"{sorted(LAYER_TYPE_KINDS)}"
        )
    width = _score_width(decoder, index)
    if kind == "none":
        return LayerAttention(kind, 0.0)
    if kind == "full":
        return LayerAttention(kind, width)
    if kind == "window":
        return LayerAttention(kind, width, keys=_positive(decoder, index, "sliding_window", layer_type))
    if kind == "chunked":
        return LayerAttention(kind, width, keys=_positive(decoder, index, "attention_chunk_size", layer_type))
    indexed = layer_type in _INDEXED_LAYER_TYPES
    index_width, index_pool = _indexer(decoder, index) if indexed else (0.0, 1)
    if kind == "sparse":
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
        # MLA stacks a RoPE slice on a nope slice; a family declaring only qk_rope_head_dim
        # (DeepSeek-V4) rotates a slice of head_dim, which already is the whole q/k width.
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
