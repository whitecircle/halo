"""Per-document segment markers for the conv / linear-attention mixers of a multi-document row.

Attention isolates a row's documents from ``position_ids``, which restart at 0 at every document;
some families' conv and linear-attention mixers read the boundaries from forward kwargs instead and
carry state across documents without them. Held here: who reads which markers, the refusals for rows
no marker can isolate, and the markers themselves."""

from dataclasses import dataclass

import torch
from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available

from src.models.attention_layout import compressed_layer_types
from src.models.config_access import model_type_matches

# Families interleaving GatedDeltaNet (linear-attention) layers with softmax attention. Prefixes
# rather than exact spellings, so text-tower variants (``qwen3_5_moe_text``) match; shared by the
# segment-marker gates below and the FA4 backward gate in ``src/models/patches/attention.py``.
GDN_MODEL_TYPE_PREFIXES = ("qwen3_5", "qwen3_next")

# Families whose forward never passes ``position_ids`` into mask construction, so on a dense backend a
# packed row runs as one causal sequence; packing is refused for them.
DENSE_PACKING_LEAK_MODEL_TYPES = frozenset({"gpt_oss"})

# transformers' own predicates selecting its segment-aware GatedDeltaNet kernels (a bare ``find_spec``
# would drop the ``fla>=0.2.2`` floor and the CUDA-capable-torch conjunct); the torch fallbacks ignore
# the markers.
GDN_SEGMENT_AWARE_BACKENDS = (
    ("causal_conv1d", is_causal_conv1d_available),
    ("fla>=0.2.2", is_flash_linear_attention_available),
)


@dataclass(frozen=True)
class SegmentMarkers:
    """Which segment markers a family's forward reads: ``seq_idx``, the per-token document index (LFM2
    ShortConv, the GatedDeltaNet conv), and ``cu_seq_lens``, the varlen set ``cu_seq_lens_q/k`` +
    ``max_length_q/k`` (GatedDeltaNet's chunked delta rule; flash attention consumes it too).
    """

    seq_idx: bool = False
    cu_seq_lens: bool = False


def segment_markers_for(model_config) -> SegmentMarkers:
    """The markers this model's mixers need to keep a multi-document row's documents apart; family-gated,
    read through the text sub-config so a VLM wrapper resolves like its decoder."""
    if model_config is None:
        return SegmentMarkers()
    return SegmentMarkers(
        seq_idx=model_type_matches(model_config, "lfm2", *GDN_MODEL_TYPE_PREFIXES),
        cu_seq_lens=model_type_matches(model_config, *GDN_MODEL_TYPE_PREFIXES),
    )


def reject_compressed_kv_rows(model_config, mode: str) -> None:
    """Refuse a multi-document row (``mode``, the flag that asks for one) on a model with
    compressed-KV layers (:func:`~src.models.attention_layout.compressed_layer_types`), on every
    backend: no segment marker or ``position_ids`` reaches windows cut by row index."""
    compressed = compressed_layer_types(model_config) if model_config is not None else []
    if not compressed:
        return
    raise ValueError(
        f"{mode} is refused for model_type={getattr(model_config, 'model_type', None)!r}: its {compressed} "
        f"layers pool KV over windows cut at fixed indices of the row and judge their causality by "
        f"position, so in a multi-document row, whose positions restart at every document, each document "
        f"after the first attends the compressed KV of the row's first tokens instead of its own. Train "
        f"one document per row (turn {mode} off)."
    )


def require_segment_aware_kernels(model_config, mode: str) -> None:
    """Refuse a multi-document row (``mode``) on a GatedDeltaNet family lacking its segment-aware
    kernels, the only ones that read the markers."""
    if model_config is None or not model_type_matches(model_config, *GDN_MODEL_TYPE_PREFIXES):
        return
    missing = ", ".join(name for name, is_available in GDN_SEGMENT_AWARE_BACKENDS if not is_available())
    if missing:
        raise ValueError(
            f"{mode}=True is refused for the GatedDeltaNet family "
            f"model_type={getattr(model_config, 'model_type', None)!r}: {missing} unavailable. "
            f"transformers selects its segment-aware linear-attention kernels on exactly these "
            f"checks (package installed, at the version floor, CUDA-capable torch), and the torch "
            f"fallbacks it takes instead IGNORE the segment markers a {mode} row carries — the "
            f"conv drops seq_idx, the chunked delta rule drops cu_seq_lens_q — so conv and "
            f"recurrent state cross document boundaries silently, while attention stays "
            f"isolated. Install {missing} (the production images pin both), or disable {mode} for "
            f"this family."
        )


def document_ids(position_ids: torch.Tensor) -> torch.Tensor:
    """Per-token document index of each row, counting the ``position_ids`` restarts at 0."""
    return (position_ids == 0).cumsum(dim=-1) - 1


def flattened_document_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """``position_ids`` of a padded batch flattened into one ``[1, real_tokens]`` row: each row's real
    tokens become one document, counted from 0 wherever the row's padding sat."""
    return (attention_mask.cumsum(dim=1) - 1)[attention_mask.bool()].unsqueeze(0)


def segment_marker_kwargs(position_ids: torch.Tensor, markers: SegmentMarkers) -> dict[str, torch.Tensor | int]:
    """The forward kwargs ``markers`` selects, built from rows whose ``position_ids`` restart at 0.
    ``seq_idx`` is per row; the ``cu_seq_lens`` set exists only for one flattened ``[1, total]`` row."""
    kwargs: dict[str, torch.Tensor | int] = {}
    if markers.seq_idx:
        kwargs["seq_idx"] = document_ids(position_ids).to(torch.int32)
    if markers.cu_seq_lens:
        if position_ids.shape[0] != 1:
            raise ValueError(
                f"cu_seq_lens markers need one flattened [1, total] row, got position_ids of shape "
                f"{tuple(position_ids.shape)}"
            )
        positions = position_ids[0]
        starts = (positions == 0).nonzero(as_tuple=True)[0]
        cu_seq_lens = torch.cat([starts, starts.new_tensor([positions.numel()])]).to(torch.int32)
        kwargs["cu_seq_lens_q"] = kwargs["cu_seq_lens_k"] = cu_seq_lens
        kwargs["max_length_q"] = kwargs["max_length_k"] = int(cu_seq_lens.diff().max())
    return kwargs
