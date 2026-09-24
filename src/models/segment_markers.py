"""Per-document segment markers for the conv / linear-attention mixers of a multi-document row.

Attention isolates the documents of a packed or flattened row from its ``position_ids``, which
restart at 0 at every document. Some families' conv and linear-attention mixers read their document
boundaries from forward kwargs instead, and carry state across documents when those are absent. This
module holds which families read which markers, the refusal for the families whose torch fallbacks
drop them, and the markers themselves — shared by the SFT packing / padding-free collators and SMPO's
padding-free forward. Per-family isolation matrix: ``agent-docs/data/collators.md``.
"""

from dataclasses import dataclass

import torch
from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available

from src.models.patches.attention import GDN_MODEL_TYPE_PREFIXES, model_type_matches

# Backends whose availability selects transformers' segment-aware GatedDeltaNet kernels; the torch
# fallbacks ignore ``seq_idx``/``cu_seqlens`` and run a packed row as one document while attention
# stays isolated. These are transformers' own fast-path predicates; re-spelling them as bare
# ``find_spec`` checks would drop the ``fla>=0.2.2`` floor and the CUDA-capable-torch conjunct.
GDN_SEGMENT_AWARE_BACKENDS = (
    ("causal_conv1d", is_causal_conv1d_available),
    ("fla>=0.2.2", is_flash_linear_attention_available),
)


@dataclass(frozen=True)
class SegmentMarkers:
    """Which segment markers a family's forward reads.

    ``seq_idx`` is the per-token document index (LFM2 ShortConv, the GatedDeltaNet conv).
    ``cu_seq_lens`` is the varlen set ``cu_seq_lens_q/k`` + ``max_length_q/k``: GatedDeltaNet's chunked
    delta rule reads ``cu_seq_lens_q``, and flash attention consumes the set in place of re-deriving
    it from ``position_ids``.
    """

    seq_idx: bool = False
    cu_seq_lens: bool = False


def segment_markers_for(model_config) -> SegmentMarkers:
    """The markers this model's mixers need to keep a multi-document row's documents apart.

    Family-gated: every other family's forward reads neither, so emitting them universally would
    push unread kwargs everywhere. Reads the text sub-config too, so a VLM wrapper resolves like its
    decoder.
    """
    if model_config is None:
        return SegmentMarkers()
    return SegmentMarkers(
        seq_idx=model_type_matches(model_config, "lfm2", *GDN_MODEL_TYPE_PREFIXES),
        cu_seq_lens=model_type_matches(model_config, *GDN_MODEL_TYPE_PREFIXES),
    )


def require_segment_aware_kernels(model_config, mode: str) -> None:
    """Refuse a multi-document row (``mode``) on a GatedDeltaNet family lacking its segment-aware kernels.

    Emitting the markers is not enough on its own: only the fast-path kernels read them.
    """
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


def segment_marker_kwargs(position_ids: torch.Tensor, markers: SegmentMarkers) -> dict[str, torch.Tensor | int]:
    """The forward kwargs ``markers`` selects, built from rows whose ``position_ids`` restart at 0.

    ``seq_idx`` is per row. The ``cu_seq_lens`` set is only defined on one flattened ``[1, total]``
    row — the delta rule's varlen boundaries have no per-row convention — and refuses anything else.
    Tensors land on ``position_ids``' device.
    """
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
