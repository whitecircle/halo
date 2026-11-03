"""Per-document segment markers for the conv / linear-attention mixers of a multi-document row, which read its
boundaries from forward kwargs rather than ``position_ids``: who reads which markers, the refusals for rows no
marker isolates, the markers, and the varlen set flash attention would otherwise derive from a packed row, or test
a one-document row for, in every layer."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available

from halo.log import warn_once
from halo.models.attention.layout import (
    SHORT_CONV_MODEL_TYPE_PREFIXES,
    compressed_layer_types,
    declares_linear_layer_groups,
    has_gated_delta_net_layers,
    has_linear_attention_layers,
)
from halo.models.config_access import model_type_matches
from halo.models.moe.family_specs import find_moe_family_spec_for_config
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# (model_type, mode) pairs whose allowed cross-document linear attention was reported: a script refuses
# before the model load and the collator again after it.
_crossing_allowed_warned: set[tuple[str | None, str]] = set()

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


def has_dense_packing_leak(model_config) -> bool:
    """Whether a packed row runs as one causal sequence on a dense backend: the family's forward passes no
    ``position_ids`` into mask construction (``MoEFamilySpec.dense_packing_leaks``), read through a VLM wrapper."""
    spec = find_moe_family_spec_for_config(model_config)
    return spec is not None and spec.dense_packing_leaks


def segment_markers_for(model_config) -> SegmentMarkers:
    """The markers this model's mixers need to keep a multi-document row's documents apart, read through the
    text sub-config so a VLM wrapper resolves like its decoder: ShortConv by family, GatedDeltaNet by its
    declared layers (:func:`~halo.models.attention.layout.has_gated_delta_net_layers`)."""
    if model_config is None:
        return SegmentMarkers()
    gated_delta_net = has_gated_delta_net_layers(model_config)
    return SegmentMarkers(
        seq_idx=gated_delta_net or model_type_matches(model_config, *SHORT_CONV_MODEL_TYPE_PREFIXES),
        cu_seq_lens=gated_delta_net,
    )


def find_unmarked_sequence_mixer(model_config) -> str | None:
    """The sequence mixers of the model no segment marker reaches, named for a refusal: the family's declared ones
    (``MoEFamilySpec.unmarked_sequence_mixer``, read through a VLM wrapper: Inkling's short convolutions), or
    linear-attention layers, by hybrid layer groups
    (:func:`~halo.models.attention.layout.declares_linear_layer_groups`) or ``linear_attention`` layer types
    (:func:`~halo.models.attention.layout.has_linear_attention_layers`), outside the families whose kernels read
    ``cu_seq_lens``; ``None`` where every mixer is reached. Such mixers carry each document of a multi-document row
    into the next on every backend."""
    if model_config is None:
        return None
    spec = find_moe_family_spec_for_config(model_config)
    if spec is not None and spec.unmarked_sequence_mixer is not None:
        return spec.unmarked_sequence_mixer
    declares = declares_linear_layer_groups(model_config) or has_linear_attention_layers(model_config)
    return "linear-attention layers" if declares and not segment_markers_for(model_config).cu_seq_lens else None


def reject_unmarked_linear_attention_rows(
    model_config, mode: str, *, allow_cross_document_linear_attention: bool
) -> None:
    """Refuse a multi-document row (``mode``) on a model whose linear-attention layers, or a family's other
    sequence mixers, no segment marker reaches (:func:`find_unmarked_sequence_mixer`), on every attention
    backend, unless the run sets ``allow_cross_document_linear_attention``; an allowed run is warned once."""
    mixer = find_unmarked_sequence_mixer(model_config)
    if mixer is None:
        return
    model_type = getattr(model_config, "model_type", None)
    crossing = (
        f"its {mixer} receive no document boundary, so their state carries from each document of a row into "
        f"the next on every attention backend"
    )
    table = "agent-docs/data/collators.md (Document isolation under packing) lists the layers each family's rows cross"
    if not allow_cross_document_linear_attention:
        raise ValueError(
            f"{mode}=True is refused for model_type={model_type!r}: {crossing}, and nothing in the loss shows "
            f"it. Turn {mode} off, or set allow_cross_document_linear_attention: true where that mixing is "
            f"acceptable; {table}."
        )
    if is_global_main_process():
        warn_once(
            logger,
            _crossing_allowed_warned,
            (model_type, mode),
            "%s=True on model_type=%r under allow_cross_document_linear_attention: %s; %s.",
            mode,
            model_type,
            crossing,
            table,
        )


def reject_pipeline_packed_rows(model_config) -> None:
    """Refuse packed rows under pipeline parallelism on a model whose sequence mixers read no document boundary
    (:func:`find_unmarked_sequence_mixer`). The segment markers a family's mixers read
    (:func:`segment_markers_for`) are rebuilt on every stage from its microbatch's ``position_ids``
    (:func:`packed_row_segment_kwargs`), so their kernels must be the segment-aware ones that read them."""
    require_segment_aware_kernels(model_config, "packing")
    unmarked = find_unmarked_sequence_mixer(model_config)
    if unmarked is None:
        return
    raise ValueError(
        f"packing under pipeline parallelism is refused for model_type={getattr(model_config, 'model_type', None)!r}: "
        f"its {unmarked} read no document boundary on any path, so their state carries from each document of a packed "
        f"row into the next. Disable packing (padded rows hold one document each), or train without pipeline "
        f"parallelism."
    )


def reject_compressed_kv_rows(model_config, mode: str) -> None:
    """Refuse a multi-document row (``mode``, the flag that asks for one) on a model with
    compressed-KV layers (:func:`~halo.models.attention.layout.compressed_layer_types`), on every
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
    """Refuse a multi-document row (``mode``) on a model with GatedDeltaNet layers lacking their
    segment-aware kernels, the only ones that read the markers."""
    if not has_gated_delta_net_layers(model_config):
        return
    require_linear_attention_kernels(getattr(model_config, "model_type", None), mode)


def require_linear_attention_kernels(model_type: str | None, mode: str) -> None:
    """Refuse ``mode`` on a linear-attention family (``model_type``) whose segment-aware conv and
    delta-rule kernels are missing; tree packing segments GLM-5 Next's KDA through the same two."""
    missing = ", ".join(name for name, is_available in GDN_SEGMENT_AWARE_BACKENDS if not is_available())
    if missing:
        raise ValueError(
            f"{mode}=True is refused for the linear-attention family "
            f"model_type={model_type!r}: {missing} unavailable. "
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


def document_boundaries(seq_idx: torch.Tensor | None, batch: int, length: int, device: torch.device) -> torch.Tensor:
    """``[batch, length]`` bool, True at the first token of every document: each row's first token and every
    change of ``seq_idx`` (:func:`document_ids`) along it. ``None`` is one document per row."""
    boundaries = torch.zeros(batch, length, dtype=torch.bool, device=device)
    boundaries[:, :1] = True
    if seq_idx is None:
        return boundaries
    if seq_idx.shape != (batch, length) or seq_idx.dtype.is_floating_point or seq_idx.dtype == torch.bool:
        raise ValueError(
            f"seq_idx must be an integer [batch, sequence] = {(batch, length)} tensor of document indices, got "
            f"{seq_idx.dtype} {tuple(seq_idx.shape)}"
        )
    boundaries[:, 1:] = seq_idx[:, 1:] != seq_idx[:, :-1]
    return boundaries


def document_starts(seq_idx: torch.Tensor | None, batch: int, length: int, device: torch.device) -> torch.Tensor:
    """``[batch, length]`` long: the index of the first token of each token's document."""
    positions = torch.arange(length, device=device)
    starts = torch.where(document_boundaries(seq_idx, batch, length, device), positions, 0)
    return starts.cummax(dim=1).values


def flattened_document_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """``position_ids`` of a padded batch flattened into one ``[1, real_tokens]`` row: each row's real
    tokens become one document, counted from 0 wherever the row's padding sat."""
    return (attention_mask.cumsum(dim=1) - 1)[attention_mask.bool()].unsqueeze(0)


def positions_restart(position_ids: torch.Tensor) -> bool:
    """Whether a row's positions restart past its first token: transformers' own test for a packed row
    (``_is_packed_sequence``), on which its flash path takes the varlen kernel, for rows that start at 0."""
    return bool((position_ids[..., 1:] == 0).any())


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


def single_row_varlen_kwargs(batch: Mapping[str, Any]) -> dict[str, torch.Tensor | int]:
    """The varlen set of a collated batch that is one ``[1, S]`` document without padding: ``cu_seq_lens_q/k``
    ``[0, S]`` and host-int ``max_length_q/k`` ``S``, what transformers derives from the row's positions. Without it,
    transformers' flash path tests the row for packing in every attention layer (``_is_packed_sequence``, a device
    sync) before running its dense kernel. Empty for several rows, a padded row (unpadded by its mask instead) and
    restarting ``position_ids``."""
    input_ids, attention_mask = batch["input_ids"], batch.get("attention_mask")
    position_ids = batch.get("position_ids")
    if input_ids.shape[0] != 1 or (attention_mask is not None and not bool(attention_mask.all())):
        return {}
    if position_ids is not None and positions_restart(position_ids):
        return {}
    return segment_marker_kwargs(torch.arange(input_ids.shape[1]).unsqueeze(0), SegmentMarkers(cu_seq_lens=True))


def packed_row_segment_kwargs(position_ids: torch.Tensor, markers: SegmentMarkers) -> dict[str, torch.Tensor | int]:
    """The forward kwargs ``markers`` selects (:func:`segment_marker_kwargs`) for one ``[1, L]`` row, built from a
    single host copy of its ``position_ids``: tensors on the row's device, ``max_length_q/k`` as host ints. Empty for
    a row whose positions do not restart, which is one document."""
    host = position_ids.cpu()
    if not positions_restart(host):
        return {}
    kwargs = segment_marker_kwargs(host, markers)
    # One copy per tensor: the q and k entries of the varlen set are one tensor.
    tensors = {id(value): value for value in kwargs.values() if isinstance(value, torch.Tensor)}
    on_device = {key: tensor.to(position_ids.device, non_blocking=True) for key, tensor in tensors.items()}
    return {key: on_device.get(id(value), value) for key, value in kwargs.items()}


def packed_row_varlen_kwargs(position_ids: torch.Tensor) -> dict[str, torch.Tensor | int]:
    """The varlen set flash attention would derive from one ``[1, L]`` row's ``position_ids`` in every layer
    (:func:`packed_row_segment_kwargs`); empty for a row transformers runs on its dense kernel."""
    return packed_row_segment_kwargs(position_ids, SegmentMarkers(cu_seq_lens=True))
