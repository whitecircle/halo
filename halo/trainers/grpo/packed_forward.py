"""The flattened varlen input of a packed GRPO micro-batch: its rows concatenated into one ``[1, N]`` sequence whose
positions restart per row, the document markers the family's mixers read, and the hidden positions each row scores."""

from typing import NamedTuple

import torch

from halo.models.segment_markers import SegmentMarkers, packed_row_segment_kwargs


class PackedRows(NamedTuple):
    """``input_ids`` and ``position_ids`` are ``[1, N]``; ``kwargs`` the markers to forward beside them (empty for one
    document, which runs as the per-row path does); ``hidden_index`` the flat positions whose hidden states predict each
    row's real completion tokens, row after row."""

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    kwargs: dict
    hidden_index: torch.Tensor


def build_packed_rows(
    input_ids: torch.Tensor,
    spans: list[tuple[int, int]],
    widths: list[int],
    markers: SegmentMarkers,
) -> PackedRows:
    """Rows ``input_ids[i, lo:hi]`` for ``(lo, hi)`` in ``spans``, flattened. ``widths[i]`` is how many trailing
    positions row ``i`` scores (the per-row path's ``hidden[:, :-1][:, -width:]``), so its hidden positions are the
    ``width`` before its last token."""
    row_width = input_ids.size(1)
    lengths = [hi - lo for lo, hi in spans]
    ends = torch.tensor(lengths).cumsum(0).tolist()
    source = torch.cat([torch.arange(i * row_width + lo, i * row_width + hi) for i, (lo, hi) in enumerate(spans)])
    positions = torch.cat([torch.arange(length) for length in lengths]).unsqueeze(0)
    hidden_index = torch.cat([torch.arange(end - 1 - width, end - 1) for end, width in zip(ends, widths, strict=True)])
    device = input_ids.device
    kwargs = packed_row_segment_kwargs(positions, markers)
    moved = {id(value): value.to(device, non_blocking=True) for value in kwargs.values() if torch.is_tensor(value)}
    return PackedRows(
        input_ids.reshape(-1)[source.to(device, non_blocking=True)].unsqueeze(0),
        positions.to(device, non_blocking=True),
        {key: moved.get(id(value), value) for key, value in kwargs.items()},
        hidden_index.to(device, non_blocking=True),
    )
