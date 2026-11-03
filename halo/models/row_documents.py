"""The documents of a training row as spans: where each sits and which of its tokens padding leaves real. What a forward
replaced to keep packed documents apart walks per document."""

from dataclasses import dataclass

import torch

from halo.models.segment_markers import document_boundaries


@dataclass(frozen=True)
class RowDocument:
    """``[start, stop)`` is the document's span of ``row``, ``[key_start, key_stop)`` its unpadded tokens."""

    row: int
    start: int
    stop: int
    key_start: int
    key_stop: int

    def visible_keys(self, lo: int, hi: int, device: torch.device) -> torch.Tensor:
        """Per query of ``[lo, hi)``, how many of the document's keys it sees, those up to itself."""
        return (torch.arange(lo, hi, device=device) - self.key_start + 1).clamp(0, self.key_stop - self.key_start)

    def tile_keys(self, hi: int) -> int:
        """The span every tile's K/V slice covers."""
        return min(max(hi - self.key_start, 0), self.key_stop - self.key_start)


def row_documents(
    seq_idx: torch.Tensor | None, attention_mask: torch.Tensor | None, batch: int, length: int, device: torch.device
) -> tuple[RowDocument, ...]:
    """In row-major order. A document's unpadded tokens must be contiguous, as left or right padding leaves them."""
    if attention_mask is not None and attention_mask.shape != (batch, length):
        raise ValueError(
            f"Row documents take a 2-D [batch, sequence] = {(batch, length)} padding mask, got "
            f"{tuple(attention_mask.shape)}"
        )
    boundaries = document_boundaries(seq_idx, batch, length, device)
    rows, starts = boundaries.nonzero(as_tuple=True)
    document = boundaries.flatten().cumsum(0) - 1
    last_of_row = torch.ones_like(rows, dtype=torch.bool)
    last_of_row[:-1] = rows[1:] != rows[:-1]
    stops = torch.where(last_of_row, length, starts.roll(-1))
    valid = (
        torch.ones(batch * length, dtype=torch.bool, device=device)
        if attention_mask is None
        else attention_mask.to(device=device, dtype=torch.bool).flatten()
    )
    positions = torch.arange(length, device=device).repeat(batch)
    count = torch.zeros_like(starts).scatter_add_(0, document, valid.long())
    first = starts.clone().scatter_reduce_(0, document[valid], positions[valid], "amin", include_self=False)
    last = (starts - 1).scatter_reduce_(0, document[valid], positions[valid], "amax", include_self=False)
    if bool(((last - first + 1) != count).any()):
        raise ValueError(
            "Row documents need each document's unpadded tokens to be contiguous (left or right padding); the padding "
            "mask interleaves padding inside a document."
        )
    spans = torch.stack([rows, starts, stops, first, first + count]).T.tolist()
    return tuple(RowDocument(*span) for span in spans)
