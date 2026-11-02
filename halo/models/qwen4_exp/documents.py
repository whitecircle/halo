"""Document boundaries of a ``qwen4_exp`` training row, and the per-layer embedding (PLE) computed within them.

A multi-document row marks its documents with ``seq_idx``, the per-token document index of
:func:`halo.models.segment_markers.document_ids`. Stock PLE crosses them twice: its n-gram history resets only
after ``eos_token_id``, and its dilated depthwise conv never resets. Here the history resets at every document
start as well as after eos, and the conv runs as one masked shifted multiply-add per tap whose taps stop at the
document start, so a one-document row computes what the stock layer computes and each packed document computes
what it would alone. Uncached rows only: a cached call keeps the stock forward.
"""

import math

import torch
import torch.nn.functional as F
import transformers.models.qwen4_exp.modeling_qwen4_exp as qwen4_exp_modeling
from torch import nn

from halo.models.qwen4_exp.ngram_table import HostNGramTable


def document_boundaries(seq_idx: torch.Tensor | None, batch: int, length: int, device: torch.device) -> torch.Tensor:
    """``[batch, length]`` bool, True at the first token of every document: each row's first token and every
    change of ``seq_idx`` along it. ``None`` is one document per row."""
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


def reject_unmarked_documents(position_ids: torch.Tensor | None, seq_idx: torch.Tensor | None) -> None:
    """Refuse a row whose text ``position_ids`` restart at 0 (a packed row) unless ``seq_idx`` marks the same
    documents: the QSA indexer, the PLE n-gram history and its conv read documents from ``seq_idx`` alone."""
    if position_ids is None and seq_idx is None:
        return
    reference = seq_idx if seq_idx is not None else position_ids
    batch, length = reference.shape
    restarts = torch.zeros(batch, length, dtype=torch.bool, device=reference.device)
    if position_ids is not None:
        restarts[:, 1:] = position_ids[:, 1:] == 0
    if seq_idx is None:
        if bool(restarts.any()):
            raise ValueError(
                "A qwen4_exp row whose position_ids restart at 0 (a packed row) arrived without seq_idx: its QSA "
                "indexer, PLE n-gram history and PLE conv would read across the documents. The packing collators "
                "emit seq_idx for this family (segment_markers_for); pass it with every packed row."
            )
        return
    starts = document_boundaries(seq_idx, batch, length, reference.device)
    starts[:, :1] = False
    if not torch.equal(starts, restarts):
        raise ValueError(
            "seq_idx marks different documents than the position_ids restarts of the same row: build it with "
            "segment_marker_kwargs from the row's own position_ids."
        )


def shift_within_documents(
    token_ids: torch.Tensor, shift: int, eos_token_id: int, starts: torch.Tensor
) -> torch.Tensor:
    """``token_ids`` moved ``shift`` places right, ``eos_token_id`` wherever the source token lies before the
    token's document start or before the eos that ends the previous segment: transformers'
    ``Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos`` with a document start as a second reset."""
    if shift == 0:
        return token_ids
    positions = torch.arange(token_ids.shape[1], device=token_ids.device)
    eos_positions = torch.where(token_ids == eos_token_id, positions, -1)
    previous_eos = F.pad(eos_positions.cummax(dim=1).values[:, :-1], (1, 0), value=-1)
    segment_start = torch.maximum(previous_eos + 1, starts)
    sources = (positions - shift).clamp_min(0).expand_as(token_ids)
    shifted = token_ids.gather(dim=1, index=sources)
    return torch.where(positions - segment_start >= shift, shifted, token_ids.new_full((), eos_token_id))


def document_ngram_ids(embedding: nn.Module, input_ids: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
    """The ``[batch, length, ngram_heads]`` hashed n-gram table rows of a ``Qwen4ExpTextNGramEmbedding`` for an
    uncached row, its history reset at document ``starts`` (:func:`shift_within_documents`)."""
    input_ids = input_ids.long()
    shifted = [
        shift_within_documents(input_ids, shift, embedding.eos_token_id, starts)
        for shift in range(embedding.ngram_size)
    ]
    multipliers = embedding.layer_multipliers
    blocks = []
    for ngram in range(2, embedding.ngram_size + 1):
        heads = slice((ngram - 2) * embedding.heads_per_ngram, (ngram - 1) * embedding.heads_per_ngram)
        mixed = shifted[0] * multipliers[0]
        for position in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[position] * multipliers[position])
        ngram_ids = torch.remainder(mixed.unsqueeze(-1), embedding.ngram_heads_vocab_sizes[heads].view(1, 1, -1))
        blocks.append(ngram_ids + embedding.ngram_heads_offsets[heads].view(1, 1, -1))
    return torch.cat(blocks, dim=-1)


def ngram_rows(embedding: nn.Module, ngram_ids: torch.Tensor) -> torch.Tensor:
    """The concatenated table rows of ``ngram_ids`` on their device: a :class:`HostNGramTable`'s own lookup, or the
    stock embedding read where its weight lives, as the stock forward reads it."""
    table = embedding.ngram_embedding
    if isinstance(table, HostNGramTable):
        return table(ngram_ids).flatten(-2)
    device = table.weight.device if table.weight.device.type != "meta" else None
    return table(ngram_ids.to(device)).to(ngram_ids.device).flatten(-2)


def document_short_conv(conv: nn.Module, hidden_states: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
    """``silu`` of the PLE dilated depthwise causal conv over ``[batch, length, channels]`` states, every tap that
    reaches before its token's document start reading zero, as at a row start.

    One multiply-add per tap over a view of the left-padded states keeps the backward to that one padded copy and
    off cuDNN; taps accumulate in fp32 and the sum returns in the states' dtype, as the conv's own output does.
    """
    if (
        not isinstance(conv, nn.Conv1d)
        or conv.bias is not None
        or conv.stride != (1,)
        or conv.padding != (0,)
        or not conv.in_channels == conv.out_channels == conv.groups
    ):
        raise TypeError(
            f"The document-aware PLE conv reads the weights of an unbiased, stride-1, unpadded depthwise nn.Conv1d "
            f"directly, got {conv!r}; an adapter on the PLE conv1d would be bypassed. Leave it out of "
            f"lora_target_modules."
        )
    length = hidden_states.shape[1]
    kernel, dilation = conv.kernel_size[0], conv.dilation[0]
    reach = (kernel - 1) * dilation
    padded = F.pad(hidden_states, (0, 0, reach, 0))
    offsets = torch.arange(length, device=hidden_states.device) - starts
    weight = conv.weight[:, 0, :]
    out = None
    for tap in range(kernel):
        lag = reach - tap * dilation
        term = padded[:, reach - lag : reach - lag + length] * weight[:, tap]
        if lag:
            term = term * (offsets >= lag).unsqueeze(-1).to(term.dtype)
        out = term.float() if out is None else out + term.float()
    return F.silu(out.to(hidden_states.dtype))


def document_ple_forward(
    ple: nn.Module,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    seq_idx: torch.Tensor | None,
    conv_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """``Qwen4ExpTextPLELayer.forward`` for an uncached row, with the n-gram history and the conv kept inside each
    ``seq_idx`` document (:func:`document_ngram_ids`, :func:`document_short_conv`)."""
    batch, length = input_ids.shape
    starts = document_starts(seq_idx, batch, length, input_ids.device)
    embeddings = ngram_rows(ple.ple_embedding, document_ngram_ids(ple.ple_embedding, input_ids, starts))
    streams = (ple.hc_count, ple.hidden_size)
    key_normed = ple.norm_key(ple.key_proj(embeddings)).unflatten(-1, streams)
    value = ple.value_proj(embeddings)
    query_normed = ple.norm_query(hidden_states).unflatten(-1, streams)
    gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(ple.hidden_size)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
    gated_value_normed = ple.norm_conv(gated_value.flatten(-2))
    gated_value = gated_value.flatten(-2)
    if conv_mask is not None:
        gated_value = qwen4_exp_modeling.apply_mask_to_padding_states(gated_value, conv_mask)
        gated_value_normed = qwen4_exp_modeling.apply_mask_to_padding_states(gated_value_normed, conv_mask)
    return gated_value + document_short_conv(ple.conv1d, gated_value_normed, starts)
