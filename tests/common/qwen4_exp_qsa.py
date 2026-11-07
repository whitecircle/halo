"""Flash-Next QSA test fixtures: packed and padded rows, and an fp64 reference that reads the key set off
``(blocks, spans)`` alone, as the block-sparse kernel does."""

from dataclasses import dataclass

import torch
from transformers.masking_utils import create_causal_mask

from tests.common.models import TINY_QWEN4_EXP_CONFIG
from tests.common.qwen4_exp import (
    packed_positions,
    packed_seq_idx,
    rotary_embeddings,
    tiny_qwen4_exp_text_config,
)

INPUT_SEED = 1235


@dataclass(frozen=True)
class Rows:
    """Each row's document lengths; in ``padding``, ``+n`` pads the last n tokens and ``-n`` the first n."""

    documents: tuple[tuple[int, ...], ...]
    padding: tuple[int, ...] | None = None


ROWS = {
    "one-document": Rows(((300,),)),
    "packed": Rows(((13, 150, 77, 60),)),
    "packed-two-rows": Rows(((100, 200), (300,))),
    "right-padded": Rows(((300,), (300,)), padding=(80, 0)),
    "left-padded": Rows(((300,), (300,)), padding=(-80, 0)),
}
# A document under one block has no complete block, so its queries attend only the tail.
SHORT_DOCUMENT_ROWS = Rows(((2, 3, 90, 1),))


@dataclass(frozen=True)
class Inputs:
    hidden: torch.Tensor
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    padding_mask: torch.Tensor | None
    seq_idx: torch.Tensor | None
    dense_mask: torch.Tensor


def build_inputs(config, rows: Rows, hidden: torch.Tensor | None = None) -> Inputs:
    """Packed rows isolate their documents through ``position_ids``, padded rows through the 2-D mask."""
    position_ids = packed_positions([list(lengths) for lengths in rows.documents])
    batch, length = position_ids.shape
    if hidden is None:
        torch.manual_seed(INPUT_SEED)
        hidden = torch.randn(batch, length, config.hidden_size)
    padding_mask, seq_idx = None, packed_seq_idx(position_ids)
    if rows.padding is not None:
        padding_mask, seq_idx = torch.ones(batch, length, dtype=torch.long), None
        for row, pad in enumerate(rows.padding):
            if pad > 0:
                padding_mask[row, length - pad :] = 0
            elif pad < 0:
                padding_mask[row, :-pad] = 0
    dense_mask = create_causal_mask(
        config=config,
        inputs_embeds=hidden,
        attention_mask=padding_mask,
        past_key_values=None,
        position_ids=position_ids,
        allow_is_causal_skip=False,
    )
    return Inputs(hidden, rotary_embeddings(config, hidden, position_ids), padding_mask, seq_idx, dense_mask)


def layout_config(heads: int, kv_heads: int, head_dim: int, **overrides):
    """The tiny text config over another attention layout: its M-RoPE sections sum to half the rotary width and its
    index head is at least as wide as the rotary width, which the config validates."""
    rope = dict(TINY_QWEN4_EXP_CONFIG["rope_parameters"])
    rotary_dim = int(head_dim * rope["partial_rotary_factor"])
    half = rotary_dim // 2
    rope["mrope_section"] = [half // 3 + (1 if index < half % 3 else 0) for index in range(3)]
    overrides.setdefault("indexer_head_dim", max(TINY_QWEN4_EXP_CONFIG["indexer_head_dim"], rotary_dim))
    return tiny_qwen4_exp_text_config(
        num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=head_dim, rope_parameters=rope, **overrides
    )


def attention_operands(config, batch: int, length: int, seed: int) -> tuple[torch.Tensor, ...]:
    """Query ``[batch, heads, length, dim]``, key and value ``[batch, kv_heads, length, dim]`` and an upstream gradient
    ``[batch, length, heads, dim]``, shaped from ``config``."""
    generator = torch.Generator().manual_seed(seed)
    heads, kv_heads, dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    shapes = [(batch, heads, length, dim), (batch, kv_heads, length, dim), (batch, kv_heads, length, dim)]
    query, key, value = (torch.randn(shape, generator=generator) for shape in shapes)
    return query, key, value, torch.randn(batch, length, heads, dim, generator=generator)


def key_set_from_spans(blocks: torch.Tensor, spans: torch.Tensor, ratio: int) -> torch.Tensor:
    """``[batch, length, length]`` bool, on ``blocks``' device: the keys each query attends given only its block ids and
    ``(first, visible)``. Slots past ``min(visible // ratio, block_topk)`` are ignored, as the kernel does."""
    batch, length, block_topk = blocks.shape
    device = blocks.device
    first, visible = spans[..., 0].long(), spans[..., 1].long()
    relative = torch.arange(length, device=device)[None, None, :] - first[..., None]
    in_document = (relative >= 0) & (relative < visible[..., None])
    block = relative.clamp(min=0) // ratio
    complete = visible // ratio
    selected = complete.clamp(max=block_topk)
    slot_is_selected = torch.arange(block_topk, device=device) < selected[..., None]
    count = -(-length // ratio)
    chosen = torch.zeros(batch, length, count + 1, dtype=torch.bool, device=device)
    chosen.scatter_(2, torch.where(slot_is_selected, blocks.long(), count), True)
    in_chosen_block = chosen.gather(2, block.clamp(max=count - 1))
    in_tail = relative >= (complete * ratio)[..., None]
    return in_document & (in_chosen_block | in_tail)


def gather_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    blocks: torch.Tensor,
    spans: torch.Tensor,
    ratio: int,
    scale: float,
) -> torch.Tensor:
    """fp64 ``[batch, heads, length, dim]``; a query that attends no key reads zeros."""
    allowed = key_set_from_spans(blocks, spans, ratio).unsqueeze(1)
    groups = query.shape[1] // key.shape[1]
    key, value = (t.double().repeat_interleave(groups, dim=1) for t in (key, value))
    scores = (query.double() @ key.transpose(-1, -2) * scale).masked_fill(~allowed, -torch.inf)
    weights = torch.nan_to_num(scores.softmax(-1))
    return weights @ value


def error_ratio(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Largest absolute error over the largest magnitude of ``expected``."""
    return ((actual.double() - expected.double()).abs().max() / expected.abs().max().clamp_min(1e-30)).item()
