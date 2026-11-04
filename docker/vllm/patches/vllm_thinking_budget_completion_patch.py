"""Finish a forced thinking-budget end sequence that contains the natural end marker.

Model Runner V2 (``vllm.v1.worker.gpu.sample.thinking_budget``) counts reasoning while the last
start marker sits after the last natural end marker, and once the budget runs out it forces the
configured end sequence one token per step. It stops forcing as soon as a natural end marker follows
the start marker, so a forced sequence that carries the natural end before its last token is cut
short there. gpt-oss needs exactly that shape: its count ends at any channel header (``<|channel|>``,
so a tool call or the final answer is never counted), while the cut it forces is the whole move into
the final channel (``<|end|><|start|>assistant<|channel|>final<|message|>``). Unpatched, the cut stops
at ``<|channel|>`` and the model picks the channel itself, including a fresh, uncounted analysis.

Fix: when the kernel forces the token that completes the inner natural end, the request's slot
records that position; on that step and the next ones, a row whose tail is the forced prefix anchored
there gets the following forced token, up to the end of the sequence. The anchor tells a forced cut
from the same header written by the model; admitting a budgeted request clears its slot's anchor.
Configurations whose natural end is the forced sequence's tail (every upstream reasoning parser) have
no inner end and are untouched. The V1 runner has no separate natural end marker and is not patched.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Stamped on the replacements so ``Dockerfile.vllm`` can assert the patch took.
PATCH_MARKER = "_halo_thinking_budget_completion"

# The logit the budget kernel writes on a forced token.
FORCED_LOGIT = 1.0e9

_APPLIED = False


def inner_end_length(forced_ids: list[int], natural_end_ids: list[int]) -> int:
    """Length of the shortest forced prefix ending in the natural end marker, short of the full sequence.

    Zero when the natural end only closes the forced sequence (nothing to finish).
    """
    width = len(natural_end_ids)
    for length in range(width, len(forced_ids)):
        if forced_ids[length - width : length] == natural_end_ids:
            return length
    return 0


def _effective_tail(state, ctx, slots: torch.Tensor, eff_len: torch.Tensor, width: int) -> torch.Tensor:
    """``[rows, width]``: column ``j`` is the token at ``eff_len - 1 - j`` (``-1`` before the sequence).

    Committed tokens come from the request's history, this step's draft tokens from ``input_ids``, the
    same split the budget kernel reads.
    """
    total = state.req_states.total_len.gpu[slots].long()
    offsets = torch.arange(1, width + 1, device=eff_len.device)
    pos = eff_len[:, None] - offsets[None, :]
    history = state.req_states.all_token_ids.gpu[slots[:, None], pos.clamp(min=0)].long()
    first_row = torch.arange(slots.shape[0], device=eff_len.device) - ctx.expanded_local_pos.long()
    draft_index = (first_row[:, None] + pos - total[:, None] + 1).clamp(0, ctx.input_ids.shape[0] - 1)
    drafts = ctx.input_ids.long()[draft_index]
    tail = torch.where(pos < total[:, None], history, drafts)
    return torch.where(pos >= 0, tail, torch.full_like(tail, -1))


def _finish_forced_sequences(state, logits: torch.Tensor, ctx) -> None:
    forced_ids = state.reasoning_end_token_ids.long()
    inner = state._halo_inner_end_length
    slots = ctx.expanded_idx_mapping.long()
    rows = torch.arange(slots.shape[0], device=logits.device)
    active = state.thinking_token_budget.gpu[slots] >= 0
    eff_len = state.req_states.total_len.gpu[slots].long() + ctx.expanded_local_pos.long()

    # Record where the kernel forced the token that completes the inner natural end. A slot owns one
    # row per draft token, and only its earliest forcing row is sure to be sampled: a later one sits
    # behind a draft the forced token already rejected.
    completes_inner = active & (logits[rows, forced_ids[inner - 1]] >= FORCED_LOGIT)
    unset = torch.iinfo(torch.int32).max
    forced_at = torch.where(completes_inner, eff_len, torch.full_like(eff_len, unset)).int()
    earliest = torch.full_like(state._halo_inner_end_pos, unset).scatter_reduce_(0, slots, forced_at, reduce="amin")
    state._halo_inner_end_pos.copy_(torch.where(earliest != unset, earliest, state._halo_inner_end_pos))

    anchor = state._halo_inner_end_pos[slots].long()
    tail = _effective_tail(state, ctx, slots, eff_len, forced_ids.shape[0] - 1)
    for length in range(inner, forced_ids.shape[0]):
        written = (tail[:, :length] == forced_ids[:length].flip(0)[None, :]).all(dim=1)
        anchored = anchor == eff_len - length + inner - 1
        hit = active & written & anchored
        column = logits[:, forced_ids[length]]
        logits[:, forced_ids[length]] = torch.where(hit, torch.full_like(column, FORCED_LOGIT), column)


def apply() -> None:
    """Idempotently patch V2's ``ThinkingBudgetState`` to finish forced sequences past an inner end."""
    global _APPLIED
    if _APPLIED:
        return

    from vllm.v1.worker.gpu.sample import thinking_budget  # noqa: PLC0415 — vLLM-only lazy import

    state_cls = thinking_budget.ThinkingBudgetState
    original_init = state_cls.__init__
    original_staged = state_cls.apply_staged_writes
    original_apply = state_cls.apply

    def __init__(self, req_states, reasoning_config):
        original_init(self, req_states, reasoning_config)
        self._halo_inner_end_length = 0
        if not self.enabled:
            return
        self._halo_inner_end_length = inner_end_length(
            self.reasoning_end_token_ids.tolist(), self.natural_reasoning_end_token_ids.tolist()
        )
        self._halo_inner_end_pos = torch.full((self.max_num_reqs,), -1, dtype=torch.int32, device=self.device)

    def apply_staged_writes(self):
        if self.enabled and self._halo_inner_end_length and self._reset_reqs:
            reset = thinking_budget.async_tensor_h2d(self._reset_reqs, dtype=torch.int64, device=self.device)
            self._halo_inner_end_pos.index_fill_(0, reset, -1)
        original_staged(self)

    def apply(self, logits, ctx):
        original_apply(self, logits, ctx)
        if self.enabled and self._halo_inner_end_length and np.any(self.use_thinking_budget[ctx.idx_mapping_np]):
            _finish_forced_sequences(self, logits, ctx)

    for replacement in (__init__, apply_staged_writes, apply):
        setattr(replacement, PATCH_MARKER, True)
    state_cls.__init__ = __init__
    state_cls.apply_staged_writes = apply_staged_writes
    state_cls.apply = apply
    _APPLIED = True
    logger.info("vllm_thinking_budget_completion_patch applied to %s", state_cls.__name__)
