"""Per-batch token accounting: the loss mask, loss-contributing tokens and each batch's attention-score work.

Nothing syncs the host per micro-batch: counts fold into on-device accumulators, the token total gathers
once per log step and the attention work is read once per optimizer step, once a callback binds as its sink.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch

from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.models.attention.tree_plan import TREE_ATTENTION_KEY
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


def effective_loss_mask(masks: Mapping[str, torch.Tensor | Any]) -> torch.Tensor | None:
    """The mask of loss-contributing tokens from a batch/result mapping, or ``None`` without one:
    ``completion_mask ∧ tool_mask`` where a ``tool_mask`` exists (tool use), else ``completion_mask``, which
    alone still counts tool-output tokens the loss never trains on."""
    completion_mask = masks.get("completion_mask")
    if completion_mask is None:
        return None
    tool_mask = masks.get("tool_mask")
    return completion_mask if tool_mask is None else completion_mask * tool_mask


class TokenMetricsMixin:
    """Per-rank loss-token and attention-work accumulation and their cross-rank totals."""

    # Set by the first ``training_step``: the efficiency callback binds here and hands over the layout.
    _attention_work_bound: bool = False
    _attention_layout = None

    def _extract_document_lengths(self, inputs):
        """Token length of every document the batch's attention scores (zeros allowed), or ``None`` without
        ``input_ids``: ``cu_seq_lens_q`` when padding-free, the ``position_ids`` resets of packed rows, else one
        document per row over its ``attention_mask``. Every buffer is sized from a shape: no host sync."""
        if not isinstance(inputs, Mapping):
            return None
        cu_seq_lens = inputs.get("cu_seq_lens_q")
        if cu_seq_lens is not None:
            return (cu_seq_lens[1:] - cu_seq_lens[:-1]).long()
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            return None
        attention_mask = inputs.get("attention_mask")
        valid = attention_mask.bool() if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)
        position_ids = inputs.get("position_ids")
        if position_ids is None or position_ids.shape != input_ids.shape:
            return valid.sum(dim=1)
        # A document starts where the positions reset and at every row's first valid token, so rows
        # never merge; a running count of starts over the flattened batch numbers the documents.
        starts = ((position_ids == 0) | (valid.cumsum(dim=1) == 1)) & valid
        document = torch.cumsum(starts.flatten().long(), dim=0)
        lengths = torch.zeros(document.numel() + 1, dtype=torch.long, device=document.device)
        lengths.scatter_add_(0, document, valid.flatten().long())
        return lengths

    def _tree_node_lengths(self, inputs):
        """``(path length, own tokens)`` of every node of a tree-packed batch, or ``None`` for any other: a
        node attends along its root-to-leaf path, its last token's position plus one."""
        plan = inputs.get(TREE_ATTENTION_KEY) if isinstance(inputs, Mapping) else None
        if plan is None:
            return None
        bounds = plan["cu_seqlens_q"].long()
        return inputs["position_ids"][0, bounds[1:] - 1] + 1, bounds[1:] - bounds[:-1]

    def _accumulate_attention_flops(self, inputs):
        """Cost this micro-batch's documents through the bound layout into the on-device accumulator,
        read once per optimizer step by :meth:`drain_attention_flops`."""
        if not self._attention_work_bound:
            # Offered to every callback that consumes measured attention work (duck-typed, so the trainer
            # package imports nothing from the callbacks); one answers with the layout it costs batches with.
            self._attention_work_bound = True
            for callback in self.callback_handler.callbacks:
                bind = getattr(callback, "bind_attention_work_source", None)
                if bind is not None:
                    self._attention_layout = bind(self)
        if self._attention_layout is None:
            return
        nodes = self._tree_node_lengths(inputs)
        lengths, own_tokens = nodes if nodes is not None else (self._extract_document_lengths(inputs), None)
        if lengths is None:
            return
        lengths = lengths.detach()
        flops = self._attention_layout.flops_for_documents(lengths, own_tokens)
        step = torch.stack((flops, (lengths > 0).sum().to(flops)))
        prev = getattr(self, "_local_attention_work_accum", None)
        self._local_attention_work_accum = step if prev is None else prev + step

    def drain_attention_flops(self) -> tuple[float, int]:
        """``(this rank's attention-score FLOPs, documents)`` since the last drain, then reset: one host read per
        optimizer step. Rank-local: each pipeline stage costs the documents through its own layer slice."""
        accum = getattr(self, "_local_attention_work_accum", None)
        if accum is None:
            return 0.0, 0
        self._local_attention_work_accum = None
        flops, documents = accum.tolist()
        return flops, int(documents)

    def _extract_output_token_count(self, inputs):
        """Per-rank count of loss-contributing tokens: causal-LM ``labels``, online/env GRPO's
        :func:`effective_loss_mask`, offline GRPO's ``completion_attention_mask``; ``None`` for sequence-level
        and preference trainers, whose batches carry no token-level loss mask."""
        if not isinstance(inputs, Mapping):  # dict or transformers BatchEncoding (UserDict)
            return None
        labels, input_ids = inputs.get("labels"), inputs.get("input_ids")
        if labels is not None and input_ids is not None and labels.shape == input_ids.shape:
            return (labels != LABEL_IGNORE_INDEX).sum()
        loss_mask = effective_loss_mask(inputs)
        if loss_mask is None:
            loss_mask = inputs.get("completion_attention_mask")
        return loss_mask.sum() if loss_mask is not None else None

    def _accumulate_unmasked_output_tokens(self, local_count):
        """Add this micro-batch's loss-token count to the on-device accumulator, gathered once per log
        (:meth:`_drain_unmasked_output_tokens`)."""
        if local_count is None:
            return
        # int64 always: the gather zero-fills absent ranks, so a float sum would make its dtype rank-dependent.
        local = local_count.detach().long()
        prev = getattr(self, "_local_unmasked_token_accum", None)
        self._local_unmasked_token_accum = local if prev is None else prev + local

    def _drain_unmasked_output_tokens(self):
        """COLLECTIVE. Fold every rank's token accumulator into the run total, corrected for non-DP replication
        (TP/ETP/CP), and reset. A rank with nothing accumulated enters with a zero; the total stays unset until
        some rank contributes, so sequence-level trainers log nothing."""
        accum = getattr(self, "_local_unmasked_token_accum", None)
        if accum is None:
            accum = torch.zeros((), dtype=torch.long, device=self.accelerator.device)
        gathered = int(self.accelerator.gather(accum).sum().item())
        non_dp_factor = self.parallelism_config.non_dp_replication_factor
        if non_dp_factor > 1:
            gathered //= non_dp_factor
        previous = getattr(self, "_cumulative_unmasked_output_tokens", None)
        if previous is not None or gathered > 0:
            self._cumulative_unmasked_output_tokens = (previous or 0) + gathered
        self._local_unmasked_token_accum = None

    def _warn_once_on_empty_labels(self, inputs: dict) -> None:
        """Warn once if a batch carries no training signal (every label -100, pad or eos). First optimizer
        step only: the collator config is static, and the ``.all()`` reduction is a host sync."""
        if self._warned_empty_labels:
            return
        if getattr(self, "state", None) is not None and self.state.global_step > 0:
            return
        labels = inputs.get("labels")
        if labels is None:
            return
        non_training = labels == LABEL_IGNORE_INDEX
        tokenizer = getattr(self, "processing_class", None)
        for attr in ("pad_token_id", "eos_token_id"):
            token_id = getattr(tokenizer, attr, None)
            if token_id is not None:
                non_training = non_training | (labels == token_id)
        if non_training.all():
            self._warned_empty_labels = True
            if is_global_main_process():
                logger.warning(
                    "Batch contains no training signal — all labels are -100, pad, or eos. "
                    "This typically means the collator could not find the assistant_message_template "
                    "in the tokenized input, or the eval set is too small after packing. "
                    f"Labels shape: {list(labels.shape)}"
                )
