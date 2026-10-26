"""Per-batch token accounting: loss-contributing tokens and the attention-score work each batch costs.

Nothing syncs the host per micro-batch: counts fold into on-device accumulators, the token total gathers
once per log step and the attention work is read once per optimizer step, accumulated only once a callback
binds itself as its sink (``bind_attention_work_source``).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from src.kernels.logprobs import LABEL_IGNORE_INDEX
from src.models.attention.tree_plan import TREE_ATTENTION_KEY
from src.trainers.mixins.loss_masks import effective_loss_mask


class TokenMetricsMixin:
    """Per-rank loss-token and attention-work accumulation and their cross-rank totals."""

    # Set by the first ``training_step``: the efficiency callback binds here and hands over the layout.
    _attention_work_bound: bool = False
    _attention_layout = None

    def _bind_attention_work_sources(self):
        """Offer this trainer to every callback that consumes measured attention work (duck-typed, so
        the trainer package imports nothing from the callbacks); the callback answers with the
        attention layout it costs the batches with, or ``None`` when it could not build one."""
        self._attention_work_bound = True
        for callback in self.callback_handler.callbacks:
            bind = getattr(callback, "bind_attention_work_source", None)
            if bind is not None:
                self._attention_layout = bind(self)

    def _extract_document_lengths(self, inputs):
        """Token length of every document the batch's attention scores (zeros allowed), or ``None``
        when the batch exposes no token layout (a trainer whose collator emits no ``input_ids``).

        Padding-free batches carry ``cu_seq_lens_q``; packed rows carry ``position_ids`` that reset
        at each document (the collators' boundary contract); a padded batch is one document per row
        over its ``attention_mask``; a bare ``input_ids`` is one full-width document per row. Every
        buffer is sized from a shape, so nothing here syncs the host.
        """
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

    def _extract_tree_nodes(self, inputs):
        """``(path length, own tokens)`` of every node of a tree-packed batch, or ``None`` for any other.

        A node's tokens attend along its root-to-leaf path, whose length is the position of the
        node's last token plus one; position resets would read a whole tree as one document.
        """
        plan = inputs.get(TREE_ATTENTION_KEY) if isinstance(inputs, Mapping) else None
        if plan is None:
            return None
        bounds = plan["cu_seqlens_q"].long()
        return inputs["position_ids"][0, bounds[1:] - 1] + 1, bounds[1:] - bounds[:-1]

    def _accumulate_attention_flops(self, inputs):
        """Cost this micro-batch's documents through the bound layout into the on-device accumulator,
        read once per optimizer step by :meth:`drain_attention_flops`."""
        if not self._attention_work_bound:
            self._bind_attention_work_sources()
        if self._attention_layout is None:
            return
        nodes = self._extract_tree_nodes(inputs)
        lengths, own_tokens = nodes if nodes is not None else (self._extract_document_lengths(inputs), None)
        if lengths is None:
            return
        lengths = lengths.detach()
        flops = self._attention_layout.flops_for_documents(lengths, own_tokens)
        step = torch.stack((flops, (lengths > 0).sum().to(flops)))
        prev = getattr(self, "_local_attention_work_accum", None)
        self._local_attention_work_accum = step if prev is None else prev + step

    def drain_attention_flops(self) -> tuple[float, int]:
        """``(this rank's attention-score FLOPs, documents)`` since the last drain, then reset — one
        host read per optimizer step.

        Rank-local by design: every pipeline stage costs the same documents through its own layer
        slice, so a world average would hand stage 0 the chain's mean against its own ``6·N``; and
        with no collective, a rank whose layout failed to build simply reports nothing.
        """
        accum = getattr(self, "_local_attention_work_accum", None)
        if accum is None:
            return 0.0, 0
        self._local_attention_work_accum = None
        flops, documents = accum.tolist()
        return flops, int(documents)

    def _extract_output_token_count(self, inputs):
        """Per-rank count of loss-contributing tokens, or ``None`` when the batch has no token-level mask.

        Covers trainers whose loss mask is present in the per-step ``inputs``: causal-LM (``labels``
        aligned to ``input_ids``), online/env GRPO (``completion_mask``, intersected with
        ``tool_mask`` when present), offline GRPO (``completion_attention_mask``). Returns ``None``
        for sequence-level trainers (classification / reward / embedding) and preference trainers
        (chosen/rejected split until ``compute_loss``).
        """
        if not isinstance(inputs, Mapping):  # dict or transformers BatchEncoding (UserDict)
            return None
        labels, input_ids = inputs.get("labels"), inputs.get("input_ids")
        if labels is not None and input_ids is not None and labels.shape == input_ids.shape:
            return (labels != LABEL_IGNORE_INDEX).sum()
        # online/env GRPO: "completion_mask" (∧ tool_mask); offline GRPO: "completion_attention_mask".
        loss_mask = effective_loss_mask(inputs)
        if loss_mask is None:
            loss_mask = inputs.get("completion_attention_mask")
        return loss_mask.sum() if loss_mask is not None else None

    def _accumulate_unmasked_output_tokens(self, local_count):
        """Add this micro-batch's per-rank loss-contributing token count to the on-device accumulator.

        Not gathered here — the cross-rank gather is deferred to :meth:`_drain_unmasked_output_tokens`
        (once per log) so it doesn't stall every micro-batch.
        """
        if local_count is None:
            return
        # int64 always: the gather zero-fills absent ranks, so a float sum would make its dtype rank-dependent.
        local = local_count.detach().long()
        prev = getattr(self, "_local_unmasked_token_accum", None)
        self._local_unmasked_token_accum = local if prev is None else prev + local

    def _drain_unmasked_output_tokens(self):
        """Fold the on-device per-rank token accumulator into the run total (once per log).

        Gathers across all ranks, corrects for non-DP replication (TP/ETP/CP), and resets. Called
        from ``_add_parallelism_to_logs``, which every rank reaches together.

        A rank with nothing accumulated enters with a zero rather than skipping the collective. The
        run total stays unset until some rank contributes, so sequence-level trainers log nothing.
        """
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
