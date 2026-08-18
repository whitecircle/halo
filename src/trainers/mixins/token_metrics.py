"""Loss-contributing token counting for the ``train/total_output_tokens`` series.

Two phases so no collective lands on the per-micro-batch path: each step folds this rank's count
into an on-device accumulator, and one gather per log step turns those into the run total. Mixed
into :class:`~src.trainers.mixins.base.DistributedTrainerMixin`; the drain reads the trainer's
accelerator and ``parallelism_config``.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from src.data.spans import LABEL_IGNORE_INDEX
from src.trainers.mixins.loss_masks import effective_loss_mask


class TokenMetricsMixin:
    """Per-rank loss-token accumulation and its once-per-log cross-rank total."""

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
