"""The attended spans of a training round's rows, built once with the round's batch, carried through TRL's shuffle and
split, and read by the loss forward in place of a device-to-host copy of each micro-batch's mask."""

import logging

import torch

from halo.runtime.consensus import reject_across_ranks
from halo.runtime.ranks import is_global_main_process
from halo.trainers.grpo.logprob_scoring import reject_malformed_rows, row_spans_on_device

logger = logging.getLogger(__name__)

# CPU int64 ``[rows, 3]`` of ``lo, hi, completion`` per row, riding the batch dict row-aligned like the routing masks.
ROW_SPANS_KEY = "row_spans"


class RowSpanStashMixin:
    """Mixed in ahead of ``DistributedTrainerMixin``. The host provides ``_trains_rows_densely`` and calls
    :meth:`_resolve_row_span_stash` once the model is placed and :meth:`_with_row_spans` on the last statement of each
    train round's batch build."""

    _stashes_row_spans = False

    def _resolve_row_span_stash(self) -> None:
        """Stash where rows forward densely. Not under TP/ETP, where every micro-batch is re-broadcast from the group
        leader as device tensors (TRL's per-rank shuffle differs across the group), so a table built before the
        shuffle would not follow the leader's rows."""
        if not self._trains_rows_densely:
            return
        if self._get_tp_or_etp_process_group() is not None:
            if is_global_main_process():
                logger.info("Row spans: read per call (TP/ETP re-sync).")
            return
        self._stashes_row_spans = True

    def _with_row_spans(self, batch: dict) -> dict:
        """The train round's final batch with its spans attached, which must follow every stage that narrows a mask, and
        any TP/ETP broadcast of the batch. Raises on a malformed row, as the dense forward would."""
        if not self._stashes_row_spans or not self.model.training:
            return batch
        completion_width = batch["completion_mask"].size(1)
        attended = torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1)
        spans = row_spans_on_device(attended, completion_width).cpu()
        reject_malformed_rows(spans)
        batch[ROW_SPANS_KEY] = spans[:, :3].contiguous()
        return batch

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._armed_row_spans = inputs.get(ROW_SPANS_KEY) if isinstance(inputs, dict) else None
        try:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        finally:
            self._armed_row_spans = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        """COLLECTIVE on a window's last micro-step: the spans its micro-batches read are checked against their masks,
        so a misalignment raises on every rank where it would otherwise slice the wrong tokens."""
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self._stashes_row_spans and self.accelerator.sync_gradients:
            self._reject_misaligned_row_spans()
        return loss

    def _reject_misaligned_row_spans(self) -> None:
        tally, self._row_span_mismatches = self._row_span_mismatches, None
        rows = 0 if tally is None else int(tally)
        reject_across_ranks(
            f"{rows} micro-batch row(s) read row spans built from other attention masks than the loss forward "
            f"attended: a stage narrowed a mask or reordered rows after the round's spans were built. Call "
            f"_with_row_spans after it."
            if rows
            else None,
            "The row-span check",
        )
