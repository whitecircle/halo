"""The length-ordered deal of a generation round's rows into its micro-batches. Under ZeRO-3 or an EP dispatch the
ranks forward their rows in lock-step, so each row costs the longest row its peers run beside it; the deal ranks
every rank's rows longest first and spreads them over the micro-batches alike, so the ``i``-th longest rows of all
ranks run together. Only the order changes, and only objectives that read no micro-batch composition are admitted."""

import contextlib
import logging
from collections.abc import Iterator

import torch
import trl.trainer.grpo_trainer as trl_grpo_trainer

from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# The TRL loss types this trainer accepts whose loss over a generation round is the same however its rows fall
# into the round's equal-sized micro-batches: a row mean, a fixed-width or a round-global normalizer. bnpo is
# not one, as it divides each micro-batch by that micro-batch's own token count.
MICROBATCH_INVARIANT_LOSS_TYPES = ("grpo", "sapo", "dr_grpo", "dapo", "cispo", "luspo")


def length_deal_order(lengths: torch.Tensor, chunks: int) -> torch.Tensor:
    """The deal of ``len(lengths)`` rows into ``chunks`` equal contiguous micro-batches: rows ranked longest first
    (ties by index), the ``i``-th to micro-batch ``i % chunks`` at slot ``i // chunks``. Returns the CPU index
    tensor whose ``p``-th entry is the row placed at position ``p``."""
    rows = lengths.numel()
    if chunks < 1 or rows % chunks:
        raise ValueError(f"Cannot deal {rows} rows into {chunks} equal micro-batches.")
    ranked = torch.sort(lengths.detach().to("cpu"), descending=True, stable=True).indices
    rank = torch.arange(rows)
    order = torch.empty_like(ranked)
    order[(rank % chunks) * (rows // chunks) + rank // chunks] = ranked
    return order


def permute_rows(batch: dict, order: torch.Tensor) -> dict:
    """``batch`` with every row-aligned value taken in ``order``, the way TRL's ``shuffle_sequence_dict`` permutes
    a generation batch: ``None`` and 0-dim tensors pass through, lists are reindexed."""

    def permute(value):
        if value is None or (isinstance(value, torch.Tensor) and value.ndim == 0):
            return value
        if isinstance(value, torch.Tensor):
            return value[order.to(value.device)]
        return [value[i] for i in order.tolist()]

    return {key: permute(value) for key, value in batch.items()}


class LengthOrderedRowsMixin:
    """Deals each training round's rows by length in place of TRL's buffer shuffle, and orders the round's no-grad
    forwards the same way; ``length_ordered_microbatches`` resolves once, at construction. Mixed in ahead of
    ``GenerationRoundMixin``, whose ``_prepare_inputs`` re-syncs the TP group after TRL's."""

    _orders_rows_by_length = False

    def _resolve_length_ordered_rows(self) -> None:
        """Turn the deal on where it changes only the row order (unset) or must (``true``), refusing ``true``
        where it would change the objective or order nothing."""
        requested = self.async_config.length_ordered_microbatches
        blocker = self._length_order_blocker()
        if requested and blocker is not None:
            raise ValueError(f"length_ordered_microbatches: true, but {blocker}. Unset it or set it to false.")
        self._orders_rows_by_length = requested is not False and blocker is None
        if requested is None and is_global_main_process():
            logger.info(
                "Training rows are dealt by length across micro-batches."
                if self._orders_rows_by_length
                else f"Training rows keep TRL's shuffle: {blocker}."
            )

    def _length_order_blocker(self) -> str | None:
        """Why the deal cannot run on this config, ``None`` when it can."""
        args = self.args
        if self._misaligned_accumulation:
            return (
                f"gradient_accumulation_steps ({args.gradient_accumulation_steps}) straddles a generation round, "
                "so an optimizer step takes a subset of its micro-batches and the order picks which"
            )
        if args.loss_type not in MICROBATCH_INVARIANT_LOSS_TYPES:
            return f"loss_type={args.loss_type!r} normalizes each micro-batch by its own token count"
        if args.top_entropy_quantile < 1.0:
            return f"top_entropy_quantile={args.top_entropy_quantile} thresholds entropy within each micro-batch"
        if self._entropy_bonus_enabled:
            return "the entropy bonus averages entropy within each micro-batch"
        if not self._trains_rows_densely:
            return (
                "rows are not forwarded one per backbone call (that needs use_chunked_grpo_logprobs, at "
                "per_device_train_batch_size 1 or under FA4), so every row costs the batch's padded width"
            )
        return None

    def _prepare_inputs(self, generation_batch):
        """TRL's buffering with the deal in place of its shuffle on a train round's refill."""
        if not (self._orders_rows_by_length and self.model.training):
            return super()._prepare_inputs(generation_batch)
        buffered = self._buffered_inputs
        with self._dealing_in_place_of_trl_shuffle() as dealt:
            inputs = super()._prepare_inputs(generation_batch)
        if self._buffered_inputs is not buffered and not dealt:
            raise RuntimeError(
                "TRL refilled its generation buffer without calling shuffle_sequence_dict, which the "
                "length-ordered deal replaces: the installed TRL moved its buffer shuffle. Set "
                "length_ordered_microbatches: false until the deal follows it."
            )
        return inputs

    @contextlib.contextmanager
    def _dealing_in_place_of_trl_shuffle(self) -> Iterator[list[bool]]:
        """Swap TRL's ``shuffle_sequence_dict`` for the deal for the block; the yielded list records each deal."""
        dealt: list[bool] = []

        def deal(batch: dict) -> dict:
            dealt.append(True)
            lengths = torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1).sum(dim=1)
            return permute_rows(batch, length_deal_order(lengths, self.args.steps_per_generation))

        shuffle = trl_grpo_trainer.shuffle_sequence_dict
        trl_grpo_trainer.shuffle_sequence_dict = deal
        try:
            yield dealt
        finally:
            trl_grpo_trainer.shuffle_sequence_dict = shuffle

    def _no_grad_row_order(self, attention_mask: torch.Tensor, mode: str) -> torch.Tensor | None:
        """The deal of a train round's rows by the lengths its no-grad forwards run, on the mask's device;
        ``None`` when the deal is off or in eval, where rows keep their batch order."""
        if not self._orders_rows_by_length or mode != "train":
            return None
        return length_deal_order(attention_mask.sum(dim=1), self.args.steps_per_generation).to(attention_mask.device)
