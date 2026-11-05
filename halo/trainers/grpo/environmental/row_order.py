"""The length-ordered deal of a generation round's rows into its micro-batches. Under ZeRO-3 or an EP dispatch each row
costs the longest row its peers run beside it, so the ``i``-th longest rows of all ranks run together. Only the order
changes, and only objectives that read no micro-batch composition are admitted."""

import contextlib
import logging
from collections.abc import Callable, Iterator

import torch
import trl.trainer.grpo_trainer as trl_grpo_trainer

from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# TRL loss types that average over a micro-batch's rows, whose per-slot value therefore scales with the row count.
ROW_MEAN_LOSS_TYPES = ("grpo", "sapo", "dr_grpo", "luspo")
# Those normalized by a round-global token count, which no micro-batch's composition enters.
TOKEN_GLOBAL_LOSS_TYPES = ("dapo", "cispo")
# The round loss is the same however rows fall into micro-batches. bnpo is not one, as it divides each micro-batch by
# its own token count.
MICROBATCH_INVARIANT_LOSS_TYPES = ROW_MEAN_LOSS_TYPES + TOKEN_GLOBAL_LOSS_TYPES


def length_deal_order(lengths: torch.Tensor, chunks: int) -> torch.Tensor:
    """Rows ranked longest first (ties by index), the ``i``-th to micro-batch ``i % chunks`` at slot ``i // chunks``.
    Returns the CPU index tensor whose ``p``-th entry is the row at position ``p``."""
    rows = lengths.numel()
    if chunks < 1 or rows % chunks:
        raise ValueError(f"Cannot deal {rows} rows into {chunks} equal micro-batches.")
    ranked = torch.sort(lengths.detach().to("cpu"), descending=True, stable=True).indices
    rank = torch.arange(rows)
    order = torch.empty_like(ranked)
    order[(rank % chunks) * (rows // chunks) + rank // chunks] = ranked
    return order


def permute_rows(batch: dict, order: torch.Tensor) -> dict:
    """Like TRL's ``shuffle_sequence_dict``: ``None`` and 0-dim tensors pass through, lists are reindexed."""

    def permute(value):
        if value is None or (isinstance(value, torch.Tensor) and value.ndim == 0):
            return value
        if isinstance(value, torch.Tensor):
            return value[order.to(value.device)]
        return [value[i] for i in order.tolist()]

    return {key: permute(value) for key, value in batch.items()}


@contextlib.contextmanager
def replacing_trl_shuffle(replacement: Callable[[dict], dict]) -> Iterator[list[bool]]:
    """Swap TRL's ``shuffle_sequence_dict`` for ``replacement`` for the block; the yielded list records each call."""
    called: list[bool] = []

    def replaced(batch: dict) -> dict:
        called.append(True)
        return replacement(batch)

    shuffle = trl_grpo_trainer.shuffle_sequence_dict
    trl_grpo_trainer.shuffle_sequence_dict = replaced
    try:
        yield called
    finally:
        trl_grpo_trainer.shuffle_sequence_dict = shuffle


class LengthOrderedRowsMixin:
    """Deals each training round's rows by length in place of TRL's buffer shuffle, and orders the no-grad forwards
    alike. Mixed in ahead of ``GenerationRoundMixin``, whose ``_prepare_inputs`` re-syncs the TP group after TRL's."""

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

    def _dealing_in_place_of_trl_shuffle(self) -> contextlib.AbstractContextManager[list[bool]]:
        def deal(batch: dict) -> dict:
            lengths = torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1).sum(dim=1)
            return permute_rows(batch, length_deal_order(lengths, self.args.steps_per_generation))

        return replacing_trl_shuffle(deal)

    def _no_grad_row_order(self, attention_mask: torch.Tensor, mode: str) -> torch.Tensor | None:
        """On the mask's device; ``None`` when the deal is off or in eval, where rows keep their batch order."""
        if not self._orders_rows_by_length or mode != "train":
            return None
        return length_deal_order(attention_mask.sum(dim=1), self.args.steps_per_generation).to(attention_mask.device)
