"""Row order of a GRPO generation round shared by the on-policy trainers: the length-ordered deal, the row permutation
and the swap of TRL's buffer shuffle that a trainer owning the round layout installs, with the loss types whose value
does not depend on how rows fall into micro-batches."""

import contextlib
from collections.abc import Callable, Iterator

import torch
import trl.trainer.grpo_trainer as trl_grpo_trainer

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
