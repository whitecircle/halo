"""Checks at trainer construction whether a loss that builds the full ``[rows, tokens, vocab]`` logits fits in the free
device memory: warns when it takes more than half, refuses the run when it does not fit."""

import logging
from dataclasses import dataclass

import torch

from halo.runtime.consensus import rank_consensus
from halo.runtime.ranks import is_global_main_process
from halo.trainers.construction.training_args_overrides import uses_evaluation

logger = logging.getLogger(__name__)

# Warn when the full logits take more than this share of the free memory. At construction the optimizer state,
# gradients and activations are not allocated yet, so they still need the rest.
_FULL_LOGITS_WARN_FRACTION = 0.5


@dataclass(frozen=True)
class LogitsWidth:
    tokens: int
    set_by: str


def full_logits_verdict(
    rows: int,
    rows_set_by: str,
    width: LogitsWidth,
    vocab: int,
    free_bytes: int,
    *,
    bytes_per_logit: int,
    loss: str,
    remedy: str,
) -> tuple[bool, str] | None:
    """``(fatal, message)``, or ``None`` when the full logits fit easily. ``fatal`` is true when they do not fit at
    all, false when they take more than :data:`_FULL_LOGITS_WARN_FRACTION`. ``remedy`` names the flag that avoids the
    full logits."""
    plane = rows * width.tokens * vocab * bytes_per_logit
    if plane <= _FULL_LOGITS_WARN_FRACTION * free_bytes:
        return None
    gib = 1024**3
    fatal = plane > free_bytes
    share = (
        f"more than the {free_bytes / gib:.1f} GiB this device has free"
        if fatal
        else f"{plane / free_bytes:.0%} of the {free_bytes / gib:.1f} GiB free before the optimizer state and "
        f"activations are allocated"
    )
    return fatal, (
        f"The {loss} forward materializes full logits: {rows} rows × {width.tokens} tokens × {vocab} vocab × "
        f"{bytes_per_logit} B = {plane / gib:.1f} GiB, {share}. {remedy}, or lower {rows_set_by} ({rows}) or "
        f"{width.set_by}, which sets the {width.tokens}-token width."
    )


def verify_full_logits_fit(
    args, width: LogitsWidth, vocab: int, device, *, bytes_per_logit: int, loss: str, remedy: str
) -> None:
    """COLLECTIVE. Raise on every rank when the full logits do not fit in the free memory of any rank's ``device``;
    warn when they take a large share. Uses the larger of the train batch and, if the run evaluates, the eval batch."""
    rows, rows_set_by = args.per_device_train_batch_size, "per_device_train_batch_size"
    if uses_evaluation(args) and args.per_device_eval_batch_size > rows:
        rows, rows_set_by = args.per_device_eval_batch_size, "per_device_eval_batch_size"
    free_bytes, _ = torch.cuda.mem_get_info(device)
    verdict = full_logits_verdict(
        rows, rows_set_by, width, vocab, free_bytes, bytes_per_logit=bytes_per_logit, loss=loss, remedy=remedy
    )
    fatal = verdict is not None and verdict[0]
    if not rank_consensus(not fatal).all_ok:
        raise RuntimeError(
            verdict[1]
            if fatal
            else f"Another rank's full-logits plane cannot fit in its free device memory (its error states the "
            f"size). {remedy}."
        )
    if verdict is not None and is_global_main_process():
        logger.warning(verdict[1])
