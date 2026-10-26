"""Trainer construction refusals for the parallelism modes."""

from __future__ import annotations

from src.distributed.context_parallel.config import CPConfigError


def reject_cp_trainer_args(
    training_args,
    compute_metrics=None,
    preprocess_logits_for_metrics=None,
) -> None:
    """Reject trainer settings whose loss/metric path needs full-sequence logits.

    A CP rank's forward returns logits for its own chunk only, and
    :class:`~src.distributed.context_parallel.ulysses_model.UlyssesModel` computes the
    boundary-aware loss itself. Anything that re-derives the loss or a metric from
    ``(logits, labels)`` outside the wrapper pairs a chunk of logits with the full-length labels.

    Raises:
        CPConfigError: If a rejected setting is active.
    """
    if getattr(training_args, "label_smoothing_factor", 0.0):
        raise CPConfigError(
            "label_smoothing_factor > 0 is not supported with Context Parallelism: the Trainer pops "
            "`labels` before the model call and smooths the loss itself, so the CP wrapper returns "
            "no loss and the smoother compares FULL-length labels against this rank's "
            "sequence-chunk logits. Set label_smoothing_factor=0, or train without CP."
        )

    if getattr(training_args, "loss_type", None) == "dft":
        raise CPConfigError(
            "loss_type='dft' is not supported with Context Parallelism: TRL implements it either by "
            "installing a compute_loss_func (the Trainer then pops `labels`, leaving the CP wrapper "
            "with no loss and the func with full-length labels) or, under Liger, by a "
            "`use_token_scaling` forward flag the CP wrapper's own loss ignores. Use "
            "loss_type='nll', or train without CP."
        )

    # Not gated on eval_strategy: a trainer built with eval_strategy='no' can still reach the
    # misaligned metric path through a direct evaluate()/predict() call.
    if compute_metrics is not None or preprocess_logits_for_metrics is not None:
        raise CPConfigError(
            "Evaluation under Context Parallelism is loss-only: compute_metrics / "
            "preprocess_logits_for_metrics receive this rank's sequence-chunk logits gathered "
            "against the full-length labels, so the two are misaligned. Drop them (eval_loss still "
            "reports), or evaluate from a saved checkpoint without CP."
        )
