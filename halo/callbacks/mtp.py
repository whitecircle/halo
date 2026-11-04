"""The held-out gate of a ``head_only`` fit: acceptance at the fit's end against its reading before the first step. An
oversized learning rate lowers the loss and acceptance together, so only the served quantity can tell a fit helped."""

from __future__ import annotations

import logging
import math
import os

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from halo.callbacks.early_stop import StopsTrainingEarly
from halo.models.mtp.acceptance import MTP_ACCEPTANCE_GREEDY, MTP_ACCEPTANCE_TV, MTP_SCORED_POSITIONS
from halo.runtime.collectives import broadcast_from_rank0
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

_EVAL_PREFIX = "eval_"
_GATED_METRICS = (f"{_EVAL_PREFIX}{MTP_ACCEPTANCE_TV}", f"{_EVAL_PREFIX}{MTP_ACCEPTANCE_GREEDY}")
# What a fit of a released head's weights has been seen to take.
_FIT_MAX_LEARNING_RATE = 1e-5
_FIT_MIN_WARMUP_STEPS = 20


def _held_checkpoint(args, state) -> str:
    if args.save_strategy == "no":
        return "This run saved no checkpoint of it."
    checkpoint = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
    return f"{checkpoint} holds the rejected head."


class MtpAcceptanceGate(StopsTrainingEarly):
    """Scores the last step whatever ``eval_steps`` says, and fails the run if either acceptance ended below its first
    logged reading plus ``min_gain``. Reading the restored log, a resume keeps the baseline of the head the fit started
    from, though ``eval_on_start`` scores the resumed head again. The last step's save already holds the head judged, so
    the verdict names that checkpoint."""

    def __init__(self, min_gain: float):
        self.min_gain = min_gain

    def on_train_begin(self, args, state, control, **kwargs):
        warmup = args.get_warmup_steps(state.max_steps)
        if is_global_main_process() and (
            args.learning_rate > _FIT_MAX_LEARNING_RATE or warmup < _FIT_MIN_WARMUP_STEPS
        ):
            logger.warning(
                f"A head_only fit at learning_rate {args.learning_rate} with {warmup} warmup steps: a Qwen3.6-35B head "
                "at 1e-4 with no warmup fell from 0.8188 to 0.7361 held-out acceptance by step 30 while its loss "
                f"fell. {_FIT_MAX_LEARNING_RATE} over {_FIT_MIN_WARMUP_STEPS} warmup steps held it."
            )

    def on_step_end(self, args, state, control, **kwargs):
        # After DefaultFlowCallback, which stops the run and arms its last save here.
        if control.should_training_stop:
            control.should_evaluate = True

    def on_train_end(self, args, state, control, **kwargs):
        """COLLECTIVE: rank 0's verdict on every rank, which all exit on it."""
        reason = self._failure(args, state) if is_global_main_process() else None
        self.stop_reason = broadcast_from_rank0(reason)
        self.stopped = self.stop_reason is not None
        if self.stopped and is_global_main_process():
            logger.error(self.stop_reason)

    def _failure(self, args, state) -> str | None:
        """Why this fit failed, or ``None``."""
        held = _held_checkpoint(args, state)
        for metric in _GATED_METRICS:
            readings = [entry for entry in state.log_history if metric in entry]
            if not readings:
                return (
                    f"The MTP acceptance gate found no {metric} in this run's log, so it cannot judge the fit. {held}"
                )
            first, last = readings[0], readings[-1]
            if not last[f"{_EVAL_PREFIX}{MTP_SCORED_POSITIONS}"]:
                return (
                    f"The MTP acceptance gate scored no position at step {last['step']}: every held-out row is "
                    f"shorter than a draft's target or masked. {held}"
                )
            bar = first[metric] + self.min_gain
            if not (math.isfinite(last[metric]) and last[metric] >= bar):
                return (
                    f"The MTP head fit failed its acceptance gate: {metric} is {last[metric]:.4f} at step "
                    f"{last['step']}, under {bar:.4f}, its step {first['step']} reading {first[metric]:.4f} plus "
                    f"mtp_min_acceptance_gain {self.min_gain}. {held} Serve the head this fit started from, or refit "
                    "at a lower learning_rate with warmup_steps, on rows like the ones you serve. "
                    "mtp_min_acceptance_gain: null turns the gate off."
                )
        return None
