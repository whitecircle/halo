"""Early stop for a GRPO run whose policy has left its healthy regime (:class:`EarlyStopConfig`).

A KL-free policy drifts slowly before it fails: entropy leaves its band, the trainer-vs-sampler log-prob
gap widens with it, and the trust-region breaker ends up skipping every update. Each condition counts
its breaching readings in a row; once one reaches ``patience`` training ends there, without saving or
evaluating that step, so the run keeps only the periodic checkpoints taken before it.
"""

from accelerate.logging import get_logger
from transformers import TrainerCallback

from src.args.mixins import EarlyStopConfig
from src.distributed.runtime import rank_consensus
from src.trainers.grpo.objective.logratio import UPDATE_SKIPPED_KEY

logger = get_logger(__name__)

# TRL's metrics the conditions read: the entropy and, under its vLLM IS correction, the online trainer's
# gap (a mean absolute difference). The environmental trainer's own gap (a signed mean, read by
# magnitude) and the breaker's verdict are keyed in ``objective/logratio.py``, beside the IS ratio and
# the mask stages they read.
ENTROPY_KEY = "entropy"
SAMPLING_LOGP_GAP_KEY = "sampling/sampling_logp_difference/mean"


class GRPOEarlyStopCallback(TrainerCallback):
    """Ends training once an :class:`EarlyStopConfig` condition breaches on ``patience`` readings in a row.

    Reads training logs only (an eval log carries ``eval_``-prefixed keys). The gap and the breaker's
    verdict are logged once per generation round, so a logged step that reuses a round carries neither:
    a missing reading leaves its condition's streak where it was, and only a healthy one resets it. The
    stop verdict is taken across ranks: the readings are world-reduced already, but a split verdict would
    leave the ranks that carry on waiting in the next step's collectives.
    """

    def __init__(self, config: EarlyStopConfig, gap_key: str):
        self.config = config
        self.gap_key = gap_key
        self._streaks: dict[str, int] = {}
        self._evidence: dict[str, str] = {}
        self.stopped = False

    def _readings(self, logs: dict[str, float]) -> dict[str, str | None]:
        """Each armed condition this log reads: the breaching reading's description, or ``None`` when healthy."""
        config, found = self.config, {}
        entropy = logs.get(ENTROPY_KEY)
        if config.entropy_band is not None and entropy is not None:
            low, high = config.entropy_band
            found["entropy"] = None if low <= entropy <= high else f"entropy {entropy:.3f} outside [{low}, {high}]"
        gap = logs.get(self.gap_key)
        if config.logratio_gap is not None and gap is not None:
            breach = abs(gap) > config.logratio_gap
            found["logratio_gap"] = f"log-prob gap {abs(gap):.4f} above {config.logratio_gap}" if breach else None
        skipped = logs.get(UPDATE_SKIPPED_KEY)
        if config.on_skipped_updates and skipped is not None:
            found["skipped_updates"] = "every update skipped by the trust-region breaker" if skipped >= 1.0 else None
        return found

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or ENTROPY_KEY not in logs:
            return
        for name, breach in self._readings(logs).items():
            self._streaks[name] = 0 if breach is None else self._streaks.get(name, 0) + 1
            if breach is not None:
                self._evidence[name] = breach
        held = [name for name, streak in self._streaks.items() if streak >= self.config.patience]
        _, any_stop = rank_consensus(bool(held))
        if not any_stop:
            return
        self.stopped = True
        # Runs before the step's evaluation and save: a checkpoint of this step would be one from inside the drift.
        control.should_training_stop = True
        control.should_save = False
        control.should_evaluate = False
        reasons = "; ".join(self._evidence[name] for name in held) or "a condition held on another rank"
        logger.error(
            f"Early stop at step {state.global_step}: {reasons}, on {self.config.patience} readings in a row. "
            "Training ends without saving this step; resume from a checkpoint taken before the drift began."
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        # The loop breaks into one more epoch-end save/eval pass, which an epoch save strategy would arm.
        if self.stopped:
            control.should_save = False
            control.should_evaluate = False


def build_early_stop_callback(
    config: EarlyStopConfig, *, gap_key: str, gap_logged: bool
) -> GRPOEarlyStopCallback | None:
    """The callback a trainer attaches for ``config``, or ``None`` when no condition is set.

    Refuses a log-prob gap condition the trainer would never feed: ``gap_logged`` is whether its run logs
    ``gap_key`` at all (both trainers log it only under the importance-sampling correction)."""
    if not config.active:
        return None
    if config.logratio_gap is not None and not gap_logged:
        raise ValueError(
            f"early_stop_logratio_gap reads {gap_key}, which only the importance-sampling correction "
            "(vllm_importance_sampling_correction) logs, and the correction is off in this run: the stop would "
            "never fire."
        )
    return GRPOEarlyStopCallback(config, gap_key)
