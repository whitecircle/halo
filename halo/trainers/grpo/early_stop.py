"""Early stop for a GRPO run whose policy has left its healthy regime. Each condition counts breaching readings in a
row; once one reaches ``patience`` training ends there, without saving or evaluating that step."""

import logging

from halo.callbacks.early_stop import StopsTrainingEarly
from halo.configs.grpo import EarlyStopConfig
from halo.runtime.consensus import rank_consensus
from halo.runtime.ranks import is_global_main_process
from halo.trainers.grpo.objective.logratio import UPDATE_SKIPPED_KEY

logger = logging.getLogger(__name__)

# TRL's entropy and, under its vLLM IS correction, the online trainer's log-prob gap (a mean absolute difference).
ENTROPY_KEY = "entropy"
SAMPLING_LOGP_GAP_KEY = "sampling/sampling_logp_difference/mean"


class GRPOEarlyStopCallback(StopsTrainingEarly):
    """Reads training logs only. A missing reading leaves a streak where it was, as the gap and the breaker's verdict
    log once per round. The verdict is taken across ranks, since a split one would strand the ranks that carry on."""

    def __init__(self, config: EarlyStopConfig, gap_key: str):
        self.config = config
        self.gap_key = gap_key
        # Consecutive breaching readings per condition; a checkpoint's sidecar carries them across a resume.
        self.streaks: dict[str, int] = {}
        self._evidence: dict[str, str] = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or ENTROPY_KEY not in logs:
            return
        for name, breach in self._readings(logs).items():
            self.streaks[name] = 0 if breach is None else self.streaks.get(name, 0) + 1
            if breach is not None:
                self._evidence[name] = breach
        held = [name for name, streak in self.streaks.items() if streak >= self.config.patience]
        if not rank_consensus(bool(held)).any_ok:
            return
        self.stopped = True
        # Runs before the step's evaluation and save: a checkpoint of this step would be one from inside the drift.
        control.should_training_stop = True
        control.should_save = False
        control.should_evaluate = False
        reasons = "; ".join(self._evidence[name] for name in held) or "a condition held on another rank"
        if is_global_main_process():
            logger.error(
                f"Early stop at step {state.global_step}: {reasons}, on {self.config.patience} readings in a row. "
                "Training ends without saving this step; resume from a checkpoint taken before the drift began."
            )

    def on_epoch_end(self, args, state, control, **kwargs):
        # The loop breaks into one more epoch-end save/eval pass, which an epoch save strategy would arm.
        if self.stopped:
            control.should_save = False
            control.should_evaluate = False

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


def build_early_stop_callback(
    config: EarlyStopConfig, *, gap_key: str, gap_logged: bool
) -> GRPOEarlyStopCallback | None:
    """``None`` when no condition is set. Refuses a log-prob gap condition the run never feeds: ``gap_logged`` is
    whether it logs ``gap_key`` (both trainers only under the importance-sampling correction)."""
    if not config.active:
        return None
    if config.logratio_gap is not None and not gap_logged:
        raise ValueError(
            f"early_stop_logratio_gap reads {gap_key}, which only the importance-sampling correction "
            "(vllm_importance_sampling_correction) logs, and the correction is off in this run: the stop would "
            "never fire."
        )
    return GRPOEarlyStopCallback(config, gap_key)
