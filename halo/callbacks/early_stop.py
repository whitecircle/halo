"""The marker of a callback that can end a run early; the script runner reads it so a stopped run exits non-zero."""

from transformers import TrainerCallback


class StopsTrainingEarly(TrainerCallback):
    """``stopped`` is set once it has ended training; take its verdict across ranks, since every rank exits on it.
    ``stop_reason`` is what the exit says, where the run's step count alone would not say why."""

    stopped: bool = False
    stop_reason: str | None = None
