"""The marker of a callback that can end a run early; the script runner reads it so a stopped run exits non-zero."""

from transformers import TrainerCallback


class StopsTrainingEarly(TrainerCallback):
    """``stopped`` is set once it has ended training; take its verdict across ranks, since every rank exits on it."""

    stopped: bool = False
