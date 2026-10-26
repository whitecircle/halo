"""The marker a callback that can end a run before its plan carries, read by the script runner after
``train()`` so a stopped run exits non-zero whichever method's callback stopped it."""

from transformers import TrainerCallback


class StopsTrainingEarly(TrainerCallback):
    """A callback that may end training early; ``stopped`` is set once it has. Its verdict must be taken
    across ranks, since every rank exits on it."""

    stopped: bool = False
