"""Schedule an arbitrary numeric model attribute over training steps."""

import math
from typing import Any

from transformers import TrainerCallback

_SCHEDULE_TYPES = ("cosine", "linear", "exponential")


class VariableSchedulerCallback(TrainerCallback):
    """Schedule an attribute on the model, cosine/linear/exponential."""

    def __init__(
        self,
        attribute_name: str,
        initial_value: float,
        final_value: float,
        schedule_type: str = "cosine",
        warmup_steps: int = 0,
    ):
        """
        Args:
            schedule_type: 'cosine', 'linear', or 'exponential'.
            warmup_steps: steps holding initial_value before the schedule starts.
        """
        self.attribute_name = attribute_name
        self.initial_value = initial_value
        self.final_value = final_value
        self.schedule_type = schedule_type
        self.warmup_steps = warmup_steps
        self.total_steps = None

        # Validated at construction, not on the first scheduled step, which is only reached after
        # dataset preparation and the trainer's own setup.
        if self.schedule_type not in _SCHEDULE_TYPES:
            raise ValueError(f"Unknown schedule type: {schedule_type}. Must be one of {_SCHEDULE_TYPES}.")
        if self.schedule_type == "exponential" and (initial_value <= 0 or final_value <= 0):
            raise ValueError(
                f"Exponential schedule requires positive initial and final values, "
                f"got {initial_value} -> {final_value}"
            )

    @staticmethod
    def _get_target_object(kwargs: dict) -> Any:
        """The model from the callback kwargs, unwrapped so the attribute lands on the module that reads it."""
        target_obj = kwargs.get("model")
        if target_obj is None:
            raise ValueError(
                "Could not find 'model' in callback arguments. This callback should be used with a Trainer."
            )
        return target_obj.module if hasattr(target_obj, "module") else target_obj

    def _calculate_value(self, current_step: int) -> float:
        """Calculate the scheduled value for the current step."""
        if current_step < self.warmup_steps:
            return self.initial_value

        progress = (current_step - self.warmup_steps) / self.total_steps
        progress = min(progress, 1.0)

        if self.schedule_type == "cosine":
            return self.final_value + 0.5 * (self.initial_value - self.final_value) * (
                1 + math.cos(math.pi * progress)
            )
        elif self.schedule_type == "linear":
            return self.initial_value + (self.final_value - self.initial_value) * progress
        else:
            return self.initial_value * (self.final_value / self.initial_value) ** progress

    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize scheduling parameters at the start of training."""
        self.total_steps = state.max_steps - self.warmup_steps
        if self.total_steps <= 0:
            raise ValueError(
                f"Total training steps ({state.max_steps}) must be greater than warmup steps ({self.warmup_steps})"
            )

        target_obj = self._get_target_object(kwargs)

        if not hasattr(target_obj, self.attribute_name):
            setattr(target_obj, self.attribute_name, self.initial_value)

    def on_step_begin(self, args, state, control, **kwargs):
        """Update variable at the beginning of each step."""
        current_value = self._calculate_value(state.global_step)

        target_obj = self._get_target_object(kwargs)

        setattr(target_obj, self.attribute_name, current_value)
