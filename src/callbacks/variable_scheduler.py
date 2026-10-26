"""Schedule an arbitrary numeric model attribute over training steps."""

from typing import Any

from transformers import TrainerCallback


class VariableSchedulerCallback(TrainerCallback):
    """Move a model attribute linearly from ``initial_value`` to ``final_value`` at ``max_steps``, then hold it."""

    def __init__(self, attribute_name: str, initial_value: float, final_value: float):
        self.attribute_name = attribute_name
        self.initial_value = initial_value
        self.final_value = final_value
        self.total_steps = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.total_steps = state.max_steps
        if self.total_steps <= 0:
            raise ValueError(f"Total training steps ({state.max_steps}) must be positive to schedule over.")

        target_obj = self._get_target_object(kwargs)
        if not hasattr(target_obj, self.attribute_name):
            setattr(target_obj, self.attribute_name, self.initial_value)

    def on_step_begin(self, args, state, control, **kwargs):
        progress = min(state.global_step / self.total_steps, 1.0)
        value = self.initial_value + (self.final_value - self.initial_value) * progress
        setattr(self._get_target_object(kwargs), self.attribute_name, value)

    @staticmethod
    def _get_target_object(kwargs: dict) -> Any:
        """The callback's model with its top-level ``.module`` peeled, so the attribute lands on the module
        that reads it; a recursive unwrap would rewrite every nested ``.module`` child each step."""
        target_obj = kwargs.get("model")
        if target_obj is None:
            raise ValueError(
                "Could not find 'model' in callback arguments. This callback should be used with a Trainer."
            )
        return target_obj.module if hasattr(target_obj, "module") else target_obj
