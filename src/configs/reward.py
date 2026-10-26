"""Script arguments for Bradley-Terry reward-model training."""

from dataclasses import dataclass, field
from typing import ClassVar

from src.configs.script_arguments import CommonScriptArguments


@dataclass
class RewardScriptArguments(CommonScriptArguments):
    PROJECT_NAME: ClassVar[str] = "reward-modeling"

    images_field: str | None = field(
        default=None,
        metadata={
            "help": "VLM only: dataset column holding the row's image(s), renamed to 'images' and merged into the "
            "prompt conversation both sides of a pair share. Setting it declares the run VLM on a multimodal "
            "checkpoint, as a dataset with an 'images'/'image' column does without it. "
            "See agent-docs/training-methods/preference/reward-modeling.md#vision-language."
        },
    )
