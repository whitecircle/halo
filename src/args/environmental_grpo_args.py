"""Script arguments for Environmental GRPO training (YAML-based)."""

from dataclasses import dataclass, field
from typing import ClassVar

from src.configs.grpo import PromptDatasetArguments
from src.configs.script_arguments import DEFAULT_ANSWER_FIELD, CommonScriptArguments


@dataclass
class EnvironmentalGRPOScriptArguments(PromptDatasetArguments, CommonScriptArguments):
    """Dataset-specific args for Environmental GRPO. Async infra is in AsyncTrainingConfig,
    env selection in EnvironmentConfig (both parsed separately)."""

    PROJECT_NAME: ClassVar[str] = "environmental-grpo"

    answer_field: str | None = field(
        default=DEFAULT_ANSWER_FIELD,
        metadata={
            "help": "Dataset column holding the expected answer, carried when present. Whether an answer is needed "
            "is the environment's requires_answer."
        },
    )

    context_fields: list[str] | None = field(
        default=None, metadata={"help": "Further dataset columns passed to the environment as context."}
    )
