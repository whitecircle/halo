"""Script arguments for Environmental GRPO training (YAML-based)."""

from dataclasses import dataclass, field

from src.args.common_script_args import CommonScriptArguments
from src.args.mixins import PromptDatasetArguments


@dataclass
class EnvironmentalGRPOScriptArguments(PromptDatasetArguments, CommonScriptArguments):
    """Dataset-specific args for Environmental GRPO. Async infra is in AsyncTrainingConfig,
    env selection in EnvironmentConfig (both parsed separately)."""

    answer_field: str | None = field(
        default="answer", metadata={"help": "Field in dataset containing the expected answer (optional)"}
    )

    context_fields: list[str] | None = field(
        default=None, metadata={"help": "Additional fields to pass as context to environment"}
    )

    def __post_init__(self):
        self._apply_default_project_name("environmental-grpo")
        self._validate_ranges()
