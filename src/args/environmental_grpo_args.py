"""Script arguments for Environmental GRPO training (YAML-based)."""

from dataclasses import dataclass, field

from src.args.common_script_args import CommonScriptArguments
from src.args.mixins import PromptDatasetArguments

# The column an answer is read from unless the config renames it. The script checks the column exists
# only for a rename (a rename resolving to nothing is a typo); under this default a dataset without
# it simply carries no answer, and the environment's ``requires_answer`` decides whether that is fatal.
DEFAULT_ANSWER_FIELD = "answer"


@dataclass
class EnvironmentalGRPOScriptArguments(PromptDatasetArguments, CommonScriptArguments):
    """Dataset-specific args for Environmental GRPO. Async infra is in AsyncTrainingConfig,
    env selection in EnvironmentConfig (both parsed separately)."""

    answer_field: str | None = field(
        default=DEFAULT_ANSWER_FIELD,
        metadata={
            "help": "Field in dataset containing the expected answer; carried when present. Whether "
            "an answer is needed at all is the environment's requires_answer declaration."
        },
    )

    context_fields: list[str] | None = field(
        default=None, metadata={"help": "Additional fields to pass as context to environment"}
    )

    def __post_init__(self):
        self._apply_default_project_name("environmental-grpo")
        self._validate_ranges()
