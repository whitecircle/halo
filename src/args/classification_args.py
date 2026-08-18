"""Script arguments for sequence-classification training."""

from dataclasses import dataclass, field

from src.args.common_script_args import CommonScriptArguments


@dataclass
class CLFScriptArguments(CommonScriptArguments):
    text_field: str | None = field(
        default=None,
        metadata={
            "help": "Raw-text column to classify (wrapped as a single user turn) when the dataset has "
            "no pre-built 'prompt' conversation — e.g. text/label datasets like imdb."
        },
    )

    def __post_init__(self):
        self._apply_default_project_name("classification")
        self._validate_ranges()
