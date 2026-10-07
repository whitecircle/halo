"""Script arguments for sequence-classification training."""

from dataclasses import dataclass, field
from typing import ClassVar

from src.args.common_script_args import CommonScriptArguments


@dataclass
class CLFScriptArguments(CommonScriptArguments):
    PROJECT_NAME: ClassVar[str] = "classification"

    text_field: str | None = field(
        default=None,
        metadata={
            "help": "Raw-text column to classify (wrapped as a single user turn) when the dataset has "
            "no pre-built 'prompt' conversation — e.g. text/label datasets like imdb."
        },
    )
