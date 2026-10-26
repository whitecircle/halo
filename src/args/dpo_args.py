"""Script arguments for DPO training."""

from dataclasses import dataclass, field
from typing import ClassVar

from src.args.common_script_args import CommonScriptArguments
from src.args.mixins import GenerationEvalArguments
from src.config_validation import present, require_positive_int


@dataclass
class DPOScriptArguments(GenerationEvalArguments, CommonScriptArguments):
    PROJECT_NAME: ClassVar[str] = "dpo-tuning"

    images_field: str | None = field(
        default=None,
        metadata={
            "help": "VLM only: dataset column holding the row's image(s), renamed to the 'images' column TRL's "
            "vision-preference route reads rather than injected into a turn. Setting it declares the run VLM on a "
            "multimodal checkpoint."
        },
    )
    generation_max_prompt_length: int | None = field(
        default=512,
        metadata={
            "help": "Prompt cap of the eval-time generation dataset only (GenerateExamplesCallback), never of what "
            "is trained on. DPO has no training-side prompt cap, since TRL truncates the concatenated prompt and "
            "completion to max_length (keep_start): filter over-long prompts in the dataset. "
            "See agent-docs/reference/configuration-reference.md#sequence-length-caps-vs-generation-budgets."
        },
    )

    def _validate_ranges(self) -> None:
        super()._validate_ranges()
        require_positive_int(
            type(self).__name__, **present(generation_max_prompt_length=self.generation_max_prompt_length)
        )
