"""Script arguments for DPO training."""

from dataclasses import dataclass, field

from src.args.common_script_args import CommonScriptArguments
from src.args.mixins import GenerationEvalArguments


@dataclass
class DPOScriptArguments(GenerationEvalArguments, CommonScriptArguments):
    images_field: str | None = field(
        default=None,
        metadata={
            "help": "VLM only: dataset column holding the row's image(s) (HF Image feature, single "
            "or list). Renamed to the 'images' column TRL's vision-preference route probes for, so "
            "hub datasets that store images under another name train without preprocessing. Unlike "
            "the SFT field it is NOT injected into a conversation turn — the vision route reads the "
            "column itself. Setting it declares the run VLM on a multimodal checkpoint."
        },
    )
    generation_max_prompt_length: int | None = field(
        default=512,
        metadata={
            "help": "Prompt cap for the eval-time GENERATION dataset only (GenerateExamplesCallback) "
            "— it bounds what is generated from, never what is trained on. DPO has no training-side "
            "prompt cap: TRL truncates the concatenated prompt + completion to DPOConfig.max_length "
            "keep_start, so an over-long prompt eats its own completion. Filter over-long prompts in "
            "the dataset instead. Deliberately NOT named max_prompt_length, which everywhere else in "
            "the toolkit is a training-side budget."
        },
    )

    def __post_init__(self):
        self._apply_default_project_name("dpo-tuning")
        self._validate_ranges()
