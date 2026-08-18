"""Script arguments for Bradley-Terry reward-model training."""

from dataclasses import dataclass, field

from src.args.common_script_args import CommonScriptArguments


@dataclass
class RMScriptArguments(CommonScriptArguments):
    images_field: str | None = field(
        default=None,
        metadata={
            "help": "VLM only: dataset column holding the row's image(s) (HF Image feature, single "
            "or list). Renamed to the 'images' column the vision path reads, so hub datasets that "
            "store images under another name train without preprocessing. The images are merged "
            "into the PROMPT conversation (both sides of a pair share it), filling unset "
            "{'type': 'image'} placeholders in order when the messages carry them and otherwise "
            "leading the first user turn. Setting it declares the run VLM on a multimodal "
            "checkpoint; a dataset that already ships an 'images'/'image' column declares it "
            "without the knob."
        },
    )

    def __post_init__(self):
        self._apply_default_project_name("reward-modeling")
        self._validate_ranges()
