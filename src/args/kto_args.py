"""Script arguments for KTO (Kahneman-Tversky Optimization) training."""

from dataclasses import dataclass, field

from src.args.common_script_args import CommonScriptArguments


@dataclass
class KTOScriptArguments(CommonScriptArguments):
    """Script arguments for KTO training.

    KTO expects an *unpaired* dataset: each row is ``{prompt, completion, label}``
    where ``label`` is a bool marking the completion desirable (True) or not (False).
    Conversational ``prompt``/``completion`` (lists of role/content dicts) are
    auto-templated by TRL's KTOTrainer via ``processing_class``.
    """

    completion_field: str = field(
        default="completion",
        metadata={"help": "Dataset field holding the completion (str or list-of-dicts conversation)"},
    )
    label_field: str = field(
        default="label",
        metadata={"help": "Dataset field holding the desirability bool (True=desirable)"},
    )
    images_field: str | None = field(
        default=None,
        metadata={
            "help": "VLM only: dataset column holding the row's images (HF Image feature list). "
            "Renamed to 'images' before training — TRL's KTOTrainer reads only the 'images'/'image' "
            "spelling, both to detect a vision dataset and in its vision collator, so a hub column "
            "under any other name would train as text with the images dropped. Selecting the vision "
            "path makes TRL refuse precompute_ref_log_probs and any chosen/rejected column."
        },
    )

    def __post_init__(self):
        self._apply_default_project_name("kto-tuning")
        self._validate_ranges()
