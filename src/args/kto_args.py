"""Script arguments for KTO (Kahneman-Tversky Optimization) training."""

from dataclasses import dataclass, field
from typing import ClassVar

from src.args.common_script_args import CommonScriptArguments


@dataclass
class KTOScriptArguments(CommonScriptArguments):
    """Script arguments for KTO training.

    KTO expects an *unpaired* dataset: each row is ``{prompt, completion, label}``
    where ``label`` is a bool marking the completion desirable (True) or not (False).
    Conversational ``prompt``/``completion`` (lists of role/content dicts) are
    auto-templated by TRL's KTOTrainer via ``processing_class``.
    """

    PROJECT_NAME: ClassVar[str] = "kto-tuning"

    completion_field: str = field(
        default="completion",
        metadata={"help": "Dataset column holding the completion, a string or a conversation list."},
    )
    label_field: str = field(
        default="label",
        metadata={"help": "Dataset column holding the desirability bool (true = desirable)."},
    )
    images_field: str | None = field(
        default=None,
        metadata={
            "help": "VLM only: dataset column holding the row's images, renamed to the 'images' spelling TRL's "
            "KTOTrainer reads (under any other name they would be dropped). The vision path makes TRL refuse "
            "precompute_ref_log_probs and any chosen/rejected column."
        },
    )
