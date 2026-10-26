"""Script arguments for off-policy teacher distillation."""

from dataclasses import dataclass, field
from typing import ClassVar

from src.args.common_script_args import CommonScriptArguments
from src.args.mixins import ConversationRenderArguments


@dataclass
class DistillScriptArguments(ConversationRenderArguments, CommonScriptArguments):
    PROJECT_NAME: ClassVar[str] = "llm_distillation"

    teacher_model: str | None = field(
        default=None,
        metadata={"help": "Name or path of the teacher model."},
    )
    teacher_model_revision: str | None = field(
        default=None,
        metadata={
            "help": "Hub revision of the teacher; unset = the teacher repo's main. The student's model_revision "
            "does not carry over, since the teacher is usually another repo."
        },
    )
    # Distillation datasets conventionally ship a "messages" column (SFT's default is "prompt").
    conversation_field: str | None = field(
        default="messages",
        metadata={"help": "Dataset column holding the conversation, a list of role/content dicts."},
    )

    def _validate_ranges(self) -> None:
        """Require ``teacher_model``: an empty value reaches the loader as a blank model id and only
        fails there."""
        super()._validate_ranges()
        if not self.teacher_model:
            raise ValueError("teacher_model must be specified for distillation")
