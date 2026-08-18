"""Script arguments for off-policy teacher distillation."""

from dataclasses import dataclass, field

from src.args.common_script_args import CommonScriptArguments
from src.args.mixins import ConversationRenderArguments


@dataclass
class DistillScriptArguments(ConversationRenderArguments, CommonScriptArguments):
    teacher_model: str | None = field(
        default=None,
        metadata={"help": "Name or path of the teacher model for distillation"},
    )
    teacher_model_revision: str | None = field(
        default=None,
        metadata={
            "help": "Hub revision for the teacher. The teacher is usually a DIFFERENT repo from the "
            "student, so it cannot share the student's model_revision — sending one repo's commit to "
            "another 404s. Defaults to the teacher repo's main."
        },
    )
    # Distillation datasets conventionally ship a "messages" column (SFT's default is "prompt").
    conversation_field: str | None = field(
        default="messages",
        metadata={"help": "Field in dataset with conversations (in list of dicts format)"},
    )

    def _validate_ranges(self) -> None:
        """Require ``teacher_model``: an empty value reaches the loader as a blank model id and only
        fails there, and the CLI override path bypasses ``__post_init__``."""
        super()._validate_ranges()
        if not self.teacher_model:
            raise ValueError("teacher_model must be specified for distillation")

    def __post_init__(self):
        self._apply_default_project_name("llm_distillation")
        self._validate_ranges()
