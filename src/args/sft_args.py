"""Script arguments for supervised fine-tuning."""

from dataclasses import dataclass, field

from src.args.common_script_args import CommonScriptArguments
from src.args.mixins import ConversationRenderArguments, GenerationEvalArguments


@dataclass
class SFTScriptArguments(ConversationRenderArguments, GenerationEvalArguments, CommonScriptArguments):
    # SFT does not generate during eval by default (the preference/GRPO trainers do).
    generate_eval_examples: bool = field(default=False, metadata={"help": "Do generate examples on eval"})
    train_on_last_assistant_only: bool = field(
        default=False,
        metadata={
            "help": "If True, train only on the LAST assistant message instead of all. "
            "Requires train_on_completions_only=True."
        },
    )
    # use_liger_kernel, liger_kernel_config, packing/packing_strategy/eval_packing/padding_free live in TRL's SFTConfig.

    def __post_init__(self):
        self._apply_default_project_name("sft-tuning")
        self._validate_ranges()
