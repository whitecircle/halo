"""Script arguments for supervised fine-tuning and the conversation-rendering fields it shares with teacher
distillation."""

from dataclasses import dataclass, field
from typing import ClassVar

from src.configs.script_arguments import CommonScriptArguments, GenerationEvalArguments


@dataclass
class ConversationRenderArguments:
    """Chat-template rendering knobs shared by conversation-rendering trainers (SFT, distillation)."""

    conversation_field: str | None = field(
        default="prompt",
        metadata={"help": "Dataset column holding the conversation, a list of role/content dicts."},
    )
    images_field: str | None = field(
        default=None,
        metadata={
            "help": "VLM only: dataset column holding the row's image(s) (HF Image feature, single or list). "
            "Injected into the first user turn ahead of its text, so hub datasets that store images "
            "outside the conversation (e.g. FineVision/the_cauldron/Docmatix) train without preprocessing."
        },
    )
    system_prompt: str | None = field(
        default=None,
        metadata={"help": "System prompt added to a conversation that has none; unset adds nothing."},
    )
    train_on_completions_only: bool = field(
        default=True, metadata={"help": "Train only on assistant completions, masking the prompt tokens."}
    )
    assistant_message_template: str | None = field(
        default=None,
        metadata={
            "help": "The rendered assistant-turn prefix of the model's chat template, like '<|im_start|>assistant\\n'. "
            "Required when train_on_completions_only is on, since no default fits every template."
        },
    )
    model_supports_system_role: bool = field(
        default=True,
        metadata={
            "help": "Whether the chat template accepts a system role; false prepends the system prompt to the first "
            "user message."
        },
    )
    interleaved_thinking: bool = field(
        default=False,
        metadata={
            "help": "Pass clear_thinking=False to apply_chat_template so earlier assistant turns keep their "
            "reasoning; only templates with that switch (the GLM family) read it. Text-only: the VLM path rejects "
            "it."
        },
    )


@dataclass
class SFTScriptArguments(ConversationRenderArguments, GenerationEvalArguments, CommonScriptArguments):
    PROJECT_NAME: ClassVar[str] = "sft-tuning"

    # SFT does not generate during eval by default (the preference/GRPO trainers do).
    generate_eval_examples: bool = field(default=False, metadata={"help": "Generate text examples during evaluation."})
    train_on_last_assistant_only: bool = field(
        default=False,
        metadata={
            "help": "If True, train only on the LAST assistant message instead of all. "
            "Requires train_on_completions_only=True."
        },
    )
    # use_liger_kernel, liger_kernel_config, packing/packing_strategy/eval_packing/padding_free live in TRL's SFTConfig.
