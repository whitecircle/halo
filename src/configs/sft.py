"""Script arguments for supervised fine-tuning, the conversation-rendering fields it shares with teacher
distillation and the tree-packing options."""

from dataclasses import dataclass, field, fields
from typing import ClassVar

from src.config_validation import ValidatedConfig, require_int
from src.configs.script_arguments import CommonScriptArguments, GenerationEvalArguments
from src.models.attention.tree_plan import FullAttentionForm


@dataclass
class ConversationRenderArguments:
    """Chat-template rendering knobs shared by conversation-rendering trainers (SFT, distillation)."""

    conversation_field: str | None = field(
        default="prompt", metadata={"help": "Dataset column holding the conversation, a list of role/content dicts."}
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
        default=None, metadata={"help": "System prompt added to a conversation that has none; unset adds nothing."}
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
class TreePackingArguments(ValidatedConfig):
    """``tree_packing`` and the ``tree_*`` options it gates (``agent-docs/optimization/tree-packing.md``)."""

    tree_packing: bool = field(
        default=False,
        metadata={
            "help": "Pack rows that share a prefix as a tree: the shared part is computed once and each row's "
            "remainder attends it and itself only. The same examples as flat packing, at fewer tokens. "
            "Requires packing and tree_share_markers."
        },
    )
    tree_share_markers: list[str] = field(
        default_factory=list,
        metadata={
            "help": "Tree packing: rows whose message before the final turn is identical up to and including one of "
            "these markers are examined together for a shared prefix."
        },
    )
    tree_share_requires: str | None = field(
        default=None,
        metadata={
            "help": "Tree packing: a row is shared only if this text occurs before its marker, e.g. the heading of "
            "the long part, so rows that put it after the marker are left alone instead of sharing a short header."
        },
    )
    tree_min_shared_tokens: int = field(
        default=16,
        metadata={"help": "Tree packing: a span shared by fewer tokens than this is repeated per row, not split off."},
    )
    tree_full_attention: FullAttentionForm = field(
        default="split",
        metadata={
            "help": "Tree packing, full-attention layers: 'split' adds a rectangular ancestor pass to the stock "
            "packed call (flash_attention_4), 'gather' builds one key sequence per tree node (flash_attention_2 as "
            "well; memory grows with fan-out x prefix length)."
        },
    )

    def _validate_fields(self) -> None:
        super()._validate_fields()
        require_int(type(self).__name__, ge=1, tree_min_shared_tokens=self.tree_min_shared_tokens)


# What a run that packs no trees must find at its default: set there, these options do nothing.
TREE_PACKING_FIELDS = tuple(f.name for f in fields(TreePackingArguments))


@dataclass
class SFTScriptArguments(
    TreePackingArguments, ConversationRenderArguments, GenerationEvalArguments, CommonScriptArguments
):
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
