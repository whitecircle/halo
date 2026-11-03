from dataclasses import dataclass, field
from typing import ClassVar, Literal

from transformers import TrainingArguments

from halo.config_validation import ValidatedConfig, present, reject_non_default_args, require_finite, require_int
from halo.configs.script_arguments import (
    CommonScriptArguments,
    CrossDocumentLinearAttentionArguments,
    DatasetNumProcArguments,
    GenerationEvalArguments,
    ModelInitKwargsArguments,
    VisionTrainingArguments,
)


@dataclass
class DPOScriptArguments(VisionTrainingArguments, GenerationEvalArguments, CommonScriptArguments):
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

    def _validate_fields(self) -> None:
        super()._validate_fields()
        require_int(
            type(self).__name__, ge=1, **present(generation_max_prompt_length=self.generation_max_prompt_length)
        )
        if not self.generate_eval_examples:
            reject_non_default_args("A run with generate_eval_examples off", self, "generation_max_prompt_length")


@dataclass
class KTOScriptArguments(VisionTrainingArguments, CommonScriptArguments):
    """KTO expects an *unpaired* dataset: each row is ``{prompt, completion, label}``, ``label`` a bool marking the
    completion desirable. TRL's KTOTrainer auto-templates conversational rows via ``processing_class``."""

    PROJECT_NAME: ClassVar[str] = "kto-tuning"

    completion_field: str = field(
        default="completion",
        metadata={"help": "Dataset column holding the completion, a string or a conversation list."},
    )
    label_field: str = field(
        default="label", metadata={"help": "Dataset column holding the desirability bool (true = desirable)."}
    )
    images_field: str | None = field(
        default=None,
        metadata={
            "help": "VLM only: dataset column holding the row's images, renamed to the 'images' spelling TRL's "
            "KTOTrainer reads (under any other name they would be dropped). The vision path makes TRL refuse "
            "precompute_ref_log_probs and any chosen/rejected column."
        },
    )


@dataclass
class SMPOScriptArguments(VisionTrainingArguments, GenerationEvalArguments, CommonScriptArguments):
    PROJECT_NAME: ClassVar[str] = "smpo-tuning"


@dataclass
class SMPOConfig(
    CrossDocumentLinearAttentionArguments,
    DatasetNumProcArguments,
    ModelInitKwargsArguments,
    ValidatedConfig,
    TrainingArguments,
):
    """SMPO: a reference-model-free, margin-based preference method with optional SFT loss."""

    learning_rate: float = field(default=1e-6, metadata={"help": "Initial learning rate for AdamW optimizer."})
    logging_steps: float = field(default=10, metadata={"help": "Log every X steps."})
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "Enable gradient checkpointing to save memory."}
    )
    bf16: bool | None = field(
        default=None, metadata={"help": "Use bf16 mixed precision. Defaults to True if fp16 is False."}
    )
    beta: float = field(default=1.2, metadata={"help": "Scale of the margin logit before the margin loss."})
    target_margin: float = field(
        default=0.35, metadata={"help": "Target margin for chosen vs rejected log prob ratio."}
    )
    loss_type: Literal["sigmoid", "hinge", "ipo", "smooth_lower_bound"] = field(
        default="smooth_lower_bound", metadata={"help": "Type of margin loss function."}
    )
    chosen_sft_ratio: float = field(
        default=0.8, metadata={"help": "Weight for chosen completions in combined SFT loss (1.0 = only chosen)."}
    )
    use_margin_schedule: bool = field(
        default=True, metadata={"help": "Linearly increase target_margin during training."}
    )
    initial_margin: float = field(
        default=0.01,
        metadata={
            "help": "Initial margin of the margin schedule (starts small for curriculum learning); refused off its "
            "default with use_margin_schedule off."
        },
    )
    lower_clip_percentile: float | None = field(
        default=0.02, metadata={"help": "Percentile for clipping low log probs in rejected (0.01-0.05 recommended)."}
    )
    upper_clip_percentile: float | None = field(
        default=None, metadata={"help": "Percentile for clipping high log probs in chosen (0.95-0.99 if used)."}
    )
    min_log_prob: float | None = field(
        default=-2.3, metadata={"help": "Absolute min log prob threshold for rejected tokens."}
    )
    padding_free: bool = field(
        default=False,
        metadata={
            "help": "Flatten the batch into one varlen sequence instead of padding it. Needs a "
            "varlen Flash Attention kernel (FA2/FA3/FA4) and raises on any other implementation, "
            "the sdpa the script defaults to under reset_sinks: true included. Incompatible with CP, PP and VLM runs, "
            "and refused on an M-RoPE decoder (Qwen3.5/3.6)."
        },
    )
    max_length: int | None = field(
        default=1024,
        metadata={
            "help": "Total sequence budget (prompt + completion) in tokens, split into the max_prompt_length and "
            "max_completion_length shares. Unset or non-positive = the model's context window."
        },
    )
    max_prompt_length: int | None = field(
        default=None,
        metadata={
            "help": "Prompt share of max_length; longer prompts are truncated per truncation_mode. "
            "Unset = half of the resolved max_length, so the split scales with a context-resolved max_length "
            "instead of pinning prompts at a small constant."
        },
    )
    max_completion_length: int | None = field(
        default=None,
        metadata={
            "help": "Completion share of max_length; a longer completion is truncated from the end. The budget "
            "includes the EOS, so a completion that reaches it ends without one, as in offline GRPO. Unset = the "
            "max_length - max_prompt_length remainder, which an explicit value must fit."
        },
    )
    truncation_mode: Literal["keep_end", "keep_start"] = field(
        default="keep_end",
        metadata={
            "help": "Which end of an over-long PROMPT to keep: 'keep_end' (default, the tokens "
            "nearest the completion) or 'keep_start'. Completions always truncate from the end. Refused off its "
            "default on a VLM run, whose collator refuses an over-long prompt instead."
        },
    )
    disable_dropout: bool = field(default=True, metadata={"help": "Disable dropout for training stability."})

    def __post_init__(self):
        if self.bf16 is None:
            self.bf16 = not self.fp16
        super().__post_init__()

    def resolve_length_budget(self) -> tuple[int, int, int]:
        """Split a resolved ``max_length`` into ``(max_length, max_prompt_length, max_completion_length)``. An unset
        prompt share takes half, an unset completion share the remainder. The shares must fit: SMPO truncates each
        independently, so an over-budget split would exceed ``max_length`` without either cut firing."""
        max_length = self.max_length
        if max_length is None or max_length <= 0:
            raise ValueError(
                "SMPOConfig.max_length must be a positive int by the time the trainer is "
                "constructed (it is the total prompt + completion budget). Set it in the config, or "
                "resolve it to the model context window first (resolve_length_to_context, as "
                "scripts/training/preference/smpo.py does)."
            )

        max_prompt_length = self.max_prompt_length if self.max_prompt_length else max_length // 2
        max_completion_length = (
            self.max_completion_length if self.max_completion_length else max_length - max_prompt_length
        )
        if max_prompt_length <= 0 or max_completion_length <= 0:
            raise ValueError(
                f"SMPO length budget leaves no room: max_length={max_length}, "
                f"max_prompt_length={max_prompt_length}, max_completion_length={max_completion_length}. "
                f"Raise max_length or lower max_prompt_length."
            )
        if max_prompt_length + max_completion_length > max_length:
            raise ValueError(
                f"SMPO length budget overflows max_length: max_prompt_length ({max_prompt_length}) + "
                f"max_completion_length ({max_completion_length}) = "
                f"{max_prompt_length + max_completion_length} > max_length ({max_length}). Prompt and "
                f"completion are truncated independently, so this split would produce sequences longer "
                f"than max_length. Raise max_length or lower one of the shares."
            )
        return max_length, max_prompt_length, max_completion_length

    def _validate_fields(self) -> None:
        super()._validate_fields()
        owner = type(self).__name__
        require_finite(
            owner,
            initial_margin=self.initial_margin,
            learning_rate=self.learning_rate,
            logging_steps=self.logging_steps,
        )
        # Scales the preference logits: zero flattens the loss, a negative value inverts the preference.
        require_finite(owner, gt=0, beta=self.beta)
        # A non-positive max_length is legal: it resolves to the model's context window at launch.
        require_int(owner, **present(max_length=self.max_length))
        require_int(
            owner,
            ge=1,
            **present(max_prompt_length=self.max_prompt_length, max_completion_length=self.max_completion_length),
        )
        require_finite(owner, ge=0, target_margin=self.target_margin)
        if self.use_margin_schedule and self.initial_margin >= self.target_margin:
            raise ValueError(
                f"initial_margin ({self.initial_margin}) must be < target_margin ({self.target_margin}) "
                "when using margin schedule"
            )
        if not self.use_margin_schedule:
            reject_non_default_args("SMPO with use_margin_schedule off", self, "initial_margin")
        # SMPO's only multi-document row is the padding-free one.
        if not self.padding_free:
            reject_non_default_args("SMPO with padding_free off", self, "allow_cross_document_linear_attention")
        require_finite(owner, gt=0, le=0.5, **present(lower_clip_percentile=self.lower_clip_percentile))
        require_finite(owner, ge=0.5, lt=1, **present(upper_clip_percentile=self.upper_clip_percentile))
        require_finite(owner, lt=0, why=" (log probabilities are <= 0)", **present(min_log_prob=self.min_log_prob))
        require_finite(owner, ge=0, le=1, chosen_sft_ratio=self.chosen_sft_ratio)
