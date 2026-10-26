"""DPO, KTO and SMPO script arguments and ``SmoothMarginPOConfig``."""

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from transformers import TrainingArguments

from src.config_validation import ValidatedConfig, present, require_finite, require_int
from src.configs.script_arguments import (
    CommonScriptArguments,
    DatasetNumProcArguments,
    GenerationEvalArguments,
    ModelInitKwargsArguments,
)


@dataclass
class DPOScriptArguments(GenerationEvalArguments, CommonScriptArguments):
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
class SMPOScriptArguments(GenerationEvalArguments, CommonScriptArguments):
    """Script-level arguments for SMPO training (not training hyperparameters)."""

    PROJECT_NAME: ClassVar[str] = "smpo-tuning"


@dataclass
class SmoothMarginPOConfig(DatasetNumProcArguments, ModelInitKwargsArguments, ValidatedConfig, TrainingArguments):
    """Config for SmoothMarginPOTrainer (SMPO): a reference-model-free, margin-based preference
    method with optional SFT loss. Per-field docs are in each field's ``help`` metadata."""

    learning_rate: float = field(default=1e-6, metadata={"help": "Initial learning rate for AdamW optimizer."})
    logging_steps: float = field(default=10, metadata={"help": "Log every X steps."})
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "Enable gradient checkpointing to save memory."}
    )
    bf16: bool | None = field(
        default=None, metadata={"help": "Use bf16 mixed precision. Defaults to True if fp16 is False."}
    )
    beta: float = field(default=1.2, metadata={"help": "Temperature parameter for loss sensitivity."})
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
        metadata={"help": "Initial margin when using margin schedule (starts small for curriculum learning)."},
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
            "the sdpa the script defaults to under reset_sinks: true included."
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
            "help": "Completion share of max_length; a longer completion is truncated from the end, keeping the "
            "terminal EOS. Unset = the max_length - max_prompt_length remainder, which an explicit value must fit."
        },
    )
    truncation_mode: Literal["keep_end", "keep_start"] = field(
        default="keep_end",
        metadata={
            "help": "Which end of an over-long PROMPT to keep: 'keep_end' (default, the tokens "
            "nearest the completion) or 'keep_start'. Completions always truncate from the end."
        },
    )
    disable_dropout: bool = field(default=True, metadata={"help": "Disable dropout for training stability."})

    def __post_init__(self):
        if self.bf16 is None:
            self.bf16 = not self.fp16
        super().__post_init__()

    def resolve_length_budget(self) -> tuple[int, int, int]:
        """Split a resolved ``max_length`` into ``(max_length, max_prompt_length, max_completion_length)``.

        An unset prompt share takes half, so it scales with a context-resolved ``max_length``; an unset completion
        share takes the remainder. The shares must fit: SMPO truncates prompt and completion independently, so an
        over-budget split would exceed ``max_length`` without either cut firing.
        """
        max_length = self.max_length
        if max_length is None or max_length <= 0:
            raise ValueError(
                "SmoothMarginPOConfig.max_length must be a positive int by the time the trainer is "
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
            target_margin=self.target_margin,
            initial_margin=self.initial_margin,
            chosen_sft_ratio=self.chosen_sft_ratio,
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
        if self.target_margin < 0:
            raise ValueError("target_margin must be >= 0")

        if self.use_margin_schedule and self.initial_margin >= self.target_margin:
            raise ValueError(
                f"initial_margin ({self.initial_margin}) must be < target_margin ({self.target_margin}) "
                "when using margin schedule"
            )

        if self.lower_clip_percentile is not None and not (0 < self.lower_clip_percentile <= 0.5):
            raise ValueError("lower_clip_percentile must be in (0, 0.5]")

        if self.upper_clip_percentile is not None and not (0.5 <= self.upper_clip_percentile < 1):
            raise ValueError("upper_clip_percentile must be in [0.5, 1)")

        if self.min_log_prob is not None:
            require_finite(owner, min_log_prob=self.min_log_prob)
            if self.min_log_prob >= 0:
                raise ValueError("min_log_prob must be negative (log probabilities are <= 0)")

        if not (0 <= self.chosen_sft_ratio <= 1):
            raise ValueError("chosen_sft_ratio must be in [0, 1]")
