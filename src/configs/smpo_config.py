"""
Configuration for Smooth Margin Preference Optimization (SMPO) Trainer.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from transformers import TrainingArguments

from src.args.validation import RangeValidatedConfig
from src.data.spans import LABEL_IGNORE_INDEX


@dataclass
class SmoothMarginPOConfig(RangeValidatedConfig, TrainingArguments):
    """Config for SmoothMarginPOTrainer (SMPO): a reference-model-free, margin-based preference
    method with optional SFT loss. Per-field docs are in each field's ``help`` metadata."""

    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "Initial learning rate for AdamW optimizer."},
    )
    logging_steps: float = field(
        default=10,
        metadata={"help": "Log every X steps."},
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Enable gradient checkpointing to save memory."},
    )
    bf16: bool | None = field(
        default=None,
        metadata={"help": "Use bf16 mixed precision. Defaults to True if fp16 is False."},
    )

    beta: float = field(
        default=1.2,
        metadata={"help": "Temperature parameter for loss sensitivity."},
    )
    target_margin: float = field(
        default=0.35,
        metadata={"help": "Target margin for chosen vs rejected log prob ratio."},
    )
    loss_type: Literal["sigmoid", "hinge", "ipo", "smooth_lower_bound"] = field(
        default="smooth_lower_bound",
        metadata={"help": "Type of margin loss function."},
    )
    chosen_sft_ratio: float = field(
        default=0.8,
        metadata={"help": "Weight for chosen completions in combined SFT loss (1.0 = only chosen)."},
    )
    use_margin_schedule: bool = field(
        default=True,
        metadata={"help": "Linearly increase target_margin during training."},
    )
    initial_margin: float = field(
        default=0.01,
        metadata={"help": "Initial margin when using margin schedule (starts small for curriculum learning)."},
    )

    lower_clip_percentile: float | None = field(
        default=0.02,
        metadata={"help": "Percentile for clipping low log probs in rejected (0.01-0.05 recommended)."},
    )
    upper_clip_percentile: float | None = field(
        default=None,
        metadata={"help": "Percentile for clipping high log probs in chosen (0.95-0.99 if used)."},
    )
    min_log_prob: float | None = field(
        default=-2.3,
        metadata={"help": "Absolute min log prob threshold for rejected tokens."},
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
            "help": "Total sequence budget (prompt + completion) in tokens. null / non-positive "
            "resolves to the model's context window at launch. Splits into a prompt and a completion "
            "share via max_prompt_length / max_completion_length."
        },
    )
    max_prompt_length: int | None = field(
        default=None,
        metadata={
            "help": "Prompt share of max_length; longer prompts are truncated per truncation_mode. "
            "null (default) = half of the resolved max_length, so the split scales with a "
            "context-resolved max_length instead of pinning prompts at a small constant."
        },
    )
    max_completion_length: int | None = field(
        default=None,
        metadata={
            "help": "Completion share of max_length; longer completions are truncated (keeping the "
            "terminal EOS). null (default) = the max_length - max_prompt_length remainder. An "
            "explicit value must still fit that remainder."
        },
    )
    truncation_mode: Literal["keep_end", "keep_start"] = field(
        default="keep_end",
        metadata={
            "help": "Which end of an over-long PROMPT to keep: 'keep_end' (default, the tokens "
            "nearest the completion) or 'keep_start'. Completions always truncate from the end."
        },
    )
    label_pad_token_id: int = field(
        default=LABEL_IGNORE_INDEX,
        metadata={"help": "Token ID for masking prompt in labels."},
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Disable dropout for training stability."},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes for dataset preprocessing."},
    )

    model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Model-config overrides on every entry-script path: written onto the loaded "
            "config's fields before the load, raising on a key that config does not declare and "
            "on dtype/torch_dtype. Model-loading kwargs only where a trainer is constructed "
            "programmatically with the model as a path string."
        },
    )

    def resolve_length_budget(self) -> tuple[int, int, int]:
        """Split ``max_length`` into ``(max_length, max_prompt_length, max_completion_length)``.

        An unset prompt share takes half of ``max_length`` so it scales with a context-resolved
        ``max_length`` (a fixed constant would truncate prompts to it on a long-context run); an
        unset completion share takes the remainder. The shares must fit inside ``max_length``: SMPO
        truncates prompt and completion independently, so an over-budget split would exceed it
        without either cut firing.

        Callers must resolve a null / non-positive ``max_length`` to the model context window first
        (:func:`~src.models.loading.tokenizer_setup.resolve_length_to_context`).
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

    def __post_init__(self):
        if self.bf16 is None:
            self.bf16 = not self.fp16

        self._validate_ranges()
        super().__post_init__()

    def _validate_ranges(self) -> None:
        super()._validate_ranges()
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

        if self.min_log_prob is not None and self.min_log_prob >= 0:
            raise ValueError("min_log_prob must be negative (log probabilities are <= 0)")

        if not (0 <= self.chosen_sft_ratio <= 1):
            raise ValueError("chosen_sft_ratio must be in [0, 1]")
