"""``DistillationConfig`` — training config for teacher distillation."""

from dataclasses import dataclass, field
from typing import Literal

from transformers import TrainingArguments

from src.args.mixins import DatasetNumProcArguments
from src.args.validation import RangeValidatedConfig, require_positive


@dataclass
class DistillationConfig(DatasetNumProcArguments, RangeValidatedConfig, TrainingArguments):
    """Configuration for distillation training, extending HuggingFace TrainingArguments."""

    # The teacher arm's names in the trainer-side divergence registry (losses.DIVERGENCES), pinned to it
    # by a test: the config layer imports no trainer.
    distill_loss: Literal[
        "kl_divergence", "mse", "soft_cross_entropy", "cosine_similarity", "jensen_shannon", "slim"
    ] = field(
        default="kl_divergence",
        metadata={
            "help": "Distillation loss type. Options: kl_divergence, mse, soft_cross_entropy, "
            "cosine_similarity, jensen_shannon, slim"
        },
    )
    distill_temperature: float = field(
        default=1.0,
        metadata={
            "help": "Softmax temperature for the temperature-based distillation losses "
            "(kl_divergence, soft_cross_entropy, jensen_shannon, slim). IGNORED by mse and "
            "cosine_similarity, whose definitions take no temperature — call_divergence forwards it "
            "only to losses that declare it."
        },
    )
    distill_alpha: float = field(
        default=1.0,
        metadata={
            "help": "Weight of the distillation term; the CLM term takes 1 - distill_alpha. 1.0 trains "
            "distillation alone (CLM is then logged, not trained)."
        },
    )
    apply_hard_labels: bool = field(
        default=False,
        metadata={
            "help": "Scale the distillation term per token by the detached gold-token gate "
            "(1 - student_prob[label]) * teacher_prob[label]. Refused under slim, which applies its own "
            "gold-token weight."
        },
    )
    max_length: int | None = field(
        default=2048,
        metadata={
            "help": "Maximum tokenized sequence length. Over-length conversations are dropped, not "
            "truncated. null / non-positive resolves to the student's context window at launch."
        },
    )

    def __post_init__(self):
        self._validate_ranges()
        super().__post_init__()

    def _validate_ranges(self) -> None:
        super()._validate_ranges()
        require_positive(type(self).__name__, distill_temperature=self.distill_temperature)
        if not 0.0 <= self.distill_alpha <= 1.0:
            raise ValueError(f"distill_alpha must be in [0, 1], got {self.distill_alpha}")
