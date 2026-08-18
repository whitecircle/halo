"""``DistillationConfig`` — training config for teacher distillation."""

from dataclasses import dataclass, field
from typing import Literal

from transformers import TrainingArguments

from src.args.validation import RangeValidatedConfig


@dataclass
class DistillationConfig(RangeValidatedConfig, TrainingArguments):
    """Configuration for distillation training, extending HuggingFace TrainingArguments."""

    distill_loss: Literal[
        "kl_divergence",
        "mse",
        "soft_cross_entropy",
        "cosine_similarity",
        "jensen_shannon",
        "earth_mover_distance",
        "alpha_beta_divergence",
        "slim",
    ] = field(
        default="kl_divergence",
        metadata={
            "help": "Distillation loss type. Options: kl_divergence, mse, soft_cross_entropy, "
            "cosine_similarity, jensen_shannon, earth_mover_distance, alpha_beta_divergence, slim"
        },
    )
    distill_temperature: float = field(
        default=1.0,
        metadata={
            "help": "Softmax temperature for the temperature-based distillation losses "
            "(kl_divergence, soft_cross_entropy, jensen_shannon, slim). IGNORED by mse, "
            "cosine_similarity, earth_mover_distance and alpha_beta_divergence, whose "
            "definitions take no temperature — call_distillation_loss forwards it only to "
            "losses that declare it."
        },
    )
    distill_alpha: float = field(
        default=1.0,
        metadata={"help": "Weight for distillation loss (vs CLM loss). 1.0 means only distillation loss"},
    )
    apply_hard_labels: bool = field(
        default=False,
        metadata={"help": "Apply hard labels coefficient to distillation loss"},
    )
    use_clm_loss: bool = field(
        default=True,
        metadata={"help": "Add the auxiliary CLM (SFT) loss alongside distillation. False = distillation only"},
    )
    max_length: int | None = field(
        default=2048,
        metadata={
            "help": "Maximum tokenized sequence length. Over-length conversations are dropped, not "
            "truncated. null / non-positive resolves to the student's context window at launch."
        },
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes for dataset preprocessing"},
    )

    def __post_init__(self):
        self._validate_ranges()
        super().__post_init__()

    def _validate_ranges(self) -> None:
        super()._validate_ranges()
        if self.distill_temperature <= 0:
            raise ValueError(f"distill_temperature must be > 0, got {self.distill_temperature}")
        if not 0.0 <= self.distill_alpha <= 1.0:
            raise ValueError(f"distill_alpha must be in [0, 1], got {self.distill_alpha}")
