"""``DistillationConfig`` — training config for teacher distillation."""

from dataclasses import dataclass, field
from typing import Literal

from transformers import TrainingArguments

from src.args.mixins import DEFAULT_JSD_BETA, DatasetNumProcArguments
from src.args.validation import (
    RangeValidatedConfig,
    present,
    require_finite,
    require_int,
    require_positive,
    require_positive_int,
)

# Top-k keeps the teacher's most likely tokens, so it only fits losses weighted by the teacher's
# probabilities. Reverse KL and JSD also weight by the student's, and the student's likely tokens may
# fall into the tail bin. MSE, cosine and the like compare raw logits, which can't be binned.
TOPK_DISTILL_LOSSES = ("kl_divergence", "soft_cross_entropy")


@dataclass
class DistillationConfig(DatasetNumProcArguments, RangeValidatedConfig, TrainingArguments):
    """Configuration for distillation training, extending HuggingFace TrainingArguments."""

    # The teacher arm's names in the trainer-side divergence registry (losses.DIVERGENCES), pinned to it
    # by a test: the config layer imports no trainer.
    distill_loss: Literal[
        "kl_divergence", "reverse_kl", "mse", "soft_cross_entropy", "cosine_similarity", "jensen_shannon", "slim"
    ] = field(
        default="kl_divergence",
        metadata={
            "help": "Distillation loss type. Options: kl_divergence (forward KL), reverse_kl, mse, "
            "soft_cross_entropy, cosine_similarity, jensen_shannon (generalized JSD; β set by "
            "distill_jsd_beta), slim"
        },
    )
    distill_temperature: float = field(
        default=1.0,
        metadata={
            "help": "Softmax temperature for the temperature-based distillation losses "
            "(kl_divergence, reverse_kl, soft_cross_entropy, jensen_shannon, slim). IGNORED by mse and "
            "cosine_similarity, whose definitions take no temperature — call_divergence forwards it "
            "only to losses that declare it."
        },
    )
    distill_jsd_beta: float = field(
        default=DEFAULT_JSD_BETA,
        metadata={
            "help": "Generalized-JSD β in [0, 1] for distill_loss: jensen_shannon. 0 = forward KL, "
            "1 = reverse KL, 0.5 = symmetric JSD. Any other value needs distill_loss: jensen_shannon."
        },
    )
    distill_topk: int | None = field(
        default=None,
        metadata={
            "help": "Score the loss on the teacher's top-k tokens plus one tail bin holding the rest "
            "of the vocabulary, instead of the full vocabulary. An approximation (a lower bound on the "
            "full-vocab loss) that saves no memory with a local teacher. kl_divergence and "
            "soft_cross_entropy only, and below the vocab size; null = full vocabulary."
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
        owner = type(self).__name__
        require_positive(owner, distill_temperature=self.distill_temperature)
        require_finite(owner, distill_alpha=self.distill_alpha, distill_jsd_beta=self.distill_jsd_beta)
        # A non-positive length is legal: it resolves to the student's context window at launch.
        require_int(owner, **present(max_length=self.max_length))
        if not 0.0 <= self.distill_alpha <= 1.0:
            raise ValueError(f"distill_alpha must be in [0, 1], got {self.distill_alpha}")
        if not 0.0 <= self.distill_jsd_beta <= 1.0:
            raise ValueError(f"distill_jsd_beta must be in [0, 1], got {self.distill_jsd_beta}")
        # Every other loss ignores β, so a set β with one of them is a config that does not train
        # what it states.
        if self.distill_jsd_beta != DEFAULT_JSD_BETA and self.distill_loss != "jensen_shannon":
            raise ValueError(
                f"distill_jsd_beta={self.distill_jsd_beta} only applies to distill_loss: jensen_shannon, "
                f"got distill_loss={self.distill_loss!r}. Set distill_loss: jensen_shannon, or drop "
                f"distill_jsd_beta."
            )
        require_positive_int(owner, **present(distill_topk=self.distill_topk))
        if self.distill_topk is not None and self.distill_loss not in TOPK_DISTILL_LOSSES:
            raise ValueError(
                f"distill_topk applies only to distill_loss in {TOPK_DISTILL_LOSSES}, got "
                f"distill_loss={self.distill_loss!r}. Drop distill_topk, or pick one of those losses."
            )
