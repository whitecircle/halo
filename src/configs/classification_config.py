"""``ClassificationConfig`` — training config for sequence classification."""

from dataclasses import dataclass, field
from typing import Literal

from transformers import TrainingArguments

from src.args.validation import RangeValidatedConfig


@dataclass
class ClassificationConfig(RangeValidatedConfig, TrainingArguments):
    r"""TrainingArguments for [`ClassificationTrainer`]; per-field docs are in each field's ``help`` metadata."""

    max_length: int | None = field(
        default=1024,
        metadata={
            "help": "Maximum tokenized sequence length. The training script truncates to it; a raw "
            "text/label dataset passed straight to the trainer is tokenized untruncated and filtered "
            "to it instead. null / non-positive resolves to the model's context window at launch."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model."},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={
            "help": "Whether to remove the columns that are not used by the model's forward pass. Can be `True` only "
            "if the dataset is pretokenized."
        },
    )

    loss_type: Literal["cross_entropy", "focal", "label_smoothing_ce"] = field(
        default="cross_entropy",
        metadata={
            "help": "Loss function type. 'cross_entropy': standard CE (default), "
            "'focal': focal loss for class imbalance, "
            "'label_smoothing_ce': CE with label smoothing."
        },
    )
    focal_gamma: float = field(
        default=2.0,
        metadata={
            "help": "Gamma (focusing parameter) for focal loss. Higher values down-weight easy examples more. "
            "Ignored unless loss_type='focal'."
        },
    )
    focal_alpha: float | None = field(
        default=None,
        metadata={
            "help": "Alpha (balancing factor) for focal loss. None means no alpha weighting. "
            "Ignored unless loss_type='focal'."
        },
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={
            "help": "Label smoothing epsilon for label_smoothing_ce loss. "
            "Typical values: 0.05-0.1. Ignored unless loss_type='label_smoothing_ce'."
        },
    )
    class_weights: list[float] | None = field(
        default=None,
        metadata={
            "help": "Manual class weights as a list of floats (one per class, ordered by label ID). "
            "Applied to cross_entropy and focal losses. None means equal weights."
        },
    )
    derive_class_weights: bool = field(
        default=False,
        metadata={
            "help": "Automatically compute class weights inversely proportional to class frequency "
            "in the training set (sklearn 'balanced' strategy). Mutually exclusive with class_weights."
        },
    )

    multi_label_threshold: float = field(
        default=0.5,
        metadata={
            "help": "Sigmoid threshold for multi-label classification predictions. "
            "Predictions with sigmoid(logit) >= threshold are considered positive."
        },
    )
    compute_per_class_metrics: bool = field(
        default=False,
        metadata={
            "help": "Log per-class precision, recall, and F1 during evaluation. "
            "Can produce many metrics with many classes."
        },
    )
    compute_auc_roc: bool = field(
        default=False,
        metadata={"help": "Compute AUC-ROC during evaluation. For multiclass uses one-vs-rest macro average."},
    )
    compute_mcc: bool = field(
        default=True,
        metadata={
            "help": "Compute Matthews Correlation Coefficient during evaluation. "
            "MCC is a balanced metric useful even with imbalanced classes."
        },
    )

    def __post_init__(self):
        self._validate_ranges()
        super().__post_init__()

    def _validate_ranges(self) -> None:
        super()._validate_ranges()
        if self.focal_gamma < 0:
            raise ValueError(f"focal_gamma must be >= 0, got {self.focal_gamma}")
        if not (0.0 <= self.label_smoothing < 1.0):
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        if not (0.0 < self.multi_label_threshold < 1.0):
            raise ValueError(f"multi_label_threshold must be in (0, 1), got {self.multi_label_threshold}")
        if self.class_weights is not None and self.derive_class_weights:
            raise ValueError(
                "class_weights and derive_class_weights are mutually exclusive: class_weights sets the "
                "per-class weights explicitly, derive_class_weights derives them from the training-set "
                "frequencies. Set exactly one."
            )
