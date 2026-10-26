"""Sequence-classification script arguments and ``ClassificationConfig``."""

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from transformers import TrainingArguments

from src.config_validation import RangeValidatedConfig, present, require_finite, require_int
from src.configs.script_arguments import CommonScriptArguments, DatasetNumProcArguments


@dataclass
class ClassificationScriptArguments(CommonScriptArguments):
    PROJECT_NAME: ClassVar[str] = "classification"

    text_field: str | None = field(
        default=None,
        metadata={
            "help": "Raw-text column to classify (wrapped as a single user turn) when the dataset has "
            "no pre-built 'prompt' conversation — e.g. text/label datasets like imdb."
        },
    )


@dataclass
class ClassificationConfig(DatasetNumProcArguments, RangeValidatedConfig, TrainingArguments):
    r"""TrainingArguments for [`ClassificationTrainer`]; per-field docs are in each field's ``help`` metadata."""

    max_length: int | None = field(
        default=1024,
        metadata={
            "help": "Maximum tokenized sequence length: the training script truncates to it, while a raw text/label "
            "dataset handed to the trainer is filtered to it. Unset or non-positive = the model's context window."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model."},
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
            "help": "Loss function: 'cross_entropy' (standard CE), 'focal' (focal loss for class imbalance) or "
            "'label_smoothing_ce' (CE with label smoothing)."
        },
    )
    focal_gamma: float = field(
        default=2.0,
        metadata={
            "help": "Focusing parameter (gamma) of the focal loss: higher values down-weight easy examples more. "
            "Ignored unless loss_type='focal'."
        },
    )
    focal_alpha: float | None = field(
        default=None,
        metadata={
            "help": "Alpha (balancing factor) of the focal loss on a multi-label (sigmoid) head; unset = no alpha "
            "weighting. Rejected on a single-label head, where a scalar alpha only rescales the loss, and ignored "
            "unless loss_type='focal'."
        },
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={
            "help": "Label-smoothing epsilon of label_smoothing_ce, typically 0.05-0.1. Ignored unless "
            "loss_type='label_smoothing_ce'."
        },
    )
    class_weights: list[float] | None = field(
        default=None,
        metadata={
            "help": "Per-class loss weights ordered by label ID, applied to every loss_type and as pos_weight on "
            "the multi-label BCE. Unset = equal weights."
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
            "help": "Log per-class precision, recall and F1 during evaluation, many metrics with many classes. "
            "Single-label only; ignored on a multi-label dataset."
        },
    )
    compute_auc_roc: bool = field(
        default=False,
        metadata={"help": "Compute AUC-ROC during evaluation. For multiclass uses one-vs-rest macro average."},
    )
    compute_mcc: bool = field(
        default=True,
        metadata={
            "help": "Compute the Matthews correlation coefficient during evaluation, a balanced metric under class "
            "imbalance. Single-label only; ignored on a multi-label dataset."
        },
    )

    def __post_init__(self):
        self._validate_ranges()
        super().__post_init__()

    def _validate_ranges(self) -> None:
        super()._validate_ranges()
        owner = type(self).__name__
        require_finite(owner, focal_gamma=self.focal_gamma, label_smoothing=self.label_smoothing)
        if self.focal_gamma < 0:
            raise ValueError(f"focal_gamma must be >= 0, got {self.focal_gamma}")
        # alpha_t = alpha*y + (1-alpha)*(1-y): outside [0, 1] one class's loss is negated.
        if self.focal_alpha is not None:
            require_finite(owner, focal_alpha=self.focal_alpha)
            if not 0.0 <= self.focal_alpha <= 1.0:
                raise ValueError(f"focal_alpha must be in [0, 1], got {self.focal_alpha}")
        # A non-positive length is legal: it resolves to the model's context window at launch.
        require_int(owner, **present(max_length=self.max_length))
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
