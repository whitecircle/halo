"""``EmbeddingConfig`` — training config for embedding fine-tuning (sentence-transformers losses)."""

from dataclasses import dataclass, field
from typing import Literal

from sentence_transformers import SentenceTransformerTrainingArguments

from src.args.validation import RangeValidatedConfig


@dataclass
class EmbeddingConfig(RangeValidatedConfig, SentenceTransformerTrainingArguments):
    r"""SentenceTransformerTrainingArguments for [`EmbeddingTrainer`] (loss, pooling,
    normalization); per-field docs are in each field's ``help`` metadata."""

    loss_type: Literal[
        "mnrl",
        "cached_mnrl",
        "cosent",
        "angle",
        "cosine_similarity",
        "triplet",
        "contrastive",
        "online_contrastive",
        "batch_all_triplet",
        "batch_hard_triplet",
    ] = field(
        default="mnrl",
        metadata={
            "help": "Loss function type. 'mnrl': MultipleNegativesRankingLoss (default), "
            "'cached_mnrl': CachedMultipleNegativesRankingLoss, "
            "'cosent': CoSENTLoss, 'angle': AnglELoss, "
            "'cosine_similarity': CosineSimilarityLoss, "
            "'triplet': TripletLoss, 'contrastive': ContrastiveLoss, "
            "'online_contrastive': OnlineContrastiveLoss, "
            "'batch_all_triplet': BatchAllTripletLoss, "
            "'batch_hard_triplet': BatchHardTripletLoss."
        },
    )

    loss_scale: float = field(
        default=20.0,
        metadata={
            "help": "Scale factor (inverse temperature) for loss functions that use it. "
            "Used by MNRL, CachedMNRL, CoSENT, and AnglE."
        },
    )

    cached_mnrl_mini_batch_size: int = field(
        default=32,
        metadata={
            "help": "Rows per gradient-cached sub-forward for loss_type='cached_mnrl'. The whole "
            "point of that loss is decoupling the in-batch negative set from activation memory: "
            "lower it to fit a larger effective batch, raise it for throughput. The loss value is "
            "unaffected."
        },
    )

    matryoshka_dimensions: list[int] | None = field(
        default=None,
        metadata={
            "help": "Truncation dimensions for Matryoshka loss wrapper. "
            "E.g., [256, 128, 64, 32]. If set, wraps the base loss with MatryoshkaLoss."
        },
    )

    matryoshka_weights: list[float] | None = field(
        default=None,
        metadata={
            "help": "Per-dimension weights for Matryoshka loss. "
            "Must match matryoshka_dimensions length (and requires it — the weights are read only "
            "by the MatryoshkaLoss wrapper, which is built from the dimensions). "
            "None means uniform weights."
        },
    )

    pooling_mode: Literal["mean", "cls", "max", "lasttoken", "weightedmean", "mean_sqrt_len_tokens"] = field(
        default="mean",
        metadata={
            "help": "Pooling strategy for generating fixed-size embeddings from token outputs. "
            "Applied on both load paths: the EP/TP branch builds the Pooling module from it, and "
            "the standard path overrides the checkpoint's own modules.json setting with it."
        },
    )

    normalize_embeddings: bool = field(
        default=True,
        metadata={"help": "Whether to L2-normalize output embeddings."},
    )

    max_length: int | None = field(
        default=512,
        metadata={
            "help": "Maximum tokenized sequence length; texts above it are truncated. Installed as "
            "SentenceTransformers' max_seq_length on both loading paths (it overrides whatever the "
            "checkpoint's sentence_bert_config.json ships). null / non-positive resolves to the "
            "backbone's context window — embeddings pool the whole sequence, so keep this small "
            "unless you really embed long documents."
        },
    )

    disable_dropout: bool = field(
        default=False,
        metadata={"help": "Whether to disable dropout in the model during training."},
    )

    def __post_init__(self):
        self._validate_ranges()
        super().__post_init__()

    def _validate_ranges(self) -> None:
        super()._validate_ranges()
        if self.loss_scale <= 0:
            raise ValueError(f"loss_scale must be > 0, got {self.loss_scale}")
        if self.cached_mnrl_mini_batch_size <= 0:
            raise ValueError(f"cached_mnrl_mini_batch_size must be > 0, got {self.cached_mnrl_mini_batch_size}")
        if self.matryoshka_dimensions is None and self.matryoshka_weights is not None:
            raise ValueError(
                f"matryoshka_weights={self.matryoshka_weights} is set without matryoshka_dimensions: "
                f"the weights only reach MatryoshkaLoss, which is built from the dimensions, so the "
                f"run would train the plain loss with the weights silently dropped. Set "
                f"matryoshka_dimensions, or remove matryoshka_weights."
            )
        if self.matryoshka_dimensions is not None:
            if len(self.matryoshka_dimensions) == 0:
                raise ValueError("matryoshka_dimensions must be non-empty if provided")
            if self.matryoshka_weights is not None and len(self.matryoshka_weights) != len(self.matryoshka_dimensions):
                raise ValueError(
                    f"matryoshka_weights length ({len(self.matryoshka_weights)}) must match "
                    f"matryoshka_dimensions length ({len(self.matryoshka_dimensions)})"
                )
