from dataclasses import dataclass, field
from typing import ClassVar, Literal

from sentence_transformers import SentenceTransformerTrainingArguments

from halo.config_validation import ValidatedConfig, present, reject_non_default_args, require_finite, require_int
from halo.configs.script_arguments import CommonScriptArguments

# The losses whose constructor takes a ``scale``; a test pins this to the trainer's registry.
SCALED_LOSSES = frozenset({"mnrl", "cached_mnrl", "cosent", "angle"})


@dataclass
class EmbeddingScriptArguments(CommonScriptArguments):
    PROJECT_NAME: ClassVar[str] = "embedding"


@dataclass
class EmbeddingConfig(ValidatedConfig, SentenceTransformerTrainingArguments):
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
            "help": "Loss function type. 'mnrl': MultipleNegativesRankingLoss, "
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
            "help": "Scale factor (inverse temperature) of MNRL, CachedMNRL, CoSENT and AnglE; refused off its "
            "default under any other loss_type."
        },
    )
    cached_mnrl_mini_batch_size: int = field(
        default=32,
        metadata={
            "help": "Rows per gradient-cached sub-forward of loss_type='cached_mnrl': lower it to fit a larger "
            "effective batch, raise it for throughput. The loss value is unaffected. Refused off its default under "
            "any other loss_type."
        },
    )
    matryoshka_dimensions: list[int] | None = field(
        default=None,
        metadata={
            "help": "Truncation dimensions, like [256, 128, 64, 32]; when set, the base loss is wrapped in "
            "MatryoshkaLoss."
        },
    )
    matryoshka_weights: list[float] | None = field(
        default=None,
        metadata={
            "help": "Per-dimension Matryoshka loss weights, one per matryoshka_dimensions entry, which it requires. "
            "Unset = uniform weights."
        },
    )
    pooling_mode: Literal["mean", "cls", "max", "lasttoken", "weightedmean", "mean_sqrt_len_tokens"] = field(
        default="mean",
        metadata={
            "help": "Pooling strategy for generating fixed-size embeddings from token outputs. "
            "Applied on both load paths: each overrides a sentence-transformers checkpoint's own Pooling setting "
            "with it, and the parallel loader (EP, TP, or the grouped-GEMM EP layers at expert_parallel_size 1) "
            "builds the Pooling module from it over a bare backbone."
        },
    )
    normalize_embeddings: bool = field(default=True, metadata={"help": "Whether to L2-normalize output embeddings."})
    max_length: int | None = field(
        default=512,
        metadata={
            "help": "Maximum tokenized sequence length, installed as SentenceTransformers' max_seq_length over the "
            "checkpoint's own; longer texts are truncated. Unset or non-positive = the backbone's context window, "
            "so keep it small unless you embed long documents."
        },
    )
    disable_dropout: bool = field(
        default=False, metadata={"help": "Whether to disable dropout in the model during training."}
    )

    def _validate_fields(self) -> None:
        super()._validate_fields()
        owner = type(self).__name__
        require_finite(owner, gt=0, loss_scale=self.loss_scale)
        require_int(owner, ge=1, cached_mnrl_mini_batch_size=self.cached_mnrl_mini_batch_size)
        # A non-positive length is legal: it resolves to the backbone's context window.
        require_int(owner, **present(max_length=self.max_length))
        context = f"Embedding training under loss_type={self.loss_type}"
        if self.loss_type not in SCALED_LOSSES:
            reject_non_default_args(context, self, "loss_scale")
        if self.loss_type != "cached_mnrl":
            reject_non_default_args(context, self, "cached_mnrl_mini_batch_size")
        if self.matryoshka_dimensions is None and self.matryoshka_weights is not None:
            raise ValueError(
                f"matryoshka_weights={self.matryoshka_weights} is set without matryoshka_dimensions: "
                f"the weights only reach MatryoshkaLoss, which is built from the dimensions, so the "
                f"run would train the plain loss with the weights silently dropped. Set "
                f"matryoshka_dimensions, or remove matryoshka_weights."
            )
        for weight in self.matryoshka_weights or ():
            require_finite(owner, ge=0, matryoshka_weights=weight)
        if self.matryoshka_dimensions is not None:
            if len(self.matryoshka_dimensions) == 0:
                raise ValueError("matryoshka_dimensions must be non-empty if provided")
            for dimension in self.matryoshka_dimensions:
                require_int(owner, ge=1, matryoshka_dimensions=dimension)
            if self.matryoshka_weights is not None and len(self.matryoshka_weights) != len(self.matryoshka_dimensions):
                raise ValueError(
                    f"matryoshka_weights length ({len(self.matryoshka_weights)}) must match "
                    f"matryoshka_dimensions length ({len(self.matryoshka_dimensions)})"
                )
