"""``OfflineGRPOConfig`` — training config for offline GRPO."""

from dataclasses import dataclass, field
from typing import Literal

from transformers import TrainingArguments

from src.args.mixins import ChunkedLogprobsArguments
from src.args.validation import RangeValidatedConfig


@dataclass
class OfflineGRPOConfig(ChunkedLogprobsArguments, RangeValidatedConfig, TrainingArguments):
    r"""Training arguments for [`OfflineGRPOTrainer`]; per-field docs are in each field's ``help`` metadata."""

    max_prompt_length: int | None = field(
        default=512,
        metadata={
            "help": "Prompt budget in tokens; a longer prompt is LEFT-truncated (the tokens nearest "
            "the completion are kept). None = no cap. Required under pipeline parallelism."
        },
    )
    max_completion_length: int | None = field(
        default=None,
        metadata={
            "help": "Completion budget in tokens; a longer stored completion is truncated from the end. "
            "None (default) = no cap. Also the loss normalizer constant for loss_type='dr_grpo', and "
            "required under pipeline parallelism — both reject None rather than normalize by a "
            "default nobody chose."
        },
    )
    max_length: int | None = field(
        default=None,
        metadata={
            "help": "Fixed total (prompt + completion) sequence length under pipeline parallelism; every "
            "batch is padded to it because the pipeline's P2P buffer shapes freeze on the first step. "
            "None defaults to max_prompt_length + max_completion_length. PP-ONLY: nothing reads it "
            "off PP (the budget there is max_prompt_length + max_completion_length), so setting it "
            "without pipeline_parallel_size > 1 is rejected at trainer construction rather than "
            "silently ignored."
        },
    )

    kl_beta: float = field(
        default=0.0,
        metadata={"help": "Coefficient of the KL penalty to the reference policy. 0 (default) = no KL term."},
    )

    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model."},
    )
    padding_value: int | None = field(
        default=None,
        metadata={"help": "Pad token id used by the collator. None takes the processing class's pad token."},
    )

    model_init_kwargs: dict | None = field(
        default=None,
        metadata={
            "help": "Model-config overrides on every entry-script path: written onto the loaded "
            "config's fields before the load, raising on a key that config does not declare and "
            "on dtype/torch_dtype. Model-loading kwargs only where a trainer is constructed "
            "programmatically with the model as a path string."
        },
    )
    advantage_method: Literal["z_norm", "minmax", "quantile_norm", "quantile_uniform", "robust"] = field(
        default="quantile_norm",
        metadata={
            "help": "How rewards map to advantages: 'z_norm' (z-score), 'minmax' ([-1,1]), "
            "'quantile_norm' (default), 'quantile_uniform' ([-1,1]), 'robust' (median/IQR, "
            "outlier-resistant)."
        },
    )
    best_completion_emphasis: float | str = field(
        default=0.0,
        metadata={
            "help": "Boost the best completion(s) per group. 0.0 (default) = none; a float > 1.0 = "
            "fixed boost (1.5 = +50%); 'auto' = std-adaptive 3.0 + 2.0*std/(1+std) (3.0 at std=0 → "
            "5.0 as std→∞). Applies to all advantage methods. Values in (0.0, 1.0] are rejected: the "
            "consumer only applies a factor > 1.0, so they would silently do nothing."
        },
    )
    min_log_prob: float | None = field(
        default=-3.0,
        metadata={
            "help": "Clip floor for completion tokens with negative advantages (-3.0 ≈ 5% probability); "
            "prevents instability from extremely low-probability tokens. None = no clipping."
        },
    )
    initial_min_log_prob: float | None = field(
        default=None,
        metadata={
            "help": "When set, linearly schedules min_log_prob from this start value to its final value "
            "over training (None = constant). Recommend ~the mean CE loss on the SFT data."
        },
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    drop_degenerate_groups: bool = field(
        default=False,
        metadata={
            "help": "Drop groups whose completions all scored EXACTLY the same reward (and groups with "
            "fewer than two completions) at tokenization. Every advantage method maps such a group to "
            "all-zero advantages, so its rows carry no gradient yet still dilute the loss normalizer "
            "and spend forward compute. Near-ties are kept — the rank-based advantage methods train "
            "them with full-scale advantages. Applies to train and eval datasets alike."
        },
    )
    loss_type: Literal["grpo", "bnpo", "dr_grpo"] = field(
        default="bnpo",
        metadata={
            "help": "Loss normalization, all group-weighted: 'grpo' (per-sequence average then weighted "
            "group mean, sequence-focused), 'bnpo' (default; global weighted token average, "
            "token-focused — token-level like online 'dapo' but not equivalent to it, no importance ratios), "
            "'dr_grpo' (normalized by effective batch size and max completion length — length-bias-free)."
        },
    )
    policy_gradient_formulation: Literal["prob_weighted", "reinforce"] = field(
        default="prob_weighted",
        metadata={
            "help": "Per-token policy gradient: 'prob_weighted' (default) L=-(π·A), ∇L=-A·π·∇log π — "
            "high-probability tokens get larger gradients, conservative, NOT online GRPO's importance "
            "ratios; 'reinforce' L=-(log π·A), ∇L=-A·∇log π — unscaled, more aggressive, less stable "
            "offline."
        },
    )

    def __post_init__(self):
        self._validate_ranges()
        super().__post_init__()

    def _validate_ranges(self) -> None:
        """The ``float | str`` union on ``best_completion_emphasis`` bypasses the parser's Literal
        gate, so a misspelled sentinel would otherwise fail only inside ``datasets.map`` on
        ``float(...)``."""
        super()._validate_ranges()
        emphasis = self.best_completion_emphasis
        if isinstance(emphasis, str):
            if emphasis != "auto":
                raise ValueError(
                    f"best_completion_emphasis must be 'auto' or a number, got {emphasis!r}; "
                    f"'auto' is the only string sentinel (std-adaptive boost)."
                )
        elif emphasis != 0.0 and not emphasis > 1.0:
            raise ValueError(
                f"best_completion_emphasis must be 0.0 (off), a value > 1.0 (a boost), or 'auto', got "
                f"{emphasis}. The consumer applies the factor only when it exceeds 1.0, so (0.0, 1.0] "
                f"and negative values are silent no-ops."
            )
