"""Offline GRPO script arguments and ``OfflineGRPOConfig``."""

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from transformers import TrainingArguments

from src.config_validation import ValidatedConfig, present, require_finite, require_int
from src.configs.grpo import ChunkedLogprobsArguments
from src.configs.script_arguments import (
    CommonScriptArguments,
    DatasetNumProcArguments,
    GenerationEvalArguments,
    ModelInitKwargsArguments,
)


@dataclass
class OfflineGRPOScriptArguments(GenerationEvalArguments, CommonScriptArguments):
    PROJECT_NAME: ClassVar[str] = "offline-grpo-tuning"


@dataclass
class OfflineGRPOConfig(
    ChunkedLogprobsArguments, DatasetNumProcArguments, ModelInitKwargsArguments, ValidatedConfig, TrainingArguments
):
    r"""Training arguments for [`OfflineGRPOTrainer`]; per-field docs are in each field's ``help`` metadata."""

    max_prompt_length: int | None = field(
        default=512,
        metadata={
            "help": "Prompt budget in tokens; a longer prompt is left-truncated, keeping the tokens nearest the "
            "completion. Unset = no cap, which pipeline parallelism refuses."
        },
    )
    max_completion_length: int | None = field(
        default=None,
        metadata={
            "help": "Completion budget in tokens; a longer stored completion is truncated from the end. Unset = no "
            "cap, which loss_type='dr_grpo' (its normalizer) and pipeline parallelism refuse."
        },
    )
    max_length: int | None = field(
        default=None,
        metadata={
            "help": "Fixed total row length under pipeline parallelism, every batch padded to it; unset = "
            "max_prompt_length + max_completion_length. Rejected at trainer construction without pipeline "
            "parallelism. See agent-docs/reference/configuration-reference.md#offlinegrpoconfig."
        },
    )
    kl_beta: float = field(
        default=0.0, metadata={"help": "Coefficient of the KL penalty to the reference policy; 0 = no KL term."}
    )
    disable_dropout: bool = field(default=True, metadata={"help": "Whether to disable dropout in the model."})
    padding_value: int | None = field(
        default=None,
        metadata={"help": "Pad token id used by the collator; unset takes the processing class's pad token."},
    )
    advantage_method: Literal["z_norm", "minmax", "quantile_norm", "quantile_uniform", "robust"] = field(
        default="quantile_norm",
        metadata={
            "help": "How rewards map to advantages: 'z_norm' (z-score), 'minmax' ([-1, 1]), 'quantile_norm' "
            "(inverse normal CDF of ranks), 'quantile_uniform' ([-1, 1]) or 'robust' (median/IQR, outlier-resistant)."
        },
    )
    best_completion_emphasis: float | str = field(
        default=0.0,
        metadata={
            "help": "Boost of the best completion(s) per group: 0.0 = off, a float > 1.0 a fixed boost (1.5 = "
            "+50%), 'auto' std-adaptive. A value in (0.0, 1.0] is rejected as a silent no-op. "
            "See agent-docs/reference/configuration-reference.md#offlinegrpoconfig."
        },
    )
    min_log_prob: float | None = field(
        default=-3.0,
        metadata={
            "help": "Clip floor for completion tokens with negative advantages (-3.0 ≈ 5% probability); "
            "prevents instability from extremely low-probability tokens. Unset = no clipping."
        },
    )
    initial_min_log_prob: float | None = field(
        default=None,
        metadata={
            "help": "When set, linearly schedules min_log_prob from this start value to its final value over "
            "training; it needs min_log_prob. A start near the mean CE loss on the SFT data works."
        },
    )
    drop_degenerate_groups: bool = field(
        default=False,
        metadata={
            "help": "Drop groups whose completions all scored exactly the same reward, or that hold fewer than two, "
            "at tokenization: their all-zero advantages carry no gradient yet dilute the loss normalizer. Near-ties "
            "are kept, since the rank-based advantage methods train them at full scale."
        },
    )
    loss_type: Literal["grpo", "bnpo", "dr_grpo"] = field(
        default="bnpo",
        metadata={
            "help": "Loss normalization, all group-weighted: 'grpo' (per-sequence average, then weighted group "
            "mean), 'bnpo' (global weighted token average, token-level like online 'dapo' without importance "
            "ratios) or 'dr_grpo' (normalized by effective batch size and max completion length, length-bias-free)."
        },
    )
    policy_gradient_formulation: Literal["prob_weighted", "reinforce"] = field(
        default="prob_weighted",
        metadata={
            "help": "Per-token policy gradient: 'prob_weighted' (L = -π·A, conservative, with no importance ratio) "
            "or 'reinforce' (L = -log π·A, unscaled, more aggressive and less stable offline)."
        },
    )

    def _validate_fields(self) -> None:
        super()._validate_fields()
        owner = type(self).__name__
        # The ``float | str`` union bypasses the parser's Literal gate, so a misspelled sentinel would
        # otherwise fail only inside ``datasets.map`` on ``float(...)``.
        emphasis = self.best_completion_emphasis
        if isinstance(emphasis, str):
            if emphasis != "auto":
                raise ValueError(
                    f"best_completion_emphasis must be 'auto' or a number, got {emphasis!r}; "
                    f"'auto' is the only string sentinel (std-adaptive boost)."
                )
        else:
            require_finite(owner, best_completion_emphasis=emphasis)
            if emphasis != 0.0 and not emphasis > 1.0:
                raise ValueError(
                    f"best_completion_emphasis must be 0.0 (off), a value > 1.0 (a boost), or 'auto', got "
                    f"{emphasis}. The consumer applies the factor only when it exceeds 1.0, so (0.0, 1.0] "
                    f"and negative values are silent no-ops."
                )
        require_finite(owner, ge=0, why=" (0 = no KL term)", kl_beta=self.kl_beta)
        require_finite(
            owner, **present(min_log_prob=self.min_log_prob, initial_min_log_prob=self.initial_min_log_prob)
        )
        require_int(
            owner,
            ge=1,
            **present(
                max_prompt_length=self.max_prompt_length,
                max_completion_length=self.max_completion_length,
                max_length=self.max_length,
            ),
        )
        # A token id indexes the embedding table.
        require_int(owner, ge=0, **present(padding_value=self.padding_value))
