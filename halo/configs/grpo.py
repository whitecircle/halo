"""Field bundles the GRPO argument and config dataclasses mix in; a subclass re-declares one to change its default."""

from dataclasses import dataclass, field, fields, make_dataclass
from typing import ClassVar, Literal, get_args

from halo.config_validation import ValidatedConfig, differs_from_default, present, require_finite, require_int
from halo.models.fp8_rollout import RolloutWeightPrecision

_RLRRMode = Literal["hrr", "prr"]
ScoreCenteringWeights = Literal["none", "tis", "mis"]


def rlrr_arg_name(config_field: str) -> str:
    """``rlrr_<field>``, except λ, whose config field is ``lam`` because ``lambda`` is a keyword."""
    return "rlrr_lambda" if config_field == "lam" else f"rlrr_{config_field}"


@dataclass
class PromptDatasetArguments(ValidatedConfig):
    """Prompt column and length cap of the GRPO-family trainers. No ``system_prompt``: environmental GRPO takes the
    environment's, so trainers that template their own declare it."""

    max_prompt_length: int | None = field(
        default=None,
        metadata={
            "help": "Prompt budget in tokens, applied as a dataset filter: a row whose rendered prompt exceeds it "
            "is dropped, never truncated. Unset = no filtering. "
            "See agent-docs/reference/configuration-reference.md#the-tokenizers-window."
        },
    )
    prompt_field: str = field(
        default="prompt", metadata={"help": "Dataset column holding the prompt, a string or a conversation list."}
    )

    def _validate_fields(self) -> None:
        super()._validate_fields()
        # A budget below one token drops every row.
        require_int(type(self).__name__, ge=1, **present(max_prompt_length=self.max_prompt_length))


@dataclass
class RLRRConfig:
    """RLRR shaping hyperparameters (defaults from the paper's Appendix A.1), from which :class:`RLRRArguments` derives
    its ``rlrr_*`` fields, help included. Messages name the YAML spelling, as the script arguments build one."""

    mode: _RLRRMode = field(
        default="hrr",
        metadata={"help": "RLRR mode: 'hrr' (hybrid rank correction, Eq. 3) or 'prr' (pure relative, Eq. 4)."},
    )
    tau: float = field(
        default=0.1, metadata={"help": "HRR rank-correction magnitude τ (Eq. 3); too high dilutes the rule reward."}
    )
    lam: float = field(
        default=2048.0,
        metadata={
            "help": "Length-bin granularity λ for re-ranking (Eq. 6): correct responses bucket by floor(len / λ)."
        },
    )
    xi_pos: float = field(default=1e-3, metadata={"help": "Advantage cap ξ⁺ for incorrect responses (Eq. 5 clip)."})
    xi_neg: float = field(default=-1e-3, metadata={"help": "Advantage floor ξ⁻ for correct responses (Eq. 5 clip)."})
    std_normalize: bool = field(
        default=False,
        metadata={"help": "Divide the centered advantage by the group std (Eq. 1); false keeps F_norm = 1 (Dr.GRPO)."},
    )
    length_rerank: bool = field(
        default=True, metadata={"help": "Apply the length-bin tie-break in hierarchical re-ranking (Eq. 6)."}
    )
    correctness_clip: bool = field(
        default=True,
        metadata={"help": "Correctness-aware advantage clipping (Eq. 5). Disable for pure PRR with no gold labels."},
    )
    correctness_threshold: float = field(
        default=0.5,
        metadata={"help": "A response is correct iff its raw reward >= this threshold, the only correctness signal."},
    )

    def __post_init__(self) -> None:
        if self.mode not in get_args(_RLRRMode):
            raise ValueError(f"{rlrr_arg_name('mode')} must be one of {get_args(_RLRRMode)}, got {self.mode!r}")
        # Both divide inside the shaping (Eq. 3 / Eq. 6): zero is a ZeroDivisionError deep in the advantage
        # pass, a negative one inverts the ranking, an infinite λ silently disables the length tie-break.
        require_finite(
            type(self).__name__, gt=0, **{rlrr_arg_name(name): getattr(self, name) for name in ("tau", "lam")}
        )
        # A NaN band or threshold fails silently: NaN clip bounds NaN every advantage, and no reward
        # ever compares >= NaN, so every response reads as incorrect.
        require_finite(
            type(self).__name__,
            **{rlrr_arg_name(name): getattr(self, name) for name in ("xi_pos", "xi_neg", "correctness_threshold")},
        )
        if self.xi_neg > self.xi_pos:
            raise ValueError(
                f"RLRR requires {rlrr_arg_name('xi_neg')} <= {rlrr_arg_name('xi_pos')}, "
                f"got {self.xi_neg} > {self.xi_pos}"
            )


# The ``rlrr_*`` script fields, derived from RLRRConfig so a tunable is declared exactly once.
_RLRRTunables = make_dataclass(
    "_RLRRTunables",
    [(rlrr_arg_name(f.name), f.type, field(default=f.default, metadata=dict(f.metadata))) for f in fields(RLRRConfig)],
    module=__name__,
)


@dataclass
class RLRRArguments(_RLRRTunables, ValidatedConfig):
    """``use_rlrr`` plus the ``rlrr_*`` tunables. Built eagerly at parse time, gate on or off, so a mistyped
    ``rlrr_tau`` cannot survive a run with ``use_rlrr`` false."""

    TUNABLES: ClassVar[tuple[str, ...]] = tuple(f.name for f in fields(_RLRRTunables))

    use_rlrr: bool = field(
        default=False,
        metadata={"help": "Enable RLRR relative-reward advantage shaping in place of group-normalized advantages."},
    )

    def _validate_fields(self) -> None:
        super()._validate_fields()
        self._rlrr_config()

    def _rlrr_config(self) -> RLRRConfig:
        return RLRRConfig(**{f.name: getattr(self, rlrr_arg_name(f.name)) for f in fields(RLRRConfig)})

    def build_rlrr_config(self) -> RLRRConfig | None:
        return self._rlrr_config() if self.use_rlrr else None


@dataclass
class AdvantageShapingArguments(ValidatedConfig):
    """Reward scaling, the degenerate-group drop and token-mass balance of the online and environmental GRPO
    advantages; the environmental config re-declares ``drop_degenerate_groups`` on by default."""

    scale_rewards_std_floor: float = field(
        default=0.0,
        metadata={
            "help": "Floor on the std divisor of scale_rewards 'batch'/'group' (divide by max(std, floor)), so a "
            "degenerate batch cannot amplify its noise into full-scale advantages. 0 = off; about 0.05 suits "
            "sparse-reward tasks. See agent-docs/training-methods/async-grpo/objective.md#advantages."
        },
    )
    drop_degenerate_groups: bool = field(
        default=False,
        metadata={
            "help": "Mask GRPO groups whose completions all scored the same reward out of the loss: their advantage "
            "is 0, yet their tokens dilute the loss normalizer. Logged as sampling/degenerate_group_frac. "
            "See agent-docs/training-methods/async-grpo/objective.md#advantages."
        },
    )
    balance_token_mass: bool = field(
        default=False,
        metadata={
            "help": "Scale down the heavier sign of each generation round's advantages so its token-weighted "
            "advantage mass nets to zero, against the entropy drift a token-sum loss takes on when failures and "
            "solves differ in length. Needs loss_type cispo, dapo or dr_grpo, top_entropy_quantile 1.0 and no "
            "off_policy_mask_threshold. See agent-docs/training-methods/async-grpo/objective.md#token-mass-balance."
        },
    )

    def _validate_fields(self) -> None:
        """A negative or NaN std floor fails silently: ``max(std, floor)`` becomes a no-op or a batch-wide NaN."""
        super()._validate_fields()
        require_finite(
            type(self).__name__, ge=0, why=" (0 = off)", scale_rewards_std_floor=self.scale_rewards_std_floor
        )


@dataclass(frozen=True)
class EarlyStopConfig:
    """What a GRPO early stop checks on logged training steps, and its validation. ``on_skipped_updates`` comes from
    the environmental config, whose trainer has a trust-region breaker that can skip an update."""

    entropy_band: tuple[float, ...] | None = None
    logratio_gap: float | None = None
    on_skipped_updates: bool = False
    patience: int = 3

    def __post_init__(self) -> None:
        band = self.entropy_band
        if band is not None:
            require_finite(type(self).__name__, **{f"early_stop_entropy_band[{i}]": v for i, v in enumerate(band)})
            if not (len(band) == 2 and 0.0 <= band[0] < band[1]):
                raise ValueError(f"early_stop_entropy_band must be [low, high] with 0 <= low < high, got {list(band)}")
        if self.logratio_gap is not None:
            require_finite(type(self).__name__, gt=0, early_stop_logratio_gap=self.logratio_gap)
        require_int(type(self).__name__, ge=1, early_stop_patience=self.patience)
        if not self.active and self.patience != EarlyStopConfig.patience:
            raise ValueError(f"early_stop_patience is {self.patience} but no early-stop condition is set to count it")

    @property
    def active(self) -> bool:
        return self.entropy_band is not None or self.logratio_gap is not None or self.on_skipped_updates


@dataclass
class GRPOEarlyStopArguments(ValidatedConfig):
    """Early stop for a KL-free GRPO run (online and environmental), whose policy drifts slowly before it fails:
    entropy leaves its band and the trainer-vs-sampler log-ratio widens. A condition breaching on
    ``early_stop_patience`` readings in a row ends training without saving or evaluating that step."""

    early_stop_entropy_band: list[float] | None = field(
        default=None,
        metadata={
            "help": "Early stop: end training once TRL's entropy metric stays outside this [low, high] band for "
            "early_stop_patience readings in a row; read the band off a healthy run. Unset = off. "
            "See agent-docs/training-methods/async-grpo/monitoring.md#early-stop."
        },
    )
    early_stop_logratio_gap: float | None = field(
        default=None,
        metadata={
            "help": "Early stop: end training once the trainer-vs-sampler log-prob gap stays above this many nats "
            "per token for early_stop_patience readings in a row; needs the importance-sampling correction, which "
            "logs it. Env GRPO reads the magnitude of sampling/logratio_mean and online GRPO "
            "sampling/sampling_logp_difference/mean, so a threshold does not carry between them. "
            "See agent-docs/training-methods/async-grpo/monitoring.md#early-stop."
        },
    )
    early_stop_patience: int = field(
        default=EarlyStopConfig.patience,
        metadata={
            "help": "Breaching readings in a row an early-stop condition needs (>= 1), a reading being one training "
            "log line. Refused at a non-default value when no early-stop condition is set."
        },
    )

    def _validate_fields(self) -> None:
        super()._validate_fields()
        self.build_early_stop_config()

    def build_early_stop_config(self) -> EarlyStopConfig:
        band = self.early_stop_entropy_band
        return EarlyStopConfig(
            entropy_band=tuple(band) if band is not None else None,
            logratio_gap=self.early_stop_logratio_gap,
            on_skipped_updates=self._stops_on_skipped_updates(),
            patience=self.early_stop_patience,
        )

    def _stops_on_skipped_updates(self) -> bool:
        """Only a config whose trainer has a trust-region breaker overrides this."""
        return False


@dataclass(frozen=True)
class ScoreCenteringConfig:
    """What a run with score centering on reads, and its knobs' defaults: ``top_k`` candidates per generated token, the
    importance weight ``weights`` and its MIS bounds. The TIS clip is the trainer's
    ``vllm_importance_sampling_clip_max``."""

    top_k: int
    weights: ScoreCenteringWeights = "tis"
    mis_low: float = 0.5
    mis_high: float = 5.0


@dataclass
class ScoreCenteringArguments(ValidatedConfig):
    """Score centering (arXiv:2609.20807, App. A) of the environmental GRPO objective; the knobs below
    ``score_centering_top_k`` are refused while it is unset."""

    score_centering_top_k: int | None = field(
        default=None,
        metadata={
            "help": "Candidates the SGLang rollout server returns per generated token for score centering, which "
            "subtracts their score-function mean from each token's gradient, so on a constant reward the drift a "
            "sampler-vs-trainer mismatch causes cancels where the candidates hold nearly all of the sampler's mass "
            "on the mismatched tokens (all of it at the full vocabulary). Unset = off; raise it until "
            "score_centering/sampler_head_mass_mean is near 1. Refused at construction beside the settings it "
            "cannot run with, each named; see agent-docs/training-methods/async-grpo/objective.md#score-centering."
        },
    )
    score_centering_weights: ScoreCenteringWeights = field(
        default=ScoreCenteringConfig.weights,
        metadata={
            "help": "Importance weight f of the score-centered gradient: 'none' (f = 1), 'tis' (truncated at "
            "vllm_importance_sampling_clip_max) or 'mis' (kept inside [score_centering_mis_low, "
            "score_centering_mis_high], else zero). Replaces the token ratio of the IS correction, which feeds the "
            "isr_* masks only. See agent-docs/training-methods/async-grpo/objective.md#score-centering."
        },
    )
    score_centering_mis_low: float = field(
        default=ScoreCenteringConfig.mis_low,
        metadata={"help": "Lower bound of the 'mis' weight on the trainer-to-sampler ratio (0 < low <= high)."},
    )
    score_centering_mis_high: float = field(
        default=ScoreCenteringConfig.mis_high,
        metadata={"help": "Upper bound of the 'mis' weight on the trainer-to-sampler ratio."},
    )

    def _validate_fields(self) -> None:
        super()._validate_fields()
        owner = type(self).__name__
        require_int(owner, ge=1, **present(score_centering_top_k=self.score_centering_top_k))
        if self.score_centering_weights not in get_args(ScoreCenteringWeights):
            raise ValueError(
                f"score_centering_weights must be one of {get_args(ScoreCenteringWeights)}, "
                f"got {self.score_centering_weights!r}"
            )
        require_finite(
            owner,
            gt=0,
            score_centering_mis_low=self.score_centering_mis_low,
            score_centering_mis_high=self.score_centering_mis_high,
        )
        if self.score_centering_mis_low > self.score_centering_mis_high:
            raise ValueError(
                f"score_centering_mis_low ({self.score_centering_mis_low}) must not exceed "
                f"score_centering_mis_high ({self.score_centering_mis_high})"
            )
        bounds_set = differs_from_default(self, "score_centering_mis_low") or differs_from_default(
            self, "score_centering_mis_high"
        )
        if self.score_centering_top_k is None and (
            bounds_set or differs_from_default(self, "score_centering_weights")
        ):
            raise ValueError(
                f"{owner}: score_centering_weights/mis_low/mis_high are set, but score_centering_top_k is unset, "
                "so score centering is off and they would do nothing."
            )
        if bounds_set and self.score_centering_weights != "mis":
            raise ValueError(
                f"{owner}: score_centering_mis_low/high bound the 'mis' weight, but score_centering_weights is "
                f"{self.score_centering_weights!r}."
            )

    def build_score_centering_config(self) -> ScoreCenteringConfig | None:
        if self.score_centering_top_k is None:
            return None
        return ScoreCenteringConfig(
            top_k=self.score_centering_top_k,
            weights=self.score_centering_weights,
            mis_low=self.score_centering_mis_low,
            mis_high=self.score_centering_mis_high,
        )


@dataclass
class ChunkedLogprobsArguments:
    """Vocab-chunked log-prob computation switch shared by the GRPO trainers (online, environmental, offline)."""

    use_chunked_grpo_logprobs: bool = field(
        default=False,
        metadata={
            "help": "Compute per-token log-probs from the backbone hidden state with a vocab-chunked softmax (at "
            "least fp32) instead of full [B, T, vocab] logits, bounding the loss-forward peak by the chunk size. "
            "For large-vocab models on long completions, where the full logits OOM. "
            "See agent-docs/training-methods/async-grpo/performance.md#chunked-log-probs."
        },
    )


@dataclass
class PackedMicrobatchArguments(ValidatedConfig):
    """The packed micro-batch budget of the on-policy GRPO trainers (online, environmental)."""

    packed_microbatch_tokens: int | None = field(
        default=None,
        metadata={
            "help": "Pack each training round's rows into micro-batches of up to this many attended tokens per rank, "
            "one flattened varlen backbone call each, in place of one call per row. The round's rows are planned "
            "into bins (a row over the budget keeps a bin of its own), every rank runs the world's bin count in the "
            "last slots of the window and no-ops before them, and row-mean losses are rescaled so the objective is "
            "the per-row one. Null keeps one backbone call per row. Needs use_chunked_grpo_logprobs, "
            "per_device_train_batch_size 1, a varlen flash backend and a family whose rows isolate; refused at "
            "construction otherwise. See "
            "agent-docs/training-methods/async-grpo/performance.md#packed-micro-batches."
        },
    )

    def _validate_fields(self) -> None:
        super()._validate_fields()
        # A one-token bin leaves no next-token target.
        require_int(type(self).__name__, ge=2, **present(packed_microbatch_tokens=self.packed_microbatch_tokens))


@dataclass
class RolloutWeightPrecisionArguments:
    """The precision the on-policy GRPO trainers (online, environmental) store the rollout servers' weights in."""

    rollout_weight_precision: RolloutWeightPrecision = field(
        default="bf16",
        metadata={
            "help": "Precision of the weights the rollout servers hold: bf16 pushes the trainer's weights as they are, "
            "fp8_block quantizes the attention and MLP projections on the sync GPU to 128x128 block FP8 "
            "(power-of-two scales) and pushes those. The trainer stays bf16, and the server must serve a checkpoint "
            "written by the block-FP8 export tool with the triton MoE backend (vLLM --moe-backend triton, SGLang "
            "--moe-runner-backend triton). Admitted only for a family whose spec validates it, refused at startup "
            "with the reason otherwise; SGLang needs SGLANG_ENABLE_JIT_DEEPGEMM=0; refused beside adapter sync and "
            "QLoRA. See agent-docs/infrastructure/rollout-servers.md#fp8-rollouts."
        },
    )


@dataclass
class RankBalanceArguments:
    """The rank-balance switch of the online GRPO trainer."""

    balance_rank_tokens: bool = field(
        default=False,
        metadata={
            "help": "Deal each training round's rows across ranks so every rank trains the same row count with a "
            "similar attended-token total, in place of the contiguous slice the rollout hands each one; a round the "
            "deal would not lighten at its heaviest rank is left as it is. Refused under TP/ETP, with a loss_type "
            "outside grpo, sapo, dr_grpo, luspo, dapo, cispo, top_entropy_quantile < 1, misaligned accumulation, or "
            "one rank; a round carrying image tensors raises. See "
            "agent-docs/training-methods/async-grpo/performance.md#rank-balance."
        },
    )


@dataclass
class PhaseTimerArguments:
    """The phase-timer flag of the on-policy GRPO trainers (online, environmental)."""

    log_phase_times: bool = field(
        default=False,
        metadata={
            "help": "Log where each optimizer step's wall time goes as phase_time/<phase>_s (rank mean) and "
            "phase_time/<phase>_max_s (slowest rank), with the step's time no phase holds as unaccounted. "
            "Costs a few CUDA events per micro-step and one all-reduce per log. "
            "Phases: agent-docs/training-methods/async-grpo/monitoring.md#phase-times."
        },
    )
