"""Field bundles the GRPO argument and config dataclasses mix in; a subclass re-declares one to change its default."""

from dataclasses import dataclass, field, fields, make_dataclass
from typing import ClassVar, Literal, get_args

from halo.config_validation import ValidatedConfig, present, require_finite, require_int

_RLRRMode = Literal["hrr", "prr"]


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
