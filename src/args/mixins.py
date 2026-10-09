"""Field bundles shared by the script-argument and trainer-config dataclasses.

Pure dataclass mixins (no ``CommonScriptArguments`` base): each holds fields that several classes
would otherwise declare identically, so a spelling or help fix lands once. A subclass re-declares a
field only to change its default (e.g. ``DistillScriptArguments``' ``conversation_field="messages"``,
``SFTScriptArguments``' ``generate_eval_examples=False``).
"""

import string
from dataclasses import dataclass, field, fields, make_dataclass
from typing import Any, ClassVar, Literal, get_args

from src.args.validation import (
    RangeValidatedConfig,
    present,
    require_finite,
    require_positive,
    require_positive_int,
)

# The RLRR shaping modes: the annotation gates YAML/CLI and RLRRConfig validates against it.
RLRRMode = Literal["hrr", "prr"]

# The script-argument spelling of each RLRRConfig field is ``rlrr_<field>``, except λ: ``lambda`` is a
# keyword, so the config field is ``lam`` while the YAML keeps the full word.
_RLRR_ARG_SPELLINGS = {"lam": "rlrr_lambda"}

# The OPD arms' names in the trainer-side divergence registry (``losses.DIVERGENCES``), pinned to it by a
# test: the args layer imports no trainer. The annotation gates YAML/CLI and SDPGArguments validates
# against it.
SelfDistillationLoss = Literal["reverse_kl", "forward_kl", "unnormalized_kl"]

# The dataset column a ground-truth answer is read from unless a config renames it.
DEFAULT_ANSWER_FIELD = "answer"

# The teacher hint both OPD flows default to, through SDPGArguments.
PRIVILEGED_HINT_TEMPLATE = "\n[Hint] The correct answer is: {answer}. Do NOT state that you were given the answer.\n"


def rlrr_arg_name(config_field: str) -> str:
    """The YAML/CLI spelling of one :class:`RLRRConfig` field."""
    return _RLRR_ARG_SPELLINGS.get(config_field, f"rlrr_{config_field}")


def format_field_names(template: str) -> set[str]:
    """The replacement-field names ``str.format`` looks up in ``template``, nested format specs
    included. A lone brace raises ``ValueError``, as ``str.format`` would."""
    names: set[str] = set()
    for _, name, spec, _ in string.Formatter().parse(template):
        if name is not None:
            names |= {name, *format_field_names(spec or "")}
    return names


@dataclass
class ConversationRenderArguments:
    """Chat-template rendering knobs shared by conversation-rendering trainers (SFT, distillation)."""

    conversation_field: str | None = field(
        default="prompt",
        metadata={"help": "Field in dataset with conversations (in list of dicts format)"},
    )
    images_field: str | None = field(
        default=None,
        metadata={
            "help": "VLM only: dataset column holding the row's image(s) (HF Image feature, single or list). "
            "Injected into the first user turn ahead of its text, so hub datasets that store images "
            "outside the conversation (e.g. FineVision/the_cauldron/Docmatix) train without preprocessing."
        },
    )
    system_prompt: str | None = field(
        default=None,
        metadata={"help": "Will use system prompt if there is no one in dialogue, set to None to disable"},
    )
    train_on_completions_only: bool = field(default=True, metadata={"help": "Do train only on completions or not"})
    assistant_message_template: str | None = field(
        default=None,
        metadata={
            "help": "The rendered assistant-turn prefix of the model's chat template (e.g. '<|im_start|>assistant\\n'); "
            "required when train_on_completions_only is on — no default fits every template"
        },
    )
    model_supports_system_role: bool = field(
        default=True,
        metadata={
            "help": "Flag that indicates if model have support for system prompt. If not, will use user for setting system prompt"
        },
    )
    interleaved_thinking: bool = field(
        default=False,
        metadata={
            "help": "Pass clear_thinking=False to tokenizer.apply_chat_template so historical "
            "assistant reasoning is preserved. Only meaningful for chat templates with a "
            "clear_thinking switch — the GLM family among supported models; a no-op for every "
            "other template. Text-only (the VLM path rejects it)."
        },
    )


@dataclass
class GenerationEvalArguments(RangeValidatedConfig):
    """Eval-time example-generation knobs (SFT, DPO, SMPO, offline GRPO)."""

    generate_eval_examples: bool = field(default=True, metadata={"help": "Do generate examples on eval"})
    num_eval_examples: int = field(default=50, metadata={"help": "Number of examples to generate on eval phase"})

    def _validate_ranges(self) -> None:
        super()._validate_ranges()
        require_positive_int(type(self).__name__, num_eval_examples=self.num_eval_examples)


@dataclass
class PromptDatasetArguments(RangeValidatedConfig):
    """Prompt-dataset shape shared by the GRPO-family trainers (prompt column + length cap).

    No ``system_prompt``: environmental GRPO builds the rollout conversation from the environment's
    system prompt, so a shared field here would be ignored on that surface. Trainers that template
    their own system turn declare it themselves.
    """

    max_prompt_length: int | None = field(
        default=None,
        metadata={
            "help": "Prompt budget in tokens, applied as a dataset FILTER: rows whose rendered prompt "
            "exceeds it are dropped, never truncated (a truncated prompt loses the question the "
            "verifier grades against). None (default) = no filtering. On the online (RLVR) arm, when "
            "BOTH this and the generation budget are set their sum also becomes the tokenizer's "
            "model_max_length for the run, and leaving either unset leaves the tokenizer's own value "
            "alone; environmental GRPO pins the model's context window instead, the limit its rollout "
            "server enforces."
        },
    )
    prompt_field: str = field(
        default="prompt",
        metadata={"help": "Field in the dataset containing the prompt (string or conversation list)"},
    )

    def _validate_ranges(self) -> None:
        super()._validate_ranges()
        # A budget below one token drops every row.
        require_positive_int(type(self).__name__, **present(max_prompt_length=self.max_prompt_length))


@dataclass
class RLRRConfig:
    """Hyperparameters for RLRR relative-reward shaping (defaults from the paper's Appendix A.1).

    The single declaration of the tunables: :class:`RLRRArguments` derives its ``rlrr_*`` script
    fields from these (same type, default and help), so the ``help`` metadata here is the CLI help.
    Every invariant lives in ``__post_init__``; the script args build the config eagerly at parse
    time, so a bad value fails before any model is loaded. Messages name the YAML spelling.
    """

    mode: RLRRMode = field(
        default="hrr",
        metadata={"help": "RLRR mode: 'hrr' (hybrid rank correction, Eq. 3) or 'prr' (pure relative, Eq. 4)"},
    )
    tau: float = field(
        default=0.1, metadata={"help": "HRR rank-correction magnitude τ (Eq. 3); too high dilutes the rule reward"}
    )
    lam: float = field(
        default=2048.0,
        metadata={
            "help": "Length-bin granularity λ for re-ranking (Eq. 6); correct responses bucketed by floor(len / λ)"
        },
    )
    xi_pos: float = field(default=1e-3, metadata={"help": "Advantage cap ξ⁺ for incorrect responses (Eq. 5 clip)"})
    xi_neg: float = field(default=-1e-3, metadata={"help": "Advantage floor ξ⁻ for correct responses (Eq. 5 clip)"})
    std_normalize: bool = field(
        default=False,
        metadata={"help": "Divide the centered advantage by the group std (Eq. 1); else F_norm = 1 (Dr.GRPO)"},
    )
    length_rerank: bool = field(
        default=True, metadata={"help": "Apply the length-bin tie-break in hierarchical re-ranking (Eq. 6)"}
    )
    correctness_clip: bool = field(
        default=True,
        metadata={"help": "Correctness-aware advantage clipping (Eq. 5). Disable for pure PRR with no gold labels."},
    )
    correctness_threshold: float = field(
        default=0.5,
        metadata={"help": "A response is correct iff raw reward >= this threshold — the only correctness signal"},
    )

    def __post_init__(self) -> None:
        if self.mode not in get_args(RLRRMode):
            raise ValueError(f"{rlrr_arg_name('mode')} must be one of {get_args(RLRRMode)}, got {self.mode!r}")
        # Both divide inside the shaping (Eq. 3 / Eq. 6): zero is a ZeroDivisionError deep in the advantage
        # pass, a negative one inverts the ranking, an infinite λ silently disables the length tie-break.
        require_positive(type(self).__name__, **{rlrr_arg_name(name): getattr(self, name) for name in ("tau", "lam")})
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
class RLRRArguments(_RLRRTunables, RangeValidatedConfig):
    """``use_rlrr`` plus the ``rlrr_*`` tunables of :class:`RLRRConfig` under their YAML spellings.

    Every knob is validated at parse time by building the config eagerly, gate on or off (fail-loud:
    a mistyped ``rlrr_tau`` must not survive a run just because ``use_rlrr`` is false).
    """

    TUNABLES: ClassVar[tuple[str, ...]] = tuple(f.name for f in fields(_RLRRTunables))

    use_rlrr: bool = field(
        default=False,
        metadata={"help": "Enable RLRR relative-reward advantage shaping (replaces group-normalized advantages)"},
    )

    def _validate_ranges(self) -> None:
        super()._validate_ranges()
        self._rlrr_config()

    def _rlrr_config(self) -> RLRRConfig:
        return RLRRConfig(**{f.name: getattr(self, rlrr_arg_name(f.name)) for f in fields(RLRRConfig)})

    def build_rlrr_config(self) -> RLRRConfig | None:
        """Return the :class:`RLRRConfig` these args describe, or ``None`` when RLRR is disabled."""
        return self._rlrr_config() if self.use_rlrr else None


@dataclass
class AdvantageShapingArguments(RangeValidatedConfig):
    """Reward scaling, the degenerate-group drop and the token-mass balance of the GRPO group-relative
    advantages.

    Shared by the online (RLVR) and environmental GRPO configs, which feed the same
    ``group_relative_advantages`` normalizer.
    ``drop_degenerate_groups`` defaults differ per trainer (opt-in online, on by default for
    env-GRPO's sparse verifiable rewards), so that one is re-declared on the env config.
    """

    scale_rewards_std_floor: float = field(
        default=0.0,
        metadata={
            "help": "Floor on the std divisor in advantage scaling (scale_rewards 'batch'/'group'): "
            "divide by max(std, floor). A degenerate batch/group (every reward within a few "
            "hundredths) otherwise divides its own noise by a near-zero std, amplifying it to "
            "full-scale advantages — the lock-in mechanism of a collapsed run. Healthy batches (std "
            "well above the floor) are unaffected. 0 (default) = off; ~0.05 on sparse-reward tasks."
        },
    )
    drop_degenerate_groups: bool = field(
        default=False,
        metadata={
            "help": "Mask GRPO groups whose completions ALL scored the same reward out of the loss. "
            "Their advantage is already 0 (no policy gradient), but their tokens still inflate the "
            "loss normalizer and dilute the groups that do carry signal (the cheap half of DAPO's "
            "dynamic sampling: drop, without resampling replacements). Logged as "
            "`sampling/degenerate_group_frac`."
        },
    )
    balance_token_mass: bool = field(
        default=False,
        metadata={
            "help": "Scale down the heavier sign of each generation round's advantages so the round's "
            "token-weighted advantage mass nets to zero. Under a token-sum loss a completion pulls with its "
            "advantage times its trained tokens; where failures run longer than solves the round pushes "
            "down the tokens the policy sampled and entropy climbs, where solves run longer it sharpens "
            "the policy. Async GRPO leaves the rows of untrainable turns, which train on a negative advantage "
            "only, out of the balance and unscaled. Needs `loss_type` `cispo`, `dapo` or `dr_grpo`, "
            "`top_entropy_quantile` 1.0 and no `off_policy_mask_threshold`, refused otherwise. The pre-balance "
            "share is logged as "
            "`advantage/net_token_mass` either way (its sign reads as the entropy push only under a "
            "token-sum loss) and the applied factor as `advantage/token_mass_scale`. Default off."
        },
    )

    def _validate_ranges(self) -> None:
        """Refuse a negative or NaN std floor, which fails silently: ``max(std, floor)`` becomes a
        no-op or a NaN that propagates to every advantage in the batch."""
        super()._validate_ranges()
        require_finite(type(self).__name__, scale_rewards_std_floor=self.scale_rewards_std_floor)
        if self.scale_rewards_std_floor < 0:
            raise ValueError(
                f"scale_rewards_std_floor must be a finite value >= 0 (0 = off), got {self.scale_rewards_std_floor}"
            )


@dataclass(frozen=True)
class EarlyStopConfig:
    """What a GRPO early stop checks on the logged training steps; the single home of its validation.

    Built by :class:`GRPOEarlyStopArguments`; ``on_skipped_updates`` comes from the environmental config,
    the one whose trainer has a trust-region breaker that can skip an update."""

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
            require_positive(type(self).__name__, early_stop_logratio_gap=self.logratio_gap)
        require_positive_int(type(self).__name__, early_stop_patience=self.patience)
        if not self.active and self.patience != EarlyStopConfig.patience:
            raise ValueError(f"early_stop_patience is {self.patience} but no early-stop condition is set to count it")

    @property
    def active(self) -> bool:
        return self.entropy_band is not None or self.logratio_gap is not None or self.on_skipped_updates


@dataclass
class GRPOEarlyStopArguments(RangeValidatedConfig):
    """Early stop for a KL-free GRPO run, shared by the online and environmental configs.

    A policy without a KL anchor drifts slowly before it fails: entropy leaves its band and the
    trainer-vs-sampler log-ratio widens with it. A condition ends training once it breaches on
    ``early_stop_patience`` readings in a row, without saving or evaluating that step; the run keeps the
    periodic checkpoints it took before.
    """

    early_stop_entropy_band: list[float] | None = field(
        default=None,
        metadata={
            "help": "Early stop: end training once TRL's `entropy` metric stays outside this [low, high] band "
            "for `early_stop_patience` readings in a row. The metric counts a micro-batch whose loss tokens "
            "were all dropped (degenerate groups, invalid episodes) as 0, so under drops it reads below the "
            "policy's entropy. A KL-free run drifts in either direction, toward collapse below the band or "
            "explosion above it, and both start slowly enough to stop on. The band is model-specific: read "
            "it off a healthy run. None (default) = off."
        },
    )
    early_stop_logratio_gap: float | None = field(
        default=None,
        metadata={
            "help": "Early stop: end training once the trainer's log-prob gap metric stays above this many "
            "nats per token for `early_stop_patience` readings in a row (one per generation round: a step "
            "reusing its round carries none and leaves the count). Env GRPO reads the magnitude "
            "of its signed mean `sampling/logratio_mean`, online GRPO TRL's mean absolute difference "
            "`sampling/sampling_logp_difference/mean`; the two read differently, so a threshold does not "
            "carry from one trainer to the other. The gap grows with entropy, and faster than it once the "
            "policy flattens, so it is the earlier signal of the two. Needs the vLLM importance-sampling "
            "correction, which logs it. None (default) = off."
        },
    )
    early_stop_patience: int = field(
        default=EarlyStopConfig.patience,
        metadata={
            "help": "Breaching readings in a row an early-stop condition needs before training ends (>= 1). "
            "A reading is one training log line, so under `logging_steps` > 1 it is the window's mean. "
            "Refused at a non-default value when no early-stop condition is set."
        },
    )

    def _validate_ranges(self) -> None:
        super()._validate_ranges()
        self.build_early_stop()

    def build_early_stop(self) -> EarlyStopConfig:
        band = self.early_stop_entropy_band
        return EarlyStopConfig(
            entropy_band=tuple(band) if band is not None else None,
            logratio_gap=self.early_stop_logratio_gap,
            on_skipped_updates=self._stops_on_skipped_updates(),
            patience=self.early_stop_patience,
        )

    def _stops_on_skipped_updates(self) -> bool:
        """Whether a step whose every update was skipped counts toward the stop; only a config whose trainer
        has a trust-region breaker overrides it."""
        return False


@dataclass
class DatasetNumProcArguments(RangeValidatedConfig):
    """Dataset-preprocessing worker count of the trainer configs that map their own data, read through
    :func:`~src.data.pipeline.processing.resolve_map_num_proc`."""

    dataset_num_proc: int | None = field(
        default=None,
        metadata={
            "help": "Worker processes for dataset preprocessing (map/filter). Unset means the toolkit "
            "default (HALO_DATASET_NUM_PROC, else max(1, min(cpu_count // 4, 4))), not one worker."
        },
    )

    def _validate_ranges(self) -> None:
        super()._validate_ranges()
        require_positive_int(type(self).__name__, **present(dataset_num_proc=self.dataset_num_proc))


@dataclass
class ModelInitKwargsArguments:
    """``model_init_kwargs`` of the trainer configs whose trainer also takes the model as a path string."""

    model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={
            "help": "Model-config overrides on every entry-script path: written onto the loaded "
            "config's fields before the load, raising on a key that config does not declare and "
            "on dtype/torch_dtype. Model-loading kwargs only where a trainer is constructed "
            "programmatically with the model as a path string."
        },
    )


@dataclass
class ChunkedLogprobsArguments:
    """Vocab-chunked log-prob computation switch shared by the GRPO trainers (online, environmental, offline)."""

    use_chunked_grpo_logprobs: bool = field(
        default=False,
        metadata={
            "help": "Compute per-token log-probs from the backbone hidden state + a vocab-chunked "
            "softmax instead of full [B, T, vocab] logits — bounds the loss-forward peak by the chunk "
            "size, not B*T*vocab. For large-vocab models (gpt-oss ~201k) on long completions where the "
            "full-logits allocation OOMs. Each chunk's log-softmax runs in at least fp32."
        },
    )


@dataclass
class SDPGArguments(RangeValidatedConfig):
    """Privileged-teacher OPD term (SDPG, arXiv:2606.04036), shared by the offline self-distillation
    SFT script and the online RLVR GRPO script.

    The hint field the teacher fills is not declared here: self-distillation reads it from a
    configurable dataset column, while RLVR's ``process_for_rlvr`` has already normalized it to
    ``answer``. Validated on construction, so the trainers that build this block from their kwargs
    hold a direct construction to the bounds a YAML run meets.
    """

    # The placeholders the arm's hint formatter fills: the on-policy trainer has the gold answer alone;
    # SelfDistillationArguments widens the set with its reference-solution column.
    HINT_PLACEHOLDERS: ClassVar[frozenset[str]] = frozenset({"answer"})

    sdpg_hint_template: str = field(
        default=PRIVILEGED_HINT_TEMPLATE,
        metadata={
            "help": "Hint the TEACHER forward sees after its prompt: appended to the last user turn on "
            "the self-distillation arm, to the rendered generation prompt on the online arm. A "
            "str.format template: {answer} on both arms, {solution} on self-distillation only "
            "(privileged_solution_field). Any other placeholder is refused at parse time."
        },
    )
    sdpg_loss: SelfDistillationLoss = field(
        default="reverse_kl",
        metadata={"help": "OPD loss: 'reverse_kl' (SDPG), 'forward_kl', or 'unnormalized_kl' (k3/UKL)."},
    )
    sdpg_temperature: float = field(
        default=1.0,
        metadata={"help": "Softmax temperature for the OPD loss (finite, > 0)."},
    )
    sdpg_beta_base: float = field(
        default=1.0,
        metadata={"help": "Base distillation coefficient beta_base (finite, >= 0; 0 drops the OPD term)."},
    )
    sdpg_beta_warmup_steps: int = field(
        default=0,
        metadata={"help": "Steps to ramp beta from 0 to sdpg_beta_base (SDPG warmup; 0 = off)."},
    )
    sdpg_beta_decay_steps: int = field(
        default=0,
        metadata={"help": "Final steps over which beta decays back to 0 (SDPG decay; 0 = off)."},
    )

    def __post_init__(self) -> None:
        self._validate_ranges()

    @classmethod
    def pop_from(cls, kwargs: dict, *, exclude: frozenset[str] = frozenset()) -> dict[str, Any]:
        """Pop this block's fields but ``exclude`` from trainer ``kwargs``: validated, defaults filled.

        A trainer adopts the result as attributes, so a directly built one runs the OPD schedule the
        identical YAML run would. A misspelt field stays in ``kwargs``, where the parent's explicit
        signature rejects it.
        """
        names = [f.name for f in fields(cls) if f.name not in exclude]
        block = cls(**{name: kwargs.pop(name) for name in names if name in kwargs})
        return {name: getattr(block, name) for name in names}

    def _validate_ranges(self) -> None:
        super()._validate_ranges()
        if self.sdpg_loss not in get_args(SelfDistillationLoss):
            raise ValueError(f"sdpg_loss must be one of {get_args(SelfDistillationLoss)}, got {self.sdpg_loss!r}")
        # Divides both distributions' logits: zero turns them infinite and NaNs the OPD loss without a
        # raise, a negative one inverts them.
        require_positive(type(self).__name__, sdpg_temperature=self.sdpg_temperature)
        # A NaN coefficient NaNs every loss; a negative one trains the student away from the teacher.
        require_finite(type(self).__name__, sdpg_beta_base=self.sdpg_beta_base)
        if self.sdpg_beta_base < 0:
            raise ValueError(
                f"sdpg_beta_base must be a finite value >= 0 (0 drops the OPD term), got {self.sdpg_beta_base}"
            )
        # The schedule reads a negative length as off, so one is a typo the run would not report.
        for name in ("sdpg_beta_warmup_steps", "sdpg_beta_decay_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an int >= 0 (0 = off), got {value!r}")
        self._validate_hint_template()

    def _validate_hint_template(self) -> None:
        """Refuse a placeholder the arm's formatter does not fill: ``str.format`` would raise it at the
        first teacher prompt, after the model load (and on the online arm, the rollout server)."""
        if not isinstance(self.sdpg_hint_template, str):
            raise ValueError(
                f"sdpg_hint_template must be a str.format template string, got {self.sdpg_hint_template!r}"
            )
        try:
            names = format_field_names(self.sdpg_hint_template)
        except ValueError as e:
            raise ValueError(f"sdpg_hint_template is not a valid str.format template: {e}") from e
        unknown = sorted(names - self.HINT_PLACEHOLDERS)
        if unknown:
            raise ValueError(
                f"sdpg_hint_template names {[f'{{{name}}}' for name in unknown]}, which "
                f"{type(self).__name__}'s hint does not fill; it fills "
                f"{[f'{{{name}}}' for name in sorted(self.HINT_PLACEHOLDERS)]}."
            )
