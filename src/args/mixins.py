"""Field bundles shared by the script-argument and trainer-config dataclasses.

Pure dataclass mixins (no ``CommonScriptArguments`` base): each bundles fields several classes
declared identically, so a spelling or help fix lands once. A subclass re-declares a field only to
change its default (e.g. ``DistillScriptArguments``' ``conversation_field="messages"``,
``SFTScriptArguments``' ``generate_eval_examples=False``).
"""

import math
from dataclasses import dataclass, field
from typing import Literal, get_args

from src.args.validation import RangeValidatedConfig

# One roster for the advantage-shaping modes: the annotation gates YAML/CLI, ``get_args`` gives
# :class:`AdvantageShaping` the same tuple to validate against, so the two cannot drift apart.
AdvantageMode = Literal["mean", "qae", "asymmetric", "neg_mask_hard"]

# The RLRR shaping modes, same contract: the annotation gates YAML/CLI and RLRRConfig validates
# against it.
RLRRMode = Literal["hrr", "prr"]

# Shared by :class:`SDPGArguments` and the SDPG trainer so both OPD flows steer the teacher alike.
PRIVILEGED_HINT_TEMPLATE = "\n[Hint] The correct answer is: {answer}. Do NOT state that you were given the answer.\n"


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
class GenerationEvalArguments:
    """Eval-time example-generation knobs (SFT, DPO, SMPO, offline GRPO)."""

    generate_eval_examples: bool = field(default=True, metadata={"help": "Do generate examples on eval"})
    num_eval_examples: int = field(default=50, metadata={"help": "Number of examples to generate on eval phase"})


@dataclass
class PromptDatasetArguments:
    """Prompt-dataset shape shared by the GRPO-family trainers (prompt column + length cap).

    No ``system_prompt``: environmental GRPO builds the rollout conversation from the ENVIRONMENT's
    system prompt, so a shared field here could only be set and silently ignored on that surface.
    The one trainer that templates its own system turn declares it itself.
    """

    max_prompt_length: int | None = field(
        default=None,
        metadata={
            "help": "Prompt budget in tokens, applied as a dataset FILTER: rows whose rendered prompt "
            "exceeds it are dropped, never truncated (a truncated prompt loses the question the "
            "verifier grades against). None (default) = no filtering. When BOTH this and the "
            "generation budget are set, their sum also becomes the tokenizer's model_max_length for "
            "the run; leaving either unset leaves the tokenizer's own value alone."
        },
    )
    prompt_field: str = field(
        default="prompt",
        metadata={"help": "Field in the dataset containing the prompt (string or conversation list)"},
    )


@dataclass(frozen=True)
class AdvantageShaping:
    """Optional advantage-channel surgery, applied by
    :func:`~src.trainers.grpo.objective.advantages.group_relative_advantages`.

    ``mode``:

    * ``"mean"`` — plain group-mean baseline (default; bit-identical to no shaping).
    * ``"qae"`` — Quantile Advantage Estimation: baseline is the per-group ``quantile``, not the mean.
    * ``"asymmetric"`` — mean baseline, then scale positive advantages by ``pos_scale`` and negative
      by ``neg_scale`` (``neg_scale=0`` is the full negative mask).
    * ``"neg_mask_hard"`` — mean baseline, then zero NEGATIVE advantages only in HARD groups (no
      member's ``gate_rewards`` reached ``hard_group_threshold``).

    Declared beside :class:`AdvantageShapingArguments`, which builds it: the GRPO objective imports
    it from here, so the args layer never has to import a trainer to construct its own return value.
    """

    mode: str = "mean"
    quantile: float = 0.4
    pos_scale: float = 1.0
    neg_scale: float = 0.4
    hard_group_threshold: float = 0.5

    _MODES = get_args(AdvantageMode)

    def __post_init__(self):
        if self.mode not in self._MODES:
            raise ValueError(f"advantage mode must be one of {self._MODES}, got {self.mode!r}")
        if not 0.0 < self.quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {self.quantile}")
        if self.pos_scale < 0 or self.neg_scale < 0:
            raise ValueError(f"pos_scale/neg_scale must be >= 0, got {self.pos_scale}/{self.neg_scale}")


@dataclass
class RLRRConfig:
    """Hyperparameters for RLRR relative-reward shaping (defaults from the paper's Appendix A.1)."""

    mode: RLRRMode = "hrr"
    tau: float = 0.1
    """HRR rank-correction magnitude ``τ`` (Eq. 3); too high dilutes the rule reward."""
    lam: float = 2048.0
    """Length-bin granularity ``λ`` (Eq. 6); correct responses bucketed by ``floor(len / λ)``."""
    xi_pos: float = 1e-3
    """Advantage cap ``ξ⁺`` for incorrect responses (Eq. 5)."""
    xi_neg: float = -1e-3
    """Advantage floor ``ξ⁻`` for correct responses (Eq. 5)."""
    std_normalize: bool = False
    """Divide the centered advantage by the group std (Eq. 1); else ``F_norm = 1`` (Dr.GRPO)."""
    length_rerank: bool = True
    """Apply the length-bin tie-break in hierarchical re-ranking (Eq. 6)."""
    correctness_clip: bool = True
    """Apply correctness-aware advantage clipping (Eq. 5); disable for pure PRR."""
    correctness_threshold: float = 0.5
    """Without explicit labels, a response is correct iff raw reward ``>= correctness_threshold``."""

    def __post_init__(self) -> None:
        if self.mode not in get_args(RLRRMode):
            raise ValueError(f"RLRR mode must be one of {get_args(RLRRMode)}, got {self.mode!r}")
        if self.lam <= 0:
            raise ValueError(f"RLRR lam (λ) must be > 0, got {self.lam}")
        if self.xi_neg > self.xi_pos:
            raise ValueError(f"RLRR requires xi_neg <= xi_pos, got xi_neg={self.xi_neg}, xi_pos={self.xi_pos}")


@dataclass
class AdvantageShapingArguments(RangeValidatedConfig):
    """Group-baseline / negative-side advantage surgery + reward-scaling guards.

    Shared by the online (RLVR) and environmental GRPO configs, which feed the same
    :class:`AdvantageShaping` and the same
    ``group_relative_advantages`` normalizer. ``drop_degenerate_groups`` defaults differ per trainer
    (opt-in online, on by default for env-GRPO's sparse verifiable rewards), so that one is
    re-declared on the env config.
    """

    advantage_mode: AdvantageMode = field(
        default="mean",
        metadata={
            "help": "Group-baseline / negative-side advantage surgery (AdvantageShaping, applied by "
            "src/trainers/grpo/objective/advantages.py). 'mean' (default) = plain "
            "GRPO group-mean baseline. 'qae' = per-group quantile baseline (QAE): on "
            "failure-dominated groups failures get ~0 advantage and only rare successes train — the "
            "entropy-safest constant baseline per regime. 'asymmetric' = mean baseline, then scale "
            "positive/negative advantages by advantage_pos_scale/advantage_neg_scale (continuous "
            "failure-cone attenuation). 'neg_mask_hard' = zero negative advantages only in groups "
            "where no member's objective reward reached advantage_hard_group_threshold. Targets entropy "
            "explosion in failure-dominated batches — the substitute that lets a strong KL anchor "
            "be retired."
        },
    )
    advantage_quantile: float = field(
        default=0.4,
        metadata={"help": "QAE baseline quantile K for advantage_mode='qae' (paper default 0.4)."},
    )
    advantage_pos_scale: float = field(
        default=1.0,
        metadata={"help": "Positive-advantage multiplier for advantage_mode='asymmetric'."},
    )
    advantage_neg_scale: float = field(
        default=0.4,
        metadata={
            "help": "Negative-advantage multiplier for advantage_mode='asymmetric' (0 = full negative "
            "mask; production analogues discard or heavily down-weight negatives)."
        },
    )
    advantage_hard_group_threshold: float = field(
        default=0.5,
        metadata={
            "help": "advantage_mode='neg_mask_hard': a group is HARD (negatives zeroed) when no "
            "member's objective reward component reaches this value. Set it to the objective value "
            "that counts as a solve for your reward scale."
        },
    )
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

    def _validate_ranges(self) -> None:
        """Guard the two knobs :class:`AdvantageShaping` does not check itself.

        Both fail silently rather than loudly: every comparison against a NaN threshold is False, so
        no group member ever "reaches" it and EVERY group is treated as HARD (all negative advantages
        zeroed); a negative or NaN std floor turns ``max(std, floor)`` into a no-op or a NaN that
        propagates to every advantage in the batch.
        """
        super()._validate_ranges()
        if not math.isfinite(self.advantage_hard_group_threshold):
            raise ValueError(
                f"advantage_hard_group_threshold must be finite, got {self.advantage_hard_group_threshold}"
            )
        if not math.isfinite(self.scale_rewards_std_floor) or self.scale_rewards_std_floor < 0:
            raise ValueError(
                f"scale_rewards_std_floor must be a finite value >= 0 (0 = off), got {self.scale_rewards_std_floor}"
            )

    def build_advantage_shaping(self) -> AdvantageShaping | None:
        """Return an :class:`AdvantageShaping` from these fields, or ``None`` at the default 'mean' mode.

        Built eagerly so the numeric knobs are validated even when the mode discards them (fail-loud:
        a mistyped ``advantage_quantile`` must not survive a run just because the mode is 'mean').
        """
        shaping = AdvantageShaping(
            mode=self.advantage_mode,
            quantile=self.advantage_quantile,
            pos_scale=self.advantage_pos_scale,
            neg_scale=self.advantage_neg_scale,
            hard_group_threshold=self.advantage_hard_group_threshold,
        )
        return shaping if shaping.mode != "mean" else None


@dataclass
class ChunkedLogprobsArguments:
    """Vocab-chunked log-prob computation switch shared by the GRPO trainers (online, environmental, offline)."""

    use_chunked_grpo_logprobs: bool = field(
        default=False,
        metadata={
            "help": "Compute per-token log-probs from the backbone hidden state + a vocab-chunked "
            "softmax instead of full [B, T, vocab] logits — bounds the loss-forward peak by the chunk "
            "size, not B*T*vocab. For large-vocab models (gpt-oss ~201k) on long completions where the "
            "full-logits allocation OOMs. Log-probs match the full path to bf16 tolerance."
        },
    )


@dataclass
class SDPGArguments:
    """Privileged-teacher OPD term (SDPG, arXiv:2606.04036), shared by the two arms that run it: the
    offline self-distillation SFT script and the online RLVR GRPO script.

    The hint field the teacher fills is NOT here — self-distillation reads it from a configurable
    dataset column, while RLVR's ``process_for_rlvr`` has already normalized it to ``answer``.
    """

    sdpg_hint_template: str = field(
        default=PRIVILEGED_HINT_TEMPLATE,
        metadata={
            "help": "Template appended to the last user turn for the TEACHER forward only. Supports "
            "an {answer} placeholder, and — on the self-distillation arm, which reads a reference "
            "solution column — a {solution} placeholder."
        },
    )
    sdpg_loss: Literal["reverse_kl", "forward_kl", "unnormalized_kl"] = field(
        default="reverse_kl",
        metadata={"help": "OPD loss: 'reverse_kl' (SDPG), 'forward_kl', or 'unnormalized_kl' (k3/UKL)."},
    )
    sdpg_temperature: float = field(
        default=1.0,
        metadata={"help": "Softmax temperature for the OPD loss."},
    )
    sdpg_beta_base: float = field(
        default=1.0,
        metadata={"help": "Base distillation coefficient beta_base."},
    )
    sdpg_beta_warmup_steps: int = field(
        default=0,
        metadata={"help": "Steps to ramp beta from 0 to sdpg_beta_base (SDPG warmup)."},
    )
    sdpg_beta_decay_steps: int = field(
        default=0,
        metadata={"help": "Final steps over which beta decays back to 0 (SDPG decay)."},
    )
