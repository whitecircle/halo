"""Distillation arguments: the privileged-teacher (SDPG) term, the self- and teacher-distillation script
arguments and ``DistillationConfig``."""

from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Literal, get_args

from transformers import TrainingArguments

from halo.config_validation import ValidatedConfig, present, require_finite, require_int
from halo.configs.script_arguments import DEFAULT_ANSWER_FIELD, CommonScriptArguments, DatasetNumProcArguments
from halo.configs.sft import ConversationRenderArguments, SFTScriptArguments
from halo.data.collators.self_distill import format_field_names

# The teacher hint both OPD flows default to, through SDPGArguments.
PRIVILEGED_HINT_TEMPLATE = "\n[Hint] The correct answer is: {answer}. Do NOT state that you were given the answer.\n"

# Keys of the trainer-side divergence registry (``divergences.DIVERGENCES``), which a test pins this to.
SelfDistillationLoss = Literal["reverse_kl", "forward_kl", "unnormalized_kl"]


@dataclass
class SDPGArguments(ValidatedConfig):
    """Privileged-teacher OPD term (SDPG, arXiv:2606.04036), shared by the offline self-distillation
    SFT script and the online RLVR GRPO script.

    The hint field the teacher fills is not declared here: self-distillation reads it from a
    configurable dataset column, while RLVR's ``process_for_rlvr`` has already normalized it to
    ``answer``. Validated on construction, so the trainers that build this block from their kwargs
    hold a direct construction to the bounds a YAML run meets.
    """

    # The placeholders the arm's hint formatter fills: the on-policy trainer has the gold answer alone;
    # SelfDistillScriptArguments widens the set with its reference-solution column.
    HINT_PLACEHOLDERS: ClassVar[frozenset[str]] = frozenset({"answer"})

    sdpg_hint_template: str = field(
        default=PRIVILEGED_HINT_TEMPLATE,
        metadata={
            "help": "Hint the teacher forward sees after its prompt (the last user turn on self-distillation, the "
            "rendered generation prompt online), a str.format template. {answer} fills on both arms, {solution} on "
            "self-distillation only, and any other placeholder is refused at parse time."
        },
    )
    sdpg_loss: SelfDistillationLoss = field(
        default="reverse_kl",
        metadata={"help": "OPD loss: 'reverse_kl' (SDPG), 'forward_kl', or 'unnormalized_kl' (k3/UKL)."},
    )
    sdpg_temperature: float = field(
        default=1.0, metadata={"help": "Softmax temperature for the OPD loss (finite, > 0)."}
    )
    sdpg_beta_base: float = field(
        default=1.0, metadata={"help": "Base distillation coefficient beta_base (finite, >= 0; 0 drops the OPD term)."}
    )
    sdpg_beta_warmup_steps: int = field(
        default=0, metadata={"help": "Steps to ramp beta from 0 to sdpg_beta_base (SDPG warmup; 0 = off)."}
    )
    sdpg_beta_decay_steps: int = field(
        default=0, metadata={"help": "Final steps over which beta decays back to 0 (SDPG decay; 0 = off)."}
    )

    @staticmethod
    def sdpg_field_names(exclude: frozenset[str] = frozenset()) -> tuple[str, ...]:
        """The :class:`SDPGArguments` field names but ``exclude``, never a subclass's own fields."""
        return tuple(f.name for f in fields(SDPGArguments) if f.name not in exclude)

    @classmethod
    def pop_from(cls, kwargs: dict, *, exclude: frozenset[str] = frozenset()) -> dict[str, Any]:
        """Pop this block's fields but ``exclude`` from trainer ``kwargs``: validated, defaults filled.

        A trainer adopts the result as attributes, so a directly built one runs the OPD schedule the
        identical YAML run would. A misspelt field stays in ``kwargs``, where the parent's explicit
        signature rejects it.
        """
        block = cls(**{name: kwargs.pop(name) for name in cls.sdpg_field_names(exclude) if name in kwargs})
        return block.sdpg_kwargs(exclude=exclude)

    def sdpg_kwargs(self, *, exclude: frozenset[str] = frozenset()) -> dict[str, Any]:
        """This block's field values by name, but ``exclude``."""
        return {name: getattr(self, name) for name in self.sdpg_field_names(exclude)}

    def _validate_fields(self) -> None:
        super()._validate_fields()
        if self.sdpg_loss not in get_args(SelfDistillationLoss):
            raise ValueError(f"sdpg_loss must be one of {get_args(SelfDistillationLoss)}, got {self.sdpg_loss!r}")
        # Divides both distributions' logits: zero turns them infinite and NaNs the OPD loss without a
        # raise, a negative one inverts them.
        require_finite(type(self).__name__, gt=0, sdpg_temperature=self.sdpg_temperature)
        # A NaN coefficient NaNs every loss; a negative one trains the student away from the teacher.
        require_finite(type(self).__name__, ge=0, sdpg_beta_base=self.sdpg_beta_base)
        # The schedule reads a negative length as off, so one is a typo the run would not report.
        require_int(
            type(self).__name__,
            ge=0,
            why=" (0 = off)",
            sdpg_beta_warmup_steps=self.sdpg_beta_warmup_steps,
            sdpg_beta_decay_steps=self.sdpg_beta_decay_steps,
        )
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


@dataclass
class SelfDistillScriptArguments(SFTScriptArguments, SDPGArguments):
    """Args for SDPG-style offline privileged-context self-distillation SFT (text or VLM).

    One model is both student (prompt only) and teacher (plus a gold-answer hint). The teacher's
    full-vocab distribution supervises the student on shared response tokens via an OPD loss, atop SFT.
    """

    PROJECT_NAME: ClassVar[str] = "self-distillation"

    # SDPG fields the script applies while building the teacher prompts; every other SDPG field is
    # forwarded to the trainer, which takes the complement.
    DATASET_SIDE_SDPG_FIELDS: ClassVar[frozenset[str]] = frozenset({"sdpg_hint_template"})
    # The field naming the column that fills each slot privileged_hint renders
    # (halo/data/collators/self_distill.py); this arm fills the reference solution too.
    HINT_SLOT_FIELDS: ClassVar[dict[str, str]] = {
        "answer": "sdpg_answer_field",
        "solution": "privileged_solution_field",
    }
    HINT_PLACEHOLDERS: ClassVar[frozenset[str]] = frozenset(HINT_SLOT_FIELDS)

    sdpg_answer_field: str | None = field(
        default=DEFAULT_ANSWER_FIELD,
        metadata={
            "help": "Dataset column holding the ground-truth answer for the hint, required while sdpg_beta_base > 0."
        },
    )
    privileged_solution_field: str | None = field(
        default="solution",
        metadata={"help": "Optional dataset column holding a reference solution, which fills {solution}."},
    )
    reference_kl_coef: float = field(
        default=0.0,
        metadata={
            "help": "Alpha for KL regularization to a frozen reference model (finite, >= 0). 0 disables "
            "(no reference model is loaded)."
        },
    )
    reference_kl_loss: SelfDistillationLoss = field(
        default="unnormalized_kl",
        metadata={"help": "Reference-policy regularizer: 'unnormalized_kl' (k3/UKL), 'reverse_kl', or 'forward_kl'."},
    )
    reference_model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Frozen reference model. Defaults to the student's init weights "
            "when reference_kl_coef > 0 and this is unset."
        },
    )
    confidence_field: str | None = field(
        default=None,
        metadata={
            "help": "Dataset column holding a per-sample confidence in [0, 1]. When set, the "
            "SFT and OPD losses are weighted by confidence**confidence_power, divided by its mean over "
            "the train split, mirroring the soft-weighted (w_conf) SFT arm."
        },
    )
    confidence_power: float = field(
        default=4.0, metadata={"help": "Exponent p in the per-sample weight conf**p (finite, > 0)."}
    )
    confidence_weight_opd: bool = field(
        default=True,
        metadata={
            "help": "Apply the confidence weight to the OPD loss too (SFT analog of "
            "SDPG positive-advantage gating). If False, only the SFT loss is weighted."
        },
    )
    opd_exclude_eos: bool = field(
        default=True,
        metadata={
            "help": "Exclude the EOS/stop token from the OPD term so SFT's hard P(EOS)->1 is "
            "not diluted by the softer teacher (prevents the no-stop / repeat failure mode)."
        },
    )

    def build_sdpg_kwargs(self) -> dict:
        """The SDPG trainer kwargs: every :class:`SDPGArguments` field but the dataset-side ones, the
        complement the trainer pops."""
        return self.sdpg_kwargs(exclude=self.DATASET_SIDE_SDPG_FIELDS)

    def _validate_fields(self) -> None:
        super()._validate_fields()
        # A NaN coefficient NaNs every loss; a negative one pushes the student away from the anchor.
        require_finite(type(self).__name__, ge=0, reference_kl_coef=self.reference_kl_coef)
        # p <= 0 inverts or flattens the weighting, and a zero confidence then divides by zero.
        require_finite(type(self).__name__, gt=0, confidence_power=self.confidence_power)
        unfilled = sorted(
            self.HINT_SLOT_FIELDS[slot]
            for slot in format_field_names(self.sdpg_hint_template)
            if not getattr(self, self.HINT_SLOT_FIELDS[slot])
        )
        if self.sdpg_beta_base > 0 and unfilled:
            raise ValueError(
                f"sdpg_hint_template names a slot whose column is unset ({unfilled}) while sdpg_beta_base > 0: "
                f"no row could render the hint, so every teacher would run on the plain prompt. Set the "
                f"column, or set sdpg_beta_base: 0."
            )
        if self.sdpg_beta_base > 0 and not self.train_on_completions_only:
            raise ValueError(
                "train_on_completions_only: false while sdpg_beta_base > 0: OPD pairs the student's and the "
                "hinted teacher's supervised tokens in order, and with the prompt supervised the hint "
                "shifts every pair after it onto a different token. Supervise the completions only "
                "(train_on_completions_only: true with assistant_message_template), or set sdpg_beta_base: 0."
            )


@dataclass
class TeacherDistillScriptArguments(ConversationRenderArguments, CommonScriptArguments):
    PROJECT_NAME: ClassVar[str] = "llm_distillation"

    teacher_model: str | None = field(default=None, metadata={"help": "Name or path of the teacher model."})
    teacher_model_revision: str | None = field(
        default=None,
        metadata={
            "help": "Hub revision of the teacher; unset = the teacher repo's main. The student's model_revision "
            "does not carry over, since the teacher is usually another repo."
        },
    )
    # Distillation datasets conventionally ship a "messages" column (SFT's default is "prompt").
    conversation_field: str | None = field(
        default="messages", metadata={"help": "Dataset column holding the conversation, a list of role/content dicts."}
    )

    def _validate_fields(self) -> None:
        """Require ``teacher_model``: an empty value reaches the loader as a blank model id and only
        fails there."""
        super()._validate_fields()
        if not self.teacher_model:
            raise ValueError("teacher_model must be specified for distillation")


@dataclass
class DistillationConfig(DatasetNumProcArguments, ValidatedConfig, TrainingArguments):
    """Configuration for distillation training, extending HuggingFace TrainingArguments."""

    distill_loss: Literal[
        "kl_divergence", "mse", "soft_cross_entropy", "cosine_similarity", "jensen_shannon", "slim"
    ] = field(
        default="kl_divergence",
        metadata={
            "help": "Distillation loss: kl_divergence, mse, soft_cross_entropy, cosine_similarity, jensen_shannon "
            "or slim."
        },
    )
    distill_temperature: float = field(
        default=1.0,
        metadata={
            "help": "Softmax temperature for the temperature-based distillation losses "
            "(kl_divergence, soft_cross_entropy, jensen_shannon, slim). Ignored by mse and "
            "cosine_similarity, whose definitions take no temperature."
        },
    )
    distill_alpha: float = field(
        default=1.0,
        metadata={
            "help": "Weight of the distillation term; the CLM term takes 1 - distill_alpha. 1.0 trains "
            "distillation alone (CLM is then logged, not trained)."
        },
    )
    apply_hard_labels: bool = field(
        default=False,
        metadata={
            "help": "Scale the distillation term per token by the detached gold-token gate "
            "(1 - student_prob[label]) * teacher_prob[label]. Refused under slim, which applies its own "
            "gold-token weight."
        },
    )
    max_length: int | None = field(
        default=2048,
        metadata={
            "help": "Maximum tokenized sequence length; an over-length conversation is dropped, not truncated. "
            "Unset or non-positive = the student's context window."
        },
    )

    def _validate_fields(self) -> None:
        super()._validate_fields()
        owner = type(self).__name__
        require_finite(owner, gt=0, distill_temperature=self.distill_temperature)
        require_finite(owner, ge=0, le=1, distill_alpha=self.distill_alpha)
        # A non-positive length is legal: it resolves to the student's context window at launch.
        require_int(owner, **present(max_length=self.max_length))
