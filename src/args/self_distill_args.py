"""Script arguments for offline privileged-context self-distillation."""

import math
from dataclasses import dataclass, field, fields
from typing import ClassVar

from src.args.mixins import DEFAULT_ANSWER_FIELD, SDPGArguments, SelfDistillationLoss, format_field_names
from src.args.sft_args import SFTScriptArguments
from src.args.validation import require_positive


@dataclass
class SelfDistillationArguments(SFTScriptArguments, SDPGArguments):
    """Args for SDPG-style offline privileged-context self-distillation SFT (text or VLM).

    One model is both student (prompt only) and teacher (plus a gold-answer hint). The teacher's
    full-vocab distribution supervises the student on shared response tokens via an OPD loss, atop SFT.
    """

    PROJECT_NAME: ClassVar[str] = "self-distillation"

    # SDPG fields the script applies while building the teacher prompts; every other SDPG field is
    # forwarded to the trainer, which takes the complement.
    DATASET_SIDE_SDPG_FIELDS: ClassVar[frozenset[str]] = frozenset({"sdpg_hint_template"})
    # The field naming the column that fills each slot privileged_hint renders
    # (src/data/collators/self_distill.py); this arm fills the reference solution too.
    HINT_SLOT_FIELDS: ClassVar[dict[str, str]] = {
        "answer": "sdpg_answer_field",
        "solution": "privileged_solution_field",
    }
    HINT_PLACEHOLDERS: ClassVar[frozenset[str]] = frozenset(HINT_SLOT_FIELDS)

    sdpg_answer_field: str | None = field(
        default=DEFAULT_ANSWER_FIELD,
        metadata={
            "help": "Dataset field holding the ground-truth answer for the hint (required while sdpg_beta_base > 0)."
        },
    )
    privileged_solution_field: str | None = field(
        default="solution",
        metadata={"help": "Optional dataset field with a reference solution for {solution}."},
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
            "help": "Dataset field with a per-sample confidence in [0, 1]. When set, the "
            "SFT and OPD losses are weighted by confidence**confidence_power, divided by its mean over "
            "the train split, mirroring the soft-weighted (w_conf) SFT arm."
        },
    )
    confidence_power: float = field(
        default=4.0,
        metadata={"help": "Exponent p in the per-sample weight conf**p (finite, > 0)."},
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
        return {
            f.name: getattr(self, f.name) for f in fields(SDPGArguments) if f.name not in self.DATASET_SIDE_SDPG_FIELDS
        }

    def _validate_ranges(self) -> None:
        super()._validate_ranges()
        # A NaN coefficient NaNs every loss; a negative one pushes the student away from the anchor.
        if not math.isfinite(self.reference_kl_coef) or self.reference_kl_coef < 0:
            raise ValueError(f"reference_kl_coef must be a finite value >= 0, got {self.reference_kl_coef}")
        # p <= 0 inverts or flattens the weighting, and a zero confidence then divides by zero.
        require_positive(type(self).__name__, confidence_power=self.confidence_power)
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
