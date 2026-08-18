"""Script arguments for offline privileged-context self-distillation."""

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from src.args.mixins import SDPGArguments
from src.args.sft_args import SFTScriptArguments


@dataclass
class SelfDistillationArguments(SDPGArguments, SFTScriptArguments):
    """Args for SDPG-style offline privileged-context self-distillation SFT (text or VLM).

    One model is both student (prompt only) and teacher (plus a gold-answer hint). The teacher's
    full-vocab distribution supervises the student on shared response tokens via an OPD loss, atop SFT.
    """

    # SDPG fields the script applies while building the teacher prompts; every other SDPG field is
    # forwarded to the trainer, which pops exactly the complement — one declaration for both sides.
    DATASET_SIDE_SDPG_FIELDS: ClassVar[frozenset[str]] = frozenset({"sdpg_hint_template"})

    sdpg_answer_field: str | None = field(
        default="answer",
        metadata={"help": "Dataset field holding the ground-truth answer for the hint."},
    )
    privileged_solution_field: str | None = field(
        default="solution",
        metadata={"help": "Optional dataset field with a reference solution for {solution}."},
    )

    reference_kl_coef: float = field(
        default=0.0,
        metadata={
            "help": "Alpha for KL regularization to a frozen reference model. 0 disables "
            "(no reference model is loaded)."
        },
    )
    reference_kl_loss: Literal["unnormalized_kl", "reverse_kl", "forward_kl"] = field(
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
            "SFT and OPD losses are weighted by confidence**confidence_power "
            "(mean-normalized), mirroring the soft-weighted (w_conf) SFT arm."
        },
    )
    confidence_power: float = field(
        default=4.0,
        metadata={"help": "Exponent p in the per-sample weight conf**p."},
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

    def __post_init__(self):
        self._apply_default_project_name("self-distillation")
        self._validate_ranges()
