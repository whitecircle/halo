"""Script arguments for RLVR online GRPO (verifiable rule-based rewards)."""

from dataclasses import dataclass, field, fields
from typing import Literal

from src.args.common_script_args import CommonScriptArguments
from src.args.mixins import (
    AdvantageShapingArguments,
    ChunkedLogprobsArguments,
    PromptDatasetArguments,
    RLRRConfig,
    SDPGArguments,
)


@dataclass
class RLVROnlineGRPOScriptArguments(
    PromptDatasetArguments,
    ChunkedLogprobsArguments,
    AdvantageShapingArguments,
    SDPGArguments,
    CommonScriptArguments,
):
    """Arguments for RLVR (Reinforcement Learning with Verifiable Rewards) Online GRPO.

    RLVR uses rule-based reward functions (math accuracy, format checking) instead of
    external API scoring — for tasks with verifiable outcomes (math, coding, etc.).
    """

    answer_field: str = field(
        default="answer",
        metadata={"help": "Field in the dataset containing the ground truth answer for verification"},
    )
    system_prompt: str | None = field(
        default=None,
        metadata={"help": "System prompt to prepend to all conversations"},
    )
    # Literal, not str: the env-GRPO path validates the same steer in BaseEnvironment.__init__, while
    # here an unknown level would reach the chat template as an unrecognized string that most
    # templates quietly ignore — the run then trains with no steer at all. The parser's Literal gate
    # covers YAML and CLI alike; tests/cpu/config pins the set against VALID_REASONING_EFFORTS
    # (src/environments/base.py), which cannot be imported here without pulling the environments
    # package — and its torch/vLLM tail — into the argument dataclasses.
    reasoning_effort: Literal["low", "medium", "high", "random"] | None = field(
        default=None,
        metadata={
            "help": "Chat-template reasoning-effort steer (low/medium/high), 'random' (sampled per "
            "prompt), or None (no steer). Passed via chat_template_kwargs; needs a template that reads "
            "reasoning_effort (e.g. gpt-oss harmony)."
        },
    )

    use_accuracy_reward: bool = field(
        default=True,
        metadata={"help": r"Enable accuracy reward: checks if \boxed{} content matches ground truth"},
    )
    use_format_reward: bool = field(
        default=False,
        metadata={"help": "Enable format reward: checks if completion matches expected format pattern"},
    )
    format_pattern: str = field(
        default=r"<think>.*?</think>\s*<answer>.*?</answer>",
        metadata={"help": "Regex pattern for format reward (used when use_format_reward=True)"},
    )

    accuracy_reward_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for accuracy reward function"},
    )
    format_reward_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for format reward function"},
    )

    # RLRR (arXiv:2601.23058) replaces group-normalized advantages with relative-ranking ones;
    # mutually exclusive with AdvantageShapingArguments' negative-side surgery (advantage_mode).
    use_rlrr: bool = field(
        default=False,
        metadata={"help": "Enable RLRR relative-reward advantage shaping (replaces group-normalized advantages)"},
    )
    rlrr_mode: Literal["hrr", "prr"] = field(
        default="hrr", metadata={"help": "RLRR mode: 'hrr' (hybrid) or 'prr' (pure relative)"}
    )
    rlrr_tau: float = field(default=0.1, metadata={"help": "HRR temperature τ (Eq. 3)"})
    rlrr_lambda: float = field(default=2048.0, metadata={"help": "Length-bin granularity λ for re-ranking (Eq. 6)"})
    rlrr_std_normalize: bool = field(default=False, metadata={"help": "Std-normalize advantages within each group"})
    rlrr_length_rerank: bool = field(default=True, metadata={"help": "Enable hierarchical length re-ranking (Eq. 6)"})
    rlrr_correctness_clip: bool = field(
        default=True,
        metadata={"help": "Correctness-aware advantage clipping (Eq. 5). Disable for pure PRR with no gold labels."},
    )
    rlrr_correctness_threshold: float = field(
        default=0.5,
        metadata={"help": "Reward >= threshold counts as correct when explicit labels are absent"},
    )
    rlrr_xi_pos: float = field(
        default=1e-3, metadata={"help": "Advantage cap ξ⁺ for incorrect responses (Eq. 5 clip)"}
    )
    rlrr_xi_neg: float = field(
        default=-1e-3, metadata={"help": "Advantage floor ξ⁻ for correct responses (Eq. 5 clip)"}
    )

    # SDPG (arXiv:2606.04036): privileged-teacher reverse-KL OPD term on positive-advantage
    # rollouts — the same model re-run with a gold-answer hint — added atop the GRPO loss.
    use_sdpg: bool = field(
        default=False,
        metadata={"help": "Enable SDPG: privileged-teacher reverse-KL OPD on positive-advantage rollouts"},
    )
    opd_positive_advantage_only: bool = field(
        default=True,
        metadata={
            "help": "Apply the OPD term only to positive-advantage tokens (SDPG as published). "
            "False distills on every completion token, negative-advantage ones included."
        },
    )

    def build_sdpg_kwargs(self) -> dict:
        """SDPG trainer kwargs from these args (empty when SDPG is disabled).

        A gate, not a translation: :class:`SDPGArguments` declares every tunable under the name the
        trainer takes it by, so the block forwards itself and a field added there cannot go missing
        here. Only the two entries that are not that mapping stay explicit.
        """
        if not self.use_sdpg:
            return {}
        return {
            **{f.name: getattr(self, f.name) for f in fields(SDPGArguments)},
            # process_for_rlvr normalizes the answer column to "answer", so use that, not answer_field.
            "sdpg_answer_field": "answer",
            # Declared here rather than on SDPGArguments: the gate is RLVR-only (the offline
            # self-distillation arm has no advantages to gate on).
            "opd_positive_advantage_only": self.opd_positive_advantage_only,
        }

    def build_rlrr_config(self) -> RLRRConfig | None:
        """Return an :class:`RLRRConfig` from these args, or ``None`` when RLRR is disabled."""
        if not self.use_rlrr:
            return None
        return RLRRConfig(
            mode=self.rlrr_mode,
            tau=self.rlrr_tau,
            lam=self.rlrr_lambda,
            std_normalize=self.rlrr_std_normalize,
            length_rerank=self.rlrr_length_rerank,
            correctness_clip=self.rlrr_correctness_clip,
            correctness_threshold=self.rlrr_correctness_threshold,
            xi_pos=self.rlrr_xi_pos,
            xi_neg=self.rlrr_xi_neg,
        )

    def _validate_ranges(self) -> None:
        super()._validate_ranges()
        # Both divide inside the RLRR shaping (Eq. 3 / Eq. 6): a zero is a ZeroDivisionError deep in
        # the advantage pass, a negative one silently inverts the ranking it is meant to correct.
        if self.rlrr_tau <= 0:
            raise ValueError(f"rlrr_tau must be > 0, got {self.rlrr_tau}")
        if self.rlrr_lambda <= 0:
            raise ValueError(f"rlrr_lambda must be > 0, got {self.rlrr_lambda}")

    def __post_init__(self):
        self._apply_default_project_name("rlvr-online-grpo")
        self._validate_ranges()
