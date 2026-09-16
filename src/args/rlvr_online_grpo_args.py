"""Script arguments for RLVR online GRPO (verifiable rule-based rewards)."""

from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Literal

from src.args.common_script_args import CommonScriptArguments
from src.args.mixins import (
    AdvantageShapingArguments,
    ChunkedLogprobsArguments,
    PromptDatasetArguments,
    RLRRArguments,
    SDPGArguments,
)
from src.rewards.spec import JudgeTerm, RewardModelTerm, RewardTerm, parse_reward_terms, sources_of
from src.rewards.verifiable import AccuracyTerm, FormatTerm

# The reward sources the RLVR arm admits: its two graders plus the externally scored terms.
RLVR_REWARD_SOURCES = sources_of(AccuracyTerm, FormatTerm, JudgeTerm, RewardModelTerm)


@dataclass
class RLVROnlineGRPOScriptArguments(
    PromptDatasetArguments,
    ChunkedLogprobsArguments,
    AdvantageShapingArguments,
    RLRRArguments,
    SDPGArguments,
    CommonScriptArguments,
):
    """Arguments for RLVR (Reinforcement Learning with Verifiable Rewards) Online GRPO.

    The reward is the ``rewards`` list of terms (:mod:`src.rewards.spec`): the ``accuracy`` and
    ``format`` graders, a generative ``judge``, a served ``reward_model`` — each ``weight * score ** exponent``.

    RLRR (arXiv:2601.23058, :class:`RLRRArguments`) replaces the group-normalized advantages with
    relative-ranking ones and is mutually exclusive with the AdvantageShapingArguments surgery.
    """

    # The tunables ``use_sdpg`` gates: the shared block plus the RLVR-only advantage gate.
    SDPG_TUNABLES: ClassVar[tuple[str, ...]] = (
        *(f.name for f in fields(SDPGArguments)),
        "opd_positive_advantage_only",
    )

    answer_field: str = field(
        default="answer",
        metadata={"help": "Field in the dataset containing the ground truth answer for verification"},
    )
    system_prompt: str | None = field(
        default=None,
        metadata={"help": "System prompt to prepend to all conversations"},
    )
    # Literal, not str: an unknown level would reach the chat template as an unrecognized string most
    # templates ignore, leaving the run with no steer. The parser's Literal gate covers YAML and CLI;
    # tests/cpu/config pins the set against VALID_REASONING_EFFORTS (src/environments/base.py), which
    # cannot be imported here without pulling the environments package and its torch/vLLM tail into
    # the argument dataclasses.
    reasoning_effort: Literal["low", "medium", "high", "random"] | None = field(
        default=None,
        metadata={
            "help": "Chat-template reasoning-effort steer (low/medium/high), 'random' (sampled per "
            "prompt), or None (no steer). Passed via chat_template_kwargs; needs a template that reads "
            "reasoning_effort (e.g. gpt-oss harmony)."
        },
    )

    rewards: list[dict[str, Any]] = field(
        default_factory=lambda: [{"source": "accuracy"}],
        metadata={
            "help": "Reward terms, each {source, name?, weight?, exponent?, ...}: sources 'accuracy' "
            "(last \\boxed{} equals the answer), 'format' (pattern), 'judge' (model, requirements, "
            "reasoning_effort, ...) and 'reward_model' (url, model, backend, ...). Each term is one "
            "TRL reward function named after it, weighted by its weight; see agent-docs/training-methods/grpo/rewards.md."
        },
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

        :class:`SDPGArguments` declares every tunable under the name the trainer takes it by, so the
        block forwards itself; the two entries outside that mapping stay explicit.
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

    @property
    def reward_terms(self) -> tuple[RewardTerm, ...]:
        """The typed reward terms of ``rewards``, parsed and validated (also at parse time)."""
        return parse_reward_terms(self.rewards, RLVR_REWARD_SOURCES)

    def _validate_ranges(self) -> None:
        super()._validate_ranges()
        if not self.rewards:
            raise ValueError("rewards must list at least one reward term")
        self.reward_terms  # noqa: B018  parse at config time so a bad term fails before any server is touched

    def __post_init__(self):
        self._apply_default_project_name("rlvr-online-grpo")
        self._validate_ranges()
