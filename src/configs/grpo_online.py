"""Script arguments for RLVR online GRPO (verifiable rule-based rewards)."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

from src.configs.distillation import SDPGArguments
from src.configs.grpo import (
    AdvantageShapingArguments,
    ChunkedLogprobsArguments,
    GRPOEarlyStopArguments,
    PromptDatasetArguments,
    RLRRArguments,
)
from src.configs.script_arguments import DEFAULT_ANSWER_FIELD, CommonScriptArguments
from src.environments.rollout_config import ReasoningEffortChoice
from src.rewards.graders.verifiable import RLVR_REWARD_SOURCES
from src.rewards.terms import RewardTerm, parse_reward_terms, refuse_veto_judges


@dataclass
class RLVROnlineGRPOScriptArguments(
    PromptDatasetArguments,
    ChunkedLogprobsArguments,
    AdvantageShapingArguments,
    GRPOEarlyStopArguments,
    RLRRArguments,
    CommonScriptArguments,
    SDPGArguments,
):
    """Arguments for RLVR (Reinforcement Learning with Verifiable Rewards) Online GRPO.

    The reward is the ``rewards`` list of terms (:mod:`src.rewards.terms`): the ``accuracy`` and
    ``format`` graders, a generative ``judge``, a served ``reward_model`` — each ``weight * score ** exponent``.

    RLRR (arXiv:2601.23058, :class:`RLRRArguments`) replaces the group-normalized advantages with
    relative-ranking ones, so the trainer refuses it beside the std floor or the degenerate-group drop.
    """

    PROJECT_NAME: ClassVar[str] = "rlvr-online-grpo"

    # The tunables ``use_sdpg`` gates, each under the name the trainer takes it by: the shared block
    # plus the advantage gate, declared here rather than on SDPGArguments because the offline
    # self-distillation arm has no advantages to gate on.
    SDPG_TUNABLES: ClassVar[tuple[str, ...]] = (*SDPGArguments.sdpg_field_names(), "opd_positive_advantage_only")

    answer_field: str = field(
        default=DEFAULT_ANSWER_FIELD,
        metadata={"help": "Dataset column holding the ground-truth answer the rewards verify against."},
    )
    system_prompt: str | None = field(
        default=None, metadata={"help": "System prompt prepended to every conversation."}
    )
    # Literal, not str: an unknown level would reach the chat template as an unrecognized string most
    # templates ignore, leaving the run with no steer.
    reasoning_effort: ReasoningEffortChoice | None = field(
        default=None,
        metadata={
            "help": "Chat-template reasoning-effort steer (low/medium/high) or 'random' (sampled per prompt); "
            "unset = no steer. Passed via chat_template_kwargs, so it needs a template that reads reasoning_effort, "
            "like gpt-oss harmony."
        },
    )
    rewards: list[dict[str, Any]] = field(
        default_factory=lambda: [{"source": "accuracy"}],
        metadata={
            "help": "Reward terms, each {source, name?, weight?, exponent?, ...} and one TRL reward function: "
            "'accuracy' (the last \\boxed{} equals the answer), 'format' (pattern), 'judge' (a veto judge listing "
            "checks is refused here) or 'reward_model'. YAML-only. "
            "See agent-docs/training-methods/grpo/rewards/README.md."
        },
    )
    # SDPG (arXiv:2606.04036): privileged-teacher reverse-KL OPD term on positive-advantage
    # rollouts — the same model re-run with a gold-answer hint — added atop the GRPO loss.
    use_sdpg: bool = field(
        default=False,
        metadata={"help": "Enable SDPG: a privileged-teacher reverse-KL OPD term on positive-advantage rollouts."},
    )
    opd_positive_advantage_only: bool = field(
        default=True,
        metadata={
            "help": "Apply the OPD term only to positive-advantage tokens (SDPG as published). "
            "False distills on every completion token, negative-advantage ones included."
        },
    )

    def build_sdpg_kwargs(self) -> dict:
        """SDPG trainer kwargs from these args (empty when SDPG is disabled)."""
        if not self.use_sdpg:
            return {}
        return {
            **self.sdpg_kwargs(),
            "opd_positive_advantage_only": self.opd_positive_advantage_only,
            # process_for_rlvr normalizes the answer column to "answer", so use that, not answer_field.
            "sdpg_answer_field": "answer",
        }

    @property
    def reward_terms(self) -> tuple[RewardTerm, ...]:
        """The typed reward terms of ``rewards``, parsed and validated (also at parse time). A veto
        judge (one listing ``checks``) gates an environment objective, which this arm's independent
        TRL reward functions have none of, so it is refused."""
        terms = parse_reward_terms(self.rewards, RLVR_REWARD_SOURCES)
        refuse_veto_judges(terms, where="the online arm")
        return terms

    def _validate_hint_template(self) -> None:
        # With use_sdpg off a null template is a value set beside the closed gate, which the script's
        # gate refusal names as such; a template that is set is held to its placeholders either way.
        if self.use_sdpg or self.sdpg_hint_template is not None:
            super()._validate_hint_template()

    def _validate_fields(self) -> None:
        super()._validate_fields()
        self.reward_terms  # noqa: B018  parse at config time so a bad or empty list fails before any server is touched
