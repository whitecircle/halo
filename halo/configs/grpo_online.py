from dataclasses import dataclass, field
from typing import Any, ClassVar

from halo.configs.distillation import SDPGArguments
from halo.configs.grpo import (
    AdvantageShapingArguments,
    ChunkedLogprobsArguments,
    GRPOEarlyStopArguments,
    PhaseTimerArguments,
    PromptDatasetArguments,
    RLRRArguments,
)
from halo.configs.script_arguments import DEFAULT_ANSWER_FIELD, CommonScriptArguments
from halo.environments.rollout_config import ReasoningEffortChoice
from halo.rewards.graders.verifiable import RLVR_REWARD_SOURCES
from halo.rewards.terms import RewardTerm, parse_reward_terms, read_context_files, reject_veto_judges


@dataclass
class OnlineGRPOScriptArguments(
    PromptDatasetArguments,
    ChunkedLogprobsArguments,
    PhaseTimerArguments,
    AdvantageShapingArguments,
    GRPOEarlyStopArguments,
    RLRRArguments,
    CommonScriptArguments,
    SDPGArguments,
):
    """RLVR Online GRPO. The reward is the ``rewards`` list of terms (:mod:`halo.rewards.terms`), each
    ``weight * score ** exponent``. RLRR (arXiv:2601.23058) replaces group-normalized advantages with relative-ranking
    ones, so the trainer refuses it beside the std floor or the degenerate-group drop."""

    PROJECT_NAME: ClassVar[str] = "rlvr-online-grpo"

    # The tunables ``use_sdpg`` gates, by trainer name; the advantage gate is here because offline self-distillation
    # has no advantages to gate on.
    SDPG_TUNABLES: ClassVar[tuple[str, ...]] = (*SDPGArguments.sdpg_field_names(), "opd_positive_advantage_only")

    answer_field: str = field(
        default=DEFAULT_ANSWER_FIELD,
        metadata={"help": "Dataset column holding the ground-truth answer the rewards verify against."},
    )
    system_prompt: str | None = field(
        default=None, metadata={"help": "System prompt prepended to every conversation."}
    )
    # An unknown level would reach the chat template as a string most templates ignore, leaving the run with no steer.
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
    # SDPG (arXiv:2606.04036): the model re-run with a gold-answer hint adds reverse-KL on positive-advantage rows.
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

    def __post_init__(self):
        self.rewards = read_context_files(self.rewards)
        super().__post_init__()

    def build_sdpg_kwargs(self) -> dict:
        if not self.use_sdpg:
            return {}
        return {
            **self.sdpg_kwargs(),
            "opd_positive_advantage_only": self.opd_positive_advantage_only,
            # process_for_rlvr normalizes the answer column to "answer".
            "sdpg_answer_field": "answer",
        }

    @property
    def reward_terms(self) -> tuple[RewardTerm, ...]:
        """A veto judge (one listing ``checks``) gates an environment objective, which this arm's independent TRL
        reward functions lack, so it is refused."""
        terms = parse_reward_terms(self.rewards, RLVR_REWARD_SOURCES)
        reject_veto_judges(terms, where="the online arm")
        return terms

    def _validate_hint_template(self) -> None:
        # With use_sdpg off, a null template is a value beside the closed gate, which the script's refusal names; a set
        # template is still held to its placeholders.
        if self.use_sdpg or self.sdpg_hint_template is not None:
            super()._validate_hint_template()

    def _validate_fields(self) -> None:
        super()._validate_fields()
        self.reward_terms  # noqa: B018  parse at config time so a bad or empty list fails before any server is touched
