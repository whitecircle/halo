"""Episode scoring: the environment's grade of a finished episode, the reward composed from it, and the
settlement of the externally scored terms (a judge, a reward model) that complete it."""

import asyncio
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.environments.episode_keys import (
    ANSWER_KEY,
    CONTEXT_KEY,
    EPISODE_ERROR_KEY,
    EPISODE_INVALID_KEY,
    EPISODE_INVALID_REASON_KEY,
    REWARD_COMPONENTS_KEY,
    REWARD_DETAILS_KEY,
    SANDBOX_FAULT_KEY,
)
from src.environments.messages import Message, Trajectory
from src.rewards.composer import RewardComposer
from src.rewards.samples import CUT_TOOL_CALLS_KEY, REFUSED_CALL_IDS_KEY, TURN_FLAG_NOTES, ScoringSample
from src.rewards.scorers.base import ScoreResult
from src.rewards.terms import (
    ENVIRONMENT_REWARD_SOURCES,
    OBJECTIVE_TERM_NAME,
    Check,
    EnvironmentTerm,
    JudgeTerm,
    RewardTerm,
    component_key,
    parse_reward_terms,
    resolve_environment_checks,
)

logger = logging.getLogger(__name__)

# Set while an episode's externally scored terms are owed (``settle_async`` clears it): its reward is partial.
REWARD_PENDING_KEY = "reward_pending"
# The external scorers' diagnostics (merged into the rollout metrics) and their failures, by term.
_REWARD_METRICS_KEY = "reward_metrics"
REWARD_ERRORS_KEY = "reward_errors"
# The component of the accrued per-turn deltas, reserved: no shaping term or reward term may take its name.
_TURN_SHAPING_COMPONENT = "turn_shaping"


@dataclass(frozen=True)
class EpisodeGrade:
    """What an environment says about a finished episode: ``objective``, its own grade of the task in
    ``[0, 1]`` (priced by the reward's environment term), and ``shaping``, the environment's own
    episode-level terms by bare name (``{"submission": 0.25}``), each added as ``reward/<name>``."""

    objective: float
    shaping: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        objective = self.objective
        if isinstance(objective, bool) or not isinstance(objective, int | float) or not (0.0 <= objective <= 1.0):
            raise ValueError(f"an episode's objective grade must lie in [0, 1], got {objective!r}")
        for name, value in self.shaping.items():
            if (
                not isinstance(name, str)
                or not name
                or "/" in name
                or name in (OBJECTIVE_TERM_NAME, _TURN_SHAPING_COMPONENT)
            ):
                raise ValueError(f"shaping term name {name!r} is malformed or reserved")
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError(f"shaping term {name!r} must be a finite number, got {value!r}")


class EpisodeScoringMixin:
    """The reward of an environment's episodes: the environment's side priced when an episode finishes, the
    external terms settled after it; the host's ``_grade_episode`` and ``get_tools_schema`` feed it."""

    # The episode-level shaping components (``reward/<name>``) a class adds, unioned over the MRO: a reward term
    # may not take one's name, and an undeclared one fails the first settled episode.
    SHAPING_COMPONENTS: tuple[str, ...] = ()
    # The episode facts a class establishes (:meth:`_episode_facts`), unioned over the MRO: what a judge check
    # may name to stand exempt under (``unless``).
    EPISODE_FACTS: tuple[str, ...] = ()
    # The checks a veto judge listing ``checks: environment`` audits with; none refuses such a judge.
    JUDGE_CHECKS: tuple[Check, ...] = ()

    def _init_rewards(self, reward_terms: Sequence[RewardTerm | Mapping[str, Any]] | None) -> None:
        terms = (
            (EnvironmentTerm(),)
            if reward_terms is None
            else resolve_environment_checks(
                parse_reward_terms(reward_terms, ENVIRONMENT_REWARD_SOURCES), self.JUDGE_CHECKS
            )
        )
        self._rewards = RewardComposer(terms)
        reserved = {_TURN_SHAPING_COMPONENT, *self.shaping_components()}
        taken = sorted(term.name for term in self._rewards.external_terms if term.name in reserved)
        if taken:
            raise ValueError(
                f"reward term name(s) {taken} are shaping components of {type(self).__name__}; "
                f"pick another name for the term"
            )
        unknown = sorted(
            f"{term.name}.{check.name}: {check.unless!r}"
            for term in self._rewards.external_terms
            if isinstance(term, JudgeTerm)
            for check in term.checks
            if check.unless is not None and check.unless not in self.episode_facts()
        )
        if unknown:
            raise ValueError(
                f"judge check(s) {unknown} stand exempt under episode facts {type(self).__name__} never establishes "
                f"(it establishes {sorted(self.episode_facts()) or 'none'})"
            )
        if hasattr(type(self), "_compute_reward"):
            raise TypeError(
                f"{type(self).__name__} defines _compute_reward, which nothing calls: an environment grades "
                f"through _grade_episode, returning an EpisodeGrade."
            )

    @property
    def reward_terms(self) -> tuple[RewardTerm, ...]:
        """The episode reward's terms, as configured."""
        return self._rewards.terms

    @property
    def objective_term(self) -> EnvironmentTerm | None:
        """The reward's term pricing this environment's own grade, or ``None`` when the reward prices none."""
        return self._rewards.environment_term

    @classmethod
    def shaping_components(cls) -> frozenset[str]:
        """The shaping component names this class may add: every ``SHAPING_COMPONENTS`` along the MRO."""
        return frozenset(name for klass in cls.__mro__ for name in getattr(klass, "SHAPING_COMPONENTS", ()))

    @classmethod
    def episode_facts(cls) -> frozenset[str]:
        """The episode fact names this class establishes: every ``EPISODE_FACTS`` along the MRO."""
        return frozenset(name for klass in cls.__mro__ for name in getattr(klass, "EPISODE_FACTS", ()))

    def verify_backend(self) -> None:
        """Probe every externally scored reward term once before training (global rank 0), so a misconfigured
        backend fails the launch; an env with a backend of its own extends this and calls ``super()``."""
        if self._rewards.external_terms:
            asyncio.run(self._rewards.verify())

    async def settle_async(self, episode_ids: list[int]) -> None:
        """Score the external terms of the named finished episodes, every scorer concurrently over the batch, and
        complete their rewards; an episode without a pending reward is a no-op."""
        pending = [trajectory for trajectory in self._episodes(episode_ids) if REWARD_PENDING_KEY in trajectory.info]
        if not pending:
            return
        verdicts = await self._rewards.score([self._scoring_sample(trajectory) for trajectory in pending])
        for trajectory, verdict in zip(pending, verdicts, strict=True):
            self._apply_external_scores(trajectory, verdict)

    def settle(self, episode_ids: list[int]) -> None:
        """:meth:`settle_async` for a caller with no running loop, on a loop of its own that releases the scorers'
        clients with it, so no later call reuses a client bound to a closed loop."""
        if not any(REWARD_PENDING_KEY in trajectory.info for trajectory in self._episodes(episode_ids)):
            return

        async def settle_and_release() -> None:
            try:
                await self.settle_async(episode_ids)
            finally:
                await self._rewards.aclose()

        asyncio.run(settle_and_release())

    def _reward_metrics(self, trajectory: Trajectory) -> dict[str, float]:
        """The reward's share of the rollout metrics: its components, the external scorers' diagnostics, and
        whether every external term scored."""
        metrics = {**trajectory.info.get(REWARD_COMPONENTS_KEY, {}), **trajectory.info.get(_REWARD_METRICS_KEY, {})}
        if (
            self._rewards.external_terms
            and REWARD_COMPONENTS_KEY in trajectory.info
            and not self._cut_short(trajectory)
        ):
            metrics["episode/reward_scored"] = 0.0 if trajectory.info.get(REWARD_ERRORS_KEY) else 1.0
        return metrics

    def _settle_grade(self, trajectory: Trajectory, context: dict[str, Any] | None) -> None:
        """Price the environment's side of the reward into ``reward_components``: the accrued per-turn deltas, the
        protocol's and the environment's shaping, and the grade through the environment term."""
        grade = self._grade_episode(trajectory, context)
        if not isinstance(grade, EpisodeGrade):
            raise TypeError(
                f"{type(self).__name__}._grade_episode must return an EpisodeGrade, got {type(grade).__name__}"
            )
        components = {component_key(_TURN_SHAPING_COMPONENT): trajectory.total_reward}
        declared = self.shaping_components()
        for shaping in (self._episode_shaping(trajectory), grade.shaping):
            for name, value in shaping.items():
                if name not in declared:
                    raise ValueError(
                        f"{type(self).__name__} produced the shaping component {name!r} without declaring it "
                        f"in SHAPING_COMPONENTS ({sorted(declared)})"
                    )
                key = component_key(name)
                if key in components:
                    raise ValueError(f"reward component {key!r} is declared twice")
                components[key] = float(value)
        term = self._rewards.environment_term
        if term is not None:
            components[component_key(term.name)] = term.price(grade.objective)
        # Every external term contributes 0 until it is scored, so the record holds the whole key set.
        for external in self._rewards.external_terms:
            components[component_key(external.name)] = 0.0
        trajectory.info[REWARD_COMPONENTS_KEY] = components
        trajectory.total_reward = sum(components.values())

    def _episode_shaping(self, trajectory: Trajectory) -> dict[str, float]:
        """The protocol's episode-level shaping terms by bare name; the base has none."""
        return {}

    @staticmethod
    def _invalid_grade(trajectory: Trajectory, reason: str) -> EpisodeGrade:
        """The grade of an episode whose grading reached no verdict: 0, marked invalid for ``reason`` so it leaves
        the group baseline rather than biasing every sibling's advantage."""
        trajectory.info[EPISODE_INVALID_KEY] = True
        trajectory.info[EPISODE_INVALID_REASON_KEY] = reason
        return EpisodeGrade(0.0)

    @classmethod
    def _null_answer_grade(cls, trajectory: Trajectory) -> EpisodeGrade:
        """The grade of an answer-graded episode whose ``answer`` cell is null: invalid, since a completion payout
        would hand the full objective to every sibling of its group."""
        logger.warning("Episode context carries a null %r; scoring it invalid, not a success", ANSWER_KEY)
        return cls._invalid_grade(trajectory, f"the row's {ANSWER_KEY!r} cell is null: nothing to grade")

    @staticmethod
    def _cut_short(trajectory: Trajectory) -> bool:
        """An episode its driver lost or a sandbox fault ended: never sent to an external scorer, whose verdict
        on the fragment would be paid for and taught."""
        return EPISODE_ERROR_KEY in trajectory.info or SANDBOX_FAULT_KEY in trajectory.info

    def _scoring_sample(self, trajectory: Trajectory) -> ScoringSample:
        """What an external scorer reads of a finished episode: the prompt turns (before the first assistant turn),
        the policy's turns, the final answer, the reference, the tools and the episode facts."""
        messages = [self._sample_message(message) for message in trajectory.messages]
        first = next((i for i, m in enumerate(trajectory.messages) if m.role == "assistant"), len(messages))
        return ScoringSample(
            prompt=messages[:first],
            completion=messages[first:],
            final_answer=self._final_answer(trajectory),
            reference=self._scoring_reference(trajectory),
            tools=self.get_tools_schema() or None,
            facts=self._episode_facts(trajectory),
        )

    @staticmethod
    def _sample_message(message: Message) -> dict[str, Any]:
        """A message as a scorer reads it: the wire fields with the reasoning, the turn flags a judge sees named,
        and apart from the calls that ran, the calls a cut turn never ran and the ids of the calls refused unrun."""
        rendered = message.to_dict(include_thinking=True)
        rendered.update({flag: True for flag in TURN_FLAG_NOTES if getattr(message, flag)})
        if message.cut_tool_calls:
            rendered[CUT_TOOL_CALLS_KEY] = message.cut_tool_calls
        if message.refused_call_ids:
            rendered[REFUSED_CALL_IDS_KEY] = list(message.refused_call_ids)
        return rendered

    def _final_answer(self, trajectory: Trajectory) -> str | None:
        """What the episode delivered (a text answer, a submitted program) for a scorer's ``final`` view; ``None``
        in the base, which knows no answer channel, so no fragment is read as the answer."""
        return None

    def _scoring_reference(self, trajectory: Trajectory) -> Any:
        """The reference a scorer may read beside the episode: the row's ``answer`` as the grader compares it
        (overridden where the cell is a grading payload or is converted before grading)."""
        return (trajectory.info.get(CONTEXT_KEY) or {}).get(ANSWER_KEY)

    def _episode_facts(self, trajectory: Trajectory) -> frozenset[str]:
        """The :data:`EPISODE_FACTS` that hold for a finished episode; the base establishes none."""
        return frozenset()

    def _apply_external_scores(self, trajectory: Trajectory, verdict: Mapping[str, ScoreResult]) -> None:
        """Settle the external terms' verdicts into the components, record their diagnostics, and close the
        episode's reward, dropping the pending mark and the grading payload."""
        settlement = self._rewards.settle(trajectory.info[REWARD_COMPONENTS_KEY], verdict)
        trajectory.info[REWARD_COMPONENTS_KEY] = settlement.components
        if settlement.metrics:
            trajectory.info.setdefault(_REWARD_METRICS_KEY, {}).update(settlement.metrics)
        if settlement.details:
            trajectory.info.setdefault(REWARD_DETAILS_KEY, {}).update(settlement.details)
        if settlement.errors:
            trajectory.info.setdefault(REWARD_ERRORS_KEY, {}).update(settlement.errors)
        if settlement.invalid_reason is not None:
            trajectory.info[EPISODE_INVALID_KEY] = True
            trajectory.info[EPISODE_INVALID_REASON_KEY] = settlement.invalid_reason
        trajectory.total_reward = settlement.reward
        del trajectory.info[REWARD_PENDING_KEY]
        self._drop_grading_payload(trajectory)

    @staticmethod
    def _drop_grading_payload(trajectory: Trajectory) -> None:
        """Strip what a settled episode no longer needs but every later hop (Ray, the TP broadcast) would carry: the
        private ``_``-prefixed stamps and the context's ``answer``. A driver's own stamps land after this."""
        for key in [key for key in trajectory.info if key.startswith("_")]:
            del trajectory.info[key]
        context = trajectory.info.get(CONTEXT_KEY)
        if context and ANSWER_KEY in context:
            trajectory.info[CONTEXT_KEY] = {key: value for key, value in context.items() if key != ANSWER_KEY}
