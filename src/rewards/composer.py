"""The composition of a reward from its terms: the external scoring the terms need, and the settlement
of their verdicts into the reward's components."""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from src.rewards.samples import ScoringSample
from src.rewards.scorers.base import Scorer, ScoreResult, scored_metric_key
from src.rewards.scorers.catalog import build_scorer
from src.rewards.terms import (
    EnvironmentTerm,
    JudgeTerm,
    OnError,
    RewardTerm,
    ScoredTerm,
    component_key,
    require_unique_names,
)


@dataclass(frozen=True)
class Settlement:
    """What the external verdicts made of a reward: its ``components`` (every ``reward/*`` key, the
    values summing to the reward), the scorers' ``metrics`` and ``details`` (a judge's rationale) by
    term, their ``errors`` by term, and the ``invalid_reason`` an outage that voids the episode leaves
    (``None`` keeps it valid)."""

    components: dict[str, float]
    metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    invalid_reason: str | None = None

    @property
    def reward(self) -> float:
        return sum(self.components.values())


class RewardComposer:
    """The terms of one reward: scores the external ones and settles their verdicts into components.

    A reward carries at least one term and at most one environment grade; every other term is scored
    through :func:`build_scorer`, built on first use so construction stays offline. A veto judge
    strips the episode's credits, the environment term's objective first, so a reward with one needs
    that term.
    """

    def __init__(self, terms: Sequence[RewardTerm]):
        self.terms = tuple(terms)
        if not self.terms:
            # Without a term the episode reward is turn shaping alone: a run with no objective, which
            # trains on noise rather than failing.
            raise ValueError("a reward needs at least one term")
        environment_terms = [term for term in self.terms if isinstance(term, EnvironmentTerm)]
        if len(environment_terms) > 1:
            raise ValueError("a reward carries at most one environment term")
        require_unique_names("reward term", [term.name for term in self.terms])
        self.environment_term: EnvironmentTerm | None = environment_terms[0] if environment_terms else None
        self.external_terms = tuple(term for term in self.terms if isinstance(term, ScoredTerm))
        unscored = [
            term for term in self.terms if term is not self.environment_term and not isinstance(term, ScoredTerm)
        ]
        if unscored:
            raise TypeError(f"reward source(s) {sorted(term.source for term in unscored)} have no external scorer")
        vetoes = [term.name for term in self.external_terms if isinstance(term, JudgeTerm) and term.is_veto]
        if vetoes and self.environment_term is None:
            raise ValueError(
                f"veto judge term(s) {vetoes} gate the objective, which only an environment term prices: add "
                f"{{source: environment}} to the reward"
            )
        self._scorers: dict[str, Scorer] | None = None

    @property
    def views(self) -> frozenset[str]:
        """The views the external terms read, so a sample builder renders only what is asked for."""
        return frozenset(term.view for term in self.external_terms)

    @property
    def scorers(self) -> dict[str, Scorer]:
        """One scorer per external term, keyed by term name; built on first access."""
        if self._scorers is None:
            self._scorers = {term.name: build_scorer(term) for term in self.external_terms}
        return self._scorers

    async def score(self, samples: Sequence[ScoringSample]) -> list[dict[str, ScoreResult]]:
        """The external terms' verdicts per sample (``{term name: result}``), every term scored concurrently."""
        if not self.external_terms:
            return [{} for _ in samples]
        per_term = await asyncio.gather(*(self.scorers[term.name].score(samples) for term in self.external_terms))
        return [
            {term.name: results[index] for term, results in zip(self.external_terms, per_term, strict=True)}
            for index in range(len(samples))
        ]

    def settle(self, components: Mapping[str, float], verdicts: Mapping[str, ScoreResult]) -> Settlement:
        """The reward's components once the external verdicts are in: each scored term priced, a
        verdict-less term priced 0 — voiding the episode unless the term is ``neutral`` on error — and
        every credit zeroed by a veto, the penalties left standing."""
        settled = dict(components)
        metrics: dict[str, float] = {}
        details: dict[str, str] = {}
        errors: dict[str, str] = {}
        invalid_reason = None
        vetoed = False
        for term in self.external_terms:
            result = verdicts[term.name]
            metrics.update(result.metrics)
            metrics[scored_metric_key(term)] = 0.0 if result.score is None else 1.0
            if result.detail is not None:
                details[term.name] = result.detail
            if result.score is None:
                # The scorer, not the policy, failed: the term contributes nothing. Unless the term is
                # neutral on error, the episode leaves the group baseline rather than teaching a forced
                # verdict, the reason stamped where the trainer's all-invalid halt reads it.
                error = result.error or "no score"
                errors[term.name] = error
                settled[component_key(term.name)] = 0.0
                if invalid_reason is None and term.on_error is not OnError.NEUTRAL:
                    invalid_reason = f"reward term {term.name!r} scored nothing: {error}"
                continue
            settled[component_key(term.name)] = term.price(result.score)
            vetoed = vetoed or result.veto
        if vetoed:
            # A vetoed episode earns nothing — not the objective, not a bonus the hack collected on the
            # way — and still pays its penalties.
            settled = {key: min(value, 0.0) for key, value in settled.items()}
        return Settlement(settled, metrics, details, errors, invalid_reason)

    async def verify(self) -> None:
        """Probe every external scorer once; the first failure raises."""
        for term in self.external_terms:
            await self.scorers[term.name].verify()

    async def aclose(self) -> None:
        scorers, self._scorers = self._scorers, None
        for scorer in (scorers or {}).values():
            await scorer.aclose()
