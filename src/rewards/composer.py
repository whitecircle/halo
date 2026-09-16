"""The composition of a reward from its terms, and the external scoring the terms need."""

import asyncio
from collections.abc import Mapping, Sequence

from src.rewards.judge import GenerativeJudge
from src.rewards.reward_model import ServedRewardModel
from src.rewards.samples import ScoringSample
from src.rewards.scoring import Scorer, ScoreResult
from src.rewards.spec import EnvironmentTerm, JudgeTerm, RewardModelTerm, RewardTerm

# Which scorer stands behind each externally scored term type.
SCORERS: dict[type[RewardTerm], type[Scorer]] = {JudgeTerm: GenerativeJudge, RewardModelTerm: ServedRewardModel}


def build_scorer(term: RewardTerm) -> Scorer:
    """The scorer for an externally scored term; a term without one (the environment's) raises."""
    scorer_type = SCORERS.get(type(term))
    if scorer_type is None:
        raise TypeError(f"reward source {term.source!r} has no external scorer")
    return scorer_type(term)


class RewardComposer:
    """The terms of one reward: prices their scores into components and scores the external ones.

    A reward carries at least one term and at most one environment grade; every other term is scored
    through :func:`build_scorer`, built on first use so construction stays offline.
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
        names = [term.name for term in self.terms]
        if len(set(names)) != len(names):
            raise ValueError(f"reward term names must be unique, got {names}")
        self.environment_term: EnvironmentTerm | None = environment_terms[0] if environment_terms else None
        self.external_terms = tuple(term for term in self.terms if not isinstance(term, EnvironmentTerm))
        self._scorers: dict[str, Scorer] | None = None

    @property
    def scorers(self) -> dict[str, Scorer]:
        """One scorer per external term, keyed by term name; built on first access."""
        if self._scorers is None:
            self._scorers = {term.name: build_scorer(term) for term in self.external_terms}
        return self._scorers

    def components(self, scores: Mapping[str, float]) -> dict[str, float]:
        """Every term's priced contribution, keyed ``reward/<name>``; a term without a score raises."""
        missing = [term.name for term in self.terms if term.name not in scores]
        if missing:
            raise ValueError(f"no score for reward term(s) {missing}")
        return {term.key: term.price(scores[term.name]) for term in self.terms}

    async def score(self, samples: Sequence[ScoringSample]) -> list[dict[str, ScoreResult]]:
        """The external terms' verdicts per sample (``{term name: result}``), every term scored concurrently."""
        if not self.external_terms:
            return [{} for _ in samples]
        per_term = await asyncio.gather(*(self.scorers[term.name].score(samples) for term in self.external_terms))
        return [
            {term.name: results[index] for term, results in zip(self.external_terms, per_term, strict=True)}
            for index in range(len(samples))
        ]

    async def verify(self) -> None:
        """Probe every external scorer once; the first failure raises."""
        for term in self.external_terms:
            await self.scorers[term.name].verify()

    async def aclose(self) -> None:
        scorers, self._scorers = self._scorers, None
        for scorer in (scorers or {}).values():
            await scorer.aclose()
