"""The scorer catalog: which scorer stands behind each externally scored term type, derived from
the term type each scorer class declares."""

from src.rewards.scorers.base import Scorer
from src.rewards.scorers.judge import GenerativeJudge
from src.rewards.scorers.reward_model import ServedRewardModel
from src.rewards.terms import RewardTerm, ScoredTerm

SCORERS: dict[type[ScoredTerm], type[Scorer]] = {
    scorer.term_type: scorer for scorer in (GenerativeJudge, ServedRewardModel)
}


def build_scorer(term: RewardTerm) -> Scorer:
    """The scorer for an externally scored term; a term without one (the environment's) raises."""
    scorer_type = SCORERS.get(type(term))
    if scorer_type is None:
        raise TypeError(f"reward source {term.source!r} has no external scorer")
    return scorer_type(term)
