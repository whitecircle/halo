"""Reward terms as TRL reward functions: one callable per term, named after it, returning the term's
shaped score (``score ** exponent``) so the term's weight rides TRL's ``reward_weights``."""

import inspect
from collections.abc import Callable, Mapping, Sequence
from statistics import fmean
from typing import Any

from src.rewards.samples import samples_from_completions
from src.rewards.scorers.base import Scorer, scored_metric_key
from src.rewards.scorers.catalog import build_scorer
from src.rewards.terms import OnError, RewardTerm, ScoredTerm, refuse_veto_judges

Grader = Callable[..., Sequence[float | None]]


class TermRewardFunction:
    """A synchronous TRL reward function over a local grader: ``grader(prompts=…, completions=…, **cols)``
    returns one score in ``[0, 1]`` (or ``None``) per completion, which the term shapes."""

    def __init__(self, term: RewardTerm, grader: Grader):
        self.term = term
        self.grader = grader
        self.__name__ = term.name

    def __call__(self, prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> list[float | None]:
        scores = self.grader(prompts=prompts, completions=completions, **kwargs)
        return [None if score is None else self.term.shape(score) for score in scores]


class ScorerRewardFunction:
    """An asynchronous TRL reward function over an external scorer. TRL runs coroutine reward functions
    together on its own loop, so a judge's requests overlap with every other async reward's.

    ``prompt_column`` names the dataset column holding the conversation the policy was prompted with,
    read in place of TRL's ``prompts`` (the rendered template text), and ``reference_column`` the one
    TRL forwards as the sample's reference. A sample the scorer reached no verdict on returns
    ``None``, which TRL sums as 0 beside the row's other terms (the row leaves the baseline only when
    every term returned ``None``), or ``0`` under ``on_error: neutral``. The scorer's fixed diagnostic keys and the term's
    ``<source>/<name>/scored`` rate are averaged into TRL's metrics through the ``log_metric``
    callback it passes — every key on every call, since TRL gathers each key across ranks and a rank
    whose samples all failed must not log a different set (0 stands in for a key no sample carried).
    """

    def __init__(self, term: ScoredTerm, scorer: Scorer, *, prompt_column: str | None, reference_column: str | None):
        self.term = term
        self.scorer = scorer
        self.prompt_column = prompt_column
        self.reference_column = reference_column
        self.__name__ = term.name
        inspect.markcoroutinefunction(self)

    async def __call__(self, prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> list[float | None]:
        conversations = kwargs.get(self.prompt_column) if self.prompt_column else None
        references = kwargs.get(self.reference_column) if self.reference_column else None
        samples = samples_from_completions(
            conversations if conversations is not None else prompts, completions, references
        )
        results = await self.scorer.score(samples)
        log_metric = kwargs.get("log_metric")
        if callable(log_metric):
            for key in self.scorer.metric_keys:
                values = [result.metrics[key] for result in results if key in result.metrics]
                log_metric(key, fmean(values) if values else 0.0)
            log_metric(scored_metric_key(self.term), fmean(0.0 if r.score is None else 1.0 for r in results))
        missing = 0.0 if self.term.on_error is OnError.NEUTRAL else None
        return [missing if result.score is None else self.term.shape(result.score) for result in results]


def reward_functions(
    terms: Sequence[RewardTerm],
    graders: Mapping[type[RewardTerm], Callable[[RewardTerm], Grader]],
    *,
    prompt_column: str | None = None,
    reference_column: str | None,
) -> tuple[list[Callable], list[float]]:
    """TRL's ``reward_funcs`` and ``reward_weights`` for ``terms``: an externally scored term becomes a
    :class:`ScorerRewardFunction`, any other a :class:`TermRewardFunction` over the grader its type
    maps to in ``graders``. A veto judge gates an environment objective, which TRL's independent
    reward functions have no way to express, so it is refused here."""
    refuse_veto_judges(terms, where="the online arm's independent reward functions")
    functions: list[Callable] = []
    weights: list[float] = []
    for term in terms:
        if isinstance(term, ScoredTerm):
            functions.append(
                ScorerRewardFunction(
                    term, build_scorer(term), prompt_column=prompt_column, reference_column=reference_column
                )
            )
        elif type(term) in graders:
            functions.append(TermRewardFunction(term, graders[type(term)](term)))
        else:
            raise TypeError(f"reward source {term.source!r} has neither a scorer nor a grader here")
        weights.append(term.weight)
    return functions, weights
