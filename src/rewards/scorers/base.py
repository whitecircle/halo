"""The scorer contract every external reward source implements, what a verdict carries, and the
OpenAI-compatible client a chat-model scorer holds."""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

from openai import AsyncOpenAI

from src.env import env_str
from src.inference.endpoints import resolve_external_api_key
from src.inference.openai_client import create_openai_client
from src.rewards.samples import ScoringSample
from src.rewards.terms import JudgeTerm, RewardTerm, ScoredTerm

# The sample a scorer's launch probe grades: tiny, so the probe costs nothing, yet in the run's exact shape.
PROBE_SAMPLE = ScoringSample(
    prompt=[{"role": "user", "content": "Reply with the single word: ready"}],
    completion=[{"role": "assistant", "content": "ready"}],
    final_answer="ready",
)
# How much of a reply an error result quotes.
REPLY_EXCERPT_CHARS = 200
# The key chain a chat-model scorer's key is read from after the term's own variable.
HOSTED_KEY_CHAIN = ("OPENROUTER_API_KEY", "OPENAI_API_KEY")


def scored_metric_key(term: RewardTerm) -> str:
    """The per-sample 1/0 of whether the term reached a verdict, ``<source>/<name>/scored``: the metric
    that says a scorer is failing, logged by both arms."""
    return f"{term.source}/{term.name}/scored"


def scorer_api_key(term: JudgeTerm) -> str:
    """A chat-model scorer's key: the term's ``api_key_env`` variable, then the hosted chain."""
    key = env_str(term.api_key_env) or resolve_external_api_key()
    if not key:
        names = " or ".join(dict.fromkeys((term.api_key_env, *HOSTED_KEY_CHAIN)))
        raise RuntimeError(f"{term.owner}: no API key — set {names}")
    return key


@dataclass(frozen=True)
class ScoreResult:
    """One sample's verdict from one scorer: the score in ``[0, 1]``, or ``None`` with the ``error``
    that prevented one. ``metrics`` are per-sample diagnostics (logged under the term's name),
    ``detail`` the scorer's own account of the verdict (a judge's rationale), ``veto`` whether the
    verdict strips the episode of its credits (a veto judge's fired veto check)."""

    score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    detail: str | None = None
    veto: bool = False


class Scorer(ABC):
    """Scores samples for one reward term through an external backend.

    A subclass declares the term type it scores (:attr:`term_type`), which the scorer catalog
    (:mod:`src.rewards.scorers.catalog`) reads. :meth:`score` never raises for a sample: a failed
    request is a :class:`ScoreResult` carrying its ``error``, so one bad call cannot take a rollout
    batch down with it. Construction is offline — the client is built on first use, on the loop that
    uses it, and :meth:`aclose` releases it — and :meth:`verify` makes one real request in the run's
    exact shape so a bad endpoint fails the launch, not every episode.
    """

    term_type: ClassVar[type[ScoredTerm]]

    def __init__(self, term: ScoredTerm):
        self.term = term
        self._client = None
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None

    @property
    def metric_keys(self) -> tuple[str, ...]:
        """The diagnostic keys a successful :class:`ScoreResult` of this scorer carries — fixed per term,
        so a trainer can log the same set on every rank whatever each rank's samples returned; a key a
        result lacks (``completion_tokens`` without a usage report) averages over the results that carry it."""
        return ()

    def _key(self, leaf: str) -> str:
        """A per-sample metric key of this scorer, ``<source>/<name>/<leaf>``."""
        return f"{self.term.source}/{self.term.name}/{leaf}"

    async def score(self, samples: Sequence[ScoringSample]) -> list[ScoreResult]:
        """One verdict per sample, scored concurrently up to the term's ``max_concurrency``."""
        return list(await asyncio.gather(*(self._bounded(self.score_one, sample) for sample in samples)))

    @abstractmethod
    async def score_one(self, sample: ScoringSample) -> ScoreResult:
        """Score one sample; failures come back as a result with ``error`` set."""

    async def verify(self) -> None:
        """Grade the probe once, through a client released before returning — the launch probe and the
        Ray actor run on different loops, and a client's pool binds to the loop it was built on; a verdict
        the scorer never reached raises."""
        try:
            result = await self.score_one(PROBE_SAMPLE)
        finally:
            await self.aclose()
        if result.score is None:
            raise RuntimeError(f"{self.term.owner} probe failed: {result.error}")

    @abstractmethod
    async def aclose(self) -> None:
        """Release the client built on first use."""

    async def _bounded(self, fn: Callable[..., Awaitable], *args):
        """Run ``fn`` under the term's concurrency cap. The semaphore is bound to the loop that first
        uses it, so a scorer driven from a new loop (a probe, then the actor loop) gets a fresh one."""
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(self.term.max_concurrency)
            self._semaphore_loop = loop
        async with self._semaphore:
            return await fn(*args)


class ChatModelScorer(Scorer):
    """A scorer over an OpenAI-compatible chat model: one lazily built client per instance (per Ray
    actor or trainer rank), shared by every sample it grades, and the usage a reply reports."""

    term: JudgeTerm
    _client: AsyncOpenAI | None

    def _connect(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = create_openai_client(
                base_url=self.term.base_url, api_key_override=scorer_api_key(self.term)
            )
        return self._client

    def _usage_metrics(self, completion) -> dict[str, float]:
        metrics: dict[str, float] = {}
        usage = getattr(completion, "usage", None)
        if usage is not None:
            metrics[self._key("completion_tokens")] = float(getattr(usage, "completion_tokens", 0) or 0)
            cost = getattr(usage, "cost", None)
            if isinstance(cost, int | float):
                metrics[self._key("cost_usd")] = float(cost)
        return metrics

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.close()
