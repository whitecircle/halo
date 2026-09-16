"""The scorer contract every external reward source implements."""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from src.rewards.samples import ScoringSample
from src.rewards.spec import RewardTerm


@dataclass(frozen=True)
class ScoreResult:
    """One sample's verdict from one scorer: the score in ``[0, 1]``, or ``None`` with the ``error``
    that prevented one. ``metrics`` are per-sample diagnostics (logged under the term's name),
    ``detail`` the scorer's own account of the verdict (a judge's rationale)."""

    score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    detail: str | None = None


class Scorer(ABC):
    """Scores samples for one reward term through an external backend.

    :meth:`score` never raises for a sample: a failed request is a :class:`ScoreResult` carrying its
    ``error``, so one bad call cannot take a rollout batch down with it. Construction is offline —
    clients are built on first use, on the loop that uses them — and :meth:`verify` makes one real
    request in the run's exact shape so a bad endpoint fails the launch, not every episode.
    """

    def __init__(self, term: RewardTerm):
        self.term = term
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None

    @property
    def metric_keys(self) -> tuple[str, ...]:
        """The diagnostic keys every successful :class:`ScoreResult` of this scorer carries — fixed per
        term, so a trainer can log the same set on every rank whatever each rank's samples returned."""
        return ()

    async def score(self, samples: Sequence[ScoringSample]) -> list[ScoreResult]:
        """One verdict per sample, scored concurrently up to the term's ``max_concurrency``."""
        return list(await asyncio.gather(*(self._bounded(self.score_one, sample) for sample in samples)))

    @abstractmethod
    async def score_one(self, sample: ScoringSample) -> ScoreResult:
        """Score one sample; failures come back as a result with ``error`` set."""

    @abstractmethod
    async def verify(self) -> None:
        """One real request in the run's shape, through a client closed before returning; raises."""

    async def aclose(self) -> None:  # noqa: B027  optional hook; scorers holding a client override
        """Release the client built by :meth:`score`."""

    async def _bounded(self, fn: Callable[..., Awaitable], *args):
        """Run ``fn`` under the term's concurrency cap. The semaphore is bound to the loop that first
        uses it, so a scorer driven from a new loop (a probe, then the actor loop) gets a fresh one."""
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(self.term.max_concurrency)
            self._semaphore_loop = loop
        async with self._semaphore:
            return await fn(*args)
