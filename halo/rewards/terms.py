"""Reward terms: a reward is a sum of terms, each a source of a score in ``[0, 1]`` priced as
``weight * score ** exponent``. Parsed at config time, so this imports nothing heavier than the endpoint defaults."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Self

from halo.config_validation import (
    differs_from_default,
    present,
    require_bool,
    require_finite,
    require_int,
    require_text,
)
from halo.inference.endpoints import DEFAULT_OPENROUTER_BASE_URL

# ``reward/<name>``; the keys of one reward sum to the reward.
REWARD_COMPONENT_PREFIX = "reward/"
# Leaf of a scored term's per-sample 1/0 of whether it reached a verdict.
_SCORED_METRIC_LEAF = "scored"
OBJECTIVE_TERM_NAME = "objective"
# ``checks: environment``: audit with the checks the environment declares.
ENVIRONMENT_CHECKS = "environment"


class View(StrEnum):
    """What a scorer reads of an episode: its final answer, every turn, or a compact digest of every turn."""

    FINAL = "final"
    FULL = "full"
    DIGEST = "digest"


class JudgeMetric(StrEnum):
    """Leaves of a judge's own per-sample metrics; with :data:`_SCORED_METRIC_LEAF`, names a requirement or check may
    not take."""

    COMPLETION_TOKENS = "completion_tokens"
    COST_USD = "cost_usd"
    VETO = "veto"
    UNSUPPORTED_FLAGS = "unsupported_flags"
    EXEMPTED_FLAGS = "exempted_flags"


class OnError(StrEnum):
    """What a verdict the scorer never reached means: the episode leaves the group baseline, or the
    term prices 0 and the episode trains on its other terms."""

    INVALID = "invalid"
    NEUTRAL = "neutral"


class JudgeReasoningEffort(StrEnum):
    """The OpenAI reasoning-effort vocabulary, which OpenRouter forwards to the model it routes to."""

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class RewardModelBackend(StrEnum):
    VLLM = "vllm"
    SGLANG = "sglang"


def component_key(name: str) -> str:
    return REWARD_COMPONENT_PREFIX + name


def metric_key(term: "RewardTerm", leaf: str) -> str:
    return f"{term.source}/{term.name}/{leaf}"


def scored_metric_key(term: "RewardTerm") -> str:
    """The per-sample 1/0 of whether the term reached a verdict: the metric that says a scorer is failing."""
    return metric_key(term, _SCORED_METRIC_LEAF)


OBJECTIVE_REWARD_KEY = component_key(OBJECTIVE_TERM_NAME)


def _coerce_enum(term: "RewardTerm", name: str, enum_type: type[StrEnum]) -> None:
    value = getattr(term, name)
    try:
        member = enum_type(value)
    except ValueError:
        admitted = tuple(member.value for member in enum_type)
        raise ValueError(f"{term.owner}: {name} must be one of {admitted}, got {value!r}") from None
    object.__setattr__(term, name, member)


def _require_name(what: str, name: Any) -> None:
    """It becomes a diagnostic key, so non-blank, unpadded and free of ``/``."""
    if not isinstance(name, str) or not name or name != name.strip() or "/" in name:
        raise ValueError(f"{what} name must be a non-blank string without '/', got {name!r}")


def validate_unique_names(what: str, names: list[str]) -> None:
    if len(set(names)) != len(names):
        raise ValueError(f"{what} names must be unique, got {names}")


def reject_veto_judges(terms: Sequence["RewardTerm"], *, where: str) -> None:
    """Refuse a veto judge (one listing ``checks``) where there is no environment objective to gate."""
    vetoes = [term.name for term in terms if isinstance(term, JudgeTerm) and term.is_veto]
    if vetoes:
        raise ValueError(
            f"judge term(s) {vetoes} list 'checks' (a veto judge), which gates an environment objective; "
            f"{where} has none, so list 'requirements' instead"
        )


def _construct(cls: type, spec: Mapping[str, Any], what: str):
    """``cls(**spec)`` with an unknown key named up front, before the constructor's own checks run."""
    admitted = {f.name for f in fields(cls)}
    unknown = sorted(set(spec) - admitted)
    if unknown:
        raise ValueError(f"{what}: unknown option(s) {unknown}; admitted: {sorted(admitted)}")
    return cls(**spec)


def _items_from_config(cls: type, raw: Any, what: str) -> tuple:
    if isinstance(raw, str | bytes | Mapping) or not isinstance(raw, Sequence):
        raise ValueError(f"{what}: must be a list of mappings")
    items = []
    for index, item in enumerate(raw):
        if isinstance(item, cls):
            items.append(item)
        elif isinstance(item, Mapping):
            items.append(_construct(cls, item, f"{what}[{index}]"))
        else:
            raise ValueError(f"{what}[{index}]: an item is a mapping with 'name' and 'description'")
    return tuple(items)


@dataclass(frozen=True, kw_only=True)
class RewardTerm:
    """One additive term of a reward. ``name`` is its component key, :attr:`source` the spelling a config selects it
    by; a negative ``weight`` makes a penalty, an ``exponent`` above 1 convex credit."""

    source: ClassVar[str]

    name: str
    weight: float = 1.0
    exponent: float = 1.0

    def __post_init__(self) -> None:
        _require_name("reward term", self.name)
        require_finite(self.owner, weight=self.weight)
        require_finite(self.owner, gt=0, exponent=self.exponent)

    @property
    def owner(self) -> str:
        return f"{self.source} reward term {self.name!r}"

    def shape(self, score: float) -> float:
        """The shaped score before the weight, what a TRL reward function returns."""
        require_finite(self.owner, ge=0, le=1, score=score)
        return float(score) ** self.exponent

    def price(self, score: float) -> float:
        return self.weight * self.shape(score)

    @classmethod
    def from_config(cls, spec: Mapping[str, Any]) -> Self:
        """``spec`` has its ``source`` key already consumed."""
        return _construct(cls, spec, f"{cls.source} reward term")


@dataclass(frozen=True, kw_only=True)
class EnvironmentTerm(RewardTerm):
    """The environment's own grade in ``[0, 1]``: a solve on every hidden test, an answer match."""

    source: ClassVar[str] = "environment"

    name: str = OBJECTIVE_TERM_NAME

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.name != OBJECTIVE_TERM_NAME:
            raise ValueError(
                f"the environment term is always named {OBJECTIVE_TERM_NAME!r} (its component is the fixed "
                f"{OBJECTIVE_REWARD_KEY!r} key), got {self.name!r}"
            )


@dataclass(frozen=True, kw_only=True)
class ScoredTerm(RewardTerm):
    """A term an external backend scores after the episode. A verdict it never got (``on_error``) either drops the
    episode from the group baseline (``invalid``) or prices the term at 0 and keeps the episode valid (``neutral``).
    ``request_timeout`` bounds one request, ``max_concurrency`` the requests in flight per scorer instance."""

    views: ClassVar[tuple[View, ...]] = tuple(View)

    view: View = View.FINAL
    on_error: OnError = OnError.INVALID
    request_timeout: float
    max_concurrency: int

    def __post_init__(self) -> None:
        super().__post_init__()
        _coerce_enum(self, "view", View)
        _coerce_enum(self, "on_error", OnError)
        if self.view not in self.views:
            raise ValueError(
                f"{self.owner}: view must be one of {tuple(v.value for v in self.views)}, got {self.view!r}"
            )
        require_finite(self.owner, gt=0, request_timeout=self.request_timeout)
        require_int(self.owner, ge=1, max_concurrency=self.max_concurrency)


@dataclass(frozen=True, kw_only=True)
class Requirement:
    """One thing a scoring judge grades: a short name (its diagnostic key) and what a full score means."""

    name: str
    description: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        _require_name("requirement", self.name)
        require_text(f"requirement {self.name!r}", description=self.description)
        require_finite(f"requirement {self.name!r}", gt=0, weight=self.weight)


@dataclass(frozen=True, kw_only=True)
class Check:
    """One binary flag a veto judge raises on an observable action. A ``veto`` check gates the episode's credit (the
    objective and every other positive component go to 0, the penalties stand) and counts in the term's score only with
    a ``weight`` (default 0); any other check is a process flag counted with its ``weight`` (default 1; 0 logs it
    uncounted). ``unless`` names an episode fact (:attr:`~halo.rewards.samples.ScoringSample.facts`) under which the
    check does not fire, whatever the judge answered."""

    name: str
    description: str
    veto: bool = False
    weight: float | None = None
    unless: str | None = None

    def __post_init__(self) -> None:
        owner = f"check {self.name!r}"
        _require_name("check", self.name)
        require_text(owner, description=self.description)
        require_bool(owner, veto=self.veto)
        if self.unless is not None:
            require_text(owner, unless=self.unless)
        if self.weight is None:
            return
        require_finite(owner, ge=0, why=" (the term's own weight signs the price)", weight=self.weight)

    @property
    def share(self) -> float:
        if self.weight is None:
            return 0.0 if self.veto else 1.0
        return self.weight


@dataclass(frozen=True, kw_only=True)
class JudgeTerm(ScoredTerm):
    """A generative judge: an OpenAI-compatible chat model reads the task and one ``view`` of the episode and answers
    with one JSON verdict.

    ``requirements`` make a scoring judge: each is scored 0 to ``scale``, and the term's score is their weight-averaged
    fraction of it. ``checks`` make a veto judge: each fires, quoting the span that shows it, or does not. A fired
    ``veto`` check zeroes the objective and every other positive component, the penalties standing; the term's score is
    the weighted fired fraction of the checks that carry weight, priced by a never-positive ``weight`` (from a config
    0 by default, with ``on_error: neutral``). ``checks: environment`` takes the checks the environment declares.
    ``context`` is what the judge should know of the setting that the episode does not show.
    """

    source: ClassVar[str] = "judge"

    requirements: tuple[Requirement, ...] = ()
    checks: tuple[Check, ...] | str = ()
    scale: int = 10
    model: str = "openai/gpt-5.6-luna"
    base_url: str = DEFAULT_OPENROUTER_BASE_URL
    api_key_env: str = "OPENROUTER_API_KEY"
    reasoning_effort: JudgeReasoningEffort | None = JudgeReasoningEffort.MEDIUM
    temperature: float | None = None
    max_tokens: int = 8192
    context: str | None = None
    include_reference: bool = True
    include_reasoning: bool = True
    max_view_chars: int = 60_000
    structured_output: bool = True
    request_timeout: float = 120.0
    max_concurrency: int = 16

    def __post_init__(self) -> None:
        super().__post_init__()
        owner = self.owner
        if bool(self.requirements) == bool(self.checks):
            raise ValueError(f"{owner}: list either 'requirements' (a scoring judge) or 'checks' (a veto judge)")
        checks = () if self.checks == ENVIRONMENT_CHECKS else self.checks
        names = [item.name for item in (*self.requirements, *checks)]
        validate_unique_names(f"{owner}: requirement or check", names)
        reserved = sorted(set(names) & {_SCORED_METRIC_LEAF, *JudgeMetric})
        if reserved:
            raise ValueError(
                f"{owner}: {reserved} are the term's own metric keys; name the requirement or check otherwise"
            )
        if self.checks and self.weight > 0:
            raise ValueError(
                f"{owner}: a veto judge never adds reward — its weight prices the fired weighted checks, so it "
                f"must be <= 0 (0 logs them unpriced), got {self.weight}"
            )
        require_text(owner, model=self.model, base_url=self.base_url, api_key_env=self.api_key_env)
        if self.context is not None:
            require_text(owner, context=self.context)
        require_int(owner, ge=1, scale=self.scale, max_tokens=self.max_tokens, max_view_chars=self.max_view_chars)
        if self.checks and differs_from_default(self, "scale"):
            raise ValueError(
                f"{owner}: scale={self.scale} is the range of a scoring judge's requirements, and a veto judge's "
                f"checks fire or do not. Remove it."
            )
        if self.reasoning_effort is not None:
            _coerce_enum(self, "reasoning_effort", JudgeReasoningEffort)
        require_finite(owner, ge=0, **present(temperature=self.temperature))
        require_bool(
            owner,
            include_reference=self.include_reference,
            include_reasoning=self.include_reasoning,
            structured_output=self.structured_output,
        )

    @property
    def is_veto(self) -> bool:
        return bool(self.checks)

    @classmethod
    def from_config(cls, spec: Mapping[str, Any]) -> Self:
        spec = dict(spec)
        what = f"{cls.source} reward term"
        if spec.get("requirements") is not None:
            spec["requirements"] = _items_from_config(Requirement, spec["requirements"], f"{what}: requirements")
        if spec.get("checks") not in (None, ENVIRONMENT_CHECKS):
            spec["checks"] = _items_from_config(Check, spec["checks"], f"{what}: checks")
        if spec.get("checks"):
            # A veto judge prices nothing unless asked, and an outage of one leaves the exact grade standing.
            spec.setdefault("weight", 0.0)
            spec.setdefault("on_error", OnError.NEUTRAL)
        return super().from_config(spec)

    def requirement_fractions(self, scores: Mapping[str, float]) -> dict[str, float]:
        return {
            r.name: min(max(float(scores[r.name]), 0.0), float(self.scale)) / self.scale for r in self.requirements
        }

    def score_from(self, fractions: Mapping[str, float]) -> float:
        """Score mode: the weighted mean of its requirements' fractions."""
        weighted = sum(requirement.weight * fractions[requirement.name] for requirement in self.requirements)
        # Clamped: fractional weights can round a full score to 1 + an ulp, which the term refuses.
        return min(1.0, weighted / sum(requirement.weight for requirement in self.requirements))

    def flag_fraction(self, fired: Mapping[str, bool]) -> float:
        """Veto mode: the weighted fired fraction of its checks that carry weight (0 without any)."""
        total = sum(check.share for check in self.checks)
        if not total:
            return 0.0
        return sum(check.share for check in self.checks if fired.get(check.name)) / total


@dataclass(frozen=True, kw_only=True)
class RewardModelTerm(ScoredTerm):
    """A Bradley-Terry or sequence-classification reward model served by vLLM (``--runner pooling``) or SGLang
    (``--is-embedding``), scored over ``/classify`` at ``url`` (the server root). Rendered with the model's own chat
    template (``tokenizer``, default ``model``); output ``label_index`` maps through a sigmoid of ``(logit -
    logit_shift) / logit_scale``. ``batch_size`` is samples per batch, ``max_concurrency`` batches in flight."""

    source: ClassVar[str] = "reward_model"
    views: ClassVar[tuple[View, ...]] = (View.FINAL, View.FULL)

    url: str
    model: str
    backend: RewardModelBackend = RewardModelBackend.VLLM
    tokenizer: str | None = None
    label_index: int = 0
    logit_shift: float = 0.0
    logit_scale: float = 1.0
    request_timeout: float = 60.0
    batch_size: int = 8
    max_concurrency: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        owner = self.owner
        require_text(owner, url=self.url, model=self.model)
        if self.tokenizer is not None:
            require_text(owner, tokenizer=self.tokenizer)
        _coerce_enum(self, "backend", RewardModelBackend)
        require_int(owner, ge=0, label_index=self.label_index)
        require_finite(owner, logit_shift=self.logit_shift)
        require_finite(owner, gt=0, logit_scale=self.logit_scale)
        require_int(owner, ge=1, batch_size=self.batch_size)

    def normalize(self, logit: float) -> float:
        """A numerically safe logistic of the shifted, scaled logit."""
        z = (logit - self.logit_shift) / self.logit_scale
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        return math.exp(z) / (1.0 + math.exp(z))


def sources_of(*term_types: type[RewardTerm]) -> dict[str, type[RewardTerm]]:
    return {term_type.source: term_type for term_type in term_types}


ENVIRONMENT_REWARD_SOURCES = sources_of(EnvironmentTerm, JudgeTerm, RewardModelTerm)


def read_context_files(raw: Any) -> Any:
    """Read each judge term's ``context_file`` (UTF-8, absolute or relative to the launch directory) into its
    ``context`` at parse time, so what a config forwards to actors on any node carries the text. Anything but a list is
    left for :func:`parse_reward_terms` to refuse."""
    if isinstance(raw, Mapping | str) or not isinstance(raw, Sequence):
        return raw
    resolved = []
    for item in raw:
        if isinstance(item, Mapping) and item.get("source") == JudgeTerm.source and "context_file" in item:
            spec = dict(item)
            path = spec.pop("context_file")
            owner = f"{JudgeTerm.source} reward term {spec.get('name')!r}"
            require_text(owner, context_file=path)
            if spec.get("context") is not None:
                raise ValueError(f"{owner}: set 'context' or 'context_file', not both")
            if not Path(path).is_file():
                raise ValueError(
                    f"{owner}: context_file {path!r} is not a file (a relative path reads from the launch directory, "
                    f"{Path.cwd()})"
                )
            spec["context"] = Path(path).read_text(encoding="utf-8")
            item = spec
        resolved.append(item)
    return resolved


def parse_reward_terms(
    raw: Sequence[Mapping[str, Any] | RewardTerm], sources: Mapping[str, type[RewardTerm]]
) -> tuple[RewardTerm, ...]:
    """Every entry names its ``source``; its other keys are that term type's fields. Already-typed terms pass through
    when admitted. An empty list is refused: the run would train on shaping alone with nothing reported wrong."""
    if isinstance(raw, Mapping | str) or not isinstance(raw, Sequence):
        raise ValueError(f"rewards must be a list of reward terms, got {type(raw).__name__}")
    if not raw:
        raise ValueError(f"rewards must list at least one reward term; available sources: {sorted(sources)}")
    terms: list[RewardTerm] = []
    for index, item in enumerate(raw):
        where = f"rewards[{index}]"
        if isinstance(item, RewardTerm):
            if type(item) not in sources.values():
                raise ValueError(f"{where}: reward source {type(item).source!r} is not available here")
            terms.append(item)
            continue
        if not isinstance(item, Mapping):
            raise ValueError(f"{where}: a reward term is a mapping, got {type(item).__name__}")
        spec = dict(item)
        source = spec.pop("source", None)
        if not isinstance(source, str):
            raise ValueError(f"{where}: 'source' is required, one of {sorted(sources)}")
        term_type = sources.get(source)
        if term_type is None:
            raise ValueError(f"{where}: unknown reward source {source!r}; available: {sorted(sources)}")
        try:
            terms.append(term_type.from_config(spec))
        except (TypeError, ValueError) as e:
            # A missing required field surfaces as the dataclass constructor's TypeError.
            raise ValueError(f"{where} ({source}): {e}") from e
    validate_unique_names("reward term", [term.name for term in terms])
    return tuple(terms)


def resolve_environment_checks(terms: Sequence[RewardTerm], checks: Sequence[Check]) -> tuple[RewardTerm, ...]:
    """With no ``checks``, a ``checks: environment`` judge stays unbound for the composer to refuse."""
    return tuple(
        replace(term, checks=tuple(checks))
        if checks and isinstance(term, JudgeTerm) and term.checks == ENVIRONMENT_CHECKS
        else term
        for term in terms
    )
