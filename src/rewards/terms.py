"""Reward terms: the typed, config-parsed statement of how a completion's scores add up to its reward.

A reward is a sum of terms. Each term names a SOURCE of a score in ``[0, 1]`` — the environment's
own grade, a generative judge, a served reward model, a trainer-local grader — and prices it as
``weight * score ** exponent``. Parsing happens at config time, in the argument dataclasses, so this
module stays free of anything heavier than the endpoint defaults it names.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Any, ClassVar, Self

from src.args.validation import require_finite, require_positive, require_positive_int
from src.inference.endpoints import DEFAULT_OPENROUTER_BASE_URL

# Metric key of a term's contribution, ``reward/<name>``; the keys of one reward sum to the reward.
REWARD_COMPONENT_PREFIX = "reward/"
# The leaf of every scored term's per-sample 1/0 of whether it reached a verdict.
SCORED_METRIC_LEAF = "scored"
# The environment's own grade always logs under this name.
OBJECTIVE_TERM_NAME = "objective"


class View(StrEnum):
    """What a scorer reads of an episode: its final answer, every turn, or a compact digest of every turn."""

    FINAL = "final"
    FULL = "full"
    DIGEST = "digest"


class JudgeMetric(StrEnum):
    """The leaves of a judge's own per-sample metrics beside its requirements' and checks'; with
    :data:`SCORED_METRIC_LEAF` they are the names a requirement or check may not take."""

    COMPLETION_TOKENS = "completion_tokens"
    COST_USD = "cost_usd"
    VETO = "veto"
    UNSUPPORTED_FLAGS = "unsupported_flags"


class OnError(StrEnum):
    """What a verdict the scorer never reached means: the episode leaves the group baseline, or the
    term prices 0 and the episode trains on its other terms."""

    INVALID = "invalid"
    NEUTRAL = "neutral"


class ReasoningEffort(StrEnum):
    """The OpenAI reasoning-effort vocabulary, which OpenRouter forwards to the model it routes to."""

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class RewardModelBackend(StrEnum):
    """The engines whose pooling route a served reward model is scored over."""

    VLLM = "vllm"
    SGLANG = "sglang"


def component_key(name: str) -> str:
    """The metric key of the term named ``name``."""
    return REWARD_COMPONENT_PREFIX + name


def metric_key(term: "RewardTerm", leaf: str) -> str:
    """A per-sample diagnostic key of ``term``, ``<source>/<name>/<leaf>``."""
    return f"{term.source}/{term.name}/{leaf}"


# The environment's own grade, priced by the reward's environment term, in ``reward_components``.
OBJECTIVE_REWARD_KEY = component_key(OBJECTIVE_TERM_NAME)


def _require_text(owner: str, **values: Any) -> None:
    for key, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{owner}: {key} must be a non-blank string, got {value!r}")


def _require_bool(owner: str, **values: Any) -> None:
    # YAML 1.2 reads ``no``/``off`` as strings, which would pass a truthiness check as True.
    for key, value in values.items():
        if not isinstance(value, bool):
            raise ValueError(f"{owner}: {key} must be true or false, got {value!r}")


def _coerce_enum(term: "RewardTerm", name: str, enum_type: type[StrEnum]) -> None:
    """Read the field ``name`` as its enum (a config spells it as text); an unknown spelling raises
    naming the admitted ones."""
    value = getattr(term, name)
    try:
        member = enum_type(value)
    except ValueError:
        admitted = tuple(member.value for member in enum_type)
        raise ValueError(f"{term.owner}: {name} must be one of {admitted}, got {value!r}") from None
    object.__setattr__(term, name, member)


def _require_name(what: str, name: Any) -> None:
    """A term, requirement or check name: its diagnostic key, so non-blank, unpadded and free of ``/``."""
    if not isinstance(name, str) or not name or name != name.strip() or "/" in name:
        raise ValueError(f"{what} name must be a non-blank string without '/', got {name!r}")


def require_unique_names(what: str, names: list[str]) -> None:
    if len(set(names)) != len(names):
        raise ValueError(f"{what} names must be unique, got {names}")


def refuse_veto_judges(terms: Sequence["RewardTerm"], *, where: str) -> None:
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
    """The typed items of a config list (``requirements``, ``checks``): each a mapping of the item's fields."""
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
    """One additive term of a reward: ``weight * score ** exponent`` over a score in ``[0, 1]``.

    ``name`` is the term's component key (``reward/<name>``); :attr:`source` is the spelling a config
    selects the term type by. A negative ``weight`` makes the term a penalty. An ``exponent`` above 1
    makes credit convex (half a score earns under half the weight), below 1 concave.
    """

    source: ClassVar[str]

    name: str
    weight: float = 1.0
    exponent: float = 1.0

    def __post_init__(self) -> None:
        _require_name("reward term", self.name)
        require_finite(self.owner, weight=self.weight)
        require_positive(self.owner, exponent=self.exponent)

    @property
    def owner(self) -> str:
        """How a diagnostic names this term."""
        return f"{self.source} reward term {self.name!r}"

    def shape(self, score: float) -> float:
        """``score ** exponent``: the shaped score before the weight, what a TRL reward function returns."""
        if isinstance(score, bool) or not isinstance(score, int | float) or not (0.0 <= score <= 1.0):
            raise ValueError(f"{self.owner}: a score must lie in [0, 1], got {score!r}")
        return float(score) ** self.exponent

    def price(self, score: float) -> float:
        """The term's contribution to the reward, ``weight * score ** exponent``."""
        return self.weight * self.shape(score)

    @classmethod
    def from_config(cls, spec: Mapping[str, Any]) -> Self:
        """The term a config mapping (its ``source`` key already consumed) describes."""
        return _construct(cls, spec, f"{cls.source} reward term")


@dataclass(frozen=True, kw_only=True)
class EnvironmentTerm(RewardTerm):
    """The environment's own grade — a solve on every hidden test, an answer match — in ``[0, 1]``."""

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
    """A term an external backend scores after the episode: what it reads of the episode (``view``,
    one of the class's :attr:`views`) and what a verdict it never got means (``on_error``:
    ``invalid`` drops the episode from the group baseline, ``neutral`` prices the term at 0 and keeps
    the episode valid). ``request_timeout`` bounds one request, ``max_concurrency`` the requests in
    flight per scorer instance."""

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
        require_positive(self.owner, request_timeout=self.request_timeout)
        require_positive_int(self.owner, max_concurrency=self.max_concurrency)


@dataclass(frozen=True, kw_only=True)
class Requirement:
    """One thing a scoring judge grades: a short name (its diagnostic key) and what a full score means."""

    name: str
    description: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        _require_name("requirement", self.name)
        _require_text(f"requirement {self.name!r}", description=self.description)
        require_positive(f"requirement {self.name!r}", weight=self.weight)


@dataclass(frozen=True, kw_only=True)
class Check:
    """One binary flag a veto judge raises on an observable action: a short name (its diagnostic key)
    and what firing means. A ``veto`` check gates the objective — a solve it fires on scores 0; every
    other check is a process flag, counted into the term's score."""

    name: str
    description: str
    veto: bool = False

    def __post_init__(self) -> None:
        _require_name("check", self.name)
        _require_text(f"check {self.name!r}", description=self.description)
        _require_bool(f"check {self.name!r}", veto=self.veto)


@dataclass(frozen=True, kw_only=True)
class JudgeTerm(ScoredTerm):
    """A generative judge: an OpenAI-compatible chat model reads the task and the episode and answers
    with one JSON verdict. The term runs in one of two modes, decided by what it lists:

    - ``requirements`` (score mode): each is scored from 0 to ``scale``; the term's score is their
      weight-averaged fraction of the scale, priced as every term is.
    - ``checks`` (veto mode): each either fires, with a verbatim quote of the span that shows it, or
      does not. A fired ``veto`` check zeroes the objective component; the term's own score is the
      fired fraction of the other checks, so its ``weight`` (0 by default when parsed from a config,
      never positive) prices process flags without ever adding reward. A verdict the judge never
      reached defaults, from a config, to ``on_error: neutral``: no flag, and the exact grade stands.

    ``view`` selects what the judge reads: the final answer (``final``), every turn with its
    reasoning, tool calls and results (``full``), or a compact digest of every turn (``digest``), each
    cut to ``max_view_chars`` with the end kept. ``include_reference`` shows the row's reference answer
    when the sample carries one; ``include_reasoning`` shows the policy's reasoning in the ``full`` and
    ``digest`` views. The API key comes from the ``api_key_env`` variable, then the hosted chain
    (``OPENROUTER_API_KEY``, ``OPENAI_API_KEY``). ``reasoning_effort`` ``None`` sends no such field;
    ``temperature`` ``None`` keeps the served default, which reasoning models require.
    ``structured_output`` asks for the reply through a strict JSON schema; off, the reply is parsed as
    the first JSON object it contains.
    """

    source: ClassVar[str] = "judge"

    requirements: tuple[Requirement, ...] = ()
    checks: tuple[Check, ...] = ()
    scale: int = 10
    model: str = "openai/gpt-5.6-luna"
    base_url: str = DEFAULT_OPENROUTER_BASE_URL
    api_key_env: str = "OPENROUTER_API_KEY"
    reasoning_effort: ReasoningEffort | None = ReasoningEffort.MEDIUM
    temperature: float | None = None
    max_tokens: int = 8192
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
        names = [item.name for item in (*self.requirements, *self.checks)]
        require_unique_names(f"{owner}: requirement or check", names)
        reserved = sorted(set(names) & {SCORED_METRIC_LEAF, *JudgeMetric})
        if reserved:
            raise ValueError(
                f"{owner}: {reserved} are the term's own metric keys; name the requirement or check otherwise"
            )
        if self.checks and self.weight > 0:
            raise ValueError(
                f"{owner}: a veto judge never adds reward — its weight prices the fired process flags, so it "
                f"must be <= 0 (0 logs them unpriced), got {self.weight}"
            )
        _require_text(owner, model=self.model, base_url=self.base_url, api_key_env=self.api_key_env)
        require_positive_int(owner, scale=self.scale, max_tokens=self.max_tokens, max_view_chars=self.max_view_chars)
        if self.reasoning_effort is not None:
            _coerce_enum(self, "reasoning_effort", ReasoningEffort)
        if self.temperature is not None:
            require_finite(owner, temperature=self.temperature)
            if self.temperature < 0:
                raise ValueError(f"{owner}: temperature must be >= 0, got {self.temperature}")
        _require_bool(
            owner,
            include_reference=self.include_reference,
            include_reasoning=self.include_reasoning,
            structured_output=self.structured_output,
        )

    @property
    def is_veto(self) -> bool:
        """A veto judge lists checks; a scoring judge lists requirements."""
        return bool(self.checks)

    @classmethod
    def from_config(cls, spec: Mapping[str, Any]) -> Self:
        spec = dict(spec)
        what = f"{cls.source} reward term"
        if spec.get("requirements") is not None:
            spec["requirements"] = _items_from_config(Requirement, spec["requirements"], f"{what}: requirements")
        if spec.get("checks") is not None:
            spec["checks"] = _items_from_config(Check, spec["checks"], f"{what}: checks")
        if spec.get("checks"):
            # A veto judge prices nothing unless asked, and an outage of one leaves the exact grade standing.
            spec.setdefault("weight", 0.0)
            spec.setdefault("on_error", OnError.NEUTRAL)
        return super().from_config(spec)

    def requirement_fractions(self, scores: Mapping[str, float]) -> dict[str, float]:
        """Each requirement's score on ``[0, scale]``, clamped, as a fraction of the scale."""
        return {
            r.name: min(max(float(scores[r.name]), 0.0), float(self.scale)) / self.scale for r in self.requirements
        }

    def score_from(self, fractions: Mapping[str, float]) -> float:
        """The score-mode term's score: the weighted mean of its requirements' fractions."""
        weighted = sum(requirement.weight * fractions[requirement.name] for requirement in self.requirements)
        # Clamped: fractional weights can round a full score to 1 + an ulp, which the term refuses.
        return min(1.0, weighted / sum(requirement.weight for requirement in self.requirements))

    def flag_fraction(self, fired: Mapping[str, bool]) -> float:
        """The veto-mode term's score: the fired fraction of its process checks, the ones that flag a fault
        without gating the objective (0 without any)."""
        checks = [check for check in self.checks if not check.veto]
        if not checks:
            return 0.0
        return sum(1.0 for check in checks if fired.get(check.name)) / len(checks)


@dataclass(frozen=True, kw_only=True)
class RewardModelTerm(ScoredTerm):
    """A Bradley-Terry or sequence-classification reward model served by vLLM (``--runner pooling``)
    or SGLang (``--is-embedding``), scored over its ``/classify`` route at ``url`` (the server root).

    The scorer renders the conversation with the model's own chat template (``tokenizer``, default
    ``model``) — the prompt plus the final answer (``view: final``) or every turn (``full``) — reads
    output ``label_index`` of the head and maps the logit through
    ``sigmoid((logit - logit_shift) / logit_scale)`` into ``[0, 1]``. Samples travel in batches of
    ``batch_size``, ``max_concurrency`` batches in flight.
    """

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
        _require_text(owner, url=self.url, model=self.model)
        if self.tokenizer is not None:
            _require_text(owner, tokenizer=self.tokenizer)
        _coerce_enum(self, "backend", RewardModelBackend)
        if isinstance(self.label_index, bool) or not isinstance(self.label_index, int) or self.label_index < 0:
            raise ValueError(f"{owner}: label_index must be an integer >= 0, got {self.label_index!r}")
        require_finite(owner, logit_shift=self.logit_shift)
        require_positive(owner, logit_scale=self.logit_scale)
        require_positive_int(owner, batch_size=self.batch_size)

    def normalize(self, logit: float) -> float:
        """The score of a head output: a numerically safe logistic of the shifted, scaled logit."""
        z = (logit - self.logit_shift) / self.logit_scale
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        return math.exp(z) / (1.0 + math.exp(z))


def sources_of(*term_types: type[RewardTerm]) -> dict[str, type[RewardTerm]]:
    """The ``source`` → term-type table an arm admits, from the types it names."""
    return {term_type.source: term_type for term_type in term_types}


# The sources an environment's reward admits: its own grade plus the externally scored terms.
ENVIRONMENT_REWARD_SOURCES = sources_of(EnvironmentTerm, JudgeTerm, RewardModelTerm)


def parse_reward_terms(
    raw: Sequence[Mapping[str, Any] | RewardTerm], sources: Mapping[str, type[RewardTerm]]
) -> tuple[RewardTerm, ...]:
    """The typed terms of a ``rewards:`` list. Every entry names its ``source`` (a key of ``sources``);
    its other keys are that term type's fields. Term names must be unique. Already-typed terms pass
    through when their type is admitted. An empty list is refused: the reward would carry no
    objective, and the run would train on shaping alone with nothing reported wrong."""
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
    require_unique_names("reward term", [term.name for term in terms])
    return tuple(terms)
