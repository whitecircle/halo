"""Reward terms: the typed, config-parsed statement of how a completion's scores add up to its reward.

A reward is a sum of terms. Each term names a SOURCE of a score in ``[0, 1]`` — the environment's
own grade, a generative judge, a served reward model, a trainer-local grader — and prices it as
``weight * score ** exponent``. Parsing happens at config time, in the argument dataclasses, so this
module stays free of anything heavier than the client defaults it names.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, ClassVar

from src.inference.openai_client import DEFAULT_OPENROUTER_BASE_URL

# Metric key of a term's contribution, ``reward/<name>``; the keys of one reward sum to the reward.
REWARD_COMPONENT_PREFIX = "reward/"
# The environment's own grade always logs under this name: component advantage shaping reads it.
OBJECTIVE_TERM_NAME = "objective"

DEFAULT_JUDGE_MODEL = "openai/gpt-5.6-luna"
DEFAULT_JUDGE_REASONING_EFFORT = "medium"
DEFAULT_JUDGE_API_KEY_ENV = "OPENROUTER_API_KEY"
# The OpenAI reasoning-effort vocabulary, which OpenRouter forwards to the model it routes to.
JUDGE_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
REWARD_MODEL_BACKENDS = ("vllm", "sglang")
# What a scorer reads of the policy's output: its final assistant message, or every turn.
TRANSCRIPT_VIEWS = ("final", "full")


def component_key(name: str) -> str:
    """The metric key of the term named ``name``."""
    return REWARD_COMPONENT_PREFIX + name


def _require_finite(owner: str, **values: Any) -> None:
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError(f"{owner}: {key} must be a finite number, got {value!r}")


def _require_positive_int(owner: str, **values: Any) -> None:
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{owner}: {key} must be an integer >= 1, got {value!r}")


def _require_text(owner: str, **values: Any) -> None:
    for key, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{owner}: {key} must be a non-blank string, got {value!r}")


def _construct(cls: type, spec: Mapping[str, Any], what: str):
    """``cls(**spec)`` with an unknown key named up front, before the constructor's own checks run."""
    admitted = {f.name for f in fields(cls)}
    unknown = sorted(set(spec) - admitted)
    if unknown:
        raise ValueError(f"{what}: unknown option(s) {unknown}; admitted: {sorted(admitted)}")
    return cls(**spec)


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
        if not isinstance(self.name, str) or not self.name or self.name != self.name.strip() or "/" in self.name:
            raise ValueError(f"reward term name must be a non-blank string without '/', got {self.name!r}")
        _require_finite(f"reward term {self.name!r}", weight=self.weight, exponent=self.exponent)
        if self.exponent <= 0:
            raise ValueError(f"reward term {self.name!r}: exponent must be > 0, got {self.exponent}")

    @property
    def key(self) -> str:
        """The component key, ``reward/<name>``."""
        return component_key(self.name)

    def shape(self, score: float) -> float:
        """``score ** exponent``: the shaped score before the weight, what a TRL reward function returns."""
        if isinstance(score, bool) or not isinstance(score, int | float) or not (0.0 <= score <= 1.0):
            raise ValueError(f"reward term {self.name!r}: a score must lie in [0, 1], got {score!r}")
        return float(score) ** self.exponent

    def price(self, score: float) -> float:
        """The term's contribution to the reward, ``weight * score ** exponent``."""
        return self.weight * self.shape(score)

    @classmethod
    def from_config(cls, spec: Mapping[str, Any]) -> "RewardTerm":
        """The term a config mapping (its ``source`` key already consumed) describes."""
        return _construct(cls, spec, f"{cls.source} reward term")


@dataclass(frozen=True, kw_only=True)
class EnvironmentTerm(RewardTerm):
    """The environment's own grade — pass fraction, answer match, judge adherence — in ``[0, 1]``."""

    source: ClassVar[str] = "environment"

    name: str = OBJECTIVE_TERM_NAME

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.name != OBJECTIVE_TERM_NAME:
            raise ValueError(
                f"the environment term is always named {OBJECTIVE_TERM_NAME!r} (its component is what "
                f"advantage shaping reads), got {self.name!r}"
            )


@dataclass(frozen=True, kw_only=True)
class Requirement:
    """One thing a judge scores: a short name (its diagnostic key) and what a full score means."""

    name: str
    description: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name != self.name.strip() or "/" in self.name:
            raise ValueError(f"requirement name must be a non-blank string without '/', got {self.name!r}")
        _require_text(f"requirement {self.name!r}", description=self.description)
        _require_finite(f"requirement {self.name!r}", weight=self.weight)
        if self.weight <= 0:
            raise ValueError(f"requirement {self.name!r}: weight must be > 0, got {self.weight}")


@dataclass(frozen=True, kw_only=True)
class JudgeTerm(RewardTerm):
    """A generative judge: an OpenAI-compatible chat model scores the response against every
    requirement from 0 to ``scale``, and the term's score is the weight-averaged fraction of the scale.

    ``instructions`` is grading guidance appended to the rubric. ``transcript`` selects what the judge
    reads: the final assistant message (``final``) or every turn with its tool calls and results
    (``full``), cut at ``max_transcript_chars``. ``include_reference`` shows the row's reference
    answer when the sample carries one. The API key comes from the ``api_key_env`` variable, then the
    hosted chain (``OPENROUTER_API_KEY``, ``OPENAI_API_KEY``). ``reasoning_effort`` ``None`` sends no
    such field; ``temperature`` ``None`` keeps the served default, which reasoning models require.
    ``structured_output`` asks for the reply through a strict JSON schema; off, the reply is parsed
    as the first JSON object it contains.
    """

    source: ClassVar[str] = "judge"

    requirements: tuple[Requirement, ...]
    instructions: str | None = None
    scale: int = 10
    model: str = DEFAULT_JUDGE_MODEL
    base_url: str = DEFAULT_OPENROUTER_BASE_URL
    api_key_env: str = DEFAULT_JUDGE_API_KEY_ENV
    reasoning_effort: str | None = DEFAULT_JUDGE_REASONING_EFFORT
    temperature: float | None = None
    max_tokens: int = 8192
    transcript: str = "final"
    include_reference: bool = True
    max_transcript_chars: int = 60_000
    structured_output: bool = True
    request_timeout: float = 120.0
    max_concurrency: int = 16

    def __post_init__(self) -> None:
        super().__post_init__()
        owner = f"judge term {self.name!r}"
        if not self.requirements:
            raise ValueError(f"{owner}: 'requirements' must list at least one requirement")
        names = [requirement.name for requirement in self.requirements]
        if len(set(names)) != len(names):
            raise ValueError(f"{owner}: requirement names must be unique, got {names}")
        _require_text(owner, model=self.model, base_url=self.base_url, api_key_env=self.api_key_env)
        _require_positive_int(
            owner,
            scale=self.scale,
            max_tokens=self.max_tokens,
            max_transcript_chars=self.max_transcript_chars,
            max_concurrency=self.max_concurrency,
        )
        if self.reasoning_effort is not None and self.reasoning_effort not in JUDGE_REASONING_EFFORTS:
            raise ValueError(f"{owner}: reasoning_effort must be one of {JUDGE_REASONING_EFFORTS} or null")
        if self.temperature is not None:
            _require_finite(owner, temperature=self.temperature)
            if self.temperature < 0:
                raise ValueError(f"{owner}: temperature must be >= 0, got {self.temperature}")
        if self.transcript not in TRANSCRIPT_VIEWS:
            raise ValueError(f"{owner}: transcript must be one of {TRANSCRIPT_VIEWS}, got {self.transcript!r}")
        # YAML 1.2 reads ``no``/``off`` as strings, which would pass a truthiness check as True.
        for key in ("include_reference", "structured_output"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{owner}: {key} must be true or false, got {getattr(self, key)!r}")
        _require_finite(owner, request_timeout=self.request_timeout)
        if self.request_timeout <= 0:
            raise ValueError(f"{owner}: request_timeout must be > 0, got {self.request_timeout}")

    @classmethod
    def from_config(cls, spec: Mapping[str, Any]) -> "JudgeTerm":
        spec = dict(spec)
        raw = spec.get("requirements")
        if raw is not None:
            if isinstance(raw, str | bytes | Mapping) or not isinstance(raw, Sequence):
                raise ValueError("judge reward term: 'requirements' must be a list of {name, description} mappings")
            spec["requirements"] = tuple(
                item if isinstance(item, Requirement) else _requirement_from_config(item, index)
                for index, item in enumerate(raw)
            )
        return super().from_config(spec)

    def score_from(self, scores: Mapping[str, float]) -> float:
        """The term's score from per-requirement scores on ``[0, scale]`` (clamped): their weighted mean
        as a fraction of the scale."""
        weighted = sum(
            requirement.weight * min(max(float(scores[requirement.name]), 0.0), float(self.scale))
            for requirement in self.requirements
        )
        # Clamped: fractional weights can round a full score to 1 + an ulp, which the term refuses.
        return min(1.0, weighted / (self.scale * sum(requirement.weight for requirement in self.requirements)))


def _requirement_from_config(raw: Any, index: int) -> Requirement:
    if not isinstance(raw, Mapping):
        raise ValueError(f"requirements[{index}]: a requirement is a mapping with 'name' and 'description'")
    return _construct(Requirement, raw, f"requirements[{index}]")


@dataclass(frozen=True, kw_only=True)
class RewardModelTerm(RewardTerm):
    """A Bradley-Terry or sequence-classification reward model served by vLLM (``--runner pooling``)
    or SGLang (``--is-embedding``), scored over its ``/classify`` route at ``url`` (the server root).

    The scorer renders the conversation with the model's own chat template (``tokenizer``, default
    ``model``), reads output ``label_index`` of the head and maps the logit through
    ``sigmoid((logit - logit_shift) / logit_scale)`` into ``[0, 1]``. Samples travel in batches of
    ``batch_size``, ``max_concurrency`` batches in flight.
    """

    source: ClassVar[str] = "reward_model"

    url: str
    model: str
    backend: str = "vllm"
    tokenizer: str | None = None
    label_index: int = 0
    logit_shift: float = 0.0
    logit_scale: float = 1.0
    transcript: str = "final"
    request_timeout: float = 60.0
    batch_size: int = 8
    max_concurrency: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        owner = f"reward_model term {self.name!r}"
        _require_text(owner, url=self.url, model=self.model)
        if self.tokenizer is not None:
            _require_text(owner, tokenizer=self.tokenizer)
        if self.backend not in REWARD_MODEL_BACKENDS:
            raise ValueError(f"{owner}: backend must be one of {REWARD_MODEL_BACKENDS}, got {self.backend!r}")
        if isinstance(self.label_index, bool) or not isinstance(self.label_index, int) or self.label_index < 0:
            raise ValueError(f"{owner}: label_index must be an integer >= 0, got {self.label_index!r}")
        _require_finite(
            owner, logit_shift=self.logit_shift, logit_scale=self.logit_scale, request_timeout=self.request_timeout
        )
        if self.logit_scale <= 0 or self.request_timeout <= 0:
            raise ValueError(f"{owner}: logit_scale and request_timeout must be > 0")
        if self.transcript not in TRANSCRIPT_VIEWS:
            raise ValueError(f"{owner}: transcript must be one of {TRANSCRIPT_VIEWS}, got {self.transcript!r}")
        _require_positive_int(owner, batch_size=self.batch_size, max_concurrency=self.max_concurrency)

    @property
    def server_root(self) -> str:
        return self.url.rstrip("/")

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
    through when their type is admitted."""
    if isinstance(raw, Mapping | str) or not isinstance(raw, Sequence):
        raise ValueError(f"rewards must be a list of reward terms, got {type(raw).__name__}")
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
        except TypeError as e:
            # A missing required field surfaces as the dataclass constructor's TypeError.
            raise ValueError(f"{where} ({source}): {e}") from e
        except ValueError as e:
            raise ValueError(f"{where} ({source}): {e}") from e
    names = [term.name for term in terms]
    if len(set(names)) != len(names):
        raise ValueError(f"reward term names must be unique, got {names}")
    return tuple(terms)
