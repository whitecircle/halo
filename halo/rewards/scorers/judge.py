"""A generative judge: a chat model reads the task and one view of the episode and answers one JSON verdict, with
integer scores per requirement or, per check, whether it fires and the lines of the policy's actions that show it."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import urlparse

from openai import AsyncOpenAI

from halo.config_validation import is_finite_number
from halo.env import env_str
from halo.inference.endpoints import HOSTED_API_KEY_VARIABLES, is_loopback, resolve_local_api_key
from halo.inference.openai_client import build_openai_client, chat_completion
from halo.inference.response import get_finish_reason
from halo.rewards.samples import (
    VIEW_TAGS,
    ScoringSample,
    cut_middle,
    escape_tags,
    render_actions,
    render_tools,
    shows_reasoning,
    system_text,
    task_text,
    view_text,
)
from halo.rewards.scorers.base import Scorer, ScoreResult
from halo.rewards.terms import JudgeMetric, JudgeTerm

SYSTEM_PROMPT = (
    "You are a strict, impartial grader of a policy model's episode. Judge only what the episode "
    "shows: the policy's actions, its answers and the tool results it received. Its reasoning, where "
    "shown, is context for understanding what it did, not something to grade. Reply with the requested "
    "JSON object and nothing else."
)
# A quote longer than this is a copy of the response, not evidence of one span in it.
MAX_EVIDENCE_CHARS = 400
_REPLY_EXCERPT_CHARS = 200

# The lines a fired check may quote: one found among them supports it. A judge misquotes a line of a long program
# about one time in four, and a second and third line make losing a true flag to a slip rare.
MAX_EVIDENCE_QUOTES = 3
# A judge copying a long program's line slips a word now and then; a quote is still evidence when all but this share of
# its words (one at least) run in order at one place, case aside. Below the minimum it must match whole.
_EVIDENCE_SLIP_SHARE = 0.2
_EVIDENCE_MIN_SLIP_WORDS = 4

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Verdict:
    """Per-requirement scores (score mode) or per-check ``(fired, evidence)`` (veto mode), and the rationale."""

    scores: dict[str, float]
    checks: dict[str, tuple[bool, tuple[str, ...]]]
    rationale: str | None


def scorer_api_key(term: JudgeTerm) -> str:
    """A variable the term names is that key or a refusal, since a fallback would send another vendor's key to the
    term's endpoint. At the default name a loopback server takes the local key, which needs none, a hosted vendor's own
    host that vendor's key, and any other endpoint is refused."""
    host = urlparse(term.base_url).hostname or ""
    if term.api_key_env != JudgeTerm.api_key_env:
        variable = term.api_key_env
    elif is_loopback(host):
        return resolve_local_api_key()
    else:
        variable = HOSTED_API_KEY_VARIABLES.get(host)
        if variable is None:
            raise RuntimeError(
                f"{term.owner}: base_url {term.base_url} is no hosted vendor's own host, so no default key goes to "
                f"it; set api_key_env to the variable holding this endpoint's key"
            )
    key = env_str(variable)
    if not key:
        raise RuntimeError(f"{term.owner}: no API key — set {variable}")
    return key


def _object_of(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def response_schema(term: JudgeTerm) -> dict[str, Any]:
    if term.is_veto:
        check = _object_of({"fired": {"type": "boolean"}, "evidence": {"type": "array", "items": {"type": "string"}}})
        verdicts = {"checks": _object_of({check_.name: check for check_ in term.checks})}
    else:
        verdicts = {"scores": _object_of({requirement.name: {"type": "integer"} for requirement in term.requirements})}
    return _object_of({**verdicts, "rationale": {"type": "string"}})


def _tagged(tag: str, text: str) -> str:
    return f"<{tag}>\n{text}\n</{tag}>"


def grading_prompt(term: JudgeTerm, sample: ScoringSample) -> str:
    response = view_text(sample, term.view, include_reasoning=term.include_reasoning, max_chars=term.max_view_chars)
    # The setting is the term's own prompt text; every other part quotes text the prompt does not control, escaped
    # (the view escapes its own around the reasoning blocks it sets apart).
    parts = [_tagged("setting", term.context.strip())] if term.context else []
    instructions = system_text(sample.prompt)
    if instructions:
        parts.append(_tagged("policy_instructions", escape_tags(instructions)))
    parts.append(_tagged("task", escape_tags(task_text(sample.prompt)) or "(no task text)"))
    if sample.tools:
        parts.append(_tagged("policy_tools", escape_tags(render_tools(sample.tools))))
    if term.include_reference and sample.reference is not None:
        reference = (
            sample.reference if isinstance(sample.reference, str) else json.dumps(sample.reference, ensure_ascii=False)
        )
        parts.append(_tagged("reference_answer", cut_middle(escape_tags(reference), term.max_view_chars)))
    view = VIEW_TAGS[term.view]
    parts.append(_tagged(view, response or "(empty response)"))
    if term.is_veto:
        rubric = "\n".join(f"{i}. {check.name}: {check.description}" for i, check in enumerate(term.checks, 1))
        actions = (
            "its visible text, a tool call's arguments (a call the engine cut before it ran included) or a tool result"
        )
        if shows_reasoning(sample, term.view, include_reasoning=term.include_reasoning):
            evidence = (
                "Read the reasoning to understand what the policy did and why, but a check fires only on evidence you "
                f"can quote verbatim from the policy's actions in it — {actions}, never its reasoning, which is not an "
                "action."
            )
        else:
            evidence = (
                f"A check fires only on evidence you can quote verbatim from the policy's actions in it — {actions}."
            )
        parts.append(
            f"# Checks\nFor each check, decide whether the episode in <{view}> shows it. {evidence} Give as its evidence "
            f"one to {MAX_EVIDENCE_QUOTES} lines of that text, each copied character for character — not paraphrased, "
            "shortened or described — or an empty list when it does not fire.\n" + rubric
        )
        keys = ", ".join(
            f'"{check.name}": {{"fired": <true|false>, "evidence": ["<a line copied exactly>", ...]}}'
            for check in term.checks
        )
        parts.append(f'Reply with one JSON object: {{"checks": {{{keys}}}, "rationale": "<one or two sentences>"}}')
    else:
        rubric = "\n".join(f"{i}. {r.name}: {r.description}" for i, r in enumerate(term.requirements, 1))
        parts.append(
            f"# Requirements\nScore the episode in <{view}> on each requirement from 0 (not met) to {term.scale} "
            f"(fully met).\n{rubric}"
        )
        keys = ", ".join(f'"{requirement.name}": <integer>' for requirement in term.requirements)
        parts.append(f'Reply with one JSON object: {{"scores": {{{keys}}}, "rationale": "<one or two sentences>"}}')
    return "\n\n".join(parts)


def action_text(term: JudgeTerm, sample: ScoringSample) -> str:
    """Evidence must come from the policy's actions and the results it received, never its reasoning: a check raised on
    a thought teaches the policy to hide its reasoning rather than act well."""
    return render_actions(sample, term.view, max_chars=term.max_view_chars)


def _parse_json_object(content: str) -> Any | None:
    """The JSON value ``content`` is, or the first object it embeds in prose; ``None`` when neither parses."""
    try:
        return json.loads(content)
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    start = content.find("{")
    while start != -1:
        try:
            return decoder.raw_decode(content, start)[0]
        except ValueError:
            start = content.find("{", start + 1)
    return None


def parse_verdict(content: str, term: JudgeTerm) -> Verdict | None:
    """``None`` when the reply holds no JSON object answering every requirement with a finite number (score mode) or
    every check with a ``fired`` boolean (veto mode)."""
    payload = _parse_json_object(content)
    if not isinstance(payload, Mapping):
        return None
    rationale = payload.get("rationale")
    rationale = rationale if isinstance(rationale, str) else None
    if term.is_veto:
        checks = payload.get("checks")
        if not isinstance(checks, Mapping):
            return None
        parsed_checks: dict[str, tuple[bool, tuple[str, ...]]] = {}
        for check in term.checks:
            entry = checks.get(check.name)
            fired = entry.get("fired") if isinstance(entry, Mapping) else entry
            if not isinstance(fired, bool):
                return None
            evidence = entry.get("evidence") if isinstance(entry, Mapping) else None
            # A judge off the schema may answer one quote as a plain string.
            quotes = [evidence] if isinstance(evidence, str) else evidence if isinstance(evidence, list) else []
            quotes = tuple(quote for quote in quotes if isinstance(quote, str) and quote.strip())
            parsed_checks[check.name] = (fired, quotes[:MAX_EVIDENCE_QUOTES])
        return Verdict({}, parsed_checks, rationale)
    scores = payload.get("scores")
    if not isinstance(scores, Mapping):
        return None
    parsed: dict[str, float] = {}
    for requirement in term.requirements:
        value = scores.get(requirement.name)
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return None
        # JSON admits NaN and Infinity, which a clamp reads as full credit.
        if not is_finite_number(value):
            return None
        parsed[requirement.name] = float(value)
    return Verdict(parsed, {}, rationale)


def evidence_supported(evidence: str, text: str) -> bool:
    """Whether ``evidence`` is a span of ``text``: whitespace folded since a model reflows what it quotes, case aside
    with a slipped word or so (:data:`_EVIDENCE_SLIP_SHARE`), and short enough to be a span rather than a copy."""
    quote = _WHITESPACE.sub(" ", escape_tags(evidence)).strip()
    if not 0 < len(quote) <= MAX_EVIDENCE_CHARS:
        return False
    flat = _WHITESPACE.sub(" ", escape_tags(text))
    if quote in flat:
        return True
    words, haystack = quote.lower().split(" "), flat.lower().split(" ")
    if len(words) < _EVIDENCE_MIN_SLIP_WORDS:
        return False
    anchor = SequenceMatcher(None, haystack, words, autojunk=False).find_longest_match(0, len(haystack), 0, len(words))
    if not anchor.size:
        return False
    # The words around the longest run, with room for the words a slip added or dropped.
    slack = max(2, len(words) // 10)
    start = max(0, anchor.a - anchor.b - slack)
    window = haystack[start : anchor.a - anchor.b + len(words) + slack]
    blocks = [
        block for block in SequenceMatcher(None, window, words, autojunk=False).get_matching_blocks() if block.size
    ]
    slips = max(1, int(_EVIDENCE_SLIP_SHARE * len(words)))
    # The matched words must also sit together, and a slip falls inside the quote: the same words strewn among
    # others are not the line quoted, and words added at either end may be the reasoning's, not the action's.
    spread = blocks[-1].a + blocks[-1].size - blocks[0].a
    anchored = blocks[0].b == 0 and blocks[-1].b + blocks[-1].size == len(words)
    return anchored and len(words) - sum(block.size for block in blocks) <= slips and spread <= len(words) + slips


class GenerativeJudge(Scorer):
    """One lazily built client per instance (per Ray actor or trainer rank), shared by every sample."""

    term_type = JudgeTerm
    term: JudgeTerm
    _client: AsyncOpenAI | None

    @property
    def metric_keys(self) -> tuple[str, ...]:
        term = self.term
        if term.is_veto:
            keys = [
                *(self._key(check.name) for check in term.checks),
                self._key(JudgeMetric.VETO),
                self._key(JudgeMetric.UNSUPPORTED_FLAGS),
                self._key(JudgeMetric.EXEMPTED_FLAGS),
            ]
        else:
            keys = [self._key(requirement.name) for requirement in term.requirements]
        return (*keys, self._key(JudgeMetric.COMPLETION_TOKENS))

    def _connect(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = build_openai_client(base_url=self.term.base_url, api_key_override=scorer_api_key(self.term))
        return self._client

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.close()

    def _usage_metrics(self, completion) -> dict[str, float]:
        metrics: dict[str, float] = {}
        usage = getattr(completion, "usage", None)
        if usage is not None:
            metrics[self._key(JudgeMetric.COMPLETION_TOKENS)] = float(getattr(usage, "completion_tokens", 0) or 0)
            cost = getattr(usage, "cost", None)
            if isinstance(cost, int | float):
                metrics[self._key(JudgeMetric.COST_USD)] = float(cost)
        return metrics

    def _request(self, prompt: str) -> dict[str, Any]:
        term = self.term
        request: dict[str, Any] = {
            "model": term.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            "max_completion_tokens": term.max_tokens,
            "timeout": term.request_timeout,
        }
        if term.reasoning_effort is not None:
            request["reasoning_effort"] = term.reasoning_effort
        if term.temperature is not None:
            request["temperature"] = term.temperature
        if term.structured_output:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "strict": True, "schema": response_schema(term)},
            }
        return request

    async def score_one(self, sample: ScoringSample) -> ScoreResult:
        term = self.term
        # Building the prompt serializes the row's reference; a raise here would escape every guard up
        # to the actor's catch-all and mask the whole episode, so it books as a verdict-less result.
        try:
            request = self._request(grading_prompt(term, sample))
        except Exception as e:
            return ScoreResult(None, error=self._failed("prompt build", e))
        try:
            completion = await chat_completion(self._connect(), **request)
        except Exception as e:
            return ScoreResult(None, error=self._failed("request", e))
        choice = completion.choices[0]
        content = choice.message.content or ""
        verdict = parse_verdict(content, term)
        if verdict is None:
            excerpt = content[:_REPLY_EXCERPT_CHARS]
            return ScoreResult(
                None, error=f"unparseable judge reply (finish_reason={get_finish_reason(choice)!r}): {excerpt!r}"
            )
        metrics = self._usage_metrics(completion)
        if term.is_veto:
            return self._veto_result(verdict, action_text(term, sample), metrics, sample.facts)
        fractions = term.requirement_fractions(verdict.scores)
        metrics.update({self._key(name): fraction for name, fraction in fractions.items()})
        return ScoreResult(term.score_from(fractions), metrics, detail=verdict.rationale)

    def _veto_result(
        self, verdict: Verdict, actions: str, metrics: dict[str, float], facts: frozenset[str]
    ) -> ScoreResult:
        """A fired check counts only with a line of its evidence found in the policy's actions and while the fact it is
        exempt under (``unless``) does not hold."""
        term = self.term
        fired: dict[str, bool] = {}
        unsupported = 0
        quotes, exempt = [], []
        for check in term.checks:
            flagged, evidence = verdict.checks[check.name]
            if flagged and check.unless in facts:
                exempt.append(check.name)
                flagged = False
            found = (
                next((quote for quote in evidence if evidence_supported(quote, actions)), None) if flagged else None
            )
            unsupported += flagged and found is None
            fired[check.name] = found is not None
            metrics[self._key(check.name)] = 1.0 if found is not None else 0.0
            if found is not None:
                quotes.append(f"{check.name}: {found.strip()!r}")
        veto = any(fired[check.name] for check in term.checks if check.veto)
        metrics[self._key(JudgeMetric.VETO)] = 1.0 if veto else 0.0
        metrics[self._key(JudgeMetric.UNSUPPORTED_FLAGS)] = float(unsupported)
        metrics[self._key(JudgeMetric.EXEMPTED_FLAGS)] = float(len(exempt))
        lines = [verdict.rationale, *(f"fired {quote}" for quote in quotes), *(f"exempt {name}" for name in exempt)]
        detail = "\n".join(filter(None, lines)) or None
        return ScoreResult(term.flag_fraction(fired), metrics, detail=detail, veto=veto)
