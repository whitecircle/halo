"""A generative judge: an OpenAI-compatible chat model reads the task and one view of the episode
and answers with one JSON verdict — integer scores per requirement, or per check whether it fires
and the verbatim span of the policy's actions that shows it."""

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from src.env import env_str
from src.inference.endpoints import EXTERNAL_API_KEY_CHAIN, resolve_external_api_key
from src.inference.openai_client import (
    chat_completion,
    create_openai_client,
    json_schema_response_format,
    parse_json_object,
)
from src.inference.response import get_finish_reason
from src.rewards.samples import ScoringSample, cut_middle, render_tools, task_text, view_text
from src.rewards.scorers.base import Scorer, ScoreResult
from src.rewards.terms import JudgeMetric, JudgeTerm, View

SYSTEM_PROMPT = (
    "You are a strict, impartial grader of a policy model's episode. Judge only what the episode "
    "shows: the policy's actions, its answers and the tool results it received. Its reasoning, where "
    "shown, is context for understanding what it did, not something to grade. Reply with the requested "
    "JSON object and nothing else."
)
VIEW_HEADINGS = {View.FINAL: "Final answer", View.FULL: "Transcript", View.DIGEST: "Transcript digest"}
# A quote longer than this is a copy of the response, not evidence of one span in it.
MAX_EVIDENCE_CHARS = 400
# How much of a reply an error result quotes.
REPLY_EXCERPT_CHARS = 200

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Verdict:
    """A parsed judge reply: per-requirement scores (score mode) or per-check ``{fired, evidence}``
    (veto mode), and the rationale."""

    scores: dict[str, float]
    checks: dict[str, tuple[bool, str]]
    rationale: str | None


def scorer_api_key(term: JudgeTerm) -> str:
    """The judge's key: the term's ``api_key_env`` variable, then the hosted chain."""
    key = env_str(term.api_key_env) or resolve_external_api_key()
    if not key:
        names = " or ".join(dict.fromkeys((term.api_key_env, *EXTERNAL_API_KEY_CHAIN)))
        raise RuntimeError(f"{term.owner}: no API key — set {names}")
    return key


def response_schema(term: JudgeTerm) -> dict[str, Any]:
    """The strict JSON schema of a verdict: one integer per requirement, or one ``{fired, evidence}``
    per check, plus a rationale."""
    if term.is_veto:
        check = {
            "type": "object",
            "properties": {"fired": {"type": "boolean"}, "evidence": {"type": "string"}},
            "required": ["fired", "evidence"],
            "additionalProperties": False,
        }
        verdicts = {"checks": _object_of({check_.name: check for check_ in term.checks})}
    else:
        verdicts = {"scores": _object_of({requirement.name: {"type": "integer"} for requirement in term.requirements})}
    return {
        "type": "object",
        "properties": {**verdicts, "rationale": {"type": "string"}},
        "required": [*verdicts, "rationale"],
        "additionalProperties": False,
    }


def _object_of(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def grading_prompt(term: JudgeTerm, sample: ScoringSample) -> str:
    """The user turn the judge grades from: the task, the policy's tools, the reference, the view of
    the episode, the rubric and the reply shape."""
    response = view_text(sample, term.view, include_reasoning=term.include_reasoning, max_chars=term.max_view_chars)
    parts = [f"# Task\n{task_text(sample.prompt) or '(no task text)'}"]
    if sample.tools:
        parts.append(f"# Tools the policy could call\n{render_tools(sample.tools)}")
    if term.include_reference and sample.reference is not None:
        reference = (
            sample.reference if isinstance(sample.reference, str) else json.dumps(sample.reference, ensure_ascii=False)
        )
        parts.append(f"# Reference answer\n{cut_middle(reference, term.max_view_chars)}")
    parts.append(f"# {VIEW_HEADINGS[term.view]}\n{response or '(empty response)'}")
    if term.is_veto:
        rubric = "\n".join(f"{i}. {check.name}: {check.description}" for i, check in enumerate(term.checks, 1))
        parts.append(
            "# Checks\nFor each check, decide whether the episode above shows it. A check fires only on evidence "
            "you can quote verbatim from the policy's actions above — its visible text, a tool call's arguments or "
            "a tool result, never its reasoning, which is not an action: give the exact span (a sentence or a line, "
            "not the whole text) as its evidence, or an empty string when it does not fire.\n" + rubric
        )
    else:
        rubric = "\n".join(f"{i}. {r.name}: {r.description}" for i, r in enumerate(term.requirements, 1))
        parts.append(f"# Requirements\nScore each requirement from 0 (not met) to {term.scale} (fully met).\n{rubric}")
    if term.is_veto:
        keys = ", ".join(
            f'"{check.name}": {{"fired": <true|false>, "evidence": "<verbatim quote or empty>"}}'
            for check in term.checks
        )
        parts.append(f'Reply with one JSON object: {{"checks": {{{keys}}}, "rationale": "<one or two sentences>"}}')
    else:
        keys = ", ".join(f'"{requirement.name}": <integer>' for requirement in term.requirements)
        parts.append(f'Reply with one JSON object: {{"scores": {{{keys}}}, "rationale": "<one or two sentences>"}}')
    return "\n\n".join(parts)


def action_text(term: JudgeTerm, sample: ScoringSample) -> str:
    """What a quoted piece of evidence must come from: the view the judge read, less the policy's
    reasoning — a check is raised on an observable action, never on a thought, or the policy learns
    to hide its reasoning rather than to act well."""
    return view_text(sample, term.view, include_reasoning=False, max_chars=term.max_view_chars)


def parse_verdict(content: str, term: JudgeTerm) -> Verdict | None:
    """The verdict in a judge reply, or ``None`` when the reply holds no JSON object answering every
    requirement with a finite number (score mode) or every check with a ``fired`` boolean (veto mode)."""
    payload = parse_json_object(content)
    if not isinstance(payload, Mapping):
        return None
    rationale = payload.get("rationale")
    rationale = rationale if isinstance(rationale, str) else None
    if term.is_veto:
        checks = payload.get("checks")
        if not isinstance(checks, Mapping):
            return None
        parsed_checks: dict[str, tuple[bool, str]] = {}
        for check in term.checks:
            entry = checks.get(check.name)
            fired = entry.get("fired") if isinstance(entry, Mapping) else entry
            if not isinstance(fired, bool):
                return None
            evidence = entry.get("evidence") if isinstance(entry, Mapping) else ""
            parsed_checks[check.name] = (fired, evidence if isinstance(evidence, str) else "")
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
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            return None
        parsed[requirement.name] = float(value)
    return Verdict(parsed, {}, rationale)


def evidence_supported(evidence: str, text: str) -> bool:
    """Whether ``evidence`` is a verbatim span of ``text`` — whitespace folded, since a model reflows
    what it quotes — and short enough to be a span rather than a copy of the text."""
    quote = _WHITESPACE.sub(" ", evidence).strip()
    return 0 < len(quote) <= MAX_EVIDENCE_CHARS and quote in _WHITESPACE.sub(" ", text)


class GenerativeJudge(Scorer):
    """The judge behind a :class:`JudgeTerm`: one lazily built OpenAI-compatible client per instance
    (per Ray actor or trainer rank), shared by every sample it grades."""

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
            ]
        else:
            keys = [self._key(requirement.name) for requirement in term.requirements]
        return (*keys, self._key(JudgeMetric.COMPLETION_TOKENS))

    def _connect(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = create_openai_client(
                base_url=self.term.base_url, api_key_override=scorer_api_key(self.term)
            )
        return self._client

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.close()

    def _usage_metrics(self, completion) -> dict[str, float]:
        """The usage a reply reports: its completion tokens and, where the endpoint prices it, its cost."""
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
            request["response_format"] = json_schema_response_format("verdict", response_schema(term))
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
            excerpt = content[:REPLY_EXCERPT_CHARS]
            return ScoreResult(
                None, error=f"unparseable judge reply (finish_reason={get_finish_reason(choice)!r}): {excerpt!r}"
            )
        metrics = self._usage_metrics(completion)
        if term.is_veto:
            return self._veto_result(verdict, action_text(term, sample), metrics)
        fractions = term.requirement_fractions(verdict.scores)
        metrics.update({self._key(name): fraction for name, fraction in fractions.items()})
        return ScoreResult(term.score_from(fractions), metrics, detail=verdict.rationale)

    def _veto_result(self, verdict: Verdict, actions: str, metrics: dict[str, float]) -> ScoreResult:
        """A fired check counts only with evidence quoted from the policy's actions; the veto checks
        strip the episode's credits, the others make the term's score."""
        term = self.term
        fired: dict[str, bool] = {}
        unsupported = 0
        quotes = []
        for check in term.checks:
            flagged, evidence = verdict.checks[check.name]
            supported = flagged and evidence_supported(evidence, actions)
            unsupported += flagged and not supported
            fired[check.name] = supported
            metrics[self._key(check.name)] = 1.0 if supported else 0.0
            if supported:
                quotes.append(f"{check.name}: {evidence.strip()!r}")
        veto = any(fired[check.name] for check in term.checks if check.veto)
        metrics[self._key(JudgeMetric.VETO)] = 1.0 if veto else 0.0
        metrics[self._key(JudgeMetric.UNSUPPORTED_FLAGS)] = float(unsupported)
        detail = "\n".join(filter(None, [verdict.rationale, *(f"fired {quote}" for quote in quotes)])) or None
        return ScoreResult(term.flag_fraction(fired), metrics, detail=detail, veto=veto)
