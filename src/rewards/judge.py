"""A generative judge: an OpenAI-compatible chat model grades a response against the term's
requirements and answers with one JSON object of integer scores."""

import asyncio
import json
import logging
import re
from collections.abc import Mapping
from typing import Any

from openai import AsyncOpenAI

from src.env import env_str
from src.inference.openai_client import create_openai_client, resolve_external_api_key
from src.rewards.samples import ScoringSample, final_assistant_text, render_transcript, task_text, truncate_text
from src.rewards.scoring import Scorer, ScoreResult
from src.rewards.spec import JudgeTerm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a strict, impartial grader. Score the response to the task against each requirement, "
    "judging only what the response contains. Reply with the requested JSON object and nothing else."
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
# An aggregator can answer 200 with no choices and an error body the SDK does not retry (OpenRouter
# spells an upstream rate limit this way); these codes are retried here with this backoff.
_RETRYABLE_UPSTREAM_CODES = (408, 429, 500, 502, 503, 504)
_UPSTREAM_RETRIES = 4
_UPSTREAM_BACKOFF_SECONDS = 2.0
# The sample the launch probe grades: tiny, so the probe costs nothing, yet in the run's exact shape.
_PROBE_SAMPLE = ScoringSample(
    prompt=[{"role": "user", "content": "Reply with the single word: ready"}],
    completion=[{"role": "assistant", "content": "ready"}],
)


def judge_api_key(term: JudgeTerm) -> str:
    """The judge's key: the term's ``api_key_env`` variable, then the hosted chain."""
    key = env_str(term.api_key_env) or resolve_external_api_key()
    if not key:
        raise RuntimeError(
            f"judge term {term.name!r}: no API key — set {term.api_key_env} (or OPENROUTER_API_KEY / OPENAI_API_KEY)"
        )
    return key


def response_schema(term: JudgeTerm) -> dict[str, Any]:
    """The strict JSON schema of a verdict: one integer per requirement plus a rationale."""
    return {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "properties": {requirement.name: {"type": "integer"} for requirement in term.requirements},
                "required": [requirement.name for requirement in term.requirements],
                "additionalProperties": False,
            },
            "rationale": {"type": "string"},
        },
        "required": ["scores", "rationale"],
        "additionalProperties": False,
    }


def grading_prompt(term: JudgeTerm, sample: ScoringSample) -> str:
    """The user turn the judge grades from: task, optional reference, response, rubric, reply shape."""
    parts = [f"# Task\n{task_text(sample.prompt) or '(no task text)'}"]
    if term.include_reference and sample.reference is not None:
        reference = sample.reference if isinstance(sample.reference, str) else json.dumps(sample.reference)
        parts.append(f"# Reference answer\n{truncate_text(reference, term.max_transcript_chars)}")
    if term.transcript == "full":
        response = render_transcript(sample.completion, max_chars=term.max_transcript_chars)
    else:
        response = truncate_text(final_assistant_text(sample.completion), term.max_transcript_chars)
    parts.append(f"# Response\n{response or '(empty response)'}")
    rubric = "\n".join(
        f"{index}. {requirement.name}: {requirement.description}"
        for index, requirement in enumerate(term.requirements, start=1)
    )
    parts.append(f"# Requirements\nScore each requirement from 0 (not met) to {term.scale} (fully met).\n{rubric}")
    if term.instructions:
        parts.append(f"# Grading instructions\n{term.instructions}")
    keys = ", ".join(f'"{requirement.name}": <integer>' for requirement in term.requirements)
    parts.append(f'Reply with one JSON object: {{"scores": {{{keys}}}, "rationale": "<one or two sentences>"}}')
    return "\n\n".join(parts)


def parse_verdict(content: str, term: JudgeTerm) -> tuple[dict[str, float], str | None] | None:
    """The per-requirement scores (and rationale) in a judge reply, or ``None`` when the reply holds
    no JSON object scoring every requirement with a number."""
    match = _JSON_OBJECT.search(content)
    for candidate in (content, match.group(0) if match else None):
        if candidate is None:
            continue
        try:
            payload = json.loads(candidate)
        except ValueError:
            continue
        scores = payload.get("scores") if isinstance(payload, Mapping) else None
        if not isinstance(scores, Mapping):
            continue
        parsed: dict[str, float] = {}
        for requirement in term.requirements:
            value = scores.get(requirement.name)
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    value = None
            if isinstance(value, bool) or not isinstance(value, int | float):
                break
            parsed[requirement.name] = float(value)
        else:
            rationale = payload.get("rationale")
            return parsed, rationale if isinstance(rationale, str) else None
    return None


class GenerativeJudge(Scorer):
    """The judge behind a :class:`JudgeTerm`: one lazily built client per scorer instance (one per
    Ray actor or trainer rank), shared by every sample it grades."""

    term: JudgeTerm

    def __init__(self, term: JudgeTerm):
        super().__init__(term)
        self._client: AsyncOpenAI | None = None

    @property
    def metric_keys(self) -> tuple[str, ...]:
        name = self.term.name
        return (*(f"judge/{name}/{r.name}" for r in self.term.requirements), f"judge/{name}/completion_tokens")

    def _create_client(self) -> AsyncOpenAI:
        return create_openai_client(base_url=self.term.base_url, api_key_override=judge_api_key(self.term))

    def _client_for_requests(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _request_kwargs(self, sample: ScoringSample, max_tokens: int) -> dict[str, Any]:
        term = self.term
        kwargs: dict[str, Any] = {
            "model": term.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": grading_prompt(term, sample)},
            ],
            "max_completion_tokens": max_tokens,
            "timeout": term.request_timeout,
        }
        if term.reasoning_effort is not None:
            kwargs["reasoning_effort"] = term.reasoning_effort
        if term.temperature is not None:
            kwargs["temperature"] = term.temperature
        if term.structured_output:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "strict": True, "schema": response_schema(term)},
            }
        return kwargs

    async def _grade(self, client: AsyncOpenAI, sample: ScoringSample, max_tokens: int) -> ScoreResult:
        term = self.term
        # Building the prompt serializes the row's reference; a raise here would escape every guard up
        # to the actor's catch-all and mask the whole episode, so it books as a verdict-less result.
        try:
            kwargs = self._request_kwargs(sample, max_tokens)
        except Exception as e:
            logger.warning("judge %r prompt build failed: %s: %s", term.name, type(e).__name__, e)
            return ScoreResult(None, error=f"prompt build failed: {type(e).__name__}: {e}")
        for attempt in range(_UPSTREAM_RETRIES + 1):
            try:
                completion = await client.chat.completions.create(**kwargs)
            except Exception as e:
                logger.warning("judge %r request failed: %s: %s", term.name, type(e).__name__, e)
                return ScoreResult(None, error=f"request failed: {type(e).__name__}: {e}")
            if completion.choices:
                break
            error = getattr(completion, "error", None)
            code = error.get("code") if isinstance(error, Mapping) else None
            if code not in _RETRYABLE_UPSTREAM_CODES or attempt == _UPSTREAM_RETRIES:
                return ScoreResult(None, error=f"judge reply carried no choices: {error!r}")
            await asyncio.sleep(_UPSTREAM_BACKOFF_SECONDS * 2**attempt)
        choice = completion.choices[0]
        content = choice.message.content or ""
        verdict = parse_verdict(content, term)
        if verdict is None:
            finish = getattr(choice, "finish_reason", None)
            return ScoreResult(None, error=f"unparseable judge reply (finish_reason={finish!r}): {content[:200]!r}")
        scores, rationale = verdict
        metrics = {
            f"judge/{term.name}/{requirement.name}": min(max(scores[requirement.name], 0.0), term.scale) / term.scale
            for requirement in term.requirements
        }
        usage = getattr(completion, "usage", None)
        if usage is not None:
            metrics[f"judge/{term.name}/completion_tokens"] = float(getattr(usage, "completion_tokens", 0) or 0)
            cost = getattr(usage, "cost", None)
            if isinstance(cost, int | float):
                metrics[f"judge/{term.name}/cost_usd"] = float(cost)
        return ScoreResult(term.score_from(scores), metrics, detail=rationale)

    async def score_one(self, sample: ScoringSample) -> ScoreResult:
        return await self._grade(self._client_for_requests(), sample, self.term.max_tokens)

    async def verify(self) -> None:
        """Grade a one-line probe through a fresh client; a request or parse failure raises."""
        client = self._create_client()
        try:
            result = await self._grade(client, _PROBE_SAMPLE, self.term.max_tokens)
        finally:
            await client.close()
        if result.score is None:
            raise RuntimeError(f"judge term {self.term.name!r} probe failed: {result.error}")

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.close()
