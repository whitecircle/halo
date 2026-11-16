"""The White Circle API as a scorer: the completion, in its conversation, checked against an environment's policies
over ``POST /api/events``."""

import asyncio
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from halo.env import env_str
from halo.inference.openai_client import TRANSIENT_CLIENT_ERROR_CODES
from halo.rewards.samples import ScoringSample, call_arguments, content_text, scored_messages
from halo.rewards.scorers.base import Scorer, ScoreResult
from halo.rewards.terms import View, WhiteCircleTerm

API_KEY_VARIABLE = "WHITECIRCLE_API_KEY"
API_VERSION = "2026-06-01"
# The API's cap on the events of one request.
MAX_EVENTS_PER_REQUEST = 50
# A Retry-After beyond this would hold a training step for longer than any retry is worth.
MAX_RETRY_AFTER_S = 60.0
_FLAGGED = "flagged"
_ERROR_BODY_CHARS = 200


def _api_key() -> str:
    key = env_str(API_KEY_VARIABLE)
    if not key:
        raise RuntimeError(f"whitecircle reward term: no API key — set {API_KEY_VARIABLE}")
    return key


def conversation_events(sample: ScoringSample, view: View) -> tuple[list[dict[str, Any]], list[str]]:
    """The conversation as API events in order, each with its ``event_id``, and the ids of the policy's own events (its
    text and its calls' arguments), the ones the verdict reads; every earlier event is their context."""
    events: list[dict[str, Any]] = []
    policy_ids: list[str] = []
    for index, message in enumerate(scored_messages(sample, view)):
        role, text = message["role"], content_text(message["content"])
        if role == "tool":
            items = [
                {"type": "tool", "name": message.get("name"), "call_id": message.get("tool_call_id"), "output": text}
            ]
        else:
            items = [{"type": "message", "role": role, "content": text}] if text else []
        for call in message.get("tool_calls") or []:
            name = call.get("function", call).get("name")
            items.append({"type": "tool", "name": name, "call_id": call.get("id"), "arguments": call_arguments(call)})
        for item in items:
            item["event_id"] = f"e{len(events)}"
            events.append(item)
            if role == "assistant" and index >= len(sample.prompt):
                policy_ids.append(item["event_id"])
    return events, policy_ids


def _retry_after(response: httpx.Response) -> float | None:
    """The ``Retry-After`` seconds, capped; ``None`` when absent or not a number."""
    try:
        return min(max(float(response.headers["retry-after"]), 0.0), MAX_RETRY_AFTER_S)
    except (KeyError, ValueError):
        return None


class WhiteCircleScorer(Scorer):
    """One request per sample, so no completion is checked against another's conversation; a conversation over
    :data:`MAX_EVENTS_PER_REQUEST` events goes in consecutive requests under one ``external_session_id``. The key is read
    from :data:`API_KEY_VARIABLE` when the client is built and lives only in its headers."""

    term_type = WhiteCircleTerm
    term: WhiteCircleTerm
    _client: httpx.AsyncClient | None

    def __init__(self, term: WhiteCircleTerm):
        super().__init__(term)
        _api_key()

    @property
    def metric_keys(self) -> tuple[str, ...]:
        return (self._key(_FLAGGED), *(self._key(policy) for policy in self.term.policies))

    def _connect(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=f"https://{self.term.region}.whitecircle.com",
                headers={"Authorization": f"Bearer {_api_key()}", "whitecircle-version": API_VERSION},
                timeout=None,
            )
        return self._client

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    async def score_one(self, sample: ScoringSample) -> ScoreResult:
        events, policy_ids = conversation_events(sample, self.term.view)
        if not policy_ids:
            # No text and no call: nothing of the policy's to check, so nothing it could violate.
            return ScoreResult(1.0, {self._key(_FLAGGED): 0.0})
        session = uuid.uuid4().hex
        results: dict[str, Mapping[str, Any]] = {}
        try:
            for start in range(0, len(events), MAX_EVENTS_PER_REQUEST):
                chunk = events[start : start + MAX_EVENTS_PER_REQUEST]
                body = {"environment_id": self.term.environment_id, "external_session_id": session, "events": chunk}
                payload = await self._post(body)
                results.update({r["event_id"]: r for kind in payload.values() if isinstance(kind, list) for r in kind})
            return self._verdict([results.get(event_id) for event_id in policy_ids])
        except Exception as e:
            return ScoreResult(None, error=self._failed("check", e))

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """Retries a timeout, a transport error, a 408, a 429 and a 5xx, waiting the ``Retry-After`` the server sent or
        an exponential backoff; any other error status raises at once."""
        term = self.term
        for attempt in range(term.max_retries + 1):
            wait = term.retry_wait * 2**attempt
            try:
                async with asyncio.timeout(term.request_timeout):
                    response = await self._connect().post("/api/events", json=body)
            except (TimeoutError, httpx.TransportError):
                if attempt == term.max_retries:
                    raise
            else:
                status = response.status_code
                if (status not in TRANSIENT_CLIENT_ERROR_CODES and status < 500) or attempt == term.max_retries:
                    if response.is_error:
                        raise RuntimeError(f"POST /api/events answered {status}: {response.text[:_ERROR_BODY_CHARS]}")
                    return response.json()
                retry_after = _retry_after(response)
                wait = wait if retry_after is None else retry_after
            await asyncio.sleep(wait)
        raise AssertionError("unreachable")

    def _verdict(self, results: Sequence[Mapping[str, Any] | None]) -> ScoreResult:
        """Raises when an event the score reads has no verdict, or a listed policy was checked on none of them."""
        term = self.term
        for result in results:
            if result is None:
                raise ValueError("the response carries no result for an event of the completion")
            if result.get("status") != "completed" or result.get(_FLAGGED) is None:
                raise ValueError(f"no verdict (status {result.get('status')!r}): {result.get('note')}")
        violated: dict[str, bool] = {}
        names: dict[str, str] = {}
        for result in results:
            for policy, outcome in (result.get("policies") or {}).items():
                violated[policy] = violated.get(policy, False) or bool(outcome[_FLAGGED])
                names[policy] = outcome.get("name", policy)
        flagged = any(result[_FLAGGED] for result in results)
        metrics = {self._key(_FLAGGED): float(flagged)}
        if term.policies:
            unchecked = [policy for policy in term.policies if policy not in violated]
            if unchecked:
                raise ValueError(
                    f"policies {unchecked} were not checked on the completion: no condition selected them"
                )
            metrics.update({self._key(policy): float(violated[policy]) for policy in term.policies})
            score = 1.0 - sum(violated[policy] for policy in term.policies) / len(term.policies)
        else:
            score = 0.0 if flagged else 1.0
        broken = [names[policy] for policy, hit in violated.items() if hit]
        return ScoreResult(score, metrics, detail=f"flagged: {', '.join(broken)}" if broken else None)
