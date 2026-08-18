"""Pluggable web search backends for environment tools, auto-selected by available API key.

Priority: Serper (SERPER_API_KEY) → Brave (BRAVE_API_KEY) → Tavily (TAVILY_API_KEY) → DuckDuckGo
(keyless fallback). Each raw result is ``{"title", "url", "snippet"}``.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import httpx
from ddgs import DDGS

from src.env import env_flag, env_str

logger = logging.getLogger(__name__)

# Per-request wall-clock cap (seconds) for every HTTP search backend.
SEARCH_TIMEOUT = 15.0


def _format_results(results: list[dict[str, Any]], max_results: int = 5) -> str:
    """Format search results into a readable string for the model."""
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results[:max_results], 1):
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        url = r.get("url", "")
        lines.append(f"{i}. {title}\n   {snippet}\n   {url}")
    return "\n\n".join(lines)


def _search_duckduckgo(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    raw = DDGS().text(query, max_results=max_results)
    return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")} for r in raw]


async def _async_search_duckduckgo(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    return await asyncio.to_thread(_search_duckduckgo, query, max_results)


@dataclass(frozen=True)
class HttpSearchSpec:
    """A keyed JSON search API, shared by the sync and async transports.

    ``build(query, max_results, key)`` returns the httpx request kwargs (``json``/``params``/
    ``headers``) for the key read from ``env_key``; ``parse(data)`` maps the decoded response to
    ``{"title", "url", "snippet"}`` dicts (the caller applies ``max_results``).
    """

    method: str
    url: str
    env_key: str
    build: Callable[[str, int, str], dict[str, Any]]
    parse: Callable[[dict[str, Any]], list[dict[str, Any]]]

    def request_kwargs(self, query: str, max_results: int) -> dict[str, Any]:
        return self.build(query, max_results, env_str(self.env_key, ""))


def _sync_call(spec: HttpSearchSpec, query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Run ``spec`` over a blocking httpx request."""
    resp = httpx.request(spec.method, spec.url, timeout=SEARCH_TIMEOUT, **spec.request_kwargs(query, max_results))
    resp.raise_for_status()
    return spec.parse(resp.json())[:max_results]


async def _async_call(spec: HttpSearchSpec, query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Run ``spec`` over an async httpx client."""
    async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
        resp = await client.request(spec.method, spec.url, **spec.request_kwargs(query, max_results))
        resp.raise_for_status()
        data = resp.json()
    return spec.parse(data)[:max_results]


_SERPER = HttpSearchSpec(
    method="POST",
    url="https://google.serper.dev/search",
    env_key="SERPER_API_KEY",
    build=lambda query, max_results, key: {
        "json": {"q": query, "num": max_results},
        "headers": {"X-API-KEY": key, "Content-Type": "application/json"},
    },
    parse=lambda data: [
        {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")}
        for r in data.get("organic", [])
    ],
)

_BRAVE = HttpSearchSpec(
    method="GET",
    url="https://api.search.brave.com/res/v1/web/search",
    env_key="BRAVE_API_KEY",
    build=lambda query, max_results, key: {
        "params": {"q": query, "count": max_results},
        "headers": {"X-Subscription-Token": key, "Accept": "application/json"},
    },
    parse=lambda data: [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description", "")}
        for r in data.get("web", {}).get("results", [])
    ],
)

_TAVILY = HttpSearchSpec(
    method="POST",
    url="https://api.tavily.com/search",
    env_key="TAVILY_API_KEY",
    build=lambda query, max_results, key: {"json": {"query": query, "max_results": max_results, "api_key": key}},
    parse=lambda data: [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in data.get("results", [])
    ],
)


def _search_mock(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "title": f"Wikipedia: {query}",
            "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
            "snippet": f"Wikipedia article about {query}.",
        },
        {
            "title": f"{query} - Official Website",
            "url": f"https://www.{query.lower().replace(' ', '')}.com",
            "snippet": f"Official website and resources for {query}.",
        },
        {
            "title": f"Learn about {query}",
            "url": f"https://www.example.com/{query.lower().replace(' ', '-')}",
            "snippet": f"Comprehensive guide to {query} with examples and tutorials.",
        },
    ][:max_results]


async def _async_search_mock(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    return _search_mock(query, max_results)


@dataclass(frozen=True)
class SearchBackend:
    """A selectable search backend: its sync/async entry points and the API-key env var (None = keyless)."""

    sync: Callable[..., list[dict[str, Any]]]
    async_: Callable[..., Awaitable[list[dict[str, Any]]]]
    env_key: str | None = None


def _http_backend(spec: HttpSearchSpec) -> SearchBackend:
    """Bind a :class:`HttpSearchSpec` to the shared sync/async transports."""
    return SearchBackend(sync=partial(_sync_call, spec), async_=partial(_async_call, spec), env_key=spec.env_key)


_BACKENDS: dict[str, SearchBackend] = {
    "serper": _http_backend(_SERPER),
    "brave": _http_backend(_BRAVE),
    "tavily": _http_backend(_TAVILY),
    "duckduckgo": SearchBackend(sync=_search_duckduckgo, async_=_async_search_duckduckgo),
}

_BACKEND_PRIORITY = ["serper", "brave", "tavily", "duckduckgo"]

# Fabricated snippets score ``tool_success_reward`` like a real search, so this backend is not
# selectable from a training YAML; the demo playground opts in through the flag below.
_MOCK_BACKEND = SearchBackend(sync=_search_mock, async_=_async_search_mock)
_ALLOW_MOCK_SEARCH_FLAG = "HALO_ALLOW_MOCK_SEARCH"


def _selectable_backends() -> dict[str, SearchBackend]:
    """The backends a caller may name. The flag is read per call, making it a runtime opt-in that
    does not depend on import order."""
    if env_flag(_ALLOW_MOCK_SEARCH_FLAG):
        return {**_BACKENDS, "mock": _MOCK_BACKEND}
    return _BACKENDS


def _auto_select_backend() -> str:
    """The first backend in priority order this environment can reach (keyless, or key present)."""
    for name in _BACKEND_PRIORITY:
        env_key = _BACKENDS[name].env_key
        if env_key is None or env_str(env_key):
            return name
    raise RuntimeError(
        f"No search backend is selectable: every entry in {_BACKEND_PRIORITY} requires an API key "
        f"and none is set. Set one of their env vars, or restore a keyless backend to the priority list."
    )


def validate_search_backend(backend: str | None) -> None:
    """Raise on a backend name this process cannot select (``None`` = auto-select, always valid).

    Called at tool construction as well as per search: every protocol turns a tool exception into an
    observation plus ``tool_error_penalty``, so a per-call raise alone would let a whole training run
    proceed against a ``web_search`` that never returns a result.
    """
    if backend is None:
        return
    selectable = _selectable_backends()
    if backend not in selectable:
        hint = (
            f" Set {_ALLOW_MOCK_SEARCH_FLAG}=1 to enable the fabricated-results backend (demo only —"
            f" its snippets score tool_success_reward)."
            if backend == "mock"
            else ""
        )
        raise ValueError(f"Unknown search backend: {backend}. Available: {list(selectable)}.{hint}")


def _resolve_backend(backend: str | None, query: str, max_results: int) -> tuple[str, SearchBackend, dict[str, Any]]:
    """Resolve the search backend and build the call kwargs shared by every entry point."""
    backend = backend or _auto_select_backend()
    validate_search_backend(backend)
    return backend, _selectable_backends()[backend], {"query": query, "max_results": max_results}


def _backend_failure(backend: str, exc: Exception) -> RuntimeError:
    """Log a backend fault and build the error the search entry points raise.

    A fault raises rather than falling back to mock results, which would score as a successful search.
    """
    logger.warning(f"Search backend '{backend}' failed: {exc}")
    return RuntimeError(f"web_search backend '{backend}' failed: {exc}")


def web_search_raw(query: str, max_results: int = 5, backend: str | None = None) -> list[dict[str, Any]]:
    """Search the web, returning the raw ``{"title", "url", "snippet"}`` results.

    ``backend=None`` auto-selects by available API keys; an unknown name raises ``ValueError`` and a
    backend fault ``RuntimeError``.
    """
    backend, impl, kwargs = _resolve_backend(backend, query, max_results)

    try:
        return impl.sync(**kwargs)
    except Exception as e:
        raise _backend_failure(backend, e) from e


def web_search(query: str, max_results: int = 5, backend: str | None = None) -> str:
    """Search the web and return results formatted as a string for LLM consumption.

    ``backend=None`` auto-selects by available API keys; pass a name to force one.
    """
    return _format_results(web_search_raw(query, max_results, backend), max_results)


async def async_web_search(query: str, max_results: int = 5, backend: str | None = None) -> str:
    """Async version of web_search."""
    backend, impl, kwargs = _resolve_backend(backend, query, max_results)

    try:
        results = await impl.async_(**kwargs)
    except Exception as e:
        raise _backend_failure(backend, e) from e
    return _format_results(results, max_results)
