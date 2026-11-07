"""Pluggable web search backends, auto-selected by available API key: Serper, Brave, Tavily, Keenable, then keyless
DuckDuckGo. Each raw result is ``{"title", "url", "snippet"}``."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import httpx
from ddgs import DDGS

from halo.env import env_flag, env_str

_SEARCH_TIMEOUT = 15.0
DEFAULT_MAX_RESULTS = 5
_ALLOW_MOCK_SEARCH_FLAG = "HALO_ALLOW_MOCK_SEARCH"
# Keenable's snippet is page text, often thousands of characters, where the other backends return a sentence or two.
_KEENABLE_SNIPPET_CHARS = 500


def _format_results(results: list[dict[str, Any]], max_results: int) -> str:
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results[:max_results], 1):
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        url = r.get("url", "")
        lines.append(f"{i}. {title}\n   {snippet}\n   {url}")
    return "\n\n".join(lines)


def _search_duckduckgo(query: str, max_results: int) -> list[dict[str, Any]]:
    raw = DDGS().text(query, max_results=max_results)
    return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")} for r in raw]


async def _async_search_duckduckgo(query: str, max_results: int) -> list[dict[str, Any]]:
    return await asyncio.to_thread(_search_duckduckgo, query, max_results)


@dataclass(frozen=True)
class _HttpSearchSpec:
    """A keyed JSON search API shared by the sync and async transports: ``build(query, max_results, key)`` gives httpx
    request kwargs, ``parse(data)`` the result dicts."""

    method: str
    url: str
    env_key: str
    build: Callable[[str, int, str], dict[str, Any]]
    parse: Callable[[dict[str, Any]], list[dict[str, Any]]]

    def request_kwargs(self, query: str, max_results: int) -> dict[str, Any]:
        return self.build(query, max_results, env_str(self.env_key, ""))


def _sync_call(spec: _HttpSearchSpec, query: str, max_results: int) -> list[dict[str, Any]]:
    resp = httpx.request(spec.method, spec.url, timeout=_SEARCH_TIMEOUT, **spec.request_kwargs(query, max_results))
    resp.raise_for_status()
    return spec.parse(resp.json())[:max_results]


async def _async_call(spec: _HttpSearchSpec, query: str, max_results: int) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=_SEARCH_TIMEOUT) as client:
        resp = await client.request(spec.method, spec.url, **spec.request_kwargs(query, max_results))
        resp.raise_for_status()
        data = resp.json()
    return spec.parse(data)[:max_results]


def _search_mock(query: str, max_results: int) -> list[dict[str, Any]]:
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


async def _async_search_mock(query: str, max_results: int) -> list[dict[str, Any]]:
    return _search_mock(query, max_results)


@dataclass(frozen=True)
class _SearchBackend:
    """``env_key`` is the API-key env var (None = keyless)."""

    sync: Callable[..., list[dict[str, Any]]]
    async_: Callable[..., Awaitable[list[dict[str, Any]]]]
    env_key: str | None = None


def _http_backend(spec: _HttpSearchSpec) -> _SearchBackend:
    return _SearchBackend(sync=partial(_sync_call, spec), async_=partial(_async_call, spec), env_key=spec.env_key)


# Insertion order is the auto-select priority; the keyless last entry always qualifies.
_BACKENDS: dict[str, _SearchBackend] = {
    "serper": _http_backend(
        _HttpSearchSpec(
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
    ),
    "brave": _http_backend(
        _HttpSearchSpec(
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
    ),
    "tavily": _http_backend(
        _HttpSearchSpec(
            method="POST",
            url="https://api.tavily.com/search",
            env_key="TAVILY_API_KEY",
            build=lambda query, max_results, key: {
                "json": {"query": query, "max_results": max_results, "api_key": key}
            },
            parse=lambda data: [
                {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
                for r in data.get("results", [])
            ],
        )
    ),
    "keenable": _http_backend(
        _HttpSearchSpec(
            method="POST",
            url="https://api.keenable.ai/v1/search",
            env_key="KEENABLE_API_KEY",
            build=lambda query, max_results, key: {
                "json": {"query": query, "max_results": max_results},
                "headers": {"X-API-Key": key},
            },
            parse=lambda data: [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": " ".join((r.get("snippet") or "").split())[:_KEENABLE_SNIPPET_CHARS],
                }
                for r in data.get("results", [])
            ],
        )
    ),
    "duckduckgo": _SearchBackend(sync=_search_duckduckgo, async_=_async_search_duckduckgo),
}

# Fabricated snippets score ``tool_success_reward`` like a real search, so naming this backend
# requires ``_ALLOW_MOCK_SEARCH_FLAG`` (read per call); tests and smoke runs set it.
_MOCK_BACKEND = _SearchBackend(sync=_search_mock, async_=_async_search_mock)


def _selectable_backends() -> dict[str, _SearchBackend]:
    """The flag is read per call, making it a runtime opt-in that does not depend on import order."""
    if env_flag(_ALLOW_MOCK_SEARCH_FLAG):
        return {**_BACKENDS, "mock": _MOCK_BACKEND}
    return _BACKENDS


def validate_search_backend(backend: str | None) -> None:
    """``None`` auto-selects. Checked at tool construction too, since every protocol turns a per-search raise into a
    priced observation and a run would train on a dead tool."""
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


def _resolve_backend(backend: str | None) -> tuple[str, _SearchBackend]:
    """The named backend, else the first in :data:`_BACKENDS` order this process can reach."""
    backend = backend or next(
        name for name, spec in _BACKENDS.items() if spec.env_key is None or env_str(spec.env_key)
    )
    validate_search_backend(backend)
    return backend, _selectable_backends()[backend]


def _backend_failure(backend: str, exc: Exception) -> RuntimeError:
    """A fault raises rather than falling back to mock results, which would score as a successful search."""
    return RuntimeError(f"web_search backend '{backend}' failed: {exc}")


def web_search_raw(
    query: str, max_results: int = DEFAULT_MAX_RESULTS, backend: str | None = None
) -> list[dict[str, Any]]:
    """An unknown name raises ``ValueError`` and a backend fault ``RuntimeError``."""
    backend, impl = _resolve_backend(backend)

    try:
        return impl.sync(query=query, max_results=max_results)
    except Exception as e:
        raise _backend_failure(backend, e) from e


def web_search(query: str, max_results: int = DEFAULT_MAX_RESULTS, backend: str | None = None) -> str:
    return _format_results(web_search_raw(query, max_results, backend), max_results)


async def async_web_search(query: str, max_results: int = DEFAULT_MAX_RESULTS, backend: str | None = None) -> str:
    backend, impl = _resolve_backend(backend)

    try:
        results = await impl.async_(query=query, max_results=max_results)
    except Exception as e:
        raise _backend_failure(backend, e) from e
    return _format_results(results, max_results)
