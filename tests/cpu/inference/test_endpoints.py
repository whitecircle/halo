#!/usr/bin/env python
"""The endpoint leaf: which key an OpenAI-compatible caller sends, which hosts name this machine only,
and that the leaf never imports the SDK — the argument dataclasses read their defaults from it, so an
``openai`` import there would pull the SDK into every config parse.

Run: pytest tests/cpu/inference/test_endpoints.py
"""

import ast
import pathlib

import pytest

from halo.inference import endpoints
from halo.inference.endpoints import (
    LOCAL_SERVER_API_KEY,
    is_loopback,
    resolve_external_api_key,
    resolve_local_api_key,
)

_KEY_VARS = ("VLLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")


@pytest.fixture
def no_keys(monkeypatch):
    """The environment with every key the resolvers read removed; returns the monkeypatch to set one."""
    for name in _KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_local_key_prefers_the_server_convention_then_the_sdk_key_then_the_placeholder(no_keys):
    assert resolve_local_api_key() == LOCAL_SERVER_API_KEY
    no_keys.setenv("OPENAI_API_KEY", "sdk-key")
    assert resolve_local_api_key() == "sdk-key"
    no_keys.setenv("VLLM_API_KEY", "server-key")
    assert resolve_local_api_key() == "server-key"


def test_external_key_prefers_openrouter_then_the_sdk_key_then_nothing(no_keys):
    assert resolve_external_api_key() is None
    no_keys.setenv("OPENAI_API_KEY", "sdk-key")
    assert resolve_external_api_key() == "sdk-key"
    no_keys.setenv("OPENROUTER_API_KEY", "router-key")
    assert resolve_external_api_key() == "router-key"


def test_an_exported_empty_key_falls_through_both_chains(no_keys):
    """``.env.example`` ships the keys blank: an exported empty value is unset, not a key."""
    for name in _KEY_VARS:
        no_keys.setenv(name, "")
    assert resolve_local_api_key() == LOCAL_SERVER_API_KEY
    assert resolve_external_api_key() is None


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.1.1", "localhost", "::1", "0.0.0.0"])
def test_a_loopback_or_wildcard_host_names_this_machine_only(host):
    """A rollout URL or group address on any of these is unreachable from another node."""
    assert is_loopback(host)


@pytest.mark.parametrize("host", ["10.0.0.7", "203.0.113.7", "rollout-server", "128.0.0.1"])
def test_a_routable_host_is_not_loopback(host):
    assert not is_loopback(host)


def _imported_modules(path: pathlib.Path) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    return modules


def test_the_endpoint_leaf_imports_nothing_from_the_sdk():
    imported = _imported_modules(pathlib.Path(endpoints.__file__))
    assert imported, "the scan read no imports, so it covers nothing"
    sdk = [name for name in imported if name == "openai" or name.startswith("openai.")]
    assert not sdk, f"halo/inference/endpoints.py imports the SDK: {sdk}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
