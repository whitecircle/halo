"""Endpoint URL defaults, API-key resolution and the loopback-host test for OpenAI-compatible callers;
SDK-free, since the argument dataclasses read their defaults from it."""

from halo.env import env_str

# A rollout server on this machine at the engines' default port: its root (weight sync, health) and its OpenAI
# API. Omitting the URL flag must never send conversations to the public OpenAI API.
DEFAULT_LOCAL_SERVER_URL = "http://localhost:8000"
DEFAULT_LOCAL_BASE_URL = f"{DEFAULT_LOCAL_SERVER_URL}/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Read in order for a hosted (non-local) endpoint's key.
EXTERNAL_API_KEY_CHAIN = ("OPENROUTER_API_KEY", "OPENAI_API_KEY")
# What a keyless local server (no --api-key) accepts: the SDK refuses to build a client without some key.
LOCAL_SERVER_API_KEY = "EMPTY"

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "0.0.0.0", "::1", "localhost"})


def resolve_local_api_key() -> str:
    return env_str("VLLM_API_KEY") or env_str("OPENAI_API_KEY") or LOCAL_SERVER_API_KEY


def resolve_external_api_key() -> str | None:
    return next((key for name in EXTERNAL_API_KEY_CHAIN if (key := env_str(name))), None)


def is_loopback(host: str) -> bool:
    return host in _LOOPBACK_HOSTS or host.startswith("127.")
