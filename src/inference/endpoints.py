"""Endpoint URL defaults and API-key resolution for OpenAI-compatible callers; SDK-free, since the
argument dataclasses read their defaults from it."""

from src.env import env_str

# Omitting the URL flag must never send conversations to the public OpenAI API.
DEFAULT_LOCAL_BASE_URL = "http://localhost:8000/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Read in order for a hosted (non-local) endpoint's key.
EXTERNAL_API_KEY_CHAIN = ("OPENROUTER_API_KEY", "OPENAI_API_KEY")
# What a keyless local server (no --api-key) accepts: the SDK refuses to build a client without some key.
LOCAL_SERVER_API_KEY = "EMPTY"


def resolve_local_api_key() -> str:
    """Key for the local rollout server: ``VLLM_API_KEY`` (the server's ``--api-key`` convention), then
    ``OPENAI_API_KEY``, then :data:`LOCAL_SERVER_API_KEY`."""
    return env_str("VLLM_API_KEY") or env_str("OPENAI_API_KEY") or LOCAL_SERVER_API_KEY


def resolve_external_api_key() -> str | None:
    """Key for a hosted (non-local) endpoint: the first set :data:`EXTERNAL_API_KEY_CHAIN` variable, or
    ``None`` for the caller to judge."""
    return next((key for name in EXTERNAL_API_KEY_CHAIN if (key := env_str(name))), None)
