"""Where an OpenAI-compatible caller points by default, and which key it sends: the leaf every
endpoint consumer reads — the argument dataclasses, the CLI flag block, the clients — so none of
them imports the SDK to learn a URL."""

from src.env import env_str

# Endpoint every OpenAI-compatible caller targets when its URL flag is omitted. Defaulting to the
# public OpenAI API instead would send conversations off-site whenever the flag is forgotten.
DEFAULT_LOCAL_BASE_URL = "http://localhost:8000/v1"

# The hosted aggregator every off-site caller targets (the judges, demo chat).
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# The key a keyless local rollout server (vLLM / SGLang without --api-key) accepts. The SDK refuses to
# construct a client without SOME key, so a None default would crash the documented local invocation.
LOCAL_SERVER_API_KEY = "EMPTY"


def resolve_local_api_key() -> str:
    """Key for the local rollout server: ``VLLM_API_KEY`` → ``OPENAI_API_KEY`` → placeholder.

    The CLI default for the shared ``--api_key`` flag; an explicit key arrives as the flag's value
    instead. ``VLLM_API_KEY`` is checked first because it is the server-side ``--api-key``
    convention.
    """
    return env_str("VLLM_API_KEY") or env_str("OPENAI_API_KEY") or LOCAL_SERVER_API_KEY


def resolve_external_api_key() -> str | None:
    """Key for a hosted (non-local) endpoint: ``OPENROUTER_API_KEY`` → ``OPENAI_API_KEY``.

    ``None`` when nothing is set — each caller decides whether that is fatal.
    """
    return env_str("OPENROUTER_API_KEY") or env_str("OPENAI_API_KEY") or None
