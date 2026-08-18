"""Backend selection for the code sandboxes: explicit args first, then ``HALO_SANDBOX_BACKEND`` / ``HALO_SANDBOX_URL``."""

from src.env import env_str
from src.environments.sandbox.base import SandboxExecutor
from src.environments.sandbox.bubblewrap import BubblewrapSandbox
from src.environments.sandbox.local import LocalSubprocessSandbox
from src.environments.sandbox.remote import RemoteSandbox


def resolve_sandbox(
    backend: str | None = None,
    url: str | None = None,
    **kwargs,
) -> SandboxExecutor:
    """Build a :class:`SandboxExecutor` from explicit args, falling back to env vars.

    Args:
        backend: ``"local"``, ``"bubblewrap"``, or ``"remote"``
            (default ``HALO_SANDBOX_BACKEND`` then ``"local"``).
        url: remote endpoint (default ``HALO_SANDBOX_URL``); required for ``"remote"``.
        **kwargs: forwarded to the backend constructor.
    """
    backend = (backend or env_str("HALO_SANDBOX_BACKEND") or "local").lower()
    url = url or env_str("HALO_SANDBOX_URL")
    if backend != "remote" and url:
        # Fail loud: a url is only meaningful for the remote backend, so ignoring it would run
        # model-generated code INSIDE the training container while the caller believes it is going
        # to an isolated sandbox service. Silence is the dangerous answer here.
        raise ValueError(
            f"A sandbox url is set ({url!r}) but the backend is {backend!r}, which executes code "
            f"locally — the url would be ignored and untrusted code would run in this container. "
            f"Set sandbox_backend='remote' (or HALO_SANDBOX_BACKEND=remote) to use it, or unset "
            f"the url to run locally on purpose."
        )
    if backend == "local":
        return LocalSubprocessSandbox(**kwargs)
    if backend == "bubblewrap":
        return BubblewrapSandbox(**kwargs)
    if backend == "remote":
        if not url:
            raise ValueError("remote sandbox requires a url (pass url= or set HALO_SANDBOX_URL)")
        return RemoteSandbox(url, **kwargs)
    raise ValueError(f"unknown sandbox backend {backend!r} (expected 'local', 'bubblewrap', or 'remote')")
