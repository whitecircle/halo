"""Backend selection for the code sandboxes: explicit args first, then ``HALO_SANDBOX_BACKEND`` /
``HALO_SANDBOX_URL``."""

import logging
from collections.abc import Hashable

from src.env import env_str
from src.environments.sandbox.base import SandboxExecutor
from src.environments.sandbox.bubblewrap import BubblewrapSandbox
from src.environments.sandbox.local import LocalSubprocessSandbox
from src.environments.sandbox.remote import RemoteSandbox
from src.log import warn_once

logger = logging.getLogger(__name__)

# ``warn_once`` scope for the unisolated-backend warning: one line per backend class per process.
_UNISOLATED_WARNED: set[Hashable] = set()


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
        # A url is only meaningful for the remote backend. Ignoring it would run model-generated code
        # inside the training container while the caller expects an isolated sandbox service.
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


def warn_if_unisolated(sandbox: SandboxExecutor, consumer: str) -> None:
    """Warn once per process when ``consumer`` runs model-written code on a backend that does not
    confine it (:attr:`SandboxExecutor.isolated` is False: ``local``, ``bubblewrap`` with
    ``allow_network``, an executor that declares nothing): the program reaches the host's filesystem or
    network, where a policy can read or rewrite what grades it and fetch a solution off the network.
    On ``local`` it also reads the grader's launch environment through ``/proc``, whatever secrets the
    trainer was started with included."""
    if sandbox.isolated:
        return
    warn_once(
        logger,
        _UNISOLATED_WARNED,
        type(sandbox),
        "%s runs model-written code on a sandbox that does not confine it (%s): the program can reach "
        "the host's filesystem or network, and on 'local' read the grader's launch environment "
        "(/proc/<pid>/environ), secrets passed to the trainer included. Use sandbox_backend="
        "'bubblewrap' without network (a privileged container) or 'remote' (HALO_SANDBOX_BACKEND / "
        "HALO_SANDBOX_URL) for RL on untrusted code.",
        consumer,
        type(sandbox).__name__,
    )
