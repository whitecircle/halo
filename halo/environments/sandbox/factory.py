"""Backend selection for the code sandboxes: explicit args first, then ``HALO_SANDBOX_BACKEND`` /
``HALO_SANDBOX_URL``."""

import logging
from collections.abc import Hashable

from halo.env import env_str
from halo.environments.sandbox.base import SandboxExecutor
from halo.environments.sandbox.bubblewrap import BubblewrapSandbox
from halo.environments.sandbox.local import LocalSubprocessSandbox
from halo.environments.sandbox.remote import RemoteSandbox
from halo.log import warn_once

logger = logging.getLogger(__name__)

_unisolated_warned: set[Hashable] = set()


def build_sandbox(
    backend: str | None = None,
    url: str | None = None,
    **kwargs,
) -> SandboxExecutor:
    backend = (backend or env_str("HALO_SANDBOX_BACKEND") or "local").lower()
    url = url or env_str("HALO_SANDBOX_URL")
    if backend != "remote" and url:
        # Ignoring a url would run model-generated code in the training container, not the expected sandbox service.
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
    """Warn when ``consumer`` runs model-written code on a backend that does not confine it: the program reaches the
    host's filesystem or network, where a policy can rewrite what grades it."""
    if sandbox.isolated:
        return
    warn_once(
        logger,
        _unisolated_warned,
        type(sandbox),
        "%s runs model-written code on a sandbox that does not confine it (%s): the program can reach "
        "the host's filesystem or network, and on 'local' read the grader's launch environment "
        "(/proc/<pid>/environ), secrets passed to the trainer included. Use sandbox_backend="
        "'bubblewrap' without network (a privileged container) or 'remote' (HALO_SANDBOX_BACKEND / "
        "HALO_SANDBOX_URL) for RL on untrusted code.",
        consumer,
        type(sandbox).__name__,
    )
