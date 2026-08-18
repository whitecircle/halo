"""Bubblewrap-isolated sandbox: the local backend's compile/run core under a namespace jail.

:class:`BubblewrapSandbox` subclasses :class:`LocalSubprocessSandbox`, overriding only the
command-wrapping hook. Each child runs through ``bwrap`` in fresh user/mount/PID/IPC/UTS namespaces
with no network (unless ``allow_network``), a read-only system, and only the per-execution working
dir writable, atop the inherited rlimits.

``bwrap`` needs unprivileged user namespaces, which a container runtime may block (Docker's default
seccomp denies namespace creation). Run the container ``--privileged`` on a host permitting them
(``kernel.unprivileged_userns_clone=1``). The constructor probes once and raises if it can't sandbox.
"""

import os
import shutil

from src.environments.sandbox.local import LocalSession, LocalSubprocessSandbox

# Bound read-only into the jail (``--ro-bind-try`` skips absent paths); ``/usr/local`` carries pip/uv interpreters.
_DEFAULT_RO_BINDS = ("/usr", "/usr/local", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/opt")


class BubblewrapSandbox(LocalSubprocessSandbox):
    """Local backend hardened with bubblewrap namespace isolation (network + filesystem).

    Args:
        allow_network: keep network reachable inside the jail (default False).
        bwrap_path: path to the ``bwrap`` binary (default: resolved from ``PATH``).
        extra_ro_binds: additional host paths to expose read-only.
        Remaining args (memory/compile limits) are inherited from the local backend.
    """

    def __init__(
        self,
        *args,
        allow_network: bool = False,
        bwrap_path: str | None = None,
        extra_ro_binds: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if bwrap_path:
            # An explicit path must point at a real executable; a typo would otherwise be accepted.
            resolved = bwrap_path if (os.path.isfile(bwrap_path) and os.access(bwrap_path, os.X_OK)) else None
        else:
            resolved = shutil.which("bwrap")
        if not resolved:
            raise RuntimeError(
                "bubblewrap sandbox requires the 'bwrap' binary "
                f"({bwrap_path!r} is not an executable; install the 'bubblewrap' package, "
                "or use the 'local'/'remote' backend)"
                if bwrap_path
                else "bubblewrap sandbox requires the 'bwrap' binary on PATH "
                "(install the 'bubblewrap' package, or use the 'local'/'remote' backend)"
            )
        self.bwrap_path = resolved
        self.allow_network = allow_network
        self.extra_ro_binds = tuple(extra_ro_binds or ())
        self._verify_can_sandbox()

    def _verify_can_sandbox(self) -> None:
        """Raise at construction if bwrap cannot create a sandbox in this environment.

        A blocked user/mount namespace makes ``bwrap`` exit non-zero before the child runs; without
        this probe that is indistinguishable from a program exit and grading reads it as a failed
        solution.
        """
        probe = self.run("pass")  # exercises the real wrapped python run path once
        if probe.ok:
            return
        detail = (probe.error or probe.stderr or "").strip().splitlines()
        tail = detail[-1] if detail else f"exit code {probe.returncode}"
        raise RuntimeError(
            f"bubblewrap cannot create a sandbox in this environment ({tail}). The container needs "
            "permission to create user + mount namespaces — run it with --privileged (or the "
            "equivalent seccomp/AppArmor relaxation) on a host that permits unprivileged user "
            "namespaces (kernel.unprivileged_userns_clone=1). Otherwise use the 'local' or 'remote' "
            "sandbox backend."
        )

    def open_session(self) -> LocalSession:
        """Open a persistent jailed session; its working dir is the only writable mount."""
        return LocalSession(self._new_workdir(), self, allow_network=self.allow_network)

    def _wrap_command(self, argv: list[str], workdir: str, *, allow_network: bool) -> list[str]:
        """Prepend the bubblewrap launcher that jails ``argv`` with only ``workdir`` writable."""
        wrapper: list[str] = [
            self.bwrap_path,
            "--die-with-parent",  # kill the jail if the worker dies
            "--new-session",  # detach controlling terminal (blocks TIOCSTI injection)
            "--unshare-all",  # user + mount + pid + net + ipc + uts + cgroup namespaces
        ]
        if allow_network:
            wrapper.append("--share-net")
        for path in (*_DEFAULT_RO_BINDS, *self.extra_ro_binds):
            wrapper += ["--ro-bind-try", path, path]
        wrapper += [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--bind",
            workdir,
            workdir,  # the only writable mount
            "--chdir",
            workdir,
        ]
        if not os.path.isdir(workdir):  # bwrap cannot bind a writable mount that does not exist
            os.makedirs(workdir, exist_ok=True)
        return wrapper + ["--", *argv]
