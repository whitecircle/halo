"""The local backend under a bubblewrap namespace jail: fresh user/mount/PID/IPC/UTS namespaces, no network unless
``allow_network``, only the working directory writable, and the jail's root a subordinate uid of this process's user,
so a program owns no host file and ``RLIMIT_NPROC`` binds it."""

import contextlib
import ctypes
import logging
import os
import shutil
import signal
import tempfile
import threading
import time

from halo.environments.sandbox.local import LocalSubprocessSandbox

logger = logging.getLogger(__name__)

# Bound read-only into the jail (``--ro-bind-try`` skips absent paths); ``/usr/local`` carries pip/uv interpreters.
_DEFAULT_RO_BINDS = ("/usr", "/usr/local", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/opt")

# <linux/prctl.h>; the stdlib exposes no prctl.
_PR_SET_CHILD_SUBREAPER = 36
_PR_GET_CHILD_SUBREAPER = 37
_LIBC = ctypes.CDLL(None, use_errno=True)

# How long a run waits for its jail's init after the group kill, holding its slot (a fork bomb leaves thousands).
_JAIL_TEARDOWN_SECONDS = 5.0
_JAIL_TEARDOWN_POLL_SECONDS = 0.002

# Fixed, so a traceback, which names its source by absolute path, shows this path rather than the host's.
_JAIL_WORKDIR = "/sandbox"

# ``<user>:<start>:<count>`` subordinate id ranges, which newuidmap/newgidmap map.
_SUBORDINATE_ID_FILES = ("/etc/subuid", "/etc/subgid")

# bwrap's messages: the kernel refused the jail a fresh /proc; the jail's root cannot reach a path it binds.
_PROC_MOUNT_REFUSED = "Can't mount proc"
_BIND_SOURCE_UNREACHABLE = "Can't find source path"

# The kernel grants a user namespace a fresh proc only beside one already fully visible, and a container's /proc is
# not (Docker masks parts of it, the NVIDIA runtime more). A clean one elsewhere is, and leaves those masks in place.
_CLEAN_PROC_MOUNT = "mkdir -p /run/fullproc && mount -t proc proc /run/fullproc"


def _is_child_subreaper() -> bool:
    flag = ctypes.c_int()
    return _LIBC.prctl(_PR_GET_CHILD_SUBREAPER, ctypes.byref(flag), 0, 0, 0) == 0 and bool(flag.value)


def _set_child_subreaper(flag: int) -> None:
    """While set, this process adopts its orphaned descendants."""
    if _LIBC.prctl(
        _PR_SET_CHILD_SUBREAPER, ctypes.c_ulong(flag), ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0)
    ):
        raise RuntimeError(
            f"bubblewrap sandbox: prctl(PR_SET_CHILD_SUBREAPER) failed ({os.strerror(ctypes.get_errno())}); "
            "without it every run leaves the jail's init a zombie. Use the 'local' or 'remote' sandbox backend."
        )


def _subordinate_id(path: str) -> int | None:
    """The first id of root's first subordinate range in ``path``, or None."""
    with contextlib.suppress(FileNotFoundError), open(path) as fh:
        for line in fh:
            name, _, rest = line.strip().partition(":")
            start, _, count = rest.partition(":")
            if name in ("root", "0") and start.isdigit() and count.isdigit() and int(count) > 0:
                return int(start)
    return None


def _reap_group(pgid: int) -> bool:
    """Reap exited children in group ``pgid`` without blocking; whether none is left there."""
    try:
        while os.waitid(os.P_PGID, pgid, os.WEXITED | os.WNOHANG) is not None:
            pass
    except ChildProcessError:
        return True
    return False


class _JailReaper:
    """Reaps the init (the jail's PID 1) that ``bwrap`` exits without reaping on every run.

    Holds ``PR_SET_CHILD_SUBREAPER`` while any jailed run is in flight, so the orphaned init is reparented here. The
    flag belongs to a process and no fork or unpickled sandbox carries it, so it is taken where a run starts; released
    between runs, it leaves other orphans (a trainer rank's Ray daemons) to the container's init. The init stays in the
    run's process group, whose id no new process can take while a member is unreaped, so a wait on the group reaps it
    and no other child. It exits only once every task in the jail has, so a later run finishes one that outlives it.
    """

    def __init__(self) -> None:
        self._reset()
        os.register_at_fork(after_in_child=self._reset)

    def _reset(self) -> None:
        self._lock = threading.Lock()
        self._runs = 0
        self._held_before = False
        self._unreaped: set[int] = set()

    @contextlib.contextmanager
    def held(self):
        with self._lock:
            if self._runs == 0:
                self._held_before = _is_child_subreaper()
                if not self._held_before:
                    _set_child_subreaper(1)
            self._runs += 1
        try:
            yield
        finally:
            with self._lock:
                self._runs -= 1
                if self._runs == 0 and not self._held_before:
                    _set_child_subreaper(0)

    def reap(self, leader: int) -> None:
        """Reap ``leader``'s group, waiting at most :data:`_JAIL_TEARDOWN_SECONDS`, then what earlier runs left."""
        deadline = time.monotonic() + _JAIL_TEARDOWN_SECONDS
        while not _reap_group(leader):
            if time.monotonic() >= deadline:
                logger.warning(
                    "bubblewrap sandbox: the jail of run %d did not exit within %gs of its kill; a later run reaps it",
                    leader,
                    _JAIL_TEARDOWN_SECONDS,
                )
                with self._lock:
                    self._unreaped.add(leader)
                break
            time.sleep(_JAIL_TEARDOWN_POLL_SECONDS)
        with self._lock:
            self._unreaped = {group for group in self._unreaped if not _reap_group(group)}


_jail_reaper = _JailReaper()


class BubblewrapSandbox(LocalSubprocessSandbox):
    """Local backend with bubblewrap namespace isolation (network + filesystem). The jail's root runs as
    :attr:`program_owner`, this process's user's first subordinate uid and gid."""

    # A session builds once, on an empty stdin, and the jailed program can neither remove its working
    # directory (a mount point) to force a rebuild nor write outside it for one to include.
    COMPILES_WITHOUT_TEST_INPUT = True

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
        if os.geteuid() != 0:
            raise RuntimeError(
                "bubblewrap sandbox runs the jail's root as a subordinate uid of root and hands it the working "
                "directory, which only root may do: run the trainer as root, as the training containers do, or use "
                "the 'local' or 'remote' sandbox backend."
            )
        ids = tuple(_subordinate_id(path) for path in _SUBORDINATE_ID_FILES)
        self._unshare_path = shutil.which("unshare")
        if None in ids or not (self._unshare_path and shutil.which("newuidmap") and shutil.which("newgidmap")):
            raise RuntimeError(
                "bubblewrap sandbox runs the jail's root as a subordinate uid and gid of root, so a jailed program "
                "owns no host file and its process count is bounded. It needs util-linux `unshare`, the uidmap "
                "package (newuidmap, newgidmap) and a range for root in /etc/subuid and /etc/subgid, which the "
                "training image ships. Otherwise use the 'local' or 'remote' sandbox backend."
            )
        self.program_owner = ids
        self._verify_can_sandbox()

    @property
    def isolated(self) -> bool:
        """Confined only while the jail has its own network namespace."""
        return not self.allow_network

    def _verify_can_sandbox(self) -> None:
        """Raise at construction if bwrap cannot create a sandbox here: a blocked namespace makes ``bwrap`` exit
        before the child runs, which grading would read as a failed solution."""
        try:
            probe = self.run("pass")  # exercises the real wrapped python run path once
        except PermissionError as exc:
            raise RuntimeError(
                f"bubblewrap sandbox cannot hand its working directory under {tempfile.gettempdir()} to the jail's "
                f"root, uid {self.program_owner[0]} ({exc}): that filesystem refuses root's chown (a root-squashed "
                "NFS mount, a user-namespace-remapped runtime). Point TMPDIR at a local filesystem, or use the "
                "'local' or 'remote' sandbox backend."
            ) from exc
        if probe.ok:
            return
        detail = (probe.error or probe.stderr or "").strip().splitlines()
        tail = detail[-1] if detail else f"exit code {probe.returncode}"
        if _PROC_MOUNT_REFUSED in tail:
            raise RuntimeError(
                f"bubblewrap cannot mount the jail's /proc ({tail}): the kernel grants one only while a proc "
                "mount in the container is fully visible, and the container's /proc has masks over parts of it "
                "(Docker's; in a GPU container, the NVIDIA runtime's). Mount a clean one beside it before the "
                f"trainer starts: `{_CLEAN_PROC_MOUNT}` (needs CAP_SYS_ADMIN). Otherwise use the 'local' or "
                "'remote' sandbox backend."
            )
        if _BIND_SOURCE_UNREACHABLE in tail:
            raise RuntimeError(
                f"bubblewrap's jail cannot reach a path it binds ({tail}): its root is uid {self.program_owner[0]}, "
                "not this process's, so every directory above a bound path must be searchable by other users, "
                f"as /tmp is. The working directories live under {tempfile.gettempdir()}: point TMPDIR at such a "
                "directory, or use the 'local' or 'remote' sandbox backend."
            )
        raise RuntimeError(
            f"bubblewrap cannot create a sandbox in this environment ({tail}). The container needs "
            "permission to create namespaces — run it with --privileged (or --cap-add SYS_ADMIN "
            "--security-opt seccomp=unconfined --security-opt apparmor=unconfined). Otherwise use the "
            "'local' or 'remote' sandbox backend."
        )

    def _run_in_new_session(
        self,
        argv: list[str],
        *,
        stdin: str,
        timeout: float,
        cwd: str,
        env: dict[str, str],
        started: tuple[int, int] | None = None,
    ) -> tuple[str, str, int | None, bool]:
        """Runs inside the subreaper window :meth:`_reap_adopted` needs. The init cannot die of its program's signal,
        so ``bwrap`` exits ``128 + n`` for a program killed by signal ``n``, returned as ``subprocess`` reports it,
        ``-n``. A program that exits with such a code itself reads as that signal."""
        with _jail_reaper.held():
            stdout, stderr, returncode, timed_out = super()._run_in_new_session(
                argv, stdin=stdin, timeout=timeout, cwd=cwd, env=env, started=started
            )
        if returncode is not None and 128 < returncode < 128 + signal.NSIG:
            returncode = 128 - returncode
        return stdout, stderr, returncode, timed_out

    @staticmethod
    def _child_env(workdir: str) -> dict[str, str]:
        return LocalSubprocessSandbox._child_env(_JAIL_WORKDIR)

    def _reap_adopted(self, leader: int) -> None:
        _jail_reaper.reap(leader)

    def _wrap_command(self, argv: list[str], workdir: str) -> list[str]:
        # Not --new-session, which takes the init out of the run's process group, where the group kill reaches it and
        # _reap_adopted finds it. The program gets its own session inside the jail instead, so a `kill(0, ...)` it
        # sends reaches neither bwrap nor the init, as on `local`.
        uid, gid = self.program_owner
        wrapper: list[str] = [
            self._unshare_path,
            "--user",
            f"--map-users=0:{uid}:1",
            f"--map-groups=0:{gid}:1",
            "--setuid",
            "0",
            "--setgid",
            "0",
            "--",
            self.bwrap_path,
            "--die-with-parent",  # kill the jail if the worker dies
            "--cap-drop",  # the jail's root keeps no capability in its namespace: it cannot remount a
            "ALL",  # read-only bind writable and reach the host files under it
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup",
        ]
        if not self.allow_network:
            wrapper.append("--unshare-net")
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
            _JAIL_WORKDIR,  # the only writable mount
            "--chdir",
            _JAIL_WORKDIR,
        ]
        return wrapper + ["--", "setsid", *argv]
