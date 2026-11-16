"""The in-sandbox half of the hosted backends (Modal, Baseten): one JSON request in, one JSON reply out. A backend
starts it as root, with the request on stdin or in the file its argument names. Standard library only, since it runs
on the sandbox image's own Python. Limits and the program's environment arrive in the request; the client
(``runner_session``) turns the reply into a verdict.

A session is a :class:`Workspace`: a directory and a user of its own. Programs run as that user, so they cannot signal
this runner or another session's programs, nor open another session's files. None of a session's processes outlives
its run, so nothing races the runner while it reads and writes the directory."""

import contextlib
import errno
import json
import os
import resource
import shutil
import signal
import stat
import subprocess
import sys
import tempfile

COMMAND_NOT_FOUND = 127
# What an open or mkdir meets where a program put a link, directory, FIFO or file in a session entry's place.
TAMPER_ERRNOS = frozenset({errno.ELOOP, errno.EISDIR, errno.ENXIO, errno.ENOTDIR, errno.EEXIST})
END_PROCESSES_ATTEMPTS = 5


class Tampered(Exception):
    """A program replaced a session entry with something the runner must not write through or read."""


class Workspace:
    """One session's directory, owned by the session's user ``uid``."""

    def __init__(self, path: str, uid: int):
        self.path = os.path.realpath(path)
        self.uid = uid

    def create(self) -> bool:
        """Whether the directory had to be made: a new one holds no build and no earlier file."""
        if os.path.isdir(self.path):
            return False
        os.makedirs(self.path)
        os.chown(self.path, self.uid, self.uid)
        os.chmod(self.path, 0o700)
        return True

    def open(self, name: str, flags: int) -> int:
        """A descriptor on the regular file ``name``, following no link on the way; a file it creates goes to the
        session's user. ``O_NONBLOCK`` keeps a FIFO from blocking the runner."""
        target = os.path.normpath(os.path.join(self.path, name))
        if os.path.realpath(target) != target:
            raise Tampered(f"session path {name!r} is a link")
        creating = bool(flags & os.O_CREAT)
        try:
            if creating:
                self._make_parents(os.path.dirname(target))
            fd = os.open(target, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o644)
        except OSError as exc:
            if exc.errno in TAMPER_ERRNOS:
                raise Tampered(f"session path {name!r} is not a regular file") from exc
            raise
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise Tampered(f"session path {name!r} is not a regular file")
        if creating:
            os.fchown(fd, self.uid, self.uid)
        return fd

    def _make_parents(self, directory: str) -> None:
        if os.path.isdir(directory):
            return
        self._make_parents(os.path.dirname(directory))
        os.mkdir(directory)
        os.chown(directory, self.uid, self.uid)

    def write(self, files: dict) -> None:
        for name, content in files.items():
            with os.fdopen(self.open(name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC), "w", encoding="utf-8") as fh:
                fh.write(content)

    def read(self, name: str):
        """The file's text, or ``None`` for anything but a regular file: a planted link carries nothing out."""
        try:
            fd = self.open(name, os.O_RDONLY)
        except (Tampered, FileNotFoundError):
            return None
        with os.fdopen(fd, encoding="utf-8", errors="replace") as fh:
            return fh.read()

    def files(self) -> list:
        found = []
        for root, _dirs, names in os.walk(self.path):
            found.extend(os.path.relpath(os.path.join(root, name), self.path) for name in names)
        return sorted(found)

    def entries(self) -> list:
        return sorted(os.listdir(self.path))

    def keep_only(self, entries: list) -> None:
        """Remove every top-level entry not in ``entries``: what the last run left behind."""
        for name in set(os.listdir(self.path)) - set(entries):
            path = os.path.join(self.path, name)
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                with contextlib.suppress(OSError):
                    os.remove(path)

    def launch(self, argv: list, *, stdin: str, timeout: float, limits: dict, env: dict) -> dict:
        """Run ``argv`` in the directory as the session's user, under ``limits``, for at most ``timeout`` seconds.
        Stdio are files, so a process left running holds nothing open."""

        def apply_limits():
            for name, value in limits.items():
                kind = getattr(resource, name)
                if name == "RLIMIT_STACK":
                    hard = resource.getrlimit(kind)[1]
                    value = value if hard == resource.RLIM_INFINITY else min(value, hard)
                resource.setrlimit(kind, (value, value))

        with tempfile.TemporaryFile() as feed, tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            feed.write(stdin.encode("utf-8"))
            feed.seek(0)
            try:
                process = subprocess.Popen(
                    argv,
                    stdin=feed,
                    stdout=out,
                    stderr=err,
                    cwd=self.path,
                    env={"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), **env},
                    start_new_session=True,
                    user=self.uid,
                    group=self.uid,
                    extra_groups=[],
                    preexec_fn=apply_limits,
                )
            except FileNotFoundError:
                return {
                    "stdout": "",
                    "stderr": f"{argv[0]}: not found",
                    "returncode": COMMAND_NOT_FOUND,
                    "timed_out": False,
                }
            timed_out = False
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            self.end_processes()
            process.wait()
            out.seek(0)
            err.seek(0)
            return {
                "stdout": out.read().decode("utf-8", errors="replace"),
                "stderr": err.read().decode("utf-8", errors="replace"),
                "returncode": process.returncode,
                "timed_out": timed_out,
            }

    def end_processes(self) -> None:
        """SIGKILL every process the session's user has, wherever it moved to. Signalled from a child that has become
        that user, so the kernel picks the targets."""
        for _ in range(END_PROCESSES_ATTEMPTS):
            child = os.fork()
            if child == 0:
                try:
                    os.setgid(self.uid)
                    os.setuid(self.uid)
                    os.kill(-1, signal.SIGKILL)
                finally:
                    os._exit(0)
            os.waitpid(child, 0)
            if not self._has_processes():
                return

    def _has_processes(self) -> bool:
        for entry in os.listdir("/proc"):
            if entry.isdigit():
                with contextlib.suppress(OSError), open(f"/proc/{entry}/status") as fh:
                    status = fh.read()
                    if f"\nUid:\t{self.uid}\t" in status and "\nState:\tZ" not in status:
                        return True
        return False


def run_program(workspace: Workspace, request: dict, fresh: bool) -> dict:
    """Write the sources, build when asked or when the directory is new, then run."""
    workspace.write({**request["files"], request["source_name"]: request["code"]})
    reply = {}
    if request.get("compile_argv") and (request["build"] or fresh):
        reply["compile"] = workspace.launch(
            request["compile_argv"],
            stdin="",
            timeout=request["compile_timeout"],
            limits=request["compile_limits"],
            env=request["env"],
        )
        if reply["compile"]["timed_out"] or reply["compile"]["returncode"] != 0:
            return reply
    reply["entries"] = workspace.entries()
    reply["run"] = workspace.launch(
        request["run_argv"],
        stdin=request["stdin"],
        timeout=request["timeout"],
        limits=request["limits"],
        env=request["env"],
    )
    return reply


def handle(request: dict) -> dict:
    for stale in request.get("remove", ()):
        shutil.rmtree(stale, ignore_errors=True)
    workspace = Workspace(request["workdir"], request["uid"])
    fresh = workspace.create()
    if request.get("reset_to") is not None:
        workspace.keep_only(request["reset_to"])
    operation = request["op"]
    try:
        if operation == "run":
            reply = run_program(workspace, request, fresh)
        elif operation == "write":
            workspace.write(request["files"])
            reply = {}
        elif operation == "read":
            reply = {"content": workspace.read(request["path"])}
        elif operation == "list":
            reply = {"files": workspace.files()}
        else:
            sys.exit(f"unknown op {operation!r}")
    except Tampered as exc:
        reply = {"tampered": str(exc)}
    return {**reply, "fresh": fresh}


def main() -> None:
    if os.geteuid() != 0:
        sys.exit("the sandbox image must run as root: the runner gives each session its own user")
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as fh:
            request = json.load(fh)
    else:
        request = json.load(sys.stdin)
    sys.stdout.write(json.dumps(handle(request)))


if __name__ == "__main__":
    main()
