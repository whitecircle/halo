"""The in-sandbox half of the hosted backends (Modal, Baseten): one JSON request in, one JSON reply out. A backend
starts it as root, with the request on stdin or in the file its argument names. Standard library only, since it runs
on the sandbox image's own Python. Limits and the program's environment arrive in the request; the client
(``runner_session``) turns the reply into a verdict.

A session is a :class:`Workspace`: a directory and a user of its own. Programs run as that user, so they cannot signal
this runner or another session's programs, nor open another session's files. None of a session's processes outlives
its run, so nothing races the runner while it reads and writes the directory."""

import functools
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

COMMAND_NOT_FOUND = 127


class Tampered(Exception):
    """A program put a link, directory or FIFO where the runner reads or writes a session file."""


class Workspace:
    """One session's directory, which only the session's user ``uid`` can enter."""

    def __init__(self, path: str, uid: int):
        self.root = Path(path).resolve()
        self.uid = uid

    def create(self) -> bool:
        """Whether the directory had to be made: a new one holds no build and no earlier file."""
        if self.root.is_dir():
            return False
        self.root.mkdir(parents=True, mode=0o700)
        shutil.chown(self.root, self.uid, self.uid)
        return True

    def member(self, name: str) -> Path:
        """The path of the session file ``name``, refused unless it is a regular file, or absent, behind no link."""
        path = self.root / name
        if path.parent.resolve() != path.parent or path.is_symlink() or (path.exists() and not path.is_file()):
            raise Tampered(f"session path {name!r} is not a regular file")
        return path

    def write(self, files: dict) -> None:
        for name, content in files.items():
            path = self.member(name)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except (FileExistsError, NotADirectoryError) as exc:
                raise Tampered(f"session path {name!r} is not a regular file") from exc
            path.write_text(content, encoding="utf-8")
            # Handed to the session's user, with the directories above it, so its programs can change them.
            for owned in (path, *(self.root / parent for parent in path.relative_to(self.root).parents[:-1])):
                shutil.chown(owned, self.uid, self.uid)

    def read(self, name: str):
        """The file's text, or ``None`` for anything but a regular file: a planted link carries nothing out."""
        try:
            return self.member(name).read_text(encoding="utf-8", errors="replace")
        except (Tampered, FileNotFoundError, NotADirectoryError):
            return None

    def files(self) -> list:
        found = []
        for directory, _subdirectories, names in os.walk(self.root):
            found.extend(os.path.relpath(os.path.join(directory, name), self.root) for name in names)
        return sorted(found)

    def entries(self) -> list:
        return sorted(path.name for path in self.root.iterdir())

    def keep_only(self, entries: list) -> None:
        """Remove every top-level entry not in ``entries``: what the last run left behind."""
        for path in self.root.iterdir():
            if path.name in entries:
                continue
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    def launch(self, argv: list, *, stdin: str, timeout: float, limits: dict, env: dict) -> dict:
        """Run ``argv`` in the directory as the session's user, under ``limits``, for at most ``timeout`` seconds.
        Stdio are files, so a process the program leaves behind holds nothing open."""
        with tempfile.TemporaryFile() as feed, tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            feed.write(stdin.encode("utf-8"))
            feed.seek(0)
            returncode, timed_out = self.wait_for(
                argv,
                timeout,
                stdin=feed,
                stdout=out,
                stderr=err,
                env={"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), **env},
                preexec_fn=functools.partial(limit_resources, limits),
            )
            return {"stdout": text_of(out), "stderr": text_of(err), "returncode": returncode, "timed_out": timed_out}

    def wait_for(self, argv: list, timeout: float, **popen) -> tuple:
        """``(returncode, timed_out)`` of ``argv`` run as the session's user, none of whose processes it leaves."""
        try:
            finished = subprocess.run(
                argv,
                timeout=timeout,
                cwd=self.root,
                start_new_session=True,
                user=self.uid,
                group=self.uid,
                extra_groups=[],
                **popen,
            )
            return finished.returncode, False
        except subprocess.TimeoutExpired:
            return None, True
        except FileNotFoundError:
            popen["stderr"].write(f"{argv[0]}: not found".encode())
            return COMMAND_NOT_FOUND, False
        finally:
            self.end_processes()

    def end_processes(self) -> None:
        """SIGKILL every process the session's user has, wherever it moved to. The signal comes from a child that has
        become that user, so the kernel picks the targets."""
        child = os.fork()
        if child == 0:
            try:
                os.setgid(self.uid)
                os.setuid(self.uid)
                os.kill(-1, signal.SIGKILL)
            finally:
                os._exit(0)
        os.waitpid(child, 0)


def limit_resources(limits: dict) -> None:
    """Set the rlimits of the process calling it: the program, between fork and exec."""
    for name, value in limits.items():
        kind = getattr(resource, name)
        hard = resource.getrlimit(kind)[1]
        value = value if hard == resource.RLIM_INFINITY else min(value, hard)
        resource.setrlimit(kind, (value, value))


def text_of(capture) -> str:
    capture.seek(0)
    return capture.read().decode("utf-8", errors="replace")


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
    for retired in request.get("remove", ()):
        shutil.rmtree(retired, ignore_errors=True)
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
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8") if len(sys.argv) > 1 else sys.stdin.read())
    sys.stdout.write(json.dumps(handle(request)))


if __name__ == "__main__":
    main()
