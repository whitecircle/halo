"""The in-sandbox half of the hosted backends (Modal, Baseten): one JSON request in, one JSON reply out. A backend
starts it as root, with the request on stdin or in the file its argument names. It is shipped with ``workspace.py``
as one file (``runner_session.RUNNER_SOURCE``) and uses the standard library only, since it runs on the sandbox
image's own Python. Limits and the program's environment arrive in the request; the client (``runner_session``) turns
the reply into a verdict.

Each session has a :class:`Workspace` and a user of its own. Programs run as that user, so they cannot signal this
runner or another session's programs, nor open another session's files, and none of a session's processes outlives
its run."""

import functools
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile

from halo.environments.sandbox.workspace import SessionPathError, Workspace, walk_tree

COMMAND_NOT_FOUND = 127


def limit_resources(limits: dict) -> None:
    """Set the rlimits of the process calling it: the program, between fork and exec."""
    for name, value in limits.items():
        kind = getattr(resource, name)
        hard = resource.getrlimit(kind)[1]
        value = value if hard == resource.RLIM_INFINITY else min(value, hard)
        resource.setrlimit(kind, (value, value))


def end_processes(uid: int) -> None:
    """SIGKILL every process user ``uid`` has, wherever it moved to. The signal comes from a child that has become
    that user, so the kernel picks the targets."""
    child = os.fork()
    if child == 0:
        try:
            os.setgid(uid)
            os.setuid(uid)
            os.kill(-1, signal.SIGKILL)
        finally:
            os._exit(0)
    os.waitpid(child, 0)


def text_of(capture) -> str:
    capture.seek(0)
    return capture.read().decode("utf-8", errors="replace")


def launch(workspace: Workspace, argv: list, *, stdin: str, timeout: float, limits: dict, env: dict) -> dict:
    """Run ``argv`` in the workspace as its user, under ``limits``, for at most ``timeout`` seconds. Stdio are files,
    so a process the program leaves behind holds nothing open; no such process outlives the call."""
    uid, gid = workspace.owner
    outcome = {"returncode": None, "timed_out": False}
    with tempfile.TemporaryFile() as feed, tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        feed.write(stdin.encode("utf-8"))
        feed.seek(0)
        try:
            outcome["returncode"] = subprocess.run(
                argv,
                stdin=feed,
                stdout=out,
                stderr=err,
                timeout=timeout,
                cwd=workspace.path,
                env={"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), **env},
                start_new_session=True,
                user=uid,
                group=gid,
                extra_groups=[],
                preexec_fn=functools.partial(limit_resources, limits),
            ).returncode
        except subprocess.TimeoutExpired:
            outcome["timed_out"] = True
        except FileNotFoundError:
            err.write(f"{argv[0]}: not found".encode())
            outcome["returncode"] = COMMAND_NOT_FOUND
        finally:
            end_processes(uid)
        return {"stdout": text_of(out), "stderr": text_of(err), **outcome}


def run_program(workspace: Workspace, request: dict, fresh: bool) -> dict:
    """Write the sources, build when asked or when the directory is new, then run."""
    for name, content in {**request["files"], request["source_name"]: request["code"]}.items():
        workspace.write(name, content)
    reply = {}
    if request.get("compile_argv") and (request["build"] or fresh):
        reply["compile"] = launch(
            workspace,
            request["compile_argv"],
            stdin="",
            timeout=request["compile_timeout"],
            limits=request["compile_limits"],
            env=request["env"],
        )
        if reply["compile"]["timed_out"] or reply["compile"]["returncode"] != 0:
            return reply
    reply["entries"] = sorted(workspace.entries())
    reply["run"] = launch(
        workspace,
        request["run_argv"],
        stdin=request["stdin"],
        timeout=request["timeout"],
        limits=request["limits"],
        env=request["env"],
    )
    return reply


def handle(request: dict) -> dict:
    for retired in request.get("remove", ()):
        walk_tree(retired, remove=True)
    workspace = Workspace(request["workdir"], (request["uid"], request["uid"]))
    fresh = not os.path.isdir(workspace.path)
    if fresh:
        # A new directory holds no build and no earlier file; only its user can enter it.
        os.makedirs(workspace.path, mode=0o700)
        workspace.hand_over()
    if request.get("reset_to") is not None:
        workspace.keep_only({tuple(entry) for entry in request["reset_to"]})
    operation = request["op"]
    try:
        if operation == "run":
            reply = run_program(workspace, request, fresh)
        elif operation == "write":
            for name, content in request["files"].items():
                workspace.write(name, content)
            reply = {}
        elif operation == "read":
            reply = {"content": workspace.read(request["path"])}
        elif operation == "list":
            reply = {"files": workspace.files()}
        else:
            sys.exit(f"unknown op {operation!r}")
    except (SessionPathError, NotADirectoryError) as exc:
        # The program put a link, directory, FIFO or file where a session file or its directory belongs.
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
