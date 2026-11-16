"""The half of the Modal and Baseten backends that runs inside the sandbox. It is started as root once per call,
reads one Request, and writes one Reply.

Each session has its own directory and its own user. A program runs as that user, so it cannot signal the runner or
reach another session, and every process of that user is killed when the run ends."""

import functools
import os
import resource
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from halo.environments.sandbox.runner_messages import (
    Command,
    FileContent,
    FileList,
    FileWritten,
    Limits,
    ListFiles,
    Outcome,
    ProgramRan,
    ReadFile,
    Reply,
    Request,
    RunProgram,
    Tampered,
    WriteFile,
    decode,
    encode,
)
from halo.environments.sandbox.workspace import Owner, SessionPathError, Workspace, walk_tree

COMMAND_NOT_FOUND = 127
DEFAULT_PATH = "/usr/local/bin:/usr/bin:/bin"


def set_limit(kind: int, wanted: int) -> None:
    hard = resource.getrlimit(kind)[1]
    value = wanted if hard == resource.RLIM_INFINITY else min(wanted, hard)
    resource.setrlimit(kind, (value, value))


def apply_limits(limits: Limits) -> None:
    set_limit(resource.RLIMIT_CPU, limits.cpu_seconds)
    set_limit(resource.RLIMIT_FSIZE, limits.file_bytes)
    set_limit(resource.RLIMIT_CORE, 0)
    if limits.memory_bytes is not None:
        set_limit(resource.RLIMIT_AS, limits.memory_bytes)
    if limits.stack_bytes is not None:
        set_limit(resource.RLIMIT_STACK, limits.stack_bytes)
    if limits.processes is not None:
        set_limit(resource.RLIMIT_NPROC, limits.processes)


def kill_processes_of(uid: int) -> None:
    # A program can leave its process group but not its user: a child that becomes that user signals all of them.
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


def execute(workspace: Workspace, command: Command, stdin: str, env: dict[str, str]) -> Outcome:
    owner = workspace.owner
    returncode = None
    timed_out = False
    with tempfile.TemporaryFile() as stdin_file, tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        stdin_file.write(stdin.encode("utf-8"))
        stdin_file.seek(0)
        try:
            finished = subprocess.run(
                command.argv,
                stdin=stdin_file,
                stdout=out,
                stderr=err,
                timeout=command.timeout,
                check=False,
                cwd=workspace.path,
                env={"PATH": os.environ.get("PATH", DEFAULT_PATH), **env},
                start_new_session=True,
                user=owner.uid,
                group=owner.gid,
                extra_groups=[],
                preexec_fn=functools.partial(apply_limits, command.limits),
            )
            returncode = finished.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
        except FileNotFoundError:
            err.write(f"{command.argv[0]}: not found".encode())
            returncode = COMMAND_NOT_FOUND
        finally:
            kill_processes_of(owner.uid)
        return Outcome(stdout=text_of(out), stderr=text_of(err), returncode=returncode, timed_out=timed_out)


def run_program(workspace: Workspace, request: RunProgram, new_workspace: bool) -> ProgramRan:
    for source in request.sources:
        workspace.write(source.name, source.content)

    build = None
    if request.build is not None and (request.build_requested or new_workspace):
        build = execute(workspace, request.build, stdin="", env=request.env)
        if build.failed:
            return ProgramRan(build=build, run=None, entries=[])

    entries = sorted(workspace.entries())
    run = execute(workspace, request.run, stdin=request.stdin, env=request.env)
    return ProgramRan(build=build, run=run, entries=entries)


def perform(workspace: Workspace, operation, new_workspace: bool):
    if isinstance(operation, RunProgram):
        return run_program(workspace, operation, new_workspace)
    if isinstance(operation, WriteFile):
        workspace.write(operation.file.name, operation.file.content)
        return FileWritten()
    if isinstance(operation, ReadFile):
        return FileContent(workspace.read(operation.name))
    if isinstance(operation, ListFiles):
        return FileList(workspace.files())
    raise TypeError(f"unknown operation {type(operation).__name__}")


def open_workspace(request: Request) -> tuple[Workspace, bool]:
    workspace = Workspace(request.workdir, Owner(uid=request.uid, gid=request.uid))
    directory = Path(workspace.path)
    if directory.is_dir():
        return workspace, False
    directory.mkdir(parents=True, mode=0o700)
    workspace.hand_over()
    return workspace, True


def handle(request: Request) -> Reply:
    for workdir in request.closed_workdirs:
        walk_tree(workdir, remove=True)

    workspace, new_workspace = open_workspace(request)
    if request.keep_only is not None:
        workspace.keep_only(set(request.keep_only))

    try:
        result = perform(workspace, request.operation, new_workspace)
    except (SessionPathError, NotADirectoryError) as problem:
        result = Tampered(str(problem))
    return Reply(new_workspace=new_workspace, result=result)


def read_request() -> Request:
    if len(sys.argv) > 1:
        return decode(Path(sys.argv[1]).read_text(encoding="utf-8"))
    return decode(sys.stdin.read())


def main() -> None:
    if os.geteuid() != 0:
        sys.exit("the sandbox image must run as root: each session runs as its own user")
    sys.stdout.write(encode(handle(read_request())))


if __name__ == "__main__":
    main()
