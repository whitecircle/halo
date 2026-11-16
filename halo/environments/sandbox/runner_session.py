"""The trainer's side of the program runner: a session that sends Requests through a backend's sandbox and reads the
Replies as verdicts. A backend supplies :meth:`RunnerSession.exchange`."""

import functools
import math
import re
import signal
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

from halo.environments.sandbox.base import SANDBOX_DEFAULT_TIMEOUT, SandboxSession
from halo.environments.sandbox.languages import INTERPRETER_PLACEHOLDER, LanguageSpec, require_language, utf8_encodable
from halo.environments.sandbox.local import (
    LOCAL_FSIZE_LIMIT,
    LOCAL_NPROC_LIMIT,
    RLIMIT_CPU_SLACK_SECONDS,
    TAMPERED_WORKDIR_RETURNCODE,
    build_program_env,
)
from halo.environments.sandbox.results import (
    COMMAND_NOT_FOUND_RETURNCODE,
    SandboxInfraError,
    SandboxResult,
    compile_failure_verdict,
    compile_limit_verdict,
)
from halo.environments.sandbox.runner_messages import (
    Command,
    FileContent,
    FileList,
    Limits,
    ListFiles,
    Outcome,
    ProgramRan,
    ReadFile,
    Reply,
    Request,
    RunProgram,
    SourceFile,
    Tampered,
    WriteFile,
    decode,
)
from halo.environments.sandbox.workspace import Entry, SessionPathError

# Seconds a call may take beyond the program's own time limits: transport and the runner's start.
EXCHANGE_OVERHEAD_SECONDS = 30.0
# The users programs run as, within the 16 bits a user-namespaced container maps.
SESSION_UIDS = range(20_000, 60_000)
MIB = 1024 * 1024

_RUNNER_FILES = ("workspace.py", "runner_messages.py", "program_runner.py")
_HALO_IMPORT = re.compile(r"^from halo\.[\w.]+ import (\([^)]*\)|[^\n(]*)\n", re.MULTILINE)


@functools.cache
def runner_source() -> str:
    """The runner and the modules it uses as one file: a sandbox has no ``halo`` package to import them from."""
    sources = [Path(__file__).with_name(name).read_text(encoding="utf-8") for name in _RUNNER_FILES]
    return "\n".join(_HALO_IMPORT.sub("", source) for source in sources)


def decode_reply(text: str) -> Reply:
    try:
        reply = decode(text)
    except (ValueError, KeyError, TypeError) as error:
        raise SandboxInfraError("sandbox runner returned malformed output") from error
    if not isinstance(reply, Reply):
        raise SandboxInfraError("sandbox runner returned malformed output")
    return reply


@dataclass(frozen=True)
class ProgramLimits:
    memory_limit_mb: int
    compile_timeout: float
    compile_memory_limit_mb: int

    def for_run(self, spec: LanguageSpec, timeout: float) -> Limits:
        memory_bytes = self.memory_limit_mb * MIB if self.memory_limit_mb else None
        return Limits(
            cpu_seconds=math.ceil(timeout + RLIMIT_CPU_SLACK_SECONDS),
            file_bytes=LOCAL_FSIZE_LIMIT,
            memory_bytes=memory_bytes,
            stack_bytes=memory_bytes if spec.is_compiled else None,
            processes=LOCAL_NPROC_LIMIT,
        )

    def for_build(self) -> Limits:
        return Limits(
            cpu_seconds=math.ceil(self.compile_timeout + RLIMIT_CPU_SLACK_SECONDS),
            file_bytes=LOCAL_FSIZE_LIMIT,
            memory_bytes=self.compile_memory_limit_mb * MIB if self.compile_memory_limit_mb else None,
            stack_bytes=None,
            processes=None,
        )


@dataclass(frozen=True)
class Program:
    language: str
    code: str
    files: tuple[SourceFile, ...]


@dataclass(frozen=True)
class Build:
    program: Program
    # None once the program is built and runnable.
    verdict: SandboxResult | None


class RunnerSession(SandboxSession):
    """A working directory in a sandbox. A compiled program is built once and rebuilt only when its language, source
    or files change, or when the sandbox lost the directory."""

    def __init__(self, workdir: str, *, uid: int, limits: ProgramLimits, interpreter: str):
        self.workdir = workdir
        self.uid = uid
        self.limits = limits
        self.interpreter = interpreter
        self._build: Build | None = None
        self._prepared: list[Entry] | None = None
        self._reset_pending = False

    @abstractmethod
    def exchange(self, request: Request, *, deadline: float) -> Reply:
        """Run the runner on ``request`` in this session's sandbox. Raises :class:`SandboxInfraError` when the
        sandbox, the transport or the runner fails."""

    def run(
        self,
        code: str,
        *,
        stdin: str = "",
        timeout: float = SANDBOX_DEFAULT_TIMEOUT,
        language: str = "python",
        files: dict[str, str] | None = None,
    ) -> SandboxResult:
        try:
            spec = require_language(language)
        except ValueError as error:
            return SandboxResult(error=str(error))
        files = files or {}
        for name in files:
            if not self.is_safe_member_name(name):
                return SandboxResult(error=f"unsafe auxiliary file path: {name!r}")

        auxiliary = tuple(SourceFile(name, utf8_encodable(content)) for name, content in sorted(files.items()))
        program = Program(spec.name, code, auxiliary)
        built = spec.is_compiled and self._build is not None and self._build.program == program
        if built and self._build.verdict is not None:
            return self._build.verdict
        if not built:
            self._build = None

        operation = RunProgram(
            sources=[*auxiliary, SourceFile(spec.source_name, utf8_encodable(code))],
            stdin=utf8_encodable(stdin),
            env=build_program_env(self.workdir),
            run=Command(self._run_argv(spec), timeout, self.limits.for_run(spec, timeout)),
            build=self._build_command(spec),
            build_requested=not built,
        )
        deadline = EXCHANGE_OVERHEAD_SECONDS + timeout + (self.limits.compile_timeout if spec.is_compiled else 0)
        try:
            result = self._send(operation, deadline=deadline)
        except SandboxInfraError as error:
            return SandboxResult(error=str(error))
        if isinstance(result, Tampered):
            self._build = None
            return SandboxResult(
                stderr=f"working directory tampered: {result.reason}", returncode=TAMPERED_WORKDIR_RETURNCODE
            )
        return self._verdict(result, spec, program)

    def reset_to_prepared(self) -> None:
        self._reset_pending = self._prepared is not None

    def write_file(self, path: str, content: str) -> None:
        self.reject_unsafe_path(path)
        self._build = None
        result = self._send(WriteFile(SourceFile(path, utf8_encodable(content))))
        self._prepared = None
        if isinstance(result, Tampered):
            raise SessionPathError(result.reason)

    def read_file(self, path: str) -> str | None:
        self.reject_unsafe_path(path)
        result = self._send(ReadFile(path))
        return result.content if isinstance(result, FileContent) else None

    def list_files(self) -> list[str]:
        result = self._send(ListFiles())
        return result.names if isinstance(result, FileList) else []

    def _send(self, operation, *, deadline: float = EXCHANGE_OVERHEAD_SECONDS):
        request = Request(
            workdir=self.workdir,
            uid=self.uid,
            closed_workdirs=[],
            keep_only=self._prepared if self._reset_pending else None,
            operation=operation,
        )
        reply = self.exchange(request, deadline=deadline)
        self._reset_pending = False
        if reply.new_workspace:
            self._build = None
            self._prepared = None
        return reply.result

    def _run_argv(self, spec: LanguageSpec) -> list[str]:
        return [self.interpreter if token == INTERPRETER_PLACEHOLDER else token for token in spec.run_argv]

    def _build_command(self, spec: LanguageSpec) -> Command | None:
        if not spec.is_compiled:
            return None
        return Command(list(spec.compile_argv), self.limits.compile_timeout, self.limits.for_build())

    def _verdict(self, ran: ProgramRan, spec: LanguageSpec, program: Program) -> SandboxResult:
        if ran.build is not None:
            build_verdict = self._build_verdict(ran.build, spec)
            self._build = Build(program, build_verdict)
            if build_verdict is not None:
                return build_verdict
        self._prepared = ran.entries
        return SandboxResult(
            stdout=ran.run.stdout,
            stderr=ran.run.stderr,
            returncode=ran.run.returncode,
            timed_out=ran.run.timed_out or ran.run.returncode == -signal.SIGXCPU,
        )

    def _build_verdict(self, build: Outcome, spec: LanguageSpec) -> SandboxResult | None:
        """``None`` when the program was built. A missing compiler is the backend's fault, not the program's."""
        if build.timed_out:
            return compile_limit_verdict(f"compilation timed out after {self.limits.compile_timeout:g} s")
        if build.returncode == COMMAND_NOT_FOUND_RETURNCODE:
            return SandboxResult(error=f"compiler not found: {spec.compile_argv[0]!r}", stderr=build.stderr.strip())
        if build.returncode != 0:
            return compile_failure_verdict(build.stderr, build.stdout, build.returncode)
        return None
