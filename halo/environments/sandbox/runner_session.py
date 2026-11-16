"""The client half of ``program_runner``, shared by the hosted backends (Modal, Baseten): what a session asks the
runner to do and how the runner's replies read as verdicts. A backend adds the transport, :meth:`RunnerSession.exchange`."""

import json
import math
import signal
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from halo.environments.sandbox.base import SANDBOX_DEFAULT_TIMEOUT, SandboxSession
from halo.environments.sandbox.languages import INTERPRETER_PLACEHOLDER, LanguageSpec, require_language, utf8_encodable
from halo.environments.sandbox.local import (
    LOCAL_FSIZE_LIMIT,
    LOCAL_NPROC_LIMIT,
    RLIMIT_CPU_SLACK_SECONDS,
    SANDBOX_DEFAULT_COMPILE_MEMORY_MB,
    SANDBOX_DEFAULT_COMPILE_TIMEOUT,
    SANDBOX_DEFAULT_MEMORY_MB,
    TAMPERED_WORKDIR_RETURNCODE,
    SessionPathError,
    build_program_env,
)
from halo.environments.sandbox.results import (
    COMMAND_NOT_FOUND_RETURNCODE,
    SandboxInfraError,
    SandboxResult,
    compile_failure_verdict,
    compile_limit_verdict,
)

RUNNER_SOURCE = Path(__file__).with_name("program_runner.py").read_text()
# Deadline slack over the runner's own limits, for transport and the runner's start.
EXCHANGE_OVERHEAD_SECONDS = 30.0
# The users sessions' programs run as. A backend holding several sessions in one sandbox hands them out in turn; one
# comes round again only after every other has been used, long after its last session's directory is gone. Within
# 16 bits, the range a user-namespaced container maps.
SESSION_UIDS = range(20_000, 60_000)
MIB = 1024 * 1024


@dataclass(frozen=True)
class ProgramLimits:
    """The limits of ``local``, applied by the runner inside the sandbox."""

    memory_limit_mb: int = SANDBOX_DEFAULT_MEMORY_MB
    compile_timeout: float = SANDBOX_DEFAULT_COMPILE_TIMEOUT
    compile_memory_limit_mb: int = SANDBOX_DEFAULT_COMPILE_MEMORY_MB

    def for_run(self, spec: LanguageSpec, timeout: float) -> dict[str, int]:
        limits = {**self._rlimits(timeout, self.memory_limit_mb), "RLIMIT_NPROC": LOCAL_NPROC_LIMIT}
        if spec.is_compiled:
            limits["RLIMIT_STACK"] = self.memory_limit_mb * MIB
        return limits

    def for_compile(self) -> dict[str, int]:
        return self._rlimits(self.compile_timeout, self.compile_memory_limit_mb)

    @staticmethod
    def _rlimits(timeout: float, memory_mb: int) -> dict[str, int]:
        limits = {
            "RLIMIT_CPU": int(math.ceil(timeout + RLIMIT_CPU_SLACK_SECONDS)),
            "RLIMIT_FSIZE": LOCAL_FSIZE_LIMIT,
            "RLIMIT_CORE": 0,
        }
        if memory_mb:
            limits["RLIMIT_AS"] = memory_mb * MIB
        return limits


class RunnerSession(SandboxSession):
    """A directory in a sandbox, driven through the runner. A compiled program rebuilds only when language, source or
    ``files`` change, or when the sandbox lost the directory, which the session then continues empty."""

    def __init__(self, workdir: str, uid: int, limits: ProgramLimits, interpreter: str = "python3"):
        self.workdir = workdir
        self.uid = uid
        self.limits = limits
        self.interpreter = interpreter
        # The built program's key and compile verdict (None once it is runnable).
        self._build: tuple[tuple, SandboxResult | None] | None = None
        # The directory's entries when the last program started; reset_to_prepared keeps only these.
        self._prepared: list[str] | None = None
        self._reset_pending = False

    @abstractmethod
    def exchange(self, request: dict[str, Any], *, deadline: float) -> dict[str, Any]:
        """Send one request to the runner in this session's sandbox and return its reply. Raises
        :class:`SandboxInfraError` when the sandbox, the transport or the runner fails."""

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
        except ValueError as exc:
            return SandboxResult(error=str(exc))
        files = files or {}
        for name in files:
            if not self.is_safe_member_name(name):
                return SandboxResult(error=f"unsafe auxiliary file path: {name!r}")
        key = (spec.name, code, tuple(sorted(files.items())))
        built = spec.is_compiled and self._build is not None and self._build[0] == key
        if built and self._build[1] is not None:
            return self._build[1]
        if not built:
            # Any other program may have removed the binary the recorded build left.
            self._build = None
        request = self._run_request(spec, code, files, stdin, timeout, build=not built)
        deadline = EXCHANGE_OVERHEAD_SECONDS + timeout + (self.limits.compile_timeout if spec.is_compiled else 0)
        try:
            reply = self._ask(request, deadline=deadline)
        except SandboxInfraError as exc:
            return SandboxResult(error=str(exc))
        return self._result(reply, spec, key)

    def _run_request(
        self, spec: LanguageSpec, code: str, files: dict[str, str], stdin: str, timeout: float, *, build: bool
    ) -> dict[str, Any]:
        return {
            "op": "run",
            "source_name": spec.source_name,
            "code": utf8_encodable(code),
            "files": {name: utf8_encodable(content) for name, content in files.items()},
            "stdin": utf8_encodable(stdin),
            "run_argv": [self.interpreter if token == INTERPRETER_PLACEHOLDER else token for token in spec.run_argv],
            "timeout": timeout,
            "limits": self.limits.for_run(spec, timeout),
            "compile_argv": list(spec.compile_argv) if spec.is_compiled else None,
            "build": build,
            "compile_timeout": self.limits.compile_timeout,
            "compile_limits": self.limits.for_compile(),
        }

    def _result(self, reply: dict[str, Any], spec: LanguageSpec, key: tuple) -> SandboxResult:
        """The verdict a run's reply comes to, recording the build and the entries a later reset keeps."""
        if "tampered" in reply:
            # The program's own runtime error, as on ``local``: an infra error would let it void its episode.
            self._build = None
            return SandboxResult(
                stderr=f"working directory tampered: {reply['tampered']}", returncode=TAMPERED_WORKDIR_RETURNCODE
            )
        if "compile" in reply:
            verdict = self._compile_verdict(reply["compile"], spec)
            self._build = (key, verdict)
            if verdict is not None:
                return verdict
        self._prepared = reply["entries"]
        outcome = reply["run"]
        return SandboxResult(
            stdout=outcome["stdout"],
            stderr=outcome["stderr"],
            returncode=None if outcome["timed_out"] else outcome["returncode"],
            timed_out=outcome["timed_out"] or outcome["returncode"] == -signal.SIGXCPU,
        )

    def reset_to_prepared(self) -> None:
        """Applied by the session's next exchange before anything else, so a graded test costs one exchange."""
        self._reset_pending = self._prepared is not None

    def write_file(self, path: str, content: str) -> None:
        self.reject_unsafe_path(path)
        self._build = None
        reply = self._ask({"op": "write", "files": {path: utf8_encodable(content)}})
        # The written file is part of what a later reset keeps.
        self._prepared = None
        if "tampered" in reply:
            raise SessionPathError(reply["tampered"])

    def read_file(self, path: str) -> str | None:
        """``None`` when ``path`` is not a regular file."""
        self.reject_unsafe_path(path)
        return self._ask({"op": "read", "path": path})["content"]

    def list_files(self) -> list[str]:
        return list(self._ask({"op": "list"})["files"])

    def _ask(self, request: dict[str, Any], *, deadline: float = EXCHANGE_OVERHEAD_SECONDS) -> dict[str, Any]:
        reply = self.exchange(
            {
                **request,
                "workdir": self.workdir,
                "uid": self.uid,
                "env": build_program_env(self.workdir),
                "reset_to": self._prepared if self._reset_pending else None,
            },
            deadline=deadline,
        )
        self._reset_pending = False
        if reply["fresh"]:
            # The runner had to create the directory: the sandbox lost the old one with its build and entries.
            self._build, self._prepared, self._reset_pending = None, None, False
        return reply

    def _compile_verdict(self, step: dict[str, Any], spec: LanguageSpec) -> SandboxResult | None:
        """As ``local`` judges a build: past the limit or rejected is the source's verdict, a missing compiler sets
        ``error``. ``None`` once the program is built."""
        if step["timed_out"]:
            return compile_limit_verdict(f"compilation timed out after {self.limits.compile_timeout:g} s")
        if step["returncode"] == COMMAND_NOT_FOUND_RETURNCODE:
            return SandboxResult(error=f"compiler not found: {spec.compile_argv[0]!r}", stderr=step["stderr"].strip())
        if step["returncode"] != 0:
            return compile_failure_verdict(step["stderr"], step["stdout"], step["returncode"])
        return None


def parse_runner_reply(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise SandboxInfraError("sandbox runner returned malformed output") from exc
