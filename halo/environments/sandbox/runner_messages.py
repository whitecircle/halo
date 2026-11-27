"""The requests the trainer sends to the program runner and the replies it gets back. Standard library only: the
runner uses this file inside the sandbox."""

import json
from dataclasses import dataclass, fields, is_dataclass

from halo.environments.sandbox.workspace import Entry


@dataclass(frozen=True)
class Limits:
    cpu_seconds: int
    file_bytes: int
    memory_bytes: int | None
    stack_bytes: int | None
    processes: int | None


@dataclass(frozen=True)
class Command:
    argv: list[str]
    timeout: float
    limits: Limits


@dataclass(frozen=True)
class SourceFile:
    name: str
    content: str


@dataclass(frozen=True)
class RunProgram:
    sources: list[SourceFile]
    stdin: str
    env: dict[str, str]
    run: Command
    build: Command | None
    build_requested: bool


@dataclass(frozen=True)
class WriteFile:
    file: SourceFile


@dataclass(frozen=True)
class ReadFile:
    name: str


@dataclass(frozen=True)
class ListFiles:
    pass


@dataclass(frozen=True)
class Request:
    workdir: str
    uid: int
    closed_workdirs: list[str]
    keep_only: list[Entry] | None
    operation: RunProgram | WriteFile | ReadFile | ListFiles


@dataclass(frozen=True)
class Outcome:
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool

    @property
    def failed(self) -> bool:
        return self.timed_out or self.returncode != 0


@dataclass(frozen=True)
class ProgramRan:
    build: Outcome | None
    run: Outcome | None
    entries: list[Entry]


@dataclass(frozen=True)
class FileWritten:
    pass


@dataclass(frozen=True)
class FileContent:
    content: str | None


@dataclass(frozen=True)
class FileList:
    names: list[str]


@dataclass(frozen=True)
class Tampered:
    reason: str


@dataclass(frozen=True)
class Reply:
    new_workspace: bool
    result: ProgramRan | FileWritten | FileContent | FileList | Tampered


MESSAGES = {
    message.__name__: message
    for message in (
        Limits,
        Command,
        SourceFile,
        Entry,
        RunProgram,
        WriteFile,
        ReadFile,
        ListFiles,
        Request,
        Outcome,
        ProgramRan,
        FileWritten,
        FileContent,
        FileList,
        Tampered,
        Reply,
    )
}

# The key that marks a JSON object as a message. `env` is the only plain mapping a message carries.
MESSAGE = "message"


def encode(message) -> str:
    return json.dumps(_to_json(message))


def decode(text: str):
    return _from_json(json.loads(text))


def _to_json(value):
    if is_dataclass(value):
        encoded = {field.name: _to_json(getattr(value, field.name)) for field in fields(value)}
        encoded[MESSAGE] = type(value).__name__
        return encoded
    if isinstance(value, list):
        return [_to_json(item) for item in value]
    return value


def _from_json(value):
    if isinstance(value, list):
        return [_from_json(item) for item in value]
    if isinstance(value, dict) and MESSAGE in value:
        message = MESSAGES[value.pop(MESSAGE)]
        return message(**{name: _from_json(item) for name, item in value.items()})
    return value
