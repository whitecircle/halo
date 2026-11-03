#!/usr/bin/env python
"""``halo`` commands the docs show must run as written: the method or tool resolves, and no ``--`` precedes a flag
the launcher does not own."""

import re
import shlex
from pathlib import Path

import click
import pytest
import typer
from typer.testing import CliRunner

from halo.cli import app
from tests.common.repo import REPO_ROOT

DOC_TREES = ("README.md", "CLAUDE.md", "human-docs", "agent-docs", "skills")
ANY_CONFIG = "examples/sft/qwen3/qwen3-4b-ultrachat.yaml"
_HALO_COMMAND = re.compile(r"\bhalo (launch|run)\b([^`\n]*)")
_PLACEHOLDER = re.compile(r"[<>{}$*…]|\.\.\.|^[A-Z_]+$")
_NAME = re.compile(r"^[a-z0-9][a-z0-9_/-]*$")

runner = CliRunner()


def _doc_files() -> list[Path]:
    files: list[Path] = []
    for tree in DOC_TREES:
        path = REPO_ROOT / tree
        files += sorted(path.rglob("*.md")) if path.is_dir() else [path]
    return files


def _joined(text: str) -> str:
    return re.sub(r"\\\n\s*", " ", text)


def _halo_commands() -> list[tuple[str, str, list[str]]]:
    """``(where, verb, tokens)`` for every ``halo launch|run`` line in the doc trees."""
    found = []
    for path in _doc_files():
        for match in _HALO_COMMAND.finditer(_joined(path.read_text(encoding="utf-8"))):
            try:
                tokens = shlex.split(match.group(2).split(" #")[0])
            except ValueError:
                continue
            found.append((str(path.relative_to(REPO_ROOT)), match.group(1), tokens))
    return found


def _launcher_flags(verb: str) -> set[str]:
    command = typer.main.get_command(app).commands[verb]
    owned = {opt for param in command.params for opt in getattr(param, "opts", ())}
    return owned | set(command.get_help_option_names(click.Context(command)))


COMMANDS = _halo_commands()


def test_the_scan_finds_the_documented_commands():
    assert len(COMMANDS) > 40, "the doc scan found almost no halo commands; the pattern is broken"
    assert {verb for _, verb, _ in COMMANDS} == {"launch", "run"}


@pytest.mark.parametrize(
    ("where", "verb", "tokens"), COMMANDS, ids=[f"{w}:{v}:{i}" for i, (w, v, _) in enumerate(COMMANDS)]
)
def test_a_documented_name_resolves(where, verb, tokens):
    names = [token for token in tokens if not token.startswith("-")]
    if not names or _PLACEHOLDER.search(names[0]) or not _NAME.match(names[0]):
        pytest.skip("no literal method or tool name on this line")
    argv = [verb, names[0], ANY_CONFIG, "--dry_run"] if verb == "launch" else [verb, names[0], "--dry_run"]
    result = runner.invoke(app, argv)
    assert result.exit_code == 0, f"{where}: `halo {verb} {names[0]}` does not resolve:\n{result.output}"


@pytest.mark.parametrize(
    ("where", "verb", "tokens"), COMMANDS, ids=[f"{w}:{v}:{i}" for i, (w, v, _) in enumerate(COMMANDS)]
)
def test_the_separator_is_shown_only_before_a_launcher_flag(where, verb, tokens):
    if "--" not in tokens:
        pytest.skip("no separator on this line")
    after = tokens[tokens.index("--") + 1 :]
    assert after, f"{where}: `--` with nothing after it"
    flag = after[0].split("=", 1)[0]
    assert _PLACEHOLDER.search(after[0]) or flag in _launcher_flags(verb), (
        f"{where}: `halo {verb} ... -- {after[0]}` — `{flag}` is not a launcher flag, so it passes through "
        f"without `--`. Drop the separator."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
