#!/usr/bin/env python
"""Commands the docs and recipe headers show must run as written.

A reader pastes them. Three things go stale silently: a ``halo`` method or tool name that no longer
resolves, a script or config path that moved, and a ``--`` separator in front of flags that never
needed one (unknown flags pass through on their own — the separator exists only for a flag the
launcher itself owns, so showing it elsewhere teaches a rule the CLI does not have).

Run: pytest tests/cpu/conventions/test_documented_commands.py
"""

import re
import shlex
from pathlib import Path

import click
import pytest
import typer
from typer.testing import CliRunner

from src.cli import app

REPO_ROOT = Path(__file__).resolve().parents[3]
DOC_TREES = ("README.md", "CLAUDE.md", "human-docs", "agent-docs", "skills")
ANY_CONFIG = "examples/sft/qwen3/qwen3-4b-ultrachat.yaml"
_HALO_COMMAND = re.compile(r"\bhalo (launch|run)\b([^`\n]*)")
_REPO_PATH = re.compile(r"(?<![\w./-])((?:scripts|examples|launcher-configs)/[\w./-]+\.(?:py|ya?ml))(?![\w-])")
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
    """Fold shell line continuations so a wrapped command reads as one line."""
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
    """Every flag the launcher parses itself, read off the CLI rather than restated here."""
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
    argv = [verb, names[0], ANY_CONFIG, "--dry-run"] if verb == "launch" else [verb, names[0], "--dry-run"]
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


def _quoted_repo_paths() -> list[tuple[str, str]]:
    """Script and config paths named in the doc trees and in the recipes' header comments."""
    quoted = []
    for path in _doc_files():
        quoted += [
            (str(path.relative_to(REPO_ROOT)), hit) for hit in _REPO_PATH.findall(path.read_text(encoding="utf-8"))
        ]
    for recipe in sorted((REPO_ROOT / "examples").rglob("*.yaml")):
        comments = "\n".join(
            line for line in recipe.read_text(encoding="utf-8").splitlines() if line.lstrip().startswith("#")
        )
        quoted += [(str(recipe.relative_to(REPO_ROOT)), hit) for hit in _REPO_PATH.findall(comments)]
    return sorted(set(quoted))


PATHS = _quoted_repo_paths()


def test_the_scan_finds_the_quoted_paths():
    assert len(PATHS) > 100, "the path scan found almost nothing; the pattern is broken"


@pytest.mark.parametrize(("where", "quoted"), PATHS, ids=[f"{w}:{q}" for w, q in PATHS])
def test_a_quoted_script_or_config_exists(where, quoted):
    assert (REPO_ROOT / quoted).exists(), f"{where} names `{quoted}`, which does not exist"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
