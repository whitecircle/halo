#!/usr/bin/env python
"""Smoke: `python <script> --help` works for every config-driven script.

``--help`` is the documented way to see a script's real flag surface (``agent-docs/reference/scripts-reference.md``),
and it is the one path no other test walks: the parser is built from the script's dataclass tuple, so a
field whose metadata/annotation argparse cannot render (a bare ``dict`` default, a help string with a stray
``%``, two dataclasses claiming one flag) breaks ``--help`` for that script alone while training still runs.
A defect in a shared dataclass breaks it on every script that parses it at once.

Each script runs in a fresh subprocess — argparse's help path exits the interpreter, and the parser is
only built inside ``main()``. The set is scanned from the tree, so a new script is covered on arrival,
and running ``--help`` executes the whole module body, so this also covers plain importability.
Each subprocess pays a full torch/transformers import (~30 s).

The `halo run` tools are argparse CLIs whose parser is built inside their entry point, so an import never
formats their help. Each is run in-process up to its own parse call, where the parser is captured and its
``--help`` rendered — the import's price, no subprocess. Every flag must carry help text (its default is
otherwise invisible in ``--help``), and no help text may hardcode a ``/mnt`` path, which is not guaranteed
to be a large volume.

Run:
    python tests/cpu/config/test_script_help.py
    pytest -m cpu tests/cpu/config/test_script_help.py
"""

import argparse
import asyncio
import importlib
import inspect
import logging
import os
import pathlib
import subprocess
import sys
from unittest.mock import patch

import pytest

from tests.common.repo import REPO_ROOT

_SCRIPTS_DIR = REPO_ROOT / "scripts"
_SCRIPTS = sorted(
    path for path in _SCRIPTS_DIR.rglob("*.py") if "parse_script_args(" in path.read_text(encoding="utf-8")
)
# The `halo run` tools: runnable, but built on argparse, so the sweep above never reaches them.
# `diagrams/` is excluded — its generators import a sibling style module off their own directory.
_TOOLS = sorted(
    path
    for path in _SCRIPTS_DIR.rglob("*.py")
    if path not in _SCRIPTS and "diagrams" not in path.parts and "__main__" in path.read_text(encoding="utf-8")
)


def _help_in_subprocess(script: pathlib.Path) -> subprocess.CompletedProcess:
    # PYTHONPATH pinned to THIS tree: the training images bake a repo copy at /workspace and put it
    # on PYTHONPATH, so a subprocess started from any other checkout would parse that copy's args
    # and pass vacuously.
    return subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        timeout=600,
    )


@pytest.mark.parametrize("script", _SCRIPTS, ids=[str(p.relative_to(_SCRIPTS_DIR)) for p in _SCRIPTS])
def test_script_help_exits_zero(script: pathlib.Path):
    result = _help_in_subprocess(script)
    assert result.returncode == 0, (
        f"{script.relative_to(_SCRIPTS_DIR)} --help exited {result.returncode}:\n{result.stderr[-2000:]}"
    )
    assert "--output_dir" in result.stdout, f"{script.relative_to(_SCRIPTS_DIR)} --help printed no flag list"


@pytest.fixture
def pristine_root_logging():
    """A tool's entry point configures the root logger before it parses (``configure_cli_logging``,
    ``force=True``), which would otherwise strip pytest's own handlers for the rest of the worker."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


class _ParserBuilt(Exception):
    """Carries a tool's parser out of its entry point, before it reads ``sys.argv``."""

    def __init__(self, parser: argparse.ArgumentParser):
        super().__init__("parser captured")
        self.parser = parser


def _tool_parser(tool: pathlib.Path) -> argparse.ArgumentParser:
    """The parser ``tool`` builds, captured at its first parse call: the module's ``parse_args`` when it
    has one, else its ``main`` (run to completion when it is a coroutine function)."""
    module = importlib.import_module(".".join(tool.relative_to(REPO_ROOT).with_suffix("").parts))
    entry = getattr(module, "parse_args", None) or module.main

    def _capture(self, *args, **kwargs):
        raise _ParserBuilt(self)

    with (
        patch.object(argparse.ArgumentParser, "parse_args", _capture),
        patch.object(argparse.ArgumentParser, "parse_known_args", _capture),
        patch.object(sys, "argv", [str(tool)]),
    ):
        try:
            result = entry()
            if inspect.iscoroutine(result):
                asyncio.run(result)
        except _ParserBuilt as built:
            return built.parser
    raise AssertionError(f"{tool.relative_to(_SCRIPTS_DIR)} built no parser before returning")


def _parsers(parser: argparse.ArgumentParser) -> list[argparse.ArgumentParser]:
    """``parser`` and every subcommand parser under it."""
    subparsers = [
        sub
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
        for sub in action.choices.values()
    ]
    return [parser, *(nested for sub in subparsers for nested in _parsers(sub))]


@pytest.mark.parametrize("tool", _TOOLS, ids=[str(p.relative_to(_SCRIPTS_DIR)) for p in _TOOLS])
def test_every_tool_help_renders_and_documents_every_flag(tool: pathlib.Path, pristine_root_logging):
    """The tools `halo run` indexes are driven by no other test end to end, so a `halo` module moved out
    from under one, or a help string argparse cannot format, breaks it silently until an operator runs it."""
    for parser in _parsers(_tool_parser(tool)):
        help_text = parser.format_help()
        assert help_text.startswith("usage:"), help_text[:200]
        undocumented = [
            "/".join(action.option_strings) or action.dest
            for action in parser._actions
            if not isinstance(action, argparse._HelpAction | argparse._SubParsersAction) and not action.help
        ]
        assert not undocumented, f"{tool.relative_to(_SCRIPTS_DIR)}: {undocumented} carry no help text"
        assert "/mnt" not in help_text, (
            f"{tool.relative_to(_SCRIPTS_DIR)} hardcodes a /mnt path in --help; /mnt is not guaranteed large, so "
            f"name the large volume as a placeholder instead"
        )


def test_the_script_scan_is_not_vacuous():
    """Anti-vacuity: both parametrizations are generated by a tree scan, so a scan that matched
    nothing — the marker string renamed, ``scripts/`` relaid out — would collect zero cases and
    report green while covering no script at all."""
    assert len(_SCRIPTS) >= 10, f"only {len(_SCRIPTS)} parse_script_args scripts found under {_SCRIPTS_DIR}"
    assert len(_TOOLS) >= 20, f"only {len(_TOOLS)} argparse tools found under {_SCRIPTS_DIR}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
