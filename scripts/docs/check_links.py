"""Resolve every relative markdown link and code reference in the doc trees; fail on a dangling one.

A ``#fragment`` must name a heading anchor GitHub generates or an explicit ``<a id>`` / ``<a name>``; a ``{#id}``
heading attribute is refused, as GitHub renders the braces as text. Code spans and fenced code are checked as
references (:mod:`scripts.docs.references`). Standard library only; parsing the package's grammar needs Python 3.12.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import unquote

from scripts.docs.markdown import fragment_page, load_document
from scripts.docs.references import PACKAGE, REPO_ROOT, RepoIndex

DEFAULT_PATHS = (
    "agent-docs",
    "human-docs",
    "skills",
    "README.md",
    "CONTRIBUTING.md",
    "CLAUDE.md",
    "AGENTS.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
)
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:")


def markdown_files(paths: list[str]) -> list[Path]:
    files = []
    for path in paths:
        if os.path.isdir(path):
            for root, _dirs, names in os.walk(path, followlinks=True):
                files.extend(Path(root, name) for name in names if name.endswith(".md"))
        elif path.endswith(".md") and os.path.isfile(path):
            files.append(Path(path))
    return sorted(files)


def check_file(file: Path, index: RepoIndex) -> list[str]:
    """Every broken link, anchor, ``{#id}`` attribute and reference in ``file``, one line each."""
    document = load_document(file.resolve())
    problems = [
        f"HEADING ID  {file}:{lineno}  {attribute}  (GitHub renders it as text; link the generated slug)"
        for lineno, attribute in document.heading_attributes
    ]
    problems.extend(
        f"UNRESOLVED  {file}:{lineno}  {reference.text}  ({reference.why})"
        for lineno, code in document.code
        for reference in index.references(code)
        if reference.why
    )
    for lineno, raw in document.links:
        target = raw.split(" ", 1)[0]
        if target.startswith(EXTERNAL_PREFIXES):
            continue
        path, _, fragment = target.partition("#")
        resolved = (file.parent / unquote(path)) if path else file
        if not resolved.exists():
            problems.append(f"BROKEN  {file}:{lineno}  ->  {path}")
            continue
        page = fragment_page(resolved) if fragment else None
        if page and unquote(fragment) not in load_document(page.resolve()).anchors:
            problems.append(f"BROKEN ANCHOR  {file}:{lineno}  ->  {target}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve every relative markdown link, heading anchor and code reference in the doc trees."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=list(DEFAULT_PATHS),
        help="Markdown files or directories to check (default: %(default)s).",
    )
    paths = parser.parse_args(argv).paths
    # A host interpreter runs this, not the project's, so the project's floor is checked here.
    if sys.version_info < (3, 12):  # noqa: UP036
        print(f"FAIL: needs Python 3.12+ to parse the {PACKAGE} package's source; this is {sys.version.split()[0]}")
        return 2
    # A path that no longer exists would leave nothing to scan and the check would pass.
    missing = [path for path in paths if not os.path.exists(path)]
    for path in missing:
        print(f"MISSING PATH  {path}")
    if missing:
        print(f"FAIL: {len(missing)} path(s) to check do not exist")
        return 1
    index = RepoIndex(REPO_ROOT)
    problems = [problem for file in markdown_files(paths) for problem in check_file(file, index)]
    for problem in problems:
        print(problem)
    if problems:
        print(f"FAIL: {len(problems)} broken link(s), anchor(s), heading id(s) or code reference(s)")
        return 1
    print("OK: every relative link, anchor and code reference resolves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
