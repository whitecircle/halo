#!/usr/bin/env python3
"""Resolve every relative markdown link in the doc trees; fail on a missing target or anchor.

All doc trees are plain GitHub-rendered markdown, so this is the link gate. A ``#fragment`` into a
markdown file (or ``#fragment`` alone, into the same file) must name a heading anchor GitHub
generates for that file or an explicit ``<a id>`` / ``<a name>``; a fragment on a directory link is
checked against the directory's ``README.md``, the page GitHub renders there. A ``{#id}`` heading
attribute is refused: GitHub renders the braces as heading text, so the id never exists. Links inside
code are not links; external URLs are skipped; a fragment into any other target is checked for its
path only. Headings are ATX (``#``) or raw ``<h1>``–``<h6>``; an underlined (setext) heading has no
anchor here, so a link to one fails.

Standard library only (Python 3.9+), so it runs on a hosted runner and on a host without the image.

usage: scripts/docs/check_links.py [path ...]   (default: agent-docs, human-docs, skills + root markdown)
"""

from __future__ import annotations

import html
import os
import re
import sys
import unicodedata
from functools import cache
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote

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
# GitHub's slug keeps letters, marks, decimal and letter numbers, connector punctuation (`_`), `-`
# and the space it then turns into `-`; every other character is dropped.
_SLUG_KEPT_CATEGORIES = frozenset({"Nd", "Nl", "Pc"})
# Private-use code points stand in for code spans while the rest of a heading is rendered.
_CODE_PLACEHOLDER_BASE = 0xE000

_BLOCKQUOTE = re.compile(r"^(?: {0,3}> ?)+")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
_ATX_HEADING = re.compile(r"^ {0,3}#{1,6}(?:[ \t]+(.*?))?[ \t]*$")
_CLOSING_HASHES = re.compile(r"(?:^|[ \t]+)#+$")
_HEADING_ATTRIBUTE = re.compile(r"\{#[^}]*\}$")
_CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")
_LINK = re.compile(r"\]\(([^)]+)\)")
_HTML_ANCHOR = re.compile(r"""<a\s[^>]*?\b(?:id|name)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_HTML_HEADING = re.compile(r"<h([1-6])\b[^>]*>(.*?)</h\1\s*>", re.IGNORECASE)
_AUTOLINK = re.compile(r"<((?:https?|mailto):[^>\s]*)>")
_INLINE_LINK = re.compile(r"(!?)\[([^\]]*)\]\([^)]*\)")
_REFERENCE_LINK = re.compile(r"\[([^\]]*)\]\[[^\]]*\]")
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_UNDERSCORE_EMPHASIS = re.compile(r"(?<![\w\\])(_{1,3})(?=\S)(.+?)(?<=\S)(?<!\\)\1(?!\w)")
_ESCAPE = re.compile(r"\\([!-/:-@\[-`{-~])")


class Document(NamedTuple):
    anchors: frozenset[str]
    heading_attributes: tuple[tuple[int, str], ...]
    links: tuple[tuple[int, str], ...]


def github_slug(text: str) -> str:
    """The anchor GitHub derives from a heading's rendered text, before de-duplication."""
    kept = []
    for ch in text.lower():
        category = unicodedata.category(ch)
        if ch in " -" or category[0] in "LM" or category in _SLUG_KEPT_CATEGORIES:
            kept.append("-" if ch == " " else ch)
    return "".join(kept)


def unique_slugs(texts: list[str]) -> list[str]:
    """Slugs for a page's headings in order; a repeat gets ``-1``, ``-2`` … as GitHub numbers them."""
    seen: dict[str, int] = {}
    slugs = []
    for text in texts:
        base = slug = github_slug(text)
        while slug in seen:
            seen[base] += 1
            slug = f"{base}-{seen[base]}"
        seen[slug] = 0
        slugs.append(slug)
    return slugs


def heading_text(source: str) -> str:
    """The text GitHub renders for an ATX heading's inline source: markup gone, code kept verbatim."""
    spans: list[str] = []

    def stash(match: re.Match[str]) -> str:
        content = match.group(2)
        if content.startswith(" ") and content.endswith(" ") and content.strip():
            content = content[1:-1]
        spans.append(content)
        return chr(_CODE_PLACEHOLDER_BASE + len(spans) - 1)

    text = _CODE_SPAN.sub(stash, source)
    text = _AUTOLINK.sub(r"\1", text)
    text = _INLINE_LINK.sub(lambda m: "" if m.group(1) else m.group(2), text)
    text = _REFERENCE_LINK.sub(r"\1", text)
    text = _HTML_TAG.sub("", text)
    while True:
        unwrapped = _UNDERSCORE_EMPHASIS.sub(r"\2", text)
        if unwrapped == text:
            break
        text = unwrapped
    text = html.unescape(_ESCAPE.sub(r"\1", text))
    return "".join(
        spans[ord(ch) - _CODE_PLACEHOLDER_BASE] if 0 <= ord(ch) - _CODE_PLACEHOLDER_BASE < len(spans) else ch
        for ch in text
    )


def parse_markdown(text: str) -> Document:
    """Anchors, ``{#id}`` heading attributes and ``(line, target)`` links of one markdown source."""
    lines = text.splitlines()
    start = 0
    if lines and lines[0] == "---":
        start = next((i + 1 for i, line in enumerate(lines[1:], 1) if line in ("---", "...")), 0)
    headings: list[str] = []
    attributes: list[tuple[int, str]] = []
    links: list[tuple[int, str]] = []
    explicit: set[str] = set()
    fence = ""
    for lineno, raw in enumerate(lines[start:], start + 1):
        line = _BLOCKQUOTE.sub("", raw)
        marker = _FENCE.match(line)
        if fence:
            closes = marker and marker.group(1)[0] == fence[0] and len(marker.group(1)) >= len(fence)
            if closes and not marker.group(2).strip():
                fence = ""
            continue
        # A backtick run with a backtick after it on the line is inline code, not a fence.
        if marker and not (marker.group(1)[0] == "`" and "`" in marker.group(2)):
            fence = marker.group(1)
            continue
        prose = _CODE_SPAN.sub("", line)
        links.extend((lineno, target) for target in _LINK.findall(prose))
        explicit.update(_HTML_ANCHOR.findall(prose))
        headings.extend(heading_text(inner) for _level, inner in _HTML_HEADING.findall(prose))
        heading = _ATX_HEADING.match(line)
        if heading:
            source = _CLOSING_HASHES.sub("", heading.group(1) or "").strip()
            attribute = _HEADING_ATTRIBUTE.search(_CODE_SPAN.sub("", source).rstrip())
            if attribute:
                attributes.append((lineno, attribute.group(0)))
            headings.append(heading_text(source))
    return Document(frozenset(unique_slugs(headings)) | explicit, tuple(attributes), tuple(links))


@cache
def load_document(path: Path) -> Document:
    return parse_markdown(path.read_text(encoding="utf-8"))


def fragment_page(target: Path) -> Path | None:
    """The markdown page a ``#fragment`` on ``target`` lands in, or None when GitHub renders none.

    A directory link renders the directory's ``README.md`` below its listing, so its anchors are that
    page's.
    """
    page = target / "README.md" if target.is_dir() else target
    return page if page.suffix == ".md" and page.is_file() else None


def markdown_files(paths: list[str]) -> list[Path]:
    files = []
    for path in paths:
        if os.path.isdir(path):
            for root, _dirs, names in os.walk(path, followlinks=True):
                files.extend(Path(root, name) for name in names if name.endswith(".md"))
        elif path.endswith(".md") and os.path.isfile(path):
            files.append(Path(path))
    return sorted(files)


def check_file(file: Path) -> list[str]:
    """Every broken link, broken anchor and ``{#id}`` heading attribute in ``file``, one line each."""
    document = load_document(file.resolve())
    problems = [
        f"HEADING ID  {file}:{lineno}  {attribute}  (GitHub renders it as text; link the generated slug)"
        for lineno, attribute in document.heading_attributes
    ]
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


def main(argv: list[str]) -> int:
    paths = argv or list(DEFAULT_PATHS)
    # A path that no longer exists would leave nothing to scan and the check would pass.
    missing = [path for path in paths if not os.path.exists(path)]
    for path in missing:
        print(f"MISSING PATH  {path}")
    if missing:
        print(f"FAIL: {len(missing)} path(s) to check do not exist")
        return 1
    problems = [problem for file in markdown_files(paths) for problem in check_file(file)]
    for problem in problems:
        print(problem)
    if problems:
        print(f"FAIL: {len(problems)} broken link(s), anchor(s) or heading id(s)")
        return 1
    print("OK: every relative link and anchor resolves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
