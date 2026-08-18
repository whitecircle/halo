"""MkDocs hook: render inline-code repository paths as links to the GitHub source.

An inline-code span whose content is a real repo path — ``src/trainers/mixins/base.py``,
``scripts/.../sft.py:42`` (line anchor), ``src/callbacks/`` (directory) — is turned
into a link to that file/dir on GitHub, displayed by its basename (filename only).

The path must exist on disk, so placeholders (``...<family>...``,
``model-XXXXX-of-...``) and stale references are left as plain code; a path-shaped
span that does not exist is logged at INFO so stale refs surface without failing the
strict build. Fenced code blocks and ATX headings are never touched. The source
markdown is unchanged — this runs only at build time, so the repo/branch live in one
place and every current and future path reference is covered automatically.
"""

from __future__ import annotations

import logging
import os
import re

log = logging.getLogger(f"mkdocs.hooks.{__name__}")

# Docs target the stable branch.
BRANCH = "main"

# Top-level dirs and root files that denote a repository path.
_PATH_PREFIXES = (
    "src/",
    "scripts/",
    "tests/",
    "examples/",
    "accelerate/",
    "docker/",
    "agent-docs/",
    "human-docs/",
    "skills/",
    ".github/",
)
_ROOT_FILES = frozenset(
    {
        "Dockerfile",
        "Dockerfile.vllm",
        "Dockerfile.sglang",
        "docker-compose.vllm.yml",
        "docker-compose.sglang.yml",
        "Makefile",
        "pyproject.toml",
        "uv.lock",
        "mkdocs.yml",
        "CLAUDE.md",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "AGENTS.md",
        "README.md",
        "SECURITY.md",
        "CITATION.cff",
    }
)

# Glob / placeholder markers that mean "not a real single path".
_PLACEHOLDER = ("<", ">", "*", "..", "XXXX", "YYYY", "{", "}")

# Extensions worth flagging when a path-shaped span does not resolve.
_CODE_EXT = (".py", ".yaml", ".yml", ".cpp", ".cu", ".cuh", ".toml", ".md", ".cff", ".sh", ".json")

_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_FENCE = re.compile(r"^\s*(```|~~~)")
_SPAN = re.compile(r"(?P<path>[^:#\s]+?)(?::(?P<line>\d+))?$")

_repo_root: str | None = None
_repo_url: str | None = None


def on_config(config):
    global _repo_root, _repo_url
    _repo_root = os.path.dirname(os.path.abspath(config["config_file_path"]))
    _repo_url = (config.get("repo_url") or "").rstrip("/")
    return config


def _resolve(span: str):
    """Return (display, url) for a repo-path code span, or None to leave it as code."""
    if not _repo_url or " " in span or any(p in span for p in _PLACEHOLDER):
        return None
    m = _SPAN.fullmatch(span)
    if not m:
        return None
    path, line = m.group("path"), m.group("line")
    clean = path.rstrip("/")
    if not ((clean + "/").startswith(_PATH_PREFIXES) or clean in _ROOT_FILES):
        return None
    disk = os.path.join(_repo_root, clean)
    if not os.path.exists(disk):
        if clean.endswith(_CODE_EXT):
            log.info("github_links: '%s' looks like a repo path but does not exist; left as code", span)
        return None
    is_dir = path.endswith("/") or os.path.isdir(disk)
    url = f"{_repo_url}/{'tree' if is_dir else 'blob'}/{BRANCH}/{clean}"
    display = clean.split("/")[-1] + ("/" if is_dir else "")
    if line and not is_dir:
        url += f"#L{line}"
        display += f":{line}"
    return display, url


def _link_paths(text: str) -> str:
    def repl(match: re.Match) -> str:
        # Leave code spans that are already a link's text (`[`code`](url)`).
        if match.start() > 0 and text[match.start() - 1] == "[":
            return match.group(0)
        resolved = _resolve(match.group(1))
        if not resolved:
            return match.group(0)
        display, url = resolved
        return f"[`{display}`]({url})"

    return _INLINE_CODE.sub(repl, text)


def on_page_markdown(markdown: str, **kwargs) -> str:
    if not _repo_url:
        return markdown
    out, in_fence, marker = [], False, ""
    for line in markdown.split("\n"):
        fence = _FENCE.match(line)
        if fence:
            if not in_fence:
                in_fence, marker = True, fence.group(1)
            elif fence.group(1) == marker:
                in_fence, marker = False, ""
            out.append(line)
        elif in_fence or line.lstrip().startswith("#"):
            out.append(line)
        else:
            out.append(_link_paths(line))
    return "\n".join(out)
