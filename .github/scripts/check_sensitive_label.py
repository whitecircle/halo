"""Fail a pull request that changes a path listed in ``.github/sensitive-paths`` without the ``sensitive`` label.

python .github/scripts/check_sensitive_label.py <base> <head> --labels '<JSON list of label names>'
"""

import argparse
import json
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SENSITIVE_PATHS = ".github/sensitive-paths"
LABEL = "sensitive"
_WILDCARD = re.compile(r"(\*\*/|\*\*|\*)")
_WILDCARD_REGEX = {"**/": "(?:.*/)?", "**": ".*", "*": "[^/]*"}


def globs_in(text: str) -> list[str]:
    """The globs a sensitive-paths list names, one per line; ``#`` starts a comment."""
    lines = (line.partition("#")[0].strip() for line in text.splitlines())
    return [line for line in lines if line]


def glob_regex(glob: str) -> re.Pattern[str]:
    """``glob`` as a pattern a whole repo-relative path matches: ``*`` stays within one path segment,
    ``**`` spans any number of them."""
    return re.compile("".join(_WILDCARD_REGEX.get(part, re.escape(part)) for part in _WILDCARD.split(glob)))


def sensitive_changes(paths: Iterable[str], globs: Iterable[str]) -> list[str]:
    """The ``paths`` a glob matches, sorted."""
    patterns = [glob_regex(glob) for glob in globs]
    return sorted(path for path in set(paths) if any(pattern.fullmatch(path) for pattern in patterns))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, stdout=subprocess.PIPE, text=True, check=True).stdout


def changed_paths(base: str, head: str) -> list[str]:
    """Every path ``head`` adds, changes or deletes since it forked from ``base``; a moved file counts under
    its old path as well as its new one."""
    return _git("diff", "--no-renames", "--name-only", f"{base}...{head}").splitlines()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", help="the branch the pull request merges into")
    parser.add_argument("head", help="the pull request's head commit")
    parser.add_argument("--labels", default="[]", help="the pull request's label names, as a JSON list")
    args = parser.parse_args(argv)

    # The base's list: a pull request that shortens it is still checked against the one it changes.
    globs = globs_in(_git("show", f"{args.base}:{SENSITIVE_PATHS}"))
    touched = sensitive_changes(changed_paths(args.base, args.head), globs)
    if touched and LABEL not in json.loads(args.labels):
        listing = "".join(f"\n  {path}" for path in touched)
        raise SystemExit(
            f"::error::This pull request changes paths listed in {SENSITIVE_PATHS} without the '{LABEL}' "
            f"label. A maintainer adds the label once the change is approved (CLAUDE.md, Process).{listing}"
        )


if __name__ == "__main__":
    main()
