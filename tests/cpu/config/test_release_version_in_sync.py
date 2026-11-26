#!/usr/bin/env python
"""Every place that states the Halo release version states the one ``pyproject.toml`` declares: ``uv.lock``, the
``Makefile``, each Dockerfile's ``ARG VERSION``, the citation, the README's newest entry (dated as the citation
is) and every documented image pin."""

import re
import tomllib
from collections.abc import Mapping
from typing import NamedTuple

import pytest

from tests.common.repo import REPO_ROOT

DOCKERFILES = sorted(path.name for path in REPO_ROOT.glob("Dockerfile*"))
ARG_VERSION = re.compile(r"^ARG VERSION=(?P<version>\S+)$", re.MULTILINE)

# Each pattern must find at least one statement, so a spelling change fails here. Anchoring on ``^version:`` keeps
# ``cff-version`` out.
FIXED_STATEMENTS: dict[str, re.Pattern[str]] = {
    "Makefile": re.compile(r"^VERSION\s*\?=\s*(?P<version>\S+)\s*$", re.MULTILINE),
    **dict.fromkeys(DOCKERFILES, ARG_VERSION),
    "CITATION.cff": re.compile(r'^version:\s*"?(?P<version>[^"\s]+)"?\s*$', re.MULTILINE),
    "uv.lock": re.compile(r'^name = "halo"\nversion = "(?P<version>[^"]+)"$', re.MULTILINE),
}

# A release version, with its release-candidate suffix where it has one (``1.3.0rc1``).
RELEASE_VERSION = r"\d+(?:\.\d+)+(?:rc\d+)?"

# The README feed is newest first, so only its first release entry states the current release.
RELEASE_ENTRY = re.compile(
    rf"^- \*\*(?P<date>\d{{4}}-\d{{2}}-\d{{2}}) — Halo (?P<version>{RELEASE_VERSION})\.\*\*", re.MULTILINE
)
DATE_RELEASED = re.compile(r'^date-released:\s*"?(?P<date>\d{4}-\d{2}-\d{2})"?\s*$', re.MULTILINE)

# A published pin: ``blackwell-X.Y.Z`` / ``hopper-X.Y.Z``, or a rollout image's
# ``vllm-<engine>-X.Y.Z``, whose engine version may carry a ``.post1`` / ``rc1`` / ``.dev0`` suffix.
# A bare engine tag (``sglang-<engine>``) names no Halo release.
IMAGE_PIN = re.compile(
    rf"\b(?:blackwell|hopper|(?:vllm|sglang)-\d+(?:\.\d+)+(?:\.?(?:post|rc|dev)\d+)?)-(?P<version>{RELEASE_VERSION})\b"
)

PIN_TREES = (
    "README.md",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    "Makefile",
    "Dockerfile*",
    "docker-compose*.yml",
    ".github/**/*.md",
    ".github/**/*.yml",
    "agent-docs/**/*.md",
    "human-docs/**/*.md",
    "skills/**/*.md",
    "examples/**/*.yaml",
    "launcher-configs/**/*.yaml",
    "launcher-configs/**/*.hcl",
    "docker/**/*.sh",
    "docker/**/*.py",
    "halo/**/*.py",
    "scripts/**/*.py",
    "tests/**/*.py",
)

# Where the release's pins are documented: the README, both doc trees, and the issue and PR templates.
DOCUMENTED_PIN_AREAS = ("README.md", "agent-docs/", "human-docs/", ".github/")


class Statement(NamedTuple):
    path: str
    line: int
    start: int  # the version's span in the file's text, which a planted mismatch rewrites
    end: int
    version: str


def repo_texts() -> dict[str, str]:
    paths = {REPO_ROOT / name for name in (*FIXED_STATEMENTS, "pyproject.toml")}
    paths |= {path for tree in PIN_TREES for path in REPO_ROOT.glob(tree)}
    return {path.relative_to(REPO_ROOT).as_posix(): path.read_text(encoding="utf-8") for path in sorted(paths)}


def declared_version(texts: Mapping[str, str]) -> str:
    """The release ``pyproject.toml`` declares: the one every other statement must repeat."""
    return tomllib.loads(texts["pyproject.toml"])["project"]["version"]


def _statement(path: str, text: str, match: re.Match[str]) -> Statement:
    start, end = match.span("version")
    return Statement(path, text.count("\n", 0, start) + 1, start, end, match["version"])


def _newest_release_entry(readme: str) -> re.Match[str]:
    entry = RELEASE_ENTRY.search(readme)
    assert entry, f"README.md has no release entry matching {RELEASE_ENTRY.pattern!r}"
    return entry


def release_statements(texts: Mapping[str, str]) -> list[Statement]:
    statements = []
    for path, pattern in FIXED_STATEMENTS.items():
        found = [_statement(path, texts[path], match) for match in pattern.finditer(texts[path])]
        assert found, f"{path} states no release version: {pattern.pattern!r} no longer matches, so it goes unchecked"
        statements += found
    statements.append(_statement("README.md", texts["README.md"], _newest_release_entry(texts["README.md"])))
    for path, text in texts.items():
        statements += [_statement(path, text, match) for match in IMAGE_PIN.finditer(text)]
    return statements


def disagreements(texts: Mapping[str, str]) -> list[str]:
    """Each statement that is not the declared release, and a newest release entry dated off the citation."""
    declared = declared_version(texts)
    reported = [
        f"{statement.path}:{statement.line} states {statement.version}"
        for statement in release_statements(texts)
        if statement.version != declared
    ]
    released = DATE_RELEASED.search(texts["CITATION.cff"])
    assert released, f"CITATION.cff has no date-released matching {DATE_RELEASED.pattern!r}"
    entry = _newest_release_entry(texts["README.md"])
    if entry["date"] != released["date"]:
        line = texts["README.md"].count("\n", 0, entry.start()) + 1
        reported.append(f"README.md:{line} dates {entry['date']}, CITATION.cff date-released {released['date']}")
    return reported


def _next_version(version: str) -> str:
    """The next release candidate of a candidate, else the next patch: a different version of the same shape."""
    final, candidate, number = version.partition("rc")
    if candidate:
        return f"{final}rc{int(number) + 1}"
    head, _, patch = version.rpartition(".")
    return f"{head}.{int(patch) + 1}"


@pytest.fixture(scope="module")
def texts() -> dict[str, str]:
    return repo_texts()


def test_every_statement_of_the_release_agrees(texts):
    reported = disagreements(texts)
    assert not reported, (
        f"{len(reported)} statement(s) of the Halo release disagree with pyproject.toml's "
        f"{declared_version(texts)}; a release bumps all of them together:\n  " + "\n  ".join(reported)
    )


def test_a_mismatch_in_any_one_statement_is_reported_alone(texts):
    """Rewrite each statement in an in-memory copy: the gate names exactly that one, at its line."""
    bumped = _next_version(declared_version(texts))
    for statement in release_statements(texts):
        original = texts[statement.path]
        planted = {**texts, statement.path: original[: statement.start] + bumped + original[statement.end :]}
        assert disagreements(planted) == [f"{statement.path}:{statement.line} states {bumped}"], statement


def test_a_bump_that_stops_at_pyproject_is_reported_everywhere_else(texts):
    declared = declared_version(texts)
    bumped = _next_version(declared)
    pyproject = re.sub(
        rf'^version = "{re.escape(declared)}"$',
        f'version = "{bumped}"',
        texts["pyproject.toml"],
        count=1,
        flags=re.MULTILINE,
    )
    planted = {**texts, "pyproject.toml": pyproject}
    assert declared_version(planted) == bumped
    assert len(disagreements(planted)) == len(release_statements(texts))


def test_both_pin_shapes_are_read_and_a_bare_engine_tag_is_not(texts):
    """Planted in memory, so each pin shape is proven whether or not a doc writes it."""
    bumped, engine = _next_version(declared_version(texts)), "1.2.3"
    doc = (
        f"docker pull public.ecr.aws/whitecircle/halo:hopper-{bumped}\n"
        f"docker pull public.ecr.aws/whitecircle/halo:vllm-{engine}-{bumped}\n"
        f"docker pull public.ecr.aws/whitecircle/halo:sglang-{engine}.post1-{bumped}\n"
        f"docker pull public.ecr.aws/whitecircle/halo:vllm-{engine}rc1-{bumped}\n"
        f"docker pull public.ecr.aws/whitecircle/halo:sglang-{engine}\n"
    )
    reported = disagreements({**texts, "human-docs/planted.md": doc})
    assert reported == [f"human-docs/planted.md:{line} states {bumped}" for line in (1, 2, 3, 4)]


def test_every_dockerfile_must_state_the_release(texts):
    """A Dockerfile whose ``ARG VERSION`` goes missing fails loud instead of dropping out of the gate."""
    assert DOCKERFILES, "no Dockerfile* at the repo root: the glob no longer finds the image build files"
    for name in DOCKERFILES:
        with pytest.raises(AssertionError, match=f"^{re.escape(name)} states no release version"):
            release_statements({**texts, name: ARG_VERSION.sub("", texts[name])})


def test_a_newest_release_entry_dated_off_the_citation_is_reported(texts):
    start, end = DATE_RELEASED.search(texts["CITATION.cff"]).span("date")
    citation = texts["CITATION.cff"][:start] + "1999-12-31" + texts["CITATION.cff"][end:]
    reported = disagreements({**texts, "CITATION.cff": citation})
    assert len(reported) == 1 and reported[0].startswith("README.md:"), reported
    assert reported[0].endswith("CITATION.cff date-released 1999-12-31"), reported


def test_the_pin_scan_reads_every_documented_area(texts):
    """A pin pattern that stopped matching the docs' spelling would pass every test above, and a glob matching
    nothing scans nothing."""
    empty = [tree for tree in PIN_TREES if not any(REPO_ROOT.glob(tree))]
    assert not empty, f"PIN_TREES entries that match no file; fix or delete them: {empty}"

    pinned = {path for path, text in texts.items() if IMAGE_PIN.search(text)}
    silent = [area for area in DOCUMENTED_PIN_AREAS if not any(path.startswith(area) for path in pinned)]
    assert not silent, f"the scan read no image pin under {silent}; the gate certifies nothing there"


def test_a_release_candidate_is_read_with_its_suffix(texts):
    """Read without its suffix, ``9.9.9rc1`` would pass as the final ``9.9.9``."""
    candidate = "9.9.9" + "rc1"  # Built, so this file states no pin of its own.
    readme = texts["README.md"]
    start = _newest_release_entry(readme).start()
    planted = {
        **texts,
        "README.md": readme[:start] + f"- **2099-01-01 — Halo {candidate}.** Planted.\n" + readme[start:],
        "human-docs/planted.md": f"docker pull public.ecr.aws/whitecircle/halo:blackwell-{candidate}\n",
    }
    reported = disagreements(planted)
    assert f"README.md:{readme.count(chr(10), 0, start) + 1} states {candidate}" in reported, reported
    assert f"human-docs/planted.md:1 states {candidate}" in reported, reported


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
