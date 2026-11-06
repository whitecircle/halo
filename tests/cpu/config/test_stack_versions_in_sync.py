#!/usr/bin/env python
"""Every stack version this repo states equals the pin it is stated about. Numbers off the pin are exempt by form (a
floor, an attribution, provenance), only for a release the stack has reached, or by ``(package, version)`` pair
with a reason; a number is held to the training pin unless a rollout image is its nearest attribution."""

import re
import tomllib
from functools import cache
from pathlib import Path
from typing import NamedTuple

import pytest

from tests.common.repo import REPO_ROOT, load_script_module

nccl_pin = load_script_module("docker/nccl_pin.py")
locked_version = load_script_module(".github/scripts/locked_version.py").locked_version

# Every tree that states a stack version, the pin sources included. Left out: ``uv.lock`` (generated) and ``Makefile``
# (tags only, checked below). Data files under agent-docs/assets record third-party baseline environments.
SCANNED_TREES = (
    "CLAUDE.md",
    "README.md",
    "pyproject.toml",
    "Dockerfile*",
    "docker-compose*.yml",
    "agent-docs/**/*.md",
    "human-docs/**/*.md",
    "halo/**/*.py",
    "tests/**/*.py",
    "scripts/**/*.py",
    "docker/**/*.py",
    "docker/**/*.sh",
    "examples/**/*.yaml",
    "skills/**/*.md",
)

# Vendored hub remote-code; this file names non-pin versions by construction.
NOT_SCANNED = ("docker/vllm/parity/fixtures",)

FILES_BY_TREE: dict[str, list[Path]] = {
    tree: sorted(
        path
        for path in REPO_ROOT.glob(tree)
        if path != Path(__file__).resolve() and not any(skip in str(path) for skip in NOT_SCANNED)
    )
    for tree in SCANNED_TREES
}
SCANNED_FILES = sorted({path for paths in FILES_BY_TREE.values() for path in paths})

# A version token, and what may sit between it and the package name. ``<`` is absent: an exclusive upper bound is the
# pin's own range end.
_VERSION = r"v?(\d+\.\d+(?:\.\d+)?)"
_LEAD = r"(?:[\s~^=≥:|`-]|>=|>|==|=)*"

# Anchored on a non-word boundary so ``flash_attn`` does not match ``attn``.
PACKAGE_ALIASES: dict[str, str] = {
    "torch": r"(?:PyTorch|torch)",
    "transformers": r"[Tt]ransformers",
    "trl": r"TRL",
    "accelerate": r"[Aa]ccelerate",
    "peft": r"PEFT",
    "liger": r"[Ll]iger(?:[ -][Kk]ernels?)?",
    "vllm": r"(?:vLLM|vllm(?:-server)?)",
    "sglang": r"(?:SGLang|sglang(?:-server)?)",
    "nccl": r"(?:nvidia-nccl-cu13|NCCL)",
    "ngc": r"nvcr\.io/nvidia/pytorch:",
    # ``-dsl`` matters: the vendored CUTLASS C++ library the FA source builds carry moves on its own, apart from the
    # ``nvidia-cutlass-dsl`` wheel.
    "cutlass_dsl": r"(?:nvidia-)?cutlass-dsl(?:-libs-[a-z0-9]+)?",
    "quack": r"quack(?:-kernels)?",
}

# A sentence's subject can sit a line above the number, so a statement is judged on this window, not its line.
WINDOW_LINES = 2

# The nearest of these words before a number owns it: a sentence routinely names both sides. An unattributed
# transformers number is held to the training pin.
ATTRIBUTION = re.compile(r"\b(?:vllm|sglang|rollout|server|training)\b", re.IGNORECASE)

# A file named for an engine is about that engine, the attribution of last resort; a nearer training word still wins.
ENGINE_PATH = re.compile(r"vllm|sglang", re.IGNORECASE)

# Sentence shapes stating a release other than the pin on purpose. The match must cover ``{version}``, so the marker
# sits beside this number in one sentence. Every verb is past tense: a present-tense claim is one a bump has to
# revisit.
HISTORICAL_FORMS: dict[str, str] = {
    r"(?:>=|≥|>|at least)\s*v?{version}\b|{version}\+(?![\w.])": (
        "a floor — the oldest upstream release that works, which moving the pin forward cannot falsify"
    ),
    r"\b(?:native|added|landed|introduced|shipped|arrived|since)\b[^.\n|]{0,40}?{version}|{version}[^.\n|]{0,25}?\bshipped\b": (
        "an attribution to the upstream release a feature first appeared in"
    ),
    r"\b(?:vendored|measured|benchmarked)\b[^.\n|]{0,60}?{version}": (
        "provenance — the release a vendored copy was taken from, or a benchmark was measured on"
    ),
}

# Each entry must still match something, so a corrected statement takes its exemption with it.
ALLOWED_EXCEPTIONS: dict[tuple[str, str], str] = {
    ("nccl", "2.28.9"): "torch's own wheel-metadata pin — the value the uv override exists to replace",
    ("cutlass_dsl", "4.4.2"): "gram-newton-schulz's own hard pin — the value the force-reinstall exists to replace",
    ("trl", "1.17.1"): "the TRL release the framework comparison benchmarked as a competitor",
    ("liger", "0.8.2"): "the Liger release the TRL arm of the framework comparison ran",
    ("transformers", "5.18.0"): "the transformers release NeMo AutoModel ran in the framework comparison",
}


class Statement(NamedTuple):
    package: str
    version: str
    path: Path
    line_number: int
    line: str
    window: str
    offset: int  # where the number sits in ``window``


@cache
def _pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _declared_lower_bound(package: str) -> str:
    """The lower bound ``pyproject.toml`` declares for a dependency, e.g. ``5.16.1``."""
    for requirement in _pyproject()["project"]["dependencies"]:
        match = re.fullmatch(rf"{re.escape(package)}(?:\[[^\]]*\])?>=([^,]+),<.+", requirement.strip())
        if match:
            return match.group(1)
    raise AssertionError(f"{package} is not a bounded dependency in pyproject.toml — this gate cannot pin it")


def _dockerfile_value(dockerfile: str, pattern: str) -> str:
    """The single occurrence of ``pattern``'s capture group in a Dockerfile."""
    matches = re.findall(pattern, (REPO_ROOT / dockerfile).read_text(encoding="utf-8"), flags=re.MULTILINE)
    assert len(set(matches)) == 1, f"{pattern!r} matched {sorted(set(matches))} in {dockerfile} — expected exactly one"
    return matches[0]


def _matches(stated: str, pinned: str) -> bool:
    """``5.16`` covers ``5.16.1`` and vice versa; ``5.1`` never covers ``5.16``. Pins hold at their source's
    precision, so a truncated pin cannot wildcard a patch level."""
    return stated == pinned or stated.startswith(f"{pinned}.") or pinned.startswith(f"{stated}.")


@cache
def stack_pins() -> dict[str, set[str]]:
    """Every version each package may be stated at anywhere, read from its single source."""
    return {
        "torch": {_declared_lower_bound("torch")},
        "transformers": {_declared_lower_bound("transformers")},
        "trl": {_declared_lower_bound("trl")},
        "accelerate": {_declared_lower_bound("accelerate")},
        "peft": {_declared_lower_bound("peft")},
        "liger": {_declared_lower_bound("liger-kernel")},
        "vllm": {_dockerfile_value("Dockerfile.vllm", r"^ARG VLLM_VERSION=(\S+)$")},
        "sglang": {_dockerfile_value("Dockerfile.sglang", r"^ARG SGLANG_VERSION=(\S+)$")},
        # The floor deep_ep._C needs is a legitimate second number for NCCL: it is what the images
        # assert against, and it moves only with a DeepEP rebuild.
        "nccl": {nccl_pin.locked_version(str(REPO_ROOT / "uv.lock")), nccl_pin.MINIMUM},
        "ngc": {_dockerfile_value("Dockerfile", r"^ARG BASE_IMAGE=nvcr\.io/nvidia/pytorch:(\S+)-py3$")},
        # The pair FA4 and the Muon / block-scaled MoE kernels share, one ARG each.
        "cutlass_dsl": {_dockerfile_value("Dockerfile", r"^ARG CUTLASS_DSL_VERSION=(\S+)$")},
        "quack": {_dockerfile_value("Dockerfile", r"^ARG QUACK_VERSION=(\S+)$")},
    }


@cache
def engine_scoped_pins() -> dict[str, set[str]]:
    """Pins claimable only where a rollout image is the number's nearest attribution. Each engine's transformers
    sits behind the training image's, so a stale training sentence could otherwise pass as a statement about a
    server."""
    return {
        "transformers": {
            _dockerfile_value("Dockerfile.vllm", r"transformers==([\d.]+)"),
            _dockerfile_value("Dockerfile.sglang", r"transformers==([\d.]+)"),
        }
    }


def all_pins() -> dict[str, set[str]]:
    """Both scopes unioned: what a statement's age is measured against and what a failure prints."""
    engine = engine_scoped_pins()
    return {package: versions | engine.get(package, set()) for package, versions in stack_pins().items()}


def stated_versions() -> list[Statement]:
    patterns = {name: re.compile(rf"(?<![\w./-]){alias}{_LEAD}{_VERSION}") for name, alias in PACKAGE_ALIASES.items()}
    found = []
    for path in SCANNED_FILES:
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            window = "\n".join(lines[max(0, index - WINDOW_LINES) : index + 1])
            for package, pattern in patterns.items():
                for match in pattern.finditer(line):
                    offset = len(window) - len(line) + match.start(1)
                    found.append(Statement(package, match.group(1), path, index + 1, line.strip(), window, offset))
    return found


def _claimed_by_a_rollout_image(statement: Statement) -> bool:
    nearest = None
    for match in ATTRIBUTION.finditer(statement.window[: statement.offset]):
        nearest = match.group(0).lower()
    if nearest is None:
        return ENGINE_PATH.search(str(statement.path)) is not None
    return nearest != "training"


def _form_matches(form: str, statement: Statement) -> bool:
    """Whether a historical shape spans this statement's number; the same digits two lines up are a different
    statement."""
    pattern = form.replace("{version}", re.escape(statement.version))
    return any(
        match.start() <= statement.offset < match.end()
        for match in re.finditer(pattern, statement.window, flags=re.IGNORECASE)
    )


def _exemption(statement: Statement, pins: set[str]) -> str | None:
    """Why a number that is not the pin is legitimate anyway, or ``None`` if it is drift."""
    if (statement.package, statement.version) in ALLOWED_EXCEPTIONS:
        return ALLOWED_EXCEPTIONS[statement.package, statement.version]
    # ``version_code`` is the lock's own version→int encoding: monotone, and it stops at a local
    # suffix (``.post1``) the way a comparison must.
    if nccl_pin.version_code(statement.version) > max(nccl_pin.version_code(pin) for pin in pins):
        return None  # nothing is history about a release this stack has not reached
    return next((reason for form, reason in HISTORICAL_FORMS.items() if _form_matches(form, statement)), None)


def unpinned_statements() -> list[Statement]:
    """Statements no pin in scope covers; each needs an exemption or it is drift."""
    anywhere, engine = stack_pins(), engine_scoped_pins()
    unpinned = []
    for statement in stated_versions():
        claimable = set(anywhere[statement.package])
        if _claimed_by_a_rollout_image(statement):
            claimable |= engine.get(statement.package, set())
        if not any(_matches(statement.version, pin) for pin in claimable):
            unpinned.append(statement)
    return unpinned


def drifted_statements() -> list[str]:
    every = all_pins()
    return [
        f"{statement.path}:{statement.line_number} states {statement.package} {statement.version} "
        f"(pinned {sorted(every[statement.package])}): {statement.line}"
        for statement in unpinned_statements()
        if not _exemption(statement, every[statement.package])
    ]


def test_stated_versions_match_their_pins():
    drifted = drifted_statements()
    assert not drifted, (
        f"{len(drifted)} version statement(s) disagree with the pin they describe — update the prose, "
        f"state the number as the floor or the history it really is, or add an ALLOWED_EXCEPTIONS "
        f"entry if it is deliberately not the pin:\n  " + "\n  ".join(drifted)
    )


def test_a_stale_number_anywhere_in_the_prose_is_reported(tmp_path, monkeypatch):
    """One planted file per shape, since the window spans lines: a bare stale pair, an unattributed rollout
    transformers line, an uncovered provenance verb, a floor ahead of the pin, a present-tense claim, a marker
    two lines above another number, and one sentence naming both images."""
    planted = {
        "bare.md": "Serve from vLLM 0.19.3, whose transformers 5.13 reads what the checkpoint writer writes.\n",
        "unattributed.py": '"""Mirrors transformers 5.14, whose default REPLACES the extra-special list."""\n',
        "provenance.py": '"""Shapes taken from a live SGLang v0.5.14 response."""\n',
        "ahead_of_the_pin.md": "The fused path needs transformers >= 9.9.\n",
        "present_tense.md": "vLLM 0.24.0 ships no Zaya implementation, so no export of it is servable.\n",
        "borrowed_marker.md": "The cache fix needs transformers >= 5.10 at minimum.\n\nWe install transformers 5.10.\n",
        "contrasted.md": "The server image ships transformers 5.19.0; training is on transformers 5.19.\n",
    }
    for name, text in planted.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    monkeypatch.setitem(globals(), "SCANNED_FILES", [tmp_path / name for name in planted])

    expected = {
        "bare.md": ("vllm 0.19.3", "transformers 5.13"),
        "unattributed.py": ("transformers 5.14",),
        "provenance.py": ("sglang 0.5.14",),
        "ahead_of_the_pin.md": ("transformers 9.9",),
        "present_tense.md": ("vllm 0.24.0",),
        "borrowed_marker.md": ("transformers 5.10",),  # the floor two lines up is a different claim
        "contrasted.md": ("transformers 5.19",),  # and not the 5.19.0 the server half claims
    }
    reported = drifted_statements()
    assert len(reported) == sum(len(statements) for statements in expected.values()), reported
    for name, statements in expected.items():
        for statement in statements:
            assert any(name in entry and statement in entry for entry in reported), (name, statement, reported)


def test_every_exemption_still_describes_something():
    """An exemption for a sentence nobody writes widens the gate. Measured against the statements no pin covers, the
    only ones an exemption can speak for."""
    unpinned = unpinned_statements()
    stated = {(statement.package, statement.version) for statement in unpinned}
    stale = sorted(pair for pair in ALLOWED_EXCEPTIONS if pair not in stated)
    assert not stale, f"ALLOWED_EXCEPTIONS entries exempt nothing any more — delete them: {stale}"

    unused = [form for form in HISTORICAL_FORMS if not any(_form_matches(form, s) for s in unpinned)]
    assert not unused, f"HISTORICAL_FORMS patterns exempt no statement any more — delete them: {unused}"


def test_image_tags_agree_with_the_dockerfile_args():
    """Every engine image tag names the exact version its Dockerfile ARG builds: the scan above accepts the line
    (``halo:vllm-0.26``), where a tag must be the whole number. ``Makefile`` is here because tags are all it
    states."""
    pins = stack_pins()
    expected = {
        "halo:vllm": _dockerfile_value("Dockerfile.vllm", r"^ARG VLLM_VERSION=(\S+)$"),
        "halo:sglang": _dockerfile_value("Dockerfile.sglang", r"^ARG SGLANG_VERSION=(\S+)$"),
    }
    assert expected["halo:vllm"] in pins["vllm"] and expected["halo:sglang"] in pins["sglang"]

    wrong, found = [], {image: 0 for image in expected}
    for path in sorted({REPO_ROOT / "Makefile", *SCANNED_FILES}):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for image, version in expected.items():
                for tagged in re.findall(rf"{image}-(\d+(?:\.\d+)*)", line):
                    found[image] += 1
                    if tagged != version:
                        wrong.append(f"{path}:{number} tags {image}-{tagged}, but the ARG builds {version}")
    assert not wrong, "image tags disagree with the Dockerfile ARG they are built from:\n  " + "\n  ".join(wrong)
    assert all(found.values()), f"no image tag left to check for {sorted(k for k, v in found.items() if not v)}"


def test_ruff_is_pinned_once_in_the_lock():
    """``uv.lock`` is ruff's one pin: CI and ``make lint`` read it, and pre-commit, which takes only a
    literal ``rev``, is held to it here, so a ruff bump is a ``uv lock`` plus that one line."""
    locked = locked_version("ruff")
    pre_commit = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    revs = re.findall(r"repo: https://github\.com/astral-sh/ruff-pre-commit\n\s+rev: v(\S+)", pre_commit)
    assert revs == [locked], f"pre-commit runs ruff {revs}, but uv.lock pins {locked}"
    for name in (".github/workflows/lint.yml", "Makefile"):
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        literal = re.findall(r"(?:ruff==|ruff@|RUFF_VERSION \?= )v?(\d[\w.]*)", text)
        assert not literal, f"{name} pins ruff {literal} by hand instead of reading uv.lock"


def test_the_scan_actually_reads_the_stack():
    """A regex or a glob that stopped matching would make every assertion above pass."""
    statements = stated_versions()
    assert set(PACKAGE_ALIASES) == set(stack_pins()), (
        "a package is spelled in one table and pinned in the other — every alias needs a pin to "
        f"check it against: {sorted(set(PACKAGE_ALIASES) ^ set(stack_pins()))}"
    )

    per_package = {statement.package for statement in statements}
    assert per_package == set(PACKAGE_ALIASES), (
        f"the scan found no version statement for {sorted(set(PACKAGE_ALIASES) - per_package)} — "
        f"its alias no longer matches the prose, so the gate certifies nothing for it"
    )

    read = {statement.path for statement in statements}
    silent = sorted(tree for tree, paths in FILES_BY_TREE.items() if not read.intersection(paths))
    assert not silent, (
        f"the scan read no version statement out of {silent} — a pattern that matches no file, or a "
        f"tree that states no version, is one this gate certifies nothing for"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
