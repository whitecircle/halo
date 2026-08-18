#!/usr/bin/env python
"""Every stack version this repo *states* must equal the pin it is stated about.

The stack lives in four images built from three Dockerfiles and one lock, and each version is
pinned in exactly one place: ``pyproject.toml`` for the training dependencies, ``uv.lock`` (through
``docker/nccl_pin.py``) for the NCCL runtime every image shares, and a ``Dockerfile`` ``ARG`` for
each engine and the NGC base. Prose and error strings across the guides, ``agent-docs/``, ``src/``,
``tests/``, ``scripts/``, ``examples/`` and the agent skills restate those numbers hundreds of
times, and a bump that updates the pin and misses one of them points a reader — or a refusal
message — at a version nobody runs. This gate reads each pin from its single source and fails on
any stated number that disagrees.

Numbers that are deliberately not the pin are exempt by **form**, not by location: a floor
(``>= X``), an attribution to the release something first shipped in (``native in X``), and
provenance (``vendored from X``, ``measured on X``) all stay true when the pin moves forward, so
they are recognized by the shape of the sentence — and only for a release this stack has already
reached, so no form can excuse a number ahead of the pin. The two numbers with no such shape are
exempt by ``(package, version)`` pair with a reason. Both lists are anti-rot checked: an entry that
describes nothing in the tree fails.

Which pin a number is measured against is read off the sentence too: the rollout images carry their
own transformers, so a number is held to the training pin unless a rollout image is the nearest
thing the sentence attributes it to.

    python tests/cpu/config/test_stack_versions_in_sync.py
"""

import re
import tomllib
from functools import cache
from pathlib import Path
from typing import NamedTuple

import pytest

from tests.common.utils import REPO_ROOT, load_script_module

nccl_pin = load_script_module("docker/nccl_pin.py")

# Every tree that states a stack version: the root guides, the docs, the error strings and
# docstrings under src/, the pinned-behaviour comments in tests/ and scripts/, the launch comments
# in examples/, the agent skills, and the pin sources themselves — a Dockerfile's ``ARG`` agrees
# with itself, but the paragraph of comment explaining why it says that number does not. Left out:
# ``uv.lock`` (generated), ``Makefile`` (tags only — checked below), and ``plans/``, whose history
# is deliberately dated. Data files under agent-docs/assets record third-party baseline environments.
SCANNED_TREES = (
    "CLAUDE.md",
    "README.md",
    "pyproject.toml",
    "Dockerfile*",
    "docker-compose*.yml",
    "agent-docs/**/*.md",
    "src/**/*.py",
    "tests/**/*.py",
    "scripts/**/*.py",
    "docker/**/*.py",
    "docker/**/*.sh",
    "examples/**/*.yaml",
    "skills/**/*.md",
)

# Vendored hub remote-code, and this file, which names non-pin versions by construction — in the
# exemption reasons and in the planted-drift fixture. Neither states a version this repo owns.
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

# A version token, and what may sit between it and the package name: a comparator, an image tag's
# `:`, a hyphenated spelling (`torch-2.11`), a markdown cell wall, the backtick closing a code-font
# name. `<` is deliberately absent — an exclusive upper bound is the pin's own range end, not a
# claim about what runs.
_VERSION = r"v?(\d+\.\d+(?:\.\d+)?)"
_LEAD = r"(?:[\s~^=≥:|`-]|>=|>|==|=)*"

# How each package is spelled where a version follows it. Anchored on a non-word boundary so
# ``flash_attn`` does not match ``attn``, and kept to the spellings that actually precede a number.
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
    # ``-dsl`` is load-bearing: CUTLASS, the vendored C++ template library the FA source builds
    # carry, is a different thing from the ``nvidia-cutlass-dsl`` wheel and moves on its own.
    "cutlass_dsl": r"(?:nvidia-)?cutlass-dsl(?:-libs-[a-z0-9]+)?",
    "quack": r"quack(?:-kernels)?",
}

# Prose wraps, and the subject of a sentence — whose image a version belongs to, the verb that makes
# it history — can sit a line above the number. A statement is judged on this window, not its line.
WINDOW_LINES = 2

# Whose stack a number belongs to is settled by the nearest of these words before it — a sentence
# routinely names both sides ("the server ships X, training is on Y"), and the subject closest to
# the number owns it. The rollout images carry their own transformers, both behind the training
# image's, so those lines are claimable only where a rollout image is what the number is about;
# an unattributed "transformers X" is a claim about the training stack and is held to its pin.
ATTRIBUTION = re.compile(r"\b(?:vllm|sglang|rollout|server|training)\b", re.IGNORECASE)

# A file named for an engine is about that engine: it is the attribution of last resort, for the
# lines inside `Dockerfile.vllm` or `docker/vllm/` that never repeat whose image they configure.
# Only a fallback — a "training" nearer the number still wins, as it does in prose.
ENGINE_PATH = re.compile(r"vllm|sglang", re.IGNORECASE)

# Sentence shapes that state a release other than the pin on purpose. ``{version}`` is replaced by
# the matched number and the match must cover it, so the marker word sits beside *this* number, in
# one sentence (``[^.\n|]`` stops at a full stop, a line break and a markdown cell wall alike).
# Every verb here is past tense: "5.14 shipped X" is history, "5.14 ships X" is a claim about the
# release in use, which a bump has to revisit.
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

# Numbers that are deliberately not the pin and have no such shape. Each entry must still match
# something, so a statement that is deleted or corrected takes its exemption with it.
ALLOWED_EXCEPTIONS: dict[tuple[str, str], str] = {
    ("torch", "2.13"): "the torch generation vLLM 0.27+ / SGLang 0.5.18 move to — why the engines are capped",
    ("nccl", "2.28.9"): "torch's own wheel-metadata pin — the value the uv override exists to replace",
    ("cutlass_dsl", "4.4.2"): "gram-newton-schulz's own hard pin — the value the force-reinstall exists to replace",
}


class Statement(NamedTuple):
    """One version a scanned file states, with the window of prose it is judged in."""

    package: str
    version: str
    path: Path
    line_number: int
    line: str
    window: str
    offset: int  # where the number sits in ``window`` — what "before the number" means


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
    """``5.16`` covers ``5.16.1`` and vice versa; ``5.1`` never covers ``5.16``.

    Pins are held at the precision their source states them, so a truncated pin cannot wildcard a
    patch level: against ``0.5.17``, ``SGLang 0.5`` still reads as that line, ``0.5.14`` does not.
    """
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
        # The pair FA4 and the Muon / block-scaled MoE kernels share: one ARG each, interpolated
        # into every install line, so the number lives in exactly one place per package.
        "cutlass_dsl": {_dockerfile_value("Dockerfile", r"^ARG CUTLASS_DSL_VERSION=(\S+)$")},
        "quack": {_dockerfile_value("Dockerfile", r"^ARG QUACK_VERSION=(\S+)$")},
    }


@cache
def engine_scoped_pins() -> dict[str, set[str]]:
    """Pins claimable only where a rollout image is the nearest thing the number is attributed to.

    Each engine ships its own transformers: vLLM's is installed here (a bounded range this image
    chooses), SGLang's is its upstream image's exact pin, recorded in its Dockerfile. Both sit a
    line or two behind the training image's, so leaving them claimable everywhere would let a stale
    training-stack sentence pass as a statement about a server.
    """
    return {
        "transformers": {
            _dockerfile_value("Dockerfile.vllm", r'pip install --no-cache-dir "transformers>=([\d.]+),<'),
            _dockerfile_value("Dockerfile.sglang", r"transformers==([\d.]+)"),
        }
    }


def all_pins() -> dict[str, set[str]]:
    """Both scopes unioned — what a statement's age is measured against, and what a failure prints."""
    engine = engine_scoped_pins()
    return {package: versions | engine.get(package, set()) for package, versions in stack_pins().items()}


def stated_versions() -> list[Statement]:
    """Every version a scanned file states."""
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
    """Whether the attribution nearest the number is a rollout image rather than the training one."""
    nearest = None
    for match in ATTRIBUTION.finditer(statement.window[: statement.offset]):
        nearest = match.group(0).lower()
    if nearest is None:
        return ENGINE_PATH.search(str(statement.path)) is not None
    return nearest != "training"


def _form_matches(form: str, statement: Statement) -> bool:
    """Whether a historical shape is written around *this* statement's number.

    The match has to span the number itself: the same digits two lines up, under a marker word of
    their own, are a different statement and cannot lend this one their exemption.
    """
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
    """Statements no pin in scope covers — each one needs an exemption or it is drift."""
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
    """Stated versions that are neither the pin nor a legitimately non-pin statement."""
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
    """Anti-vacuity for the gate itself: the tree passing must mean drift is caught, not missed.

    One planted file per shape, because the window spans lines: a bare stale pair, a sentence
    claiming a rollout image's transformers line with nothing saying it is one, a provenance-shaped
    verb the forms deliberately do not cover, a floor ahead of the pin, a present-tense capability
    claim (which a bump must revisit, unlike the past tense the forms exempt), a marker word two
    lines above a different number, and — the shape that made attribution positional — one sentence
    naming both images, where only the training half drifts.
    """
    planted = {
        "bare.md": "Serve from vLLM 0.19.3, whose transformers 5.13 reads what the checkpoint writer writes.\n",
        "unattributed.py": '"""Mirrors transformers 5.14, whose default REPLACES the extra-special list."""\n',
        "provenance.py": '"""Shapes taken from a live SGLang v0.5.14 response."""\n',
        "ahead_of_the_pin.md": "The fused path needs transformers >= 9.9.\n",
        "present_tense.md": "vLLM 0.24.0 ships no Zaya implementation, so no export of it is servable.\n",
        "borrowed_marker.md": "The cache fix needs transformers >= 5.10 at minimum.\n\nWe install transformers 5.10.\n",
        "contrasted.md": "The server image ships transformers 5.14.1; training is on transformers 5.14.\n",
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
        "contrasted.md": ("transformers 5.14",),  # and NOT the 5.14.1 the server half claims
    }
    reported = drifted_statements()
    assert len(reported) == sum(len(statements) for statements in expected.values()), reported
    for name, statements in expected.items():
        for statement in statements:
            assert any(name in entry and statement in entry for entry in reported), (name, statement, reported)


def test_every_exemption_still_describes_something():
    """Anti-rot: an exemption for a sentence nobody writes any more silently widens the gate.

    Measured against the statements no pin covers — the only ones an exemption can speak for — so an
    entry kept alive by a sentence that now states the pin is reported as the dead weight it is.
    """
    unpinned = unpinned_statements()
    stated = {(statement.package, statement.version) for statement in unpinned}
    stale = sorted(pair for pair in ALLOWED_EXCEPTIONS if pair not in stated)
    assert not stale, f"ALLOWED_EXCEPTIONS entries exempt nothing any more — delete them: {stale}"

    unused = [form for form in HISTORICAL_FORMS if not any(_form_matches(form, s) for s in unpinned)]
    assert not unused, f"HISTORICAL_FORMS patterns exempt no statement any more — delete them: {unused}"


def test_image_tags_agree_with_the_dockerfile_args():
    """Every engine image tag written down — build/push, compose default, doc, test, launch comment
    — names the version its Dockerfile ARG builds, exactly: the scan above reads a tag as a version
    statement, which accepts the line (``vllm-server:0.26``) where a tag has to be the whole number.
    ``Makefile`` is here rather than in the scan because tags are all it states."""
    pins = stack_pins()
    expected = {
        "vllm-server": _dockerfile_value("Dockerfile.vllm", r"^ARG VLLM_VERSION=(\S+)$"),
        "sglang-server": _dockerfile_value("Dockerfile.sglang", r"^ARG SGLANG_VERSION=(\S+)$"),
    }
    assert expected["vllm-server"] in pins["vllm"] and expected["sglang-server"] in pins["sglang"]

    wrong, found = [], {image: 0 for image in expected}
    for path in sorted({REPO_ROOT / "Makefile", *SCANNED_FILES}):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for image, version in expected.items():
                for tagged in re.findall(rf"{image}[:-]([\d.]+)", line):
                    found[image] += 1
                    if tagged != version:
                        wrong.append(f"{path}:{number} tags {image}:{tagged}, but the ARG builds {version}")
    assert not wrong, "image tags disagree with the Dockerfile ARG they are built from:\n  " + "\n  ".join(wrong)
    assert all(found.values()), f"no image tag left to check for {sorted(k for k, v in found.items() if not v)}"


def test_the_scan_actually_reads_the_stack():
    """Anti-vacuity: a regex or a glob that stopped matching would make every assertion above pass."""
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
