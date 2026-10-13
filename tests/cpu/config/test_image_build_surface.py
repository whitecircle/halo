#!/usr/bin/env python
"""Repo pins on the image build surface: locked dependencies and provenance labels.

A ``pip install`` in the training image that resolves dependencies for a package the lock names pulls
whatever PyPI serves that day. Ahead of the locked install, that install replaces only the packages the
lock names, so every other package the resolve brought in stays in the image, unlocked; after it, a
resolve moves locked packages off their pins. Each such install must pass ``--no-deps``, however it
spells the requirement, and the optional extras come from the lock export itself, which ``make install``
exports the same way.

Each image labels itself with the Halo commit, version and build date it was built from. Without
their own labels, the rollout images report their engine's upstream revision and the training image
reports the Ubuntu base's ``org.opencontainers.image.version``.

    python tests/cpu/config/test_image_build_surface.py
"""

import re
import tomllib

import pytest
from packaging.utils import canonicalize_name

from tests.common.utils import REPO_ROOT

# The training image's locked install: the one place its dependencies are resolved from.
LOCKED_INSTALL = "uv pip install --system --no-deps -r"
LOCK_EXPORT = "uv export --locked"
PIP_INSTALL = re.compile(r"\bpip3?\b(?:\s+-\S+)*\s+install\b")
COMMAND_SEPARATOR = re.compile(r"&&|\|\||;")
INSTALL_ARGUMENT = re.compile(r"""\"[^\"]*\"|'[^']*'|\S+""")
REQUIREMENT = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?(?:\s*(?:===?|~=|!=|[<>]=?).*)?")
# pip options whose next argument is their value, never a requirement.
VALUE_OPTIONS = frozenset({"-r", "--requirement", "-c", "--constraint", "-i", "--index-url", "--extra-index-url"})
# Packages installed by a resolve on purpose, ahead of every source build: the build tool, and the torch the
# source builds compile against, from the PyTorch index.
BOOTSTRAP_PACKAGES = frozenset({"ninja", "torch"})
EXPORTED_EXTRA = re.compile(r"--extra[ =](\S+)")
ALL_EXTRAS = "--all-extras"
DOCKERFILES = sorted(path.name for path in REPO_ROOT.glob("Dockerfile*"))
# Each provenance label and the build arg it is stamped from.
PROVENANCE_LABELS = {
    "org.opencontainers.image.revision": "SOURCE_REVISION",
    "org.opencontainers.image.version": "VERSION",
    "org.opencontainers.image.created": "BUILD_DATE",
}


def _logical_lines(text: str) -> list[str]:
    """Lines with backslash continuations joined; comment and blank lines dropped, as Docker does."""
    lines, pending = [], ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        piece = stripped.removesuffix("\\").strip()
        pending = f"{pending} {piece}".strip()
        if not stripped.endswith("\\"):
            lines.append(pending)
            pending = ""
    return lines


def _locked_packages() -> set[str]:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    return {canonicalize_name(package["name"]) for package in lock["package"]}


def _requirement_names(arguments: str) -> list[str]:
    """The package names an install's arguments name, in any spelling: bare, pinned, ranged, with extras.
    Options, their values, paths and shell variables are not requirements."""
    names, option_value = [], False
    for argument in INSTALL_ARGUMENT.findall(arguments):
        if option_value or argument.startswith(("-", "$", ".")) or "/" in argument:
            option_value = argument in VALUE_OPTIONS
            continue
        if requirement := REQUIREMENT.fullmatch(argument.strip("\"'")):
            names.append(canonicalize_name(requirement.group(1)))
    return names


def _resolving_installs(dockerfile_text: str, locked: set[str]) -> list[str]:
    """The ``pip install`` commands that resolve the dependencies of a package the lock names. Ahead of the
    locked install the :data:`BOOTSTRAP_PACKAGES` may; from it on nothing may."""
    lines = _logical_lines(dockerfile_text)
    if not any(LOCKED_INSTALL in line for line in lines):
        raise AssertionError(f"no `{LOCKED_INSTALL}` line found: the scan lost its anchor")
    offenders, allowed = [], BOOTSTRAP_PACKAGES
    for line in lines:
        if LOCKED_INSTALL in line:
            allowed = frozenset()
        if not line.startswith("RUN "):
            continue
        for command in COMMAND_SEPARATOR.split(line):
            install = PIP_INSTALL.search(command)
            if not install or "--no-deps" in command:
                continue
            if hits := [name for name in _requirement_names(command[install.end() :]) if name in locked - allowed]:
                offenders.append(f"{command.strip()}  (in the lock: {', '.join(hits)})")
    return offenders


def _exported_extras(text: str, extras: set[str]) -> list[set[str]]:
    """The extras each lock export in a Dockerfile or the Makefile carries, ``extras`` being all of them."""
    return [
        set(extras) if ALL_EXTRAS in line else set(EXPORTED_EXTRA.findall(line))
        for line in _logical_lines(text)
        if LOCK_EXPORT in line
    ]


def test_installs_take_their_dependencies_from_the_lock():
    offenders = _resolving_installs((REPO_ROOT / "Dockerfile").read_text(encoding="utf-8"), _locked_packages())
    assert not offenders, (
        "these installs resolve dependencies outside the lock; pass --no-deps, or for an optional extra "
        "export it with the lock:\n  " + "\n  ".join(offenders)
    )


def test_the_scan_flags_a_resolving_install():
    """Guards the pin above against passing vacuously on a parser that sees nothing."""
    dockerfile = (
        "RUN export A=1 \\\n"
        "\n"
        '    && pip install --no-build-isolation "einops==0.8.2"\n'
        f"RUN {LOCKED_INSTALL} /tmp/requirements.txt\n"
    )
    assert _resolving_installs(dockerfile, {"einops"}), "a resolving pre-install of a locked package went unflagged"
    assert not _resolving_installs(dockerfile.replace("--no-build-isolation", "--no-deps"), {"einops"})
    assert not _resolving_installs(dockerfile, {"other"}), "a package outside the lock was flagged"
    chained = f'RUN pip3 install --no-deps "a==1"; pip3 install "einops==0.8.2"\nRUN {LOCKED_INSTALL} r.txt\n'
    assert _resolving_installs(chained, {"einops"}), "a resolve chained after a --no-deps install went unflagged"
    same_line = f'RUN {LOCKED_INSTALL} r.txt && pip install "einops==0.8.2"\n'
    assert _resolving_installs(same_line, {"einops"}), "a resolve on the locked install's own line went unflagged"
    for spelling in ("einops", '"einops>=0.8"', "'einops==0.8.2'", "--upgrade einops", '"einops[extra]~=0.8.0"'):
        after = f"RUN {LOCKED_INSTALL} r.txt\nRUN pip --no-cache-dir install {spelling}\n"
        assert _resolving_installs(after, {"einops"}), f"`{spelling}` after the locked install went unflagged"
    not_requirements = f"RUN pip install --index-url https://einops.example/simple -r einops.txt . $EINOPS\nRUN {LOCKED_INSTALL} r.txt\n"
    assert not _resolving_installs(not_requirements, {"einops"}), (
        "an option value, path or variable was read as a requirement"
    )
    bootstrap = sorted(BOOTSTRAP_PACKAGES)[0]
    ahead = f'RUN pip install "{bootstrap}~=1.0"\nRUN {LOCKED_INSTALL} r.txt\n'
    assert not _resolving_installs(ahead, {bootstrap}), "a bootstrap install ahead of the locked one was flagged"
    behind = f'RUN {LOCKED_INSTALL} r.txt\nRUN pip install "{bootstrap}~=1.0"\n'
    assert _resolving_installs(behind, {bootstrap}), (
        "a bootstrap package resolved after the locked install went unflagged"
    )
    with pytest.raises(AssertionError, match="lost its anchor"):
        _resolving_installs('RUN pip install --no-deps "einops==0.8.2"\n', {"einops"})


def test_the_image_installs_every_optional_extra_from_the_lock():
    """An extra installed by its own ``pip install`` resolves its dependencies outside the lock, and
    ``make install`` would then check a different requirement set than the image builds from."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = set(pyproject["project"]["optional-dependencies"])
    assert extras, "pyproject.toml declares no optional extras: the scan lost its root"
    assert _exported_extras(f"RUN {LOCK_EXPORT} {ALL_EXTRAS} -o r.txt\n", extras) == [extras]
    for name in ("Dockerfile", "Makefile"):
        exports = _exported_extras((REPO_ROOT / name).read_text(encoding="utf-8"), extras)
        assert exports, f"no `{LOCK_EXPORT}` found in {name}: the scan lost its anchor"
        assert all(exported == extras for exported in exports), (
            f"{name} exports the lock with extras {[sorted(e) for e in exports]}, not {sorted(extras)}"
        )


@pytest.mark.parametrize("dockerfile", DOCKERFILES)
def test_every_image_labels_its_provenance(dockerfile):
    """Each label renders from its build arg, declared in the final stage before the label reads it."""
    lines = (REPO_ROOT / dockerfile).read_text(encoding="utf-8").splitlines()
    last_from = max(i for i, line in enumerate(lines) if line.startswith("FROM "))
    missing = []
    for label, arg in PROVENANCE_LABELS.items():
        labeled_at = [i for i, line in enumerate(lines) if line == f'LABEL {label}="${{{arg}}}"']
        declared_at = [i for i, line in enumerate(lines) if re.fullmatch(rf"ARG {arg}(=\S*)?", line)]
        if not labeled_at or not any(last_from < i < labeled_at[-1] for i in declared_at):
            missing.append(f'ARG {arg}, then LABEL {label}="${{{arg}}}", after the last FROM')
    assert not missing, f"{dockerfile} does not label its provenance: {missing}"


def test_every_image_build_passes_the_provenance_args():
    builds = [
        line for line in _logical_lines((REPO_ROOT / "Makefile").read_text(encoding="utf-8")) if "docker build" in line
    ]
    assert len(builds) >= len(DOCKERFILES), (
        f"found {len(builds)} docker builds in the Makefile: the scan lost its root"
    )
    missing = {
        build: [arg for arg in PROVENANCE_LABELS.values() if f"--build-arg {arg}=$({arg})" not in build]
        for build in builds
    }
    missing = {build: args for build, args in missing.items() if args}
    assert not missing, f"these image builds leave provenance labels empty: {missing}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
