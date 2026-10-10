#!/usr/bin/env python
"""Repo pins on the image build surface: locked dependencies and provenance labels.

Before the training image's locked install, a ``pip install`` that resolves dependencies for a
lock-pinned package pulls whatever PyPI serves that day. The locked install then replaces only the
packages the lock names, so every other package the resolve brought in stays in the image, unlocked.
Each such pre-install must pass ``--no-deps``.

Each image labels itself with the Halo commit, version and build date it was built from. Without
their own labels, the rollout images report their engine's upstream revision and the training image
reports the Ubuntu base's ``org.opencontainers.image.version``.

    python tests/cpu/config/test_image_build_surface.py
"""

import re
import tomllib

import pytest

from tests.common.utils import REPO_ROOT

# The training image's locked install; every pre-install above it must leave dependencies to it.
LOCKED_INSTALL = "uv pip install --system --no-deps -r"
PINNED_REQUIREMENT = re.compile(r'"?([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==[^\s"]+"?')
DOCKERFILES = sorted(path.name for path in REPO_ROOT.glob("Dockerfile*"))
# Each provenance label and the build arg it is stamped from.
PROVENANCE_LABELS = {
    "org.opencontainers.image.revision": "SOURCE_REVISION",
    "org.opencontainers.image.version": "VERSION",
    "org.opencontainers.image.created": "BUILD_DATE",
}


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _logical_lines(text: str) -> list[str]:
    """Lines with backslash continuations joined and comment lines dropped."""
    lines, pending = [], ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("#") or (not stripped and not pending):
            continue
        piece = stripped.removesuffix("\\").strip()
        pending = f"{pending} {piece}".strip()
        if not stripped.endswith("\\"):
            lines.append(pending)
            pending = ""
    return lines


def _locked_packages() -> set[str]:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    return {_normalize(package["name"]) for package in lock["package"]}


def _resolving_pre_installs(dockerfile_text: str, locked: set[str]) -> list[str]:
    """The ``pip install`` commands above the locked install that resolve a lock-pinned package's deps."""
    offenders = []
    for line in _logical_lines(dockerfile_text):
        if LOCKED_INSTALL in line:
            return offenders
        if not line.startswith("RUN "):
            continue
        for command in line.split("&&"):
            if "pip install" not in command or "--no-deps" in command:
                continue
            pinned = [m.group(1) for m in PINNED_REQUIREMENT.finditer(command.split("pip install", 1)[1])]
            if hits := [name for name in pinned if _normalize(name) in locked]:
                offenders.append(f"{command.strip()}  (lock-pinned: {', '.join(hits)})")
    raise AssertionError(f"no `{LOCKED_INSTALL}` line found: the scan lost its anchor")


def test_pre_installs_take_their_dependencies_from_the_lock():
    offenders = _resolving_pre_installs((REPO_ROOT / "Dockerfile").read_text(encoding="utf-8"), _locked_packages())
    assert not offenders, (
        "these pre-installs resolve dependencies the locked install will not remove; pass --no-deps:\n  "
        + "\n  ".join(offenders)
    )


def test_the_scan_flags_a_resolving_pre_install():
    """Guards the pin above against passing vacuously on a parser that sees nothing."""
    dockerfile = (
        "RUN export A=1 \\\n"
        '    && pip install --no-build-isolation "einops==0.8.2"\n'
        f"RUN {LOCKED_INSTALL} /tmp/requirements.txt\n"
    )
    assert _resolving_pre_installs(dockerfile, {"einops"}), (
        "a resolving pre-install of a locked package went unflagged"
    )
    assert not _resolving_pre_installs(dockerfile.replace("--no-build-isolation", "--no-deps"), {"einops"})
    assert not _resolving_pre_installs(dockerfile, {"other"}), "a package outside the lock was flagged"


@pytest.mark.parametrize("dockerfile", DOCKERFILES)
def test_every_image_labels_its_provenance(dockerfile):
    text = (REPO_ROOT / dockerfile).read_text(encoding="utf-8")
    missing = [
        f'LABEL {label}="${{{arg}}}" (with ARG {arg})'
        for label, arg in PROVENANCE_LABELS.items()
        if not re.search(rf'^LABEL {re.escape(label)}="\$\{{{arg}\}}"$', text, re.MULTILINE)
        or not re.search(rf"^ARG {arg}(=\S*)?$", text, re.MULTILINE)
    ]
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
