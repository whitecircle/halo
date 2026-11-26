#!/usr/bin/env python
"""``uv.lock`` and the ``uv`` override agree on one ``nvidia-cublas`` pin at or above the first release free of
the dropped-trailing-rows bf16 matmul bug (pytorch/pytorch#203481); torch's own lock carries the broken 13.1.0.3."""

import tomllib

import pytest

from tests.common.repo import REPO_ROOT

PACKAGE = "nvidia-cublas"
MINIMUM = (13, 8, 0, 4)


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def test_the_lock_and_the_override_agree_above_the_floor():
    locked = next(
        package["version"]
        for package in tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))["package"]
        if package["name"] == PACKAGE
    )
    overrides = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["uv"][
        "override-dependencies"
    ]

    assert f"{PACKAGE}=={locked}" in overrides, (
        f"uv.lock pins {PACKAGE} {locked} but override-dependencies says {overrides}; the next `uv lock` "
        f"re-resolves the images onto a different cuBLAS"
    )
    assert _version_tuple(locked) >= MINIMUM, f"{PACKAGE} {locked} is below {MINIMUM}, which drops matmul rows"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
