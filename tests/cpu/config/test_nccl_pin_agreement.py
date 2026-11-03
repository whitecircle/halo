#!/usr/bin/env python
"""``uv.lock``, the ``uv`` override and the DeepEP V2 floor agree on one NCCL pin. The images link one wheel's
``libnccl.so.2``, and a skew hangs the RL weight-sync ``ncclCommInitRank`` on the NCCL timeout with no error."""

import tomllib

import pytest

from tests.common.repo import REPO_ROOT, load_script_module

nccl_pin = load_script_module("docker/nccl_pin.py")


def test_the_lock_the_override_and_the_floor_agree():
    locked = nccl_pin.locked_version(str(REPO_ROOT / "uv.lock"))
    overrides = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["uv"][
        "override-dependencies"
    ]

    assert f"{nccl_pin.PACKAGE}=={locked}" in overrides, (
        f"uv.lock pins {nccl_pin.PACKAGE} {locked} but override-dependencies says {overrides} — the "
        f"next `uv lock` re-resolves the images onto a different NCCL than the one they were built for"
    )
    assert nccl_pin._version_tuple(locked) >= nccl_pin._version_tuple(nccl_pin.MINIMUM), (
        f"{nccl_pin.PACKAGE} {locked} is below the DeepEP V2 floor {nccl_pin.MINIMUM}: deep_ep._C "
        f"would import with an undefined Gin symbol"
    )


def test_the_version_prefix_comparison_orders_real_wheel_spellings():
    """A stdlib-only parse must order the versions it sees, ``.post`` suffixes included, and not compare strings
    lexically (``2.9`` > ``2.31``)."""
    assert nccl_pin._version_tuple("2.28.9") < nccl_pin._version_tuple("2.30.4")
    assert nccl_pin._version_tuple("2.9.0") < nccl_pin._version_tuple("2.31.2")
    assert nccl_pin._version_tuple("2.31.2.post1") == (2, 31, 2)


def test_the_runtime_check_encodes_wheel_versions_the_way_nccl_reports_them():
    """``--verify`` compares the lock against ``ncclGetVersion``'s integer, so the encoding must be NCCL's own
    ``X*10000 + Y*100 + Z``."""
    assert nccl_pin.version_code("2.31.2") == 23102  # the wheel every image installs
    assert nccl_pin.version_code("2.29.7") == 22907  # the NGC base's system copy
    assert nccl_pin.version_code("2.28.9") == 22809  # torch's own metadata pin
    assert nccl_pin.version_code("2.30.4") == nccl_pin.version_code(nccl_pin.MINIMUM)
    assert nccl_pin.version_code("2.31") == nccl_pin.version_code("2.31.0")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
