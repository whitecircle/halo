#!/usr/bin/env python
"""A GPU row that fails on an open bug still runs, as a strict ``xfail`` naming that bug.

``TestSpec.known_failures`` maps rows of a suite's ``args_matrix`` to the bug each fails on, and
``tests/gpu/conftest.py`` marks exactly those rows strict ``xfail``: the failure stays visible, and a row
that starts passing fails the tier until its entry is removed. An entry naming no row would mark
nothing and read as coverage, so the spec refuses it.

    python tests/cpu/conventions/test_gpu_known_failures.py
"""

import pytest

from tests.gpu import conftest as gpu_conftest
from tests.gpu.manifest import TestSpec


class _Metafunc:
    fixturenames = ("gpu_case",)

    def parametrize(self, name, params):
        self.params = params


def test_a_known_failure_must_name_a_row_the_suite_runs():
    with pytest.raises(ValueError, match="does not run"):
        TestSpec(nproc=2, args_matrix=("--mode a",), known_failures={"--mode b": "open bug"})


def test_only_known_failure_rows_are_collected_as_strict_xfail(monkeypatch):
    spec = TestSpec(
        nproc=2, markers=("gpu",), args_matrix=("--mode a", "--mode b"), known_failures={"--mode b": "bug"}
    )
    monkeypatch.setattr(gpu_conftest, "MANIFEST", {"suite.py": spec})
    metafunc = _Metafunc()
    gpu_conftest.pytest_generate_tests(metafunc)
    xfails = {param.id: [mark.kwargs for mark in param.marks if mark.name == "xfail"] for param in metafunc.params}
    assert xfails == {"suite.py[--mode a]": [], "suite.py[--mode b]": [{"reason": "bug", "strict": True}]}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
