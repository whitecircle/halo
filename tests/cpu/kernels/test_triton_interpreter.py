#!/usr/bin/env python
"""The interpreted kernel tests' preflight, in a fresh process, since ``TRITON_INTERPRET`` is read at kernel
definition: one failure naming the cause, beside the subprocess errors the interpreted tests raise on a Triton whose
interpreter cannot run a runtime loop bound."""

import pytest

from tests.common.repo import probe_findings

_MARKER = "TRITON_INTERPRETER_FINDINGS:"
_PROBE = f"""
import os
os.environ["TRITON_INTERPRET"] = "1"
from tests.common.triton_interpreter import interpreter_findings
print({_MARKER!r} + "|".join(interpreter_findings()))
"""


def test_the_interpreter_runs_a_runtime_loop_bound():
    findings = probe_findings(_PROBE, _MARKER)
    assert not findings, "\n".join(findings)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
