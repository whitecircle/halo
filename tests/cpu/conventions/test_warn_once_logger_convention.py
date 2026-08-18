#!/usr/bin/env python
"""Repo pin: every ``warn_once`` caller logs through a stdlib logger.

``warn_once`` forwards only ``exc_info`` (:mod:`src.log`), so it cannot ask
accelerate's ``MultiProcessAdapter`` for ``main_process_only=False`` — and that adapter defaults it
to ``True``. A module that warns once through ``accelerate.logging.get_logger`` therefore drops the
warning on every rank but 0, which is precisely the rank a per-row / per-shard condition tends to
land on, and the ``seen`` set is per-rank anyway. The trainers legitimately use the adapter for
their rank-0 progress logging; the pin is only on the modules that also warn once.

    python tests/cpu/conventions/test_warn_once_logger_convention.py
"""

import ast

import pytest

from tests.common.utils import REPO_ROOT

SRC_ROOT = REPO_ROOT / "src"


def _warn_once_callers():
    """Source files that actually call ``warn_once`` (the helper's own module aside)."""
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "warn_once"
        ]
        if calls:
            yield path, tree


def _module_logger_expression(tree: ast.Module) -> str | None:
    for node in tree.body:
        if isinstance(node, ast.Assign) and [t.id for t in node.targets if isinstance(t, ast.Name)] == ["logger"]:
            return ast.unparse(node.value)
    return None


def test_every_warn_once_caller_binds_a_stdlib_logger():
    callers = list(_warn_once_callers())
    assert len(callers) > 5, f"only {len(callers)} warn_once callers found — the sweep lost its root"

    offenders = {
        str(path.relative_to(REPO_ROOT)): _module_logger_expression(tree)
        for path, tree in callers
        if _module_logger_expression(tree) != "logging.getLogger(__name__)"
    }
    assert not offenders, (
        "these modules warn once through a non-stdlib logger, so the warning is dropped on every "
        f"rank but 0 (warn_once cannot pass main_process_only=False): {offenders}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
