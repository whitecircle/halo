#!/usr/bin/env python
"""No save/load path calls a model's ``state_dict()`` under a rank gate: under FSDP2 a pre-hook reshards, leaving ranks
in different states. Structural, since reproducing the desync needs 2 ranks and a real FSDP2 wrap (the GPU tier)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.common.repo import REPO_ROOT

# Optimizer and scheduler state is per-rank by design.
_SCANNED = (REPO_ROOT / "halo" / "distributed", REPO_ROOT / "halo" / "checkpoint")

# The gates that make a call rank-asymmetric.
_RANK_GATES = (
    "is_global_main_process",
    "is_local_main_process",
    "is_fs_save_rank",
    "is_fs_load_rank",
    "is_save_rank",
    "should_save",
    "is_writer",
    "is_pp_shard_writer",
)

# Optimizer and scheduler receivers are deliberately absent.
_MODEL_RECEIVERS = ("model", "module", "stage", "wrapper", "backbone", "policy")


def _is_model_state_dict_call(node: ast.AST) -> str | None:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return None
    if node.func.attr != "state_dict":
        return None
    receiver = ast.unparse(node.func.value)
    lowered = receiver.lower()
    return receiver if any(hint in lowered for hint in _MODEL_RECEIVERS) else None


def _gate_in(test: ast.AST) -> str | None:
    return next((name for name in _RANK_GATES if name in ast.unparse(test)), None)


def _offenders(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        gate = _gate_in(node.test)
        if gate is None:
            continue
        for inner in ast.walk(node):
            receiver = _is_model_state_dict_call(inner)
            if receiver is not None:
                rel = path.relative_to(REPO_ROOT)
                found.append(f"{rel}:{inner.lineno} — {receiver}.state_dict() under `if {gate}`")
    return found


def test_no_model_state_dict_under_a_rank_gate():
    assert all(root.is_dir() for root in _SCANNED), f"the sweep lost a root: {_SCANNED}"
    scanned = [f for root in _SCANNED for f in sorted(root.rglob("*.py"))]
    assert len(scanned) > 50, "the sweep lost its roots — rglob on a missing directory is silent"
    offenders = [msg for f in scanned for msg in _offenders(f)]
    assert not offenders, (
        "FSDP2 state_dict() reshards the module's parameters, so calling it on one rank alone "
        "splits the ranks' sharding state and desynchronizes the job. Read live names from "
        "named_parameters()/named_buffers() instead:\n  " + "\n  ".join(offenders)
    )


def test_detector_catches_the_shape():
    """Anti-vacuity: the scanner must flag the pattern."""
    source = (
        "def save(model):\n"
        "    if is_global_main_process():\n"
        "        matched = len(set(state) & set(model.state_dict()))\n"
    )
    tmp = REPO_ROOT / "halo" / "distributed" / "checkpoint" / "__init__.py"  # Any real path for relative_to.
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _gate_in(node.test):
            hits += [r for inner in ast.walk(node) if (r := _is_model_state_dict_call(inner))]
    assert hits == ["model"], f"detector failed to flag the known-bad shape (got {hits}); {tmp.name} scan is vacuous"


def test_detector_ignores_optimizer_state_dict():
    """A rank-gated optimizer state_dict is correct."""
    source = "def save(optimizer):\n    if is_global_main_process():\n        blob = optimizer.state_dict()\n"
    tree = ast.parse(source)
    hits = [
        r
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and _gate_in(node.test)
        for inner in ast.walk(node)
        if (r := _is_model_state_dict_call(inner))
    ]
    assert hits == [], f"optimizer state_dict must not be flagged (got {hits})"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
