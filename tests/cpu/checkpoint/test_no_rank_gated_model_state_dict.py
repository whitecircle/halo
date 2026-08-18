#!/usr/bin/env python
"""No save/load path may call a MODEL's ``state_dict()`` under a rank gate.

Under FSDP2 ``state_dict()`` is not a read: a registered pre-hook reshards the module's parameters
back to ``DTensor(Shard(0))``. With ``reshard_after_forward=False`` (the toolkit default) the params
sit UNSHARDED between a forward and its backward, so a call made on one rank alone leaves that rank
holding DTensors while its peers hold plain tensors. The next op over them is then collective on the
gated rank and local on everyone else — a desync that surfaces far from its cause, as a peer timing
out in an unrelated barrier.

The two sites that invite the shape are ``persistent_buffers`` (a writer-only PEFT unmerge goes
collective alone and the peer dies in DeepEP teardown) and ``CheckpointLoader._load_fsdp2``'s rank-0
coverage gate. Both read live FQNs from ``named_parameters()``/``named_buffers()`` instead, which
touch no state dict.

This is a structural test because reproducing the divergence needs 2 ranks and a real FSDP2 wrap
(covered in the GPU tier). It fails the moment the shape is reintroduced anywhere in the save/load
surface. Run: ``pytest -m cpu tests/cpu/checkpoint/test_no_rank_gated_model_state_dict.py``
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]

# The surface that writes or restores model weights; optimizer/scheduler state is per-rank by design.
_SCANNED = (_REPO_ROOT / "src" / "distributed", _REPO_ROOT / "src" / "checkpoint")

# Names that decide "this rank writes" — the gates that make a call rank-asymmetric.
_RANK_GATES = (
    "is_global_main_process",
    "is_local_main_process",
    "fs_aware_save_rank",
    "fs_aware_load_rank",
    "is_save_rank",
    "should_save",
    "is_writer",
    "is_pp_shard_writer",
)

# Receivers whose state_dict() is a MODEL's. Optimizer/scheduler receivers are deliberately absent.
_MODEL_RECEIVERS = ("model", "module", "stage", "wrapper", "backbone", "policy")


def _is_model_state_dict_call(node: ast.AST) -> str | None:
    """``<receiver>.state_dict()`` where the receiver looks like a model → the receiver's source."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return None
    if node.func.attr != "state_dict":
        return None
    receiver = ast.unparse(node.func.value)
    lowered = receiver.lower()
    return receiver if any(hint in lowered for hint in _MODEL_RECEIVERS) else None


def _gate_in(test: ast.AST) -> str | None:
    """The rank-gate name an ``if`` test depends on, if any."""
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
                rel = path.relative_to(_REPO_ROOT)
                found.append(f"{rel}:{inner.lineno} — {receiver}.state_dict() under `if {gate}`")
    return found


def test_no_model_state_dict_under_a_rank_gate():
    scanned = [f for root in _SCANNED for f in sorted(root.rglob("*.py"))]
    assert len(scanned) > 50, "the sweep lost its roots — rglob on a missing directory is silent"
    offenders = [msg for f in scanned for msg in _offenders(f)]
    assert not offenders, (
        "FSDP2 state_dict() reshards the module's parameters, so calling it on one rank alone "
        "splits the ranks' sharding state and desynchronizes the job. Read live names from "
        "named_parameters()/named_buffers() instead:\n  " + "\n  ".join(offenders)
    )


def test_detector_catches_the_shape():
    """Anti-vacuity: the scanner must actually flag the pattern it exists to forbid."""
    source = (
        "def save(model):\n"
        "    if is_global_main_process():\n"
        "        matched = len(set(state) & set(model.state_dict()))\n"
    )
    tmp = _REPO_ROOT / "src" / "distributed" / "checkpoint" / "__init__.py"  # any real path for relative_to
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _gate_in(node.test):
            hits += [r for inner in ast.walk(node) if (r := _is_model_state_dict_call(inner))]
    assert hits == ["model"], f"detector failed to flag the known-bad shape (got {hits}); {tmp.name} scan is vacuous"


def test_detector_ignores_optimizer_state_dict():
    """Anti-false-positive: a rank-gated OPTIMIZER state_dict is correct and must not be flagged."""
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
    sys.exit(pytest.main([__file__, "-v"]))
