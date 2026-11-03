#!/usr/bin/env python
"""Every training script constructs its trainer with ``model=`` as a keyword, which the pipeline seam rewrites. A
positional model leaves the script PP-unreachable while GPU tests stay green."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.common.repo import REPO_ROOT

_SCRIPTS = REPO_ROOT / "scripts" / "training"

# Explicit so the test reads scripts as text and no runtime alias satisfies it. The teacher trainer names its model
# ``student_model``, listed for the no-positional contract.
TRAINERS = {
    "DistributedSFTTrainer",
    "SMPOTrainer",
    "DistributedDPOTrainer",
    "DistributedKTOTrainer",
    "DistributedRewardTrainer",
    "ClassificationTrainer",
    "OfflineGRPOTrainer",
    "DistributedOnlineGRPOTrainer",
    "DistributedAsyncEnvironmentalGRPOTrainer",
    "DistributedTeacherDistillationTrainer",
    "DistributedSelfDistillationTrainer",
    "DistributedSDPGTrainer",
    "EmbeddingTrainer",
}


def _trainer_names(node: ast.AST) -> set[str]:
    return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name) and sub.id in TRAINERS}


def _trainer_constructions() -> list[tuple[Path, str, ast.Call]]:
    """A call on a trainer class, or on a local name bound to one."""
    found = []
    for path in sorted(_SCRIPTS.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        aliases = {
            target.id: " | ".join(sorted(names))
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and (names := _trainer_names(node.value))
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                name = node.func.id
                if name in TRAINERS:
                    found.append((path, name, node))
                elif name in aliases:
                    found.append((path, f"{name} ({aliases[name]})", node))
    return found


def test_training_scripts_are_discoverable():
    """If the walk finds nothing, the assertion below is vacuous."""
    constructions = _trainer_constructions()
    assert len(constructions) >= 8, f"expected the shipped launch scripts, found {constructions}"


def test_every_script_naming_a_trainer_has_its_construction_seen():
    """A missed construction would pass the positional check unread."""
    constructed = {path for path, _trainer, _call in _trainer_constructions()}
    unseen = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted(_SCRIPTS.rglob("*.py"))
        if _trainer_names(ast.parse(path.read_text(), filename=str(path))) and path not in constructed
    ]
    assert not unseen, f"these scripts name a trainer but no construction of it was found: {unseen}"


def test_no_script_passes_the_model_positionally():
    offenders = []
    for path, trainer, call in _trainer_constructions():
        if call.args:  # The model is always the first.
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{call.lineno} {trainer}(<positional>, ...)")
    assert not offenders, (
        "These scripts pass the model positionally, so it never reaches the kwargs the pipeline "
        "seam in halo/trainers/distributed_trainer.py splits — PP becomes unreachable:\n  " + "\n  ".join(offenders)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
