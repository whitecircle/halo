#!/usr/bin/env python
"""Every training script must construct its trainer with ``model=`` as a KEYWORD.

The pipeline seam in ``src/trainers/mixins/base.py`` must SPLIT the model into this rank's stage
before ``super().__init__`` ever sees it, and it rewrites ``kwargs["model"]`` to do so. A positional
model lands in ``*args`` instead, so pipeline parallelism raises "Pipeline parallelism requires the
model to be passed as the `model` keyword" no matter how the run is configured, leaving the shipped
script PP-unreachable. (The other setup steps read a positional model through ``ctor_args``; see
``tests/cpu/trainers/test_positional_ctor_setup.py``.)

The trainer, the config validation and every GPU test (which build trainers directly with
``model=``) stay green either way, and only the shipped script is degraded. This test reads the
scripts themselves rather than trusting review.

Run: ``pytest -m cpu tests/cpu/parallelism/test_scripts_pass_model_as_kwarg.py``
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO_ROOT / "scripts" / "training"

# Every trainer a shipped script constructs — the PP-capable roster (agent-docs/parallelism/
# pipeline-parallelism.md) and the rest alike. Kept explicit rather than imported so the test reads
# the scripts as text and cannot be satisfied by a runtime alias; a new trainer belongs here. The
# teacher-distillation trainer names its model ``student_model``, so the pipeline seam never sees it
# either way — it is listed because "no positional arguments" is the same contract for every trainer.
TRAINERS = {
    "DistributedSFTTrainer",
    "SmoothMarginPOTrainer",
    "DistributedDPOTrainer",
    "DistributedKTOTrainer",
    "DistributedRewardTrainer",
    "ClassificationTrainer",
    "OfflineGRPOTrainer",
    "DistributedGRPOTrainer",
    "DistributedAsyncEnvironmentalGRPOTrainer",
    "DistributedDistillationTrainer",
    "DistributedSelfDistillationTrainer",
    "DistributedSDPGTrainer",
    "EmbeddingTrainer",
}


def _trainer_constructions() -> list[tuple[Path, str, ast.Call]]:
    """Every ``<trainer>(...)`` call in ``scripts/training/``, with its source file."""
    found = []
    for path in sorted(_SCRIPTS.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in TRAINERS:
                found.append((path, node.func.id, node))
    return found


def test_training_scripts_are_discoverable():
    """Guard the guard: if the walk finds nothing, the assertion below is vacuous."""
    constructions = _trainer_constructions()
    assert len(constructions) >= 8, f"expected the shipped launch scripts, found {constructions}"


def test_no_script_passes_the_model_positionally():
    offenders = []
    for path, trainer, call in _trainer_constructions():
        if call.args:  # any positional argument at all — the model is always the first
            offenders.append(f"{path.relative_to(_REPO_ROOT)}:{call.lineno} {trainer}(<positional>, ...)")
    assert not offenders, (
        "These scripts pass the model positionally, so it never reaches the kwargs the pipeline "
        "seam in src/trainers/mixins/base.py splits — PP becomes unreachable:\n  " + "\n  ".join(offenders)
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
