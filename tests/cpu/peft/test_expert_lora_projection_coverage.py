#!/usr/bin/env python
"""CPU guard: every registered EP family's non-bias expert-weight attribute root must have a
``LORA_PROJECTION_COVERAGE`` entry.

``ExpertLoraSpec.adapts`` resolves ``lora_target_modules`` projections against
:data:`~src.distributed.expert_parallel.config.LORA_PROJECTION_COVERAGE`; a family adding a new
expert-weight root (``_EXPERT_WEIGHT_ATTR_ROOTS``) without a coverage entry would make expert-LoRA
on that weight a SILENT no-op (``adapts`` returns False, no adapter built, no error). This test
fails the moment such a root appears. Bias roots (``*_bias``) are 2-D and never adapted —
``_init_expert_lora`` only adapts 3-D weights — so they are exempt.

Run: ``python tests/cpu/peft/test_expert_lora_projection_coverage.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest

import src.distributed.expert_parallel.layers.roster  # noqa: F401 — registers every EP family subclass
from src.distributed.expert_parallel.config import LORA_PROJECTION_COVERAGE
from src.distributed.expert_parallel.expert_weights import expert_weight_roots


def test_every_non_bias_expert_root_has_lora_coverage():
    roots = expert_weight_roots()
    assert roots, "no EP family registered any expert-weight roots — subclass walk broken"
    missing = sorted(r for r in roots if not r.endswith("_bias") and r not in LORA_PROJECTION_COVERAGE)
    assert not missing, (
        f"expert-weight roots {missing} have no LORA_PROJECTION_COVERAGE entry — expert-LoRA on "
        f"them would be a silent no-op. Add the root → logical-projection mapping in "
        f"src/distributed/expert_parallel/config.py."
    )


def test_coverage_maps_only_known_projections():
    """Coverage values must stay within the logical projection vocabulary ExpertLoraSpec uses."""
    legal = {"gate", "up", "down"}
    for root, projections in LORA_PROJECTION_COVERAGE.items():
        assert projections and projections <= legal, f"{root}: illegal projections {projections - legal}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
