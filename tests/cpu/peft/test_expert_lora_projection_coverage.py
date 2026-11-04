#!/usr/bin/env python
"""Every registered EP family's non-bias expert-weight root must have a ``LORA_PROJECTION_COVERAGE`` entry.
Without one, expert LoRA on that weight is a silent no-op. Bias roots are 2-D, and only 3-D weights are adapted."""

from __future__ import annotations

import pytest

from halo.models.moe.expert_lora_spec import LORA_PROJECTION_COVERAGE
from halo.models.moe.family_specs import expert_weight_roots


def test_every_non_bias_expert_root_has_lora_coverage():
    roots = expert_weight_roots()
    assert roots, "no EP family registered any expert-weight roots — subclass walk broken"
    missing = sorted(r for r in roots if not r.endswith("_bias") and r not in LORA_PROJECTION_COVERAGE)
    assert not missing, (
        f"expert-weight roots {missing} have no LORA_PROJECTION_COVERAGE entry — expert-LoRA on "
        f"them would be a silent no-op. Add the root → logical-projection mapping in "
        f"halo/distributed/expert_parallel/config.py."
    )


def test_coverage_maps_only_known_projections():
    """Values stay within the logical projection vocabulary ExpertLoraSpec uses."""
    legal = {"gate", "up", "down"}
    for root, projections in LORA_PROJECTION_COVERAGE.items():
        assert projections and projections <= legal, f"{root}: illegal projections {projections - legal}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
