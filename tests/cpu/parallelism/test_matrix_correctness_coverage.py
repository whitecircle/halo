#!/usr/bin/env python3
"""Every supported parallelism combination owns a GPU test comparing it to an undistributed reference, the only
thing that catches a wrong reduction."""

import pytest

from halo.distributed.parallelism_rules import SUPPORTED_AXIS_SETS, render_axis_set
from tests.gpu.manifest import MANIFEST

# Smoke tests do not count.
CORRECTNESS_GATES: dict[frozenset[str], tuple[str, ...]] = {
    frozenset(): (
        "parallelism/test_fsdp_tied_embeddings.py",
        "parallelism/tp/test_tp_dp_correctness.py",
    ),
    frozenset({"ep"}): (
        "parallelism/ep/test_ep_modes_vs_reference.py",
        "parallelism/ep/test_ep_vs_no_ep.py",
        "parallelism/ep/test_ep_family_vs_reference.py",
        "parallelism/combined/test_combined_ref_correctness.py",
    ),
    frozenset({"etp"}): (
        "parallelism/ep/test_ep_modes_vs_reference.py",
        "parallelism/combined/test_combined_ref_correctness.py",
    ),
    frozenset({"tp"}): (
        "parallelism/tp/test_tp_correctness.py",
        # Its per-head norm reduction is invisible to the dense test above.
        "parallelism/tp/test_tp_attention_norm_grad.py",
        "parallelism/combined/test_combined_ref_correctness.py",
    ),
    frozenset({"cp"}): (
        "parallelism/cp/test_cp_correctness.py",
        "parallelism/cp/test_cp_train_correctness.py",
        "parallelism/cp/test_cp_family_vs_reference.py",
    ),
    frozenset({"pp"}): ("parallelism/pp/test_pp_vs_single_gpu.py",),
    frozenset({"ep", "tp"}): (
        "parallelism/ep/test_ep_modes_vs_reference.py",
        "parallelism/combined/test_combined_ref_correctness.py",
    ),
    frozenset({"ep", "cp"}): (
        "parallelism/combined/test_ep_cp_correctness.py",
        "parallelism/combined/test_ep_cp_train_correctness.py",
    ),
    frozenset({"ep", "etp"}): (
        "parallelism/ep/test_ep_family_vs_reference.py",
        "parallelism/combined/test_combined_ref_correctness.py",
    ),
    frozenset({"pp", "ep"}): ("parallelism/pp/test_pp_ep_vs_unsplit.py",),
    frozenset({"pp", "etp"}): ("parallelism/pp/test_pp_etp_vs_unsplit.py",),
}


def test_every_supported_combination_has_a_correctness_gate():
    missing = sorted(render_axis_set(axes) for axes in SUPPORTED_AXIS_SETS if not CORRECTNESS_GATES.get(axes))
    assert not missing, (
        f"These parallelism combinations are in SUPPORTED_AXIS_SETS but no test compares them to an "
        f"undistributed reference: {missing}. Add the combination's equivalence test and register it "
        f"here, or drop the combination from the allowlist — an unproven combination can train with "
        f"mis-scaled gradients while every CPU test passes."
    )


def test_gates_reference_tests_that_actually_exist():
    """The manifest is the source of truth."""
    unknown = sorted({path for paths in CORRECTNESS_GATES.values() for path in paths if path not in MANIFEST})
    assert not unknown, (
        f"CORRECTNESS_GATES names GPU tests that are not in tests/gpu/manifest.py: {unknown}. "
        f"They were renamed or deleted, so the combinations they claim to cover are now unguarded."
    )


def test_no_gate_is_registered_for_a_rejected_combination():
    """Dead weight that reads as coverage."""
    stale = sorted(render_axis_set(axes) for axes in CORRECTNESS_GATES if axes not in SUPPORTED_AXIS_SETS)
    assert not stale, (
        f"CORRECTNESS_GATES still claims coverage for combinations that are no longer supported: "
        f"{stale}. Drop the entry along with the combination."
    )


def test_the_coverage_map_is_not_vacuous():
    """An empty allowlist or all-empty map would pass everything above."""
    assert len(SUPPORTED_AXIS_SETS) >= 10, f"allowlist shrank unexpectedly: {len(SUPPORTED_AXIS_SETS)}"
    assert all(CORRECTNESS_GATES.values()), "a gate entry is an empty tuple, which asserts nothing"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
