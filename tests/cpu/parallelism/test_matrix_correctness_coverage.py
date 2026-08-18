#!/usr/bin/env python3
"""Every supported parallelism combination must own a numerical correctness gate.

``SUPPORTED_AXIS_SETS`` decides what may run; this map decides what must be *proven* to run
correctly. Without it a combination reaches the allowlist with no test comparing its gradients
against an unsplit reference — not a hypothetical: unguarded PP+ETP produces gradients at 0.32x the
reference, and PP itself fails with a ``TypeError`` on its default normalizer path, both while the
whole CPU suite stays green.

This test closes the loop from the other side: each allowlisted axis set names the GPU test that
compares it to a single-GPU reference, and both directions are checked — an allowlisted set with
no gate fails, and a named gate that has vanished from the manifest fails. Smoke tests (finite
loss, "it ran") deliberately do not count; only tests that compare against an undistributed
reference do, because that is the only thing that catches a wrong reduction.

Usage:
    python tests/cpu/parallelism/test_matrix_correctness_coverage.py
"""

import sys

import pytest

from src.distributed.parallelism_config import SUPPORTED_AXIS_SETS, _render_axis_set
from tests.gpu.manifest import MANIFEST

# Axis set -> GPU tests comparing it to an UNDISTRIBUTED reference; smoke tests do not count.
CORRECTNESS_GATES: dict[frozenset[str], tuple[str, ...]] = {
    frozenset(): (
        # Plain FSDP2 data parallelism, incl. the tied-embedding split.
        "parallelism/test_fsdp_tied_embeddings.py",
        "parallelism/tp/test_tp_dp_correctness.py",
    ),
    frozenset({"ep"}): (
        "parallelism/ep/test_ep_correctness.py",
        "parallelism/ep/test_ep_vs_no_ep.py",
        "parallelism/ep/test_ep_vs_reference_qwen3_moe.py",
        "parallelism/combined/test_combined_ref_correctness.py",
    ),
    frozenset({"etp"}): ("parallelism/combined/test_combined_ref_correctness.py",),
    frozenset({"tp"}): (
        "parallelism/tp/test_tp_correctness.py",
        # The toolkit's attention-only TP path (MoE): its per-head norm reduction is invisible to the dense test above.
        "parallelism/tp/test_tp_attention_norm_grad.py",
        "parallelism/combined/test_combined_ref_correctness.py",
    ),
    frozenset({"cp"}): (
        "parallelism/cp/test_cp_correctness.py",
        "parallelism/cp/test_cp_train_correctness.py",
        "parallelism/cp/test_qwen3_5_cp_correctness.py",
    ),
    frozenset({"ep", "tp"}): (
        "parallelism/combined/test_ep_tp_correctness.py",
        "parallelism/combined/test_combined_ref_correctness.py",
    ),
    frozenset({"ep", "cp"}): (
        "parallelism/combined/test_ep_cp_correctness.py",
        "parallelism/combined/test_ep_cp_train_correctness.py",
    ),
    frozenset({"ep", "etp"}): (
        "parallelism/combined/test_ep_etp_correctness.py",
        "parallelism/combined/test_combined_ref_correctness.py",
    ),
}

# PP's schedule engine is not in this release: the pp axis sets stay allowlisted for the config
# surface (rank math, gates, seams), and the runtime raise in PipelineRuntime keeps them unrunnable,
# so no equivalence gate can exist for them yet.
_ENGINE_PENDING_AXIS_SETS = frozenset(axes for axes in SUPPORTED_AXIS_SETS if "pp" in axes)


def test_every_supported_combination_has_a_correctness_gate():
    """An allowlisted combination with no reference-equivalence test is an unproven combination."""
    gated = SUPPORTED_AXIS_SETS - _ENGINE_PENDING_AXIS_SETS
    missing = sorted(_render_axis_set(axes) for axes in gated if not CORRECTNESS_GATES.get(axes))
    assert not missing, (
        f"These parallelism combinations are in SUPPORTED_AXIS_SETS but no test compares them to an "
        f"undistributed reference: {missing}. Add the combination's equivalence test and register it "
        f"here, or drop the combination from the allowlist — shipping it unproven is how PP+ETP "
        f"trained at 0.32x the reference gradient."
    )


def test_gates_reference_tests_that_actually_exist():
    """A gate naming a deleted or renamed test proves nothing; the manifest is the source of truth."""
    unknown = sorted({path for paths in CORRECTNESS_GATES.values() for path in paths if path not in MANIFEST})
    assert not unknown, (
        f"CORRECTNESS_GATES names GPU tests that are not in tests/gpu/manifest.py: {unknown}. "
        f"They were renamed or deleted, so the combinations they claim to cover are now unguarded."
    )


def test_no_gate_is_registered_for_a_rejected_combination():
    """A gate for a combination the allowlist refuses is dead weight that reads as coverage."""
    stale = sorted(_render_axis_set(axes) for axes in CORRECTNESS_GATES if axes not in SUPPORTED_AXIS_SETS)
    assert not stale, (
        f"CORRECTNESS_GATES still claims coverage for combinations that are no longer supported: "
        f"{stale}. Drop the entry along with the combination."
    )


def test_the_coverage_map_is_not_vacuous():
    """Guard the guard: an empty allowlist or an all-empty map would pass everything above."""
    assert len(SUPPORTED_AXIS_SETS) >= 10, f"allowlist shrank unexpectedly: {len(SUPPORTED_AXIS_SETS)}"
    assert all(CORRECTNESS_GATES.values()), "a gate entry is an empty tuple, which asserts nothing"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
