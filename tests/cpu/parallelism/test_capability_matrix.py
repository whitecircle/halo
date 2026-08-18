#!/usr/bin/env python3
"""The parallelism capability matrix is an ALLOWLIST, and behaves like one.

``ParallelismConfig`` decides which axis combinations may run from
:data:`SUPPORTED_AXIS_SETS`, not from a ladder of pairwise ``if x and y: raise``. The difference is
the failure direction: a denylist lets a combination nobody thought about run unvalidated, an
allowlist refuses it. These tests pin both halves — every listed set is genuinely constructible,
and every unlisted one is genuinely refused — so the matrix cannot rot into a decoration that the
real gate ignores.

Usage:
    python tests/cpu/parallelism/test_capability_matrix.py
"""

import itertools
import sys

import pytest

from src.distributed.parallelism_config import (
    AXIS_FLAGS,
    AXIS_SET_MECHANISMS,
    SUPPORTED_AXIS_SETS,
    ParallelismConfig,
)

DOMAIN = 8

# Per-axis sizes satisfying every topology rule on one NVLink domain. EP takes the whole domain so
# EP+CP's "ep_group_size == nvlink_domain_size" rule holds.
_AXIS_SIZES = {"pp": 2, "ep": DOMAIN, "etp": 2, "tp": 2, "cp": 2}


def _sizes_for(axes: frozenset[str]) -> dict:
    """Config kwargs realizing exactly ``axes`` — inactive axes pinned to 1."""
    sizes = {f"{axis}_size" if axis != "etp" else "expert_tp_size": 1 for axis in AXIS_FLAGS}
    for axis in axes:
        key = "expert_tp_size" if axis == "etp" else f"{axis}_size"
        sizes[key] = _AXIS_SIZES[axis]
    if "etp" in axes and "ep" in axes:
        # ep_size * expert_tp_size IS the full EP group; keep it to exactly one domain.
        sizes["ep_size"], sizes["expert_tp_size"] = DOMAIN // 2, 2
    return sizes


def _build(axes: frozenset[str]) -> ParallelismConfig:
    """Build on the smallest world that can host ``axes``.

    PP needs at least two NVLink domains (a stage must own whole domains); every other axis is
    domain-local, and giving them a second domain would instead trip the multi-group EP rules —
    a rejection about topology, not about capability, which is not what these tests are pinning.
    """
    world = DOMAIN * 2 if "pp" in axes else DOMAIN
    return ParallelismConfig(world_size=world, gpus_per_node=DOMAIN, nvlink_domain_size=DOMAIN, **_sizes_for(axes))


@pytest.mark.parametrize("axes", sorted(SUPPORTED_AXIS_SETS, key=lambda s: sorted(s)))
def test_every_supported_set_is_constructible(axes):
    """A listed combination must actually build — a matrix that lists shapes the validators then
    reject is worse than no matrix, because the support claim is documentation-only."""
    config = _build(axes)
    assert config.active_axes == axes, (config.active_axes, axes)


@pytest.mark.parametrize(
    "axes",
    sorted(
        (
            frozenset(combo)
            for size in range(1, len(AXIS_FLAGS) + 1)
            for combo in itertools.combinations(AXIS_FLAGS, size)
            if frozenset(combo) not in SUPPORTED_AXIS_SETS
        ),
        key=lambda s: (len(s), sorted(s)),
    ),
)
def test_every_unsupported_set_is_rejected(axes):
    """The allowlist half: anything not listed raises, whether or not a mechanism was written for
    it. A missing mechanism entry may degrade the explanation; it must never let the shape run."""
    with pytest.raises(ValueError) as excinfo:
        _build(axes)
    message = str(excinfo.value)
    # It must be the matrix that refused, not a downstream divisibility rule that happens to fire.
    assert "is not a supported parallelism combination" in message, message


def test_rejection_quotes_the_flags_the_user_typed():
    """Messages must name the CLI/YAML knobs, not the internal field names.

    A user sets ``expert_tensor_parallel_size``; an error that only says ``expert_tp_size`` names a
    symbol they cannot find in their config.
    """
    with pytest.raises(ValueError) as excinfo:
        _build(frozenset({"tp", "cp"}))
    message = str(excinfo.value)
    assert "tensor_parallel_size=2" in message, message
    assert "context_parallel_size=2" in message, message


def test_mechanisms_are_written_for_the_reachable_rejections():
    """Every hand-written mechanism must key a genuinely unsupported set.

    A mechanism whose axis set later becomes supported is dead prose that will drift; this catches
    that at the moment support is added rather than at the next reader.
    """
    stale = [sorted(axes) for axes in AXIS_SET_MECHANISMS if axes in SUPPORTED_AXIS_SETS]
    assert not stale, f"mechanism text describes now-SUPPORTED combinations: {stale}"


def test_the_allowlist_is_not_vacuous():
    """Guards the tests above: an empty or all-inclusive matrix would make both halves pass."""
    all_sets = {
        frozenset(combo) for size in range(len(AXIS_FLAGS) + 1) for combo in itertools.combinations(AXIS_FLAGS, size)
    }
    assert frozenset() in SUPPORTED_AXIS_SETS, "plain data parallelism must be supported"
    assert frozenset({"pp", "ep"}) in SUPPORTED_AXIS_SETS, "PP+EP is a supported combination"
    assert frozenset({"pp", "tp"}) not in SUPPORTED_AXIS_SETS, "PP+TP was deliberately dropped"
    assert 0 < len(SUPPORTED_AXIS_SETS) < len(all_sets), (len(SUPPORTED_AXIS_SETS), len(all_sets))


def test_unsupported_combination_is_refused_before_any_rank_math():
    """The allowlist must be checked FIRST, not after the divisibility rules.

    ``ParallelismConfig`` promises (and the docs publish) that an axis combination with no validated
    composition is refused as unsupported rather than as a divisibility error about a shape that was
    never going to run. The two rejections differ in what the user does next: "PP+TP is unsupported"
    ends the attempt, "world_size 8 must be divisible by pp_size 3" invites picking a different
    pp_size that will then hit the real wall. Each case below ALSO violates a rank-math rule, so it
    only passes while the capability check genuinely runs first.
    """
    for kwargs, axes in (
        ({"pp_size": 3, "tp_size": 2}, "PP + TP"),
        ({"pp_size": 3, "cp_size": 2}, "PP + CP"),
        ({"pp_size": 5, "tp_size": 2, "cp_size": 2}, "PP + TP"),
    ):
        with pytest.raises(ValueError) as excinfo:
            ParallelismConfig(world_size=8, gpus_per_node=8, **kwargs)
        message = str(excinfo.value)
        assert "not a supported parallelism combination" in message, (
            f"{kwargs} reported a rank-math error before the capability check:\n{message}"
        )
        assert "divisible" not in message.split("\n")[0], f"{kwargs} led with a divisibility error"
        assert axes in message, f"{kwargs} named the wrong axes: {message}"


@pytest.mark.parametrize(
    ("knob", "value"),
    [("pp_split", [1, 1]), ("pp_microbatches", 8), ("pp_schedule", "gpipe")],
)
def test_pp_only_knobs_are_refused_at_pp_size_one(knob, value):
    """A PP knob set without PP is a config that does not do what it says — nothing reads it."""
    with pytest.raises(ValueError, match="only meaningful with pipeline_parallel_size"):
        ParallelismConfig(world_size=8, gpus_per_node=8, **{knob: value})


def test_pp_knobs_at_their_defaults_are_accepted_at_pp_size_one():
    """Anti-vacuity: the rejection must key on a CHANGED value, not on the field existing."""
    config = ParallelismConfig(world_size=8, gpus_per_node=8)
    assert config.pp_size == 1 and config.pp_split is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
