#!/usr/bin/env python3
"""The capability matrix is an allowlist: every listed set is constructible and every unlisted one refused, since a
denylist lets the unconsidered run unvalidated."""

import itertools

import pytest

from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.parallelism_rules import AXES, AXIS_SET_MECHANISMS, SUPPORTED_AXIS_SETS

AXIS_NAMES = tuple(a.name for a in AXES)

DOMAIN = 8

# EP takes the whole domain so EP+CP's ``ep_group_size == nvlink_domain_size`` rule holds.
_AXIS_SIZES = {"pp": 2, "ep": DOMAIN, "etp": 2, "tp": 2, "cp": 2}


def _sizes_for(axes: frozenset[str]) -> dict:
    sizes = {a.size_field: _AXIS_SIZES[a.name] if a.name in axes else 1 for a in AXES}
    if "etp" in axes and "ep" in axes:
        # ep_size * expert_tp_size is the full EP group; keep it to exactly one domain.
        sizes["ep_size"], sizes["expert_tp_size"] = DOMAIN // 2, 2
    return sizes


def _build(axes: frozenset[str]) -> ParallelismConfig:
    """PP needs two NVLink domains; a second domain elsewhere would trip multi-group EP rules, a rejection about
    topology."""
    world = DOMAIN * 2 if "pp" in axes else DOMAIN
    return ParallelismConfig(world_size=world, gpus_per_node=DOMAIN, nvlink_domain_size=DOMAIN, **_sizes_for(axes))


@pytest.mark.parametrize("axes", sorted(SUPPORTED_AXIS_SETS, key=lambda s: sorted(s)))
def test_every_supported_set_is_constructible(axes):
    """A matrix listing shapes the validators reject is worse than no matrix."""
    config = _build(axes)
    assert config.active_axes == axes, (config.active_axes, axes)


@pytest.mark.parametrize(
    "axes",
    sorted(
        (
            frozenset(combo)
            for size in range(1, len(AXIS_NAMES) + 1)
            for combo in itertools.combinations(AXIS_NAMES, size)
            if frozenset(combo) not in SUPPORTED_AXIS_SETS
        ),
        key=lambda s: (len(s), sorted(s)),
    ),
)
def test_every_unsupported_set_is_rejected(axes):
    """A missing mechanism entry may degrade the explanation; it must never let the shape run."""
    with pytest.raises(ValueError) as excinfo:
        _build(axes)
    message = str(excinfo.value)
    # It must be the matrix that refused, not a downstream divisibility rule.
    assert "is not a supported parallelism combination" in message, message


def test_rejection_quotes_the_flags_the_user_typed():
    """An error saying only ``expert_tp_size`` names a symbol the user cannot find in their config."""
    with pytest.raises(ValueError) as excinfo:
        _build(frozenset({"tp", "cp"}))
    message = str(excinfo.value)
    assert "tensor_parallel_size=2" in message, message
    assert "context_parallel_size=2" in message, message


def test_mechanisms_are_written_for_the_reachable_rejections():
    """A mechanism whose axis set becomes supported is dead prose; this catches it when support is added."""
    stale = [sorted(axes) for axes in AXIS_SET_MECHANISMS if axes in SUPPORTED_AXIS_SETS]
    assert not stale, f"mechanism text describes now-SUPPORTED combinations: {stale}"


def test_the_allowlist_is_not_vacuous():
    """An empty or all-inclusive matrix would make both halves pass."""
    all_sets = {
        frozenset(combo) for size in range(len(AXIS_NAMES) + 1) for combo in itertools.combinations(AXIS_NAMES, size)
    }
    assert frozenset() in SUPPORTED_AXIS_SETS, "plain data parallelism must be supported"
    assert frozenset({"pp", "ep"}) in SUPPORTED_AXIS_SETS, "PP+EP is an allowlisted combination"
    assert frozenset({"pp", "tp"}) not in SUPPORTED_AXIS_SETS, "PP+TP is deliberately not allowlisted"
    assert 0 < len(SUPPORTED_AXIS_SETS) < len(all_sets), (len(SUPPORTED_AXIS_SETS), len(all_sets))


def test_unsupported_combination_is_refused_before_any_rank_math():
    """A divisibility error such as "world_size 8 must be divisible by pp_size 3" invites a pp_size that hits the
    real wall. Each case also violates a rank-math rule."""
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
    with pytest.raises(ValueError, match="only meaningful with pipeline_parallel_size"):
        ParallelismConfig(world_size=8, gpus_per_node=8, **{knob: value})


def test_pp_knobs_at_their_defaults_are_accepted_at_pp_size_one():
    """Anti-vacuity: the rejection keys on a changed value."""
    config = ParallelismConfig(world_size=8, gpus_per_node=8)
    assert config.pp_size == 1 and config.pp_split is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
