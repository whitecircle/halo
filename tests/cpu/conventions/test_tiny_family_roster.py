#!/usr/bin/env python
"""Every EP MoE family has one tiny model, and the per-family resume sweeps run each of them.

The merge-on-save and precompute-resume GPU suites take their ``--family`` names from
``tests/common/tiny_models.py``'s ``TINY_MOE_FAMILIES`` and their rows from ``tests/gpu/manifest.py``.
A family registered in ``src/distributed/expert_parallel/layers`` with no tiny model, or one the rows
skip for an adapter shape or a layout, would leave its resume path untested with nothing failing;
the roster is therefore held to the EP registry, and the rows to the roster.

    python tests/cpu/conventions/test_tiny_family_roster.py
"""

import shlex

import pytest

import src.distributed.expert_parallel.layers.roster  # noqa: F401  (registers every EP family)
from src.distributed.expert_parallel.expert_weights import ep_layer_class_by_model_type
from tests.common.merged_resume_e2e import ADAPTER_MODES, LAYOUTS, merged_resume_parser
from tests.common.preference_precompute_e2e import DENSE, precompute_resume_parser
from tests.common.tiny_models import TINY_MOE_FAMILIES
from tests.gpu.manifest import MANIFEST

MERGED_RESUME_SUITES = (
    "trainers/lora/test_lora_merged_save_resume.py",
    "trainers/lora/test_lora_merged_save_resume_families.py",
)
PRECOMPUTE_RESUME_SUITES = (
    "parallelism/ep/test_ep_preference_precompute_resume.py",
    "trainers/preference/test_preference_precompute_resume_families.py",
)
# What every MoE family runs through the precompute-resume body: DPO on both expert layouts, KTO once.
PRECOMPUTE_PER_FAMILY = (("dpo", "ep2"), ("dpo", "ep1"), ("kto", "ep2"))


def _rows(suites, parser):
    return [parser.parse_args(shlex.split(args)) for suite in suites for args in MANIFEST[suite].args_matrix]


def test_every_ep_family_has_one_tiny_model():
    registry = ep_layer_class_by_model_type()
    unregistered = sorted(name for name in TINY_MOE_FAMILIES if name not in registry)
    assert not unregistered, f"tiny families no EP layer class claims: {unregistered}"
    covered = [registry[name] for name in TINY_MOE_FAMILIES]
    missing = sorted(cls.__name__ for cls in set(registry.values()) - set(covered))
    assert not missing, f"EP families with no tiny model in TINY_MOE_FAMILIES: {missing}"
    assert len(set(covered)) == len(covered), "two tiny models stand for one EP family"


def test_the_merged_resume_rows_cover_every_family_adapter_shape_and_layout():
    rows = _rows(MERGED_RESUME_SUITES, merged_resume_parser(TINY_MOE_FAMILIES))
    covered = [(row.family, row.adapters, row.ep_size, row.cp_size) for row in rows]
    expected = {
        (family, adapters, ep, cp) for family in TINY_MOE_FAMILIES for adapters in ADAPTER_MODES for ep, cp in LAYOUTS
    }
    assert not sorted(expected - set(covered)), f"merged-resume shapes no row runs: {sorted(expected - set(covered))}"
    assert len(set(covered)) == len(covered), "a merged-resume shape runs in two rows"


def test_the_precompute_resume_rows_cover_every_family():
    rows = _rows(PRECOMPUTE_RESUME_SUITES, precompute_resume_parser((*TINY_MOE_FAMILIES, DENSE)))
    covered = {(row.trainer, row.family, row.mode) for row in rows if not row.peft}
    expected = {(trainer, family, mode) for family in TINY_MOE_FAMILIES for trainer, mode in PRECOMPUTE_PER_FAMILY}
    assert not sorted(expected - covered), f"precompute-resume shapes no row runs: {sorted(expected - covered)}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
