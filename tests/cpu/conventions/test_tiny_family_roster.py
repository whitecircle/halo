#!/usr/bin/env python
"""Every EP MoE family has one tiny model, and the per-family sweeps run each of them.

The merge-on-save, precompute-resume and LoRA sync-exactness GPU suites take their ``--family`` names
from ``tests/common/tiny_models.py``'s ``TINY_MOE_FAMILIES`` (the sync-exactness suites their dense
ones from ``TINY_DENSE_FAMILIES``) and their rows from ``tests/gpu/manifest.py``. A family registered
in ``src/distributed/expert_parallel/layers`` with no tiny model, or one the rows skip for an adapter
shape or a layout, would leave its path untested with nothing failing; the roster is therefore held to
the EP registry, and the rows to the roster (for the sync-exactness suites, the part of it some
rollout engine takes an online update for).

    python tests/cpu/conventions/test_tiny_family_roster.py
"""

import shlex

import pytest

import src.distributed.expert_parallel.layers.roster  # noqa: F401  (registers every EP family)
from src.distributed.expert_parallel.expert_weights import ep_layer_class_by_model_type
from tests.common.lora_sync_exactness import REPRESENTATIVE_FAMILIES, parse_row, row_families, syncable_moe_families
from tests.common.merged_resume_e2e import ADAPTER_MODES, LAYOUTS, merged_resume_parser
from tests.common.preference_precompute_e2e import DENSE, precompute_resume_parser
from tests.common.tiny_models import TINY_DENSE_FAMILIES, TINY_MOE_FAMILIES
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
SYNC_EXACTNESS_SUITE = "trainers/lora/test_lora_weight_sync_exact.py"
SYNC_EXACTNESS_SWEEP = "trainers/lora/test_lora_weight_sync_exact_families.py"
# What every syncable MoE family runs through the sync-exactness body, as (mode, adapters): both expert
# layouts with attention PEFT alone and mixed with expert LoRA, and pure ETP, which refuses expert LoRA.
# A dense family runs fsdp alone.
SYNC_EXACTNESS_PER_MOE_FAMILY = (
    ("ep1", "peft"),
    ("ep1", "mixed"),
    ("ep2", "peft"),
    ("ep2", "mixed"),
    ("etp2", "peft"),
)


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
    covered = [(row.family, row.adapters, row.ep_size, row.cp_size, row.fp32_masters) for row in rows]
    expected = {
        (family, adapters, ep, cp, False)
        for family in TINY_MOE_FAMILIES
        for adapters in ADAPTER_MODES
        for ep, cp in LAYOUTS
    }
    assert not sorted(expected - set(covered)), f"merged-resume shapes no row runs: {sorted(expected - set(covered))}"
    assert len(set(covered)) == len(covered), "a merged-resume shape runs in two rows"


def test_the_precompute_resume_rows_cover_every_family():
    rows = _rows(PRECOMPUTE_RESUME_SUITES, precompute_resume_parser((*TINY_MOE_FAMILIES, DENSE)))
    covered = {(row.trainer, row.family, row.mode) for row in rows if not row.peft}
    expected = {(trainer, family, mode) for family in TINY_MOE_FAMILIES for trainer, mode in PRECOMPUTE_PER_FAMILY}
    assert not sorted(expected - covered), f"precompute-resume shapes no row runs: {sorted(expected - covered)}"


def test_the_sync_exactness_rows_run_every_syncable_family_once_per_shape():
    roster = row_families()
    # Each suite with the families its script accepts.
    suites = {
        SYNC_EXACTNESS_SUITE: REPRESENTATIVE_FAMILIES,
        SYNC_EXACTNESS_SWEEP: tuple(family for family in roster if family not in REPRESENTATIVE_FAMILIES),
    }
    known = (*TINY_DENSE_FAMILIES, *TINY_MOE_FAMILIES)
    rows = [(suite, parse_row(known, shlex.split(args))) for suite in suites for args in MANIFEST[suite].args_matrix]
    outside = sorted({row.family for _, row in rows} - set(roster))
    assert not outside, f"sync-exactness rows naming a family outside the syncable roster: {outside}"
    misplaced = sorted({(suite, row.family) for suite, row in rows if row.family not in suites[suite]})
    assert not misplaced, f"sync-exactness rows naming a family their script refuses: {misplaced}"
    covered = [(row.family, row.mode, row.adapters) for _, row in rows]
    expected = {(family, "fsdp", "peft") for family in TINY_DENSE_FAMILIES} | {
        (family, mode, adapters)
        for family in syncable_moe_families()
        for mode, adapters in SYNC_EXACTNESS_PER_MOE_FAMILY
    }
    missing = sorted(expected - set(covered))
    assert not missing, f"syncable family x shape no sync-exactness row runs: {missing}"
    assert set(covered) <= expected, f"sync-exactness rows off the roster's shapes: {sorted(set(covered) - expected)}"
    assert len(set(covered)) == len(covered), "a sync-exactness shape runs in two rows"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
