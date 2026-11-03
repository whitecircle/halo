#!/usr/bin/env python
"""Every pipeline family registered as validated has a GPU row scoring its stages against the unsplit model, and
every validated MoE family one with its experts distributed."""

import functools
import shlex

import pytest

from halo.distributed.pipeline_parallel.families import PP_FAMILY_SPECS
from halo.models.module_tree import backbone_with_layers
from tests.common.tiny_models import TINY_DENSE_FAMILIES, TINY_MOE_FAMILIES, tiny_family_model
from tests.gpu.manifest import MANIFEST

# The pipeline suites that score a family's stages against its unsplit model: the family a row runs when it names
# none (``--family``), and its expert-parallel size when it names none (``--ep-size``).
PP_EQUIVALENCE_SUITES = {
    "parallelism/pp/test_pp_vs_single_gpu.py": ("qwen3", 1),
    "parallelism/pp/test_pp_ragged_mask_vs_single_gpu.py": ("qwen3", 1),
    "parallelism/pp/test_pp_hc_vs_single_gpu.py": (None, 1),
    "parallelism/pp/test_pp_ep1_gmm_vs_unsplit.py": ("gpt_oss", 1),
    "parallelism/pp/test_pp_ep_vs_unsplit.py": ("gpt_oss", 2),
    "parallelism/pp/test_pp_family_vs_reference.py": (None, 2),
    "parallelism/pp/test_pp_qwen4_exp_vs_reference.py": ("qwen4_exp_text", 2),
}
ROSTER = {**TINY_DENSE_FAMILIES, **TINY_MOE_FAMILIES}
VALIDATED = frozenset(name for spec in PP_FAMILY_SPECS if spec.validated for name in spec.hf_module_names)


def _flag(argv: list[str], flag: str, default):
    return argv[argv.index(flag) + 1] if flag in argv else default


@functools.cache
def _backbone(family: str) -> str:
    """The class of the backbone ``family``'s tiny model splits at, the name its ``PPFamilySpec`` claims."""
    return type(backbone_with_layers(tiny_family_model(ROSTER[family]))).__name__


def _rows() -> list[tuple[str, int]]:
    rows = []
    for suite, (family, ep_size) in PP_EQUIVALENCE_SUITES.items():
        for args, _spec in MANIFEST[suite].launches():
            argv = shlex.split(args)
            rows.append((_flag(argv, "--family", family), int(_flag(argv, "--ep-size", ep_size))))
    return rows


def test_every_validated_pp_family_has_an_equivalence_row():
    covered = {_backbone(family) for family, _ in _rows()}
    assert not sorted(VALIDATED - covered), (
        f"validated PP families no equivalence row runs: {sorted(VALIDATED - covered)}"
    )


def test_every_validated_pp_moe_family_has_a_row_with_its_experts_distributed():
    moe = {_backbone(family) for family in TINY_MOE_FAMILIES} & VALIDATED
    distributed = {_backbone(family) for family, ep_size in _rows() if ep_size > 1}
    assert moe, "anti-vacuity: the validated roster holds MoE families"
    assert not sorted(moe - distributed), f"validated PP MoE families no PP+EP row runs: {sorted(moe - distributed)}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
