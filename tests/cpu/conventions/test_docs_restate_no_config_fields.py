"""A configuration field's type and default are tabled on one page: the configuration reference.

A feature page explains behavior and links the field's row there. A second table of the same field
and its default drifts from the first the next time either changes, and nothing catches it.
"""

import dataclasses
import importlib
import pkgutil
import re

import pytest
from trl import DPOConfig, GRPOConfig, KTOConfig, RewardConfig, SFTConfig

import halo.configs
from halo.distributed.parallelism_config import ParallelismConfig
from halo.kernels.lowp.spec import LowpSpec
from scripts.docs.markdown import parse_tables
from tests.common.repo import REPO_ROOT

REFERENCE = "agent-docs/reference/configuration-reference.md"


def _config_field_names() -> frozenset[str]:
    """Every field a training script parses: those of the dataclasses ``halo.configs`` defines (their
    transformers, TRL and sentence-transformers bases included), the TRL trainer configs the scripts take
    as-is, and ``ParallelismConfig``'s."""
    classes = {
        obj
        for module in pkgutil.iter_modules(halo.configs.__path__)
        for obj in vars(importlib.import_module(f"halo.configs.{module.name}")).values()
        if isinstance(obj, type) and dataclasses.is_dataclass(obj) and obj.__module__.startswith("halo.configs.")
    }
    classes |= {ParallelismConfig, LowpSpec, SFTConfig, DPOConfig, KTOConfig, GRPOConfig, RewardConfig}
    return frozenset(field.name for cls in classes for field in dataclasses.fields(cls))


_FIELDS = _config_field_names()


def _field_tables(text: str) -> list[int]:
    """The 1-based header lines of the tables in ``text`` that re-table configuration fields: a ``Default``
    or ``Type`` column, and a first column whose backticked names are mostly configuration fields."""
    found = []
    for table in parse_tables(text):
        names = [match.group(1) for row in table.rows if (match := re.match(r"`([A-Za-z_]\w*)`", row[0]))]
        restates = 2 * sum(name in _FIELDS for name in names) > len(names)
        if restates and {header.lower() for header in table.headers} & {"default", "type"}:
            found.append(table.line)
    return found


def test_only_the_reference_tables_configuration_fields():
    pages = (path for tree in ("agent-docs", "human-docs", "skills") for path in (REPO_ROOT / tree).rglob("*.md"))
    offenders = {
        page: tables
        for path in pages
        if (page := str(path.relative_to(REPO_ROOT))) != REFERENCE
        if (tables := _field_tables(path.read_text(encoding="utf-8")))
    }
    assert not offenders, (
        f"configuration fields tabled with a type or default outside {REFERENCE} (page: table lines): "
        f"{offenders}. Explain the behavior and link the field's row in the reference instead."
    )


def test_the_detector_flags_a_restated_field_table_and_nothing_else():
    restated = (
        "| Knob | Default | Effect |\n|---|---|---|\n| `max_length` | `1024` | budget |\n| `beta` | `0.1` | x |\n"
    )
    effect_only = "| Knob | Effect |\n|---|---|\n| `max_length` | budget |\n"
    other_keys = "| Option | Default |\n|---|---|\n| `max_length` | 1 |\n| `rubric` | 2 |\n| `endpoint` | 3 |\n"
    assert _field_tables("Intro.\n\n" + restated) == [3]
    assert _field_tables(effect_only) == []
    assert _field_tables(other_keys) == []
    assert _field_tables((REPO_ROOT / REFERENCE).read_text(encoding="utf-8")), (
        f"the detector finds no field table in {REFERENCE} itself, so it would find none anywhere"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
