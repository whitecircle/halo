#!/usr/bin/env python
"""The parallelism docs' limitation tables must agree with the code that enforces them.

``agent-docs/parallelism/*`` and ``agent-docs/reference/trainer-architecture.md`` publish three matrices a
reader configures a run from: which axis combinations may run, which trainer supports which axis,
and which MoE family restricts an EP capability. All three are *derived* in the code — from
:data:`SUPPORTED_AXIS_SETS`, from the per-class ``_supports_*`` attributes, and from the EP layer
classes' capability flags — so a published table is a second copy, free to drift from the gate it
describes. Drift here is expensive: a wrong row sends a reader into a config-time raise, or promises
a mode the trainer refuses, after the GPUs are allocated.

These tests parse the published tables and compare them against the same code the runtime reads.
Every failure names the offending row.

Run: ``pytest -m cpu tests/cpu/parallelism/test_docs_limitation_tables.py``
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.patching import MOE_LAYER_MAP
from src.distributed.parallelism_config import AXIS_FLAGS, SUPPORTED_AXIS_SETS
from src.trainers.mixins.base import DistributedTrainerMixin
from tests.common.rosters import import_all_trainers

import_all_trainers()

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PARALLELISM_INDEX = _REPO_ROOT / "agent-docs" / "parallelism" / "README.md"
_EXPERT_PARALLELISM = _REPO_ROOT / "agent-docs" / "parallelism" / "expert-parallelism.md"
_TRAINER_ARCHITECTURE = _REPO_ROOT / "agent-docs" / "reference" / "trainer-architecture.md"

# Doc column header -> the class attribute that decides it. ETP has no attribute of its own: it
# folds into ep_group_size, so _supports_ep gates it — the table must say the same in both columns.
_TRAINER_AXIS_ATTRS = {"EP": "_supports_ep", "CP": "_supports_cp", "TP": "_supports_tp", "PP": "_supports_pp"}

# EP capability flags that default True on EPMoELayerBase, so a False IS the documented restriction.
# DERIVED from the base class, never listed: a flag restated here is a second copy of the class's own
# declaration, and the one it forgets is unpinnable — the restriction table can then omit that
# family's row and still pass. Over the whole MRO, since the base is composed of mixins (the base layer
# and the balancing mixin) that each declare part of the capability surface.
_EP_RESTRICTABLE_FLAGS = tuple(
    sorted(
        {
            name
            for klass in EPMoELayerBase.__mro__
            for name, value in vars(klass).items()
            if name.startswith("_supports_") and value is True
        }
    )
)

# The catch-all row of the supported-combinations table; it names no axis set on purpose.
_CATCH_ALL_ROW = "anything else"

# The restriction table's preamble states how many flags a family may switch off. Spelled out, so the
# count is checked against the derivation rather than left to rot beside a table that grew.
_EP_FLAG_COUNT_SENTENCE = re.compile(r"declares (\w+) capability flags")
_NUMBER_WORDS = {word: n for n, word in enumerate(("four", "five", "six", "seven", "eight", "nine"), start=4)}


def _documented_count(word: str) -> int | None:
    """The count the doc states, spelled out or as a digit. ``None`` only when it is neither — which
    the caller reports as drift rather than silently reading as a mismatch."""
    return _NUMBER_WORDS.get(word.lower(), int(word) if word.isdigit() else None)


def _markdown_tables(path: Path) -> list[tuple[list[str], list[list[str]]]]:
    """Every GitHub-flavored table in ``path`` as ``(headers, rows)`` of raw cell text."""
    tables: list[tuple[list[str], list[list[str]]]] = []
    headers: list[str] | None = None
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        is_row = stripped.startswith("|") and stripped.endswith("|")
        if not is_row:
            if headers is not None:
                tables.append((headers, rows))
            headers, rows = None, []
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if headers is None:
            headers = cells
        elif set("".join(cells)) <= set("-: "):
            continue  # the ---|--- separator
        else:
            rows.append(cells)
    if headers is not None:
        tables.append((headers, rows))
    return tables


def _table_with_headers(path: Path, *required: str) -> list[dict[str, str]]:
    """The one table in ``path`` whose header row contains every name in ``required``, as dicts."""
    matches = [
        [dict(zip(headers, row, strict=False)) for row in rows]
        for headers, rows in _markdown_tables(path)
        if set(required) <= set(headers)
    ]
    assert len(matches) == 1, (
        f"expected exactly one table in {path.name} with headers {required}, found {len(matches)}"
    )
    return matches[0]


def _plain(cell: str) -> str:
    """Cell text without markdown emphasis, code ticks or links."""
    cell = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", cell)
    return re.sub(r"[*`]", "", cell).strip()


def _axis_set_from_mode(label: str) -> frozenset[str] | None:
    """The axis set a supported-combinations row names, or ``None`` for the catch-all row."""
    text = re.sub(r"\(.*", "", _plain(label)).strip()
    lowered = text.lower()
    if lowered.startswith(_CATCH_ALL_ROW):
        return None
    if lowered.startswith("no parallelism"):
        return frozenset()
    axes = set()
    for token in text.split("+"):
        axis = token.replace("only", "").replace("Only", "").strip().lower()
        assert axis in AXIS_FLAGS, f"supported-combinations row {label!r} names {axis!r}, which is not an axis"
        axes.add(axis)
    return frozenset(axes)


def _ep_model_rows() -> list[dict[str, str]]:
    """The EP supported-models table rows."""
    return _table_with_headers(_EXPERT_PARALLELISM, "Model", "HF Class", "EP Wrapper")


def _ep_wrapper_class_name(row: dict[str, str]) -> str:
    """The wrapper class a supported-models row names (first code span; ignores 'subclasses X')."""
    spans = re.findall(r"`([^`]+)`", row["EP Wrapper"])
    assert spans, f"EP supported-models row {row['Model']!r} names no wrapper class in backticks"
    return spans[0]


def _trainer_classes() -> dict[str, type]:
    """Every concrete distributed trainer, walked from the mixin's subclass tree (never listed).

    Filtered to classes defined under ``src.trainers``: ``__subclasses__()`` is a process-global
    registry, and pytest imports every test module before this runs — so a stub trainer any other
    test file defines at module scope would otherwise be demanded of the docs table.
    """
    found: dict[str, type] = {}
    stack = list(DistributedTrainerMixin.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls.__name__ in found:
            continue
        stack.extend(cls.__subclasses__())
        if not cls.__module__.startswith("src.trainers"):
            continue
        found[cls.__name__] = cls
    return found


def test_supported_combinations_table_matches_the_allowlist():
    """``agent-docs/parallelism/README.md`` must publish exactly the sets ``ParallelismConfig`` accepts.

    A row the allowlist rejects sends a reader into a config-time raise; a missing row hides a shape
    that works. Both are the failure this table exists to prevent.
    """
    rows = _table_with_headers(_PARALLELISM_INDEX, "Mode", "Data Parallel Size", "Notes")
    documented = {}
    catch_all = 0
    for row in rows:
        axes = _axis_set_from_mode(row["Mode"])
        if axes is None:
            catch_all += 1
            continue
        assert axes not in documented, f"supported-combinations row {row['Mode']!r} is a duplicate"
        documented[axes] = row["Mode"]

    assert catch_all == 1, "the supported-combinations table must keep exactly one 'Anything else' catch-all row"

    stale = sorted(documented[a] for a in documented.keys() - SUPPORTED_AXIS_SETS)
    missing = sorted("+".join(sorted(a)) or "no parallelism" for a in SUPPORTED_AXIS_SETS - documented.keys())
    assert not stale, f"documented as supported but rejected by SUPPORTED_AXIS_SETS: {stale}"
    assert not missing, f"in SUPPORTED_AXIS_SETS but absent from agent-docs/parallelism/README.md: {missing}"


def test_trainer_compatibility_table_matches_the_support_flags():
    """Every cell of the trainer × axis matrix must equal the class attribute the mixin reads.

    ``ParallelismValidationMixin`` raises off ``_supports_ep`` / ``_supports_cp`` / ``_supports_tp``
    / ``_supports_pp``; a doc cell that disagrees promises (or denies) a mode the trainer will
    refuse (or accept) at construction.
    """
    rows = _table_with_headers(_TRAINER_ARCHITECTURE, "Trainer", "EP", "CP", "TP", "ETP", "PP")
    classes = _trainer_classes()

    documented = {_plain(row["Trainer"]) for row in rows}
    assert documented == set(classes), (
        f"trainer-compatibility rows missing from docs: {sorted(set(classes) - documented)}; "
        f"documented but not a DistributedTrainerMixin subclass: {sorted(documented - set(classes))}"
    )

    for row in rows:
        name = _plain(row["Trainer"])
        cls = classes[name]
        for column, attr in _TRAINER_AXIS_ATTRS.items():
            documented_support = _plain(row[column]).lower().startswith("yes")
            assert documented_support == getattr(cls, attr), (
                f"{name} row, {column} column: docs say {row[column]!r} but {attr}={getattr(cls, attr)!r}"
            )
        assert _plain(row["ETP"]).lower().startswith("yes") == cls._supports_ep, (
            f"{name} row, ETP column: ETP is gated by _supports_ep (there is no _supports_etp), so it "
            f"must match the EP column; docs say {row['ETP']!r} with _supports_ep={cls._supports_ep!r}"
        )


def test_ep_supported_models_table_lists_every_registered_wrapper():
    """The EP supported-models table must name exactly the wrapper classes ``MOE_LAYER_MAP`` holds.

    ``MOE_LAYER_MAP`` is derived from the ``EPMoELayerBase`` subclass tree, so a family added to
    ``layers/`` is live the moment it is imported — with no doc row telling anyone it exists.
    """
    documented = {_ep_wrapper_class_name(row) for row in _ep_model_rows()}
    registered = {cls.__name__ for cls in MOE_LAYER_MAP.values()}
    assert documented == registered, (
        f"EP wrappers registered but undocumented: {sorted(registered - documented)}; "
        f"documented but not registered: {sorted(documented - registered)}"
    )


def test_ep_restriction_table_matches_the_layer_capability_flags():
    """The per-family EP restriction table must be exactly the set of ``False`` capability flags.

    Each flag defaults ``True`` on ``EPMoELayerBase``, so a family turning one off is the whole
    limitation. A stale row promises a restriction that no longer exists; a missing row hides one
    that does (Zaya under gradient checkpointing, DeepSeek-V4 under vLLM weight sync).
    """
    by_wrapper = {_ep_wrapper_class_name(row): _plain(row["Model"]) for row in _ep_model_rows()}
    documented = set()
    for row in _table_with_headers(_EXPERT_PARALLELISM, "Family", "Restricted capability", "Class attribute"):
        family, attr = _plain(row["Family"]), _plain(row["Class attribute"])
        assert family in by_wrapper.values(), (
            f"EP restriction row {family!r} names no family in the supported-models table above it"
        )
        assert attr in _EP_RESTRICTABLE_FLAGS, f"EP restriction row {family!r} cites unknown attribute {attr!r}"
        documented.add((family, attr))

    enforced = {
        (by_wrapper[cls.__name__], attr)
        for cls in MOE_LAYER_MAP.values()
        for attr in _EP_RESTRICTABLE_FLAGS
        if not getattr(cls, attr)
    }
    assert documented == enforced, (
        f"restrictions enforced in code but undocumented: {sorted(enforced - documented)}; "
        f"documented but not enforced: {sorted(documented - enforced)}"
    )


def test_the_restriction_preamble_counts_the_flags_it_derives_from():
    """The prose above the table must state the number of flags the base actually declares.

    A capability flag added to ``EPMoELayerBase`` that no family switches off yet leaves the table
    itself correct and the sentence introducing it wrong, so nothing else here would notice.
    """
    match = _EP_FLAG_COUNT_SENTENCE.search(_EXPERT_PARALLELISM.read_text(encoding="utf-8"))
    assert match, (
        f"{_EXPERT_PARALLELISM.name} no longer states how many capability flags EPMoELayerBase "
        f"declares; the restriction table's preamble is the reader's map into them"
    )
    documented = _documented_count(match.group(1))
    assert documented is not None, (
        f"{_EXPERT_PARALLELISM.name} states the flag count as {match.group(1)!r}, which is neither a "
        f"digit nor a number word this pin can read — spell it out or use a digit"
    )
    assert documented == len(_EP_RESTRICTABLE_FLAGS), (
        f"{_EXPERT_PARALLELISM.name} says EPMoELayerBase declares {match.group(1)!r} capability "
        f"flags; it declares {len(_EP_RESTRICTABLE_FLAGS)}: {list(_EP_RESTRICTABLE_FLAGS)}"
    )


def test_every_capability_a_family_switches_off_is_a_restrictable_flag():
    """Guards the test above: a restriction the derived flag set cannot express is undocumentable.

    The derivation only sees flags that default ``True`` on ``EPMoELayerBase``. A family that turns
    off anything else — a ``_supports_*`` the base never declared, or one whose base default was
    flipped to ``False`` — states a restriction the table has no row shape for, and the comparison
    above would pass while the docs stayed silent about it.
    """
    assert len(_EP_RESTRICTABLE_FLAGS) >= 5, (
        f"only {len(_EP_RESTRICTABLE_FLAGS)} restrictable flags derived from EPMoELayerBase — the "
        f"attribute sweep lost its target: {_EP_RESTRICTABLE_FLAGS}"
    )
    # Own declarations below the base only: the base's own ``False`` defaults are opt-IN capabilities
    # (``_supports_bias_balancing``), not restrictions.
    switched_off = {
        (cls.__name__, name)
        for cls in MOE_LAYER_MAP.values()
        for klass in cls.__mro__[: cls.__mro__.index(EPMoELayerBase)]
        for name, value in vars(klass).items()
        if name.startswith("_supports_") and value is False
    }
    undocumentable = sorted(pair for pair in switched_off if pair[1] not in _EP_RESTRICTABLE_FLAGS)
    assert not undocumentable, (
        f"these EP layer classes switch off a capability the restriction table cannot express (the "
        f"flag does not default True on EPMoELayerBase): {undocumentable}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
