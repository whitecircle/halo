#!/usr/bin/env python
"""The published support tables must agree with the code that enforces them.

Each pinned table — the axis allowlist, trainer × axis, family × axis (agent and human) and the per-family EP
restrictions — is re-derived here from the registries and class declarations the runtime reads.
"""

from __future__ import annotations

import dataclasses
import importlib
import operator
import re
from pathlib import Path

import pytest
from transformers.models.auto.configuration_auto import CONFIG_MAPPING, model_type_to_module_name

from halo.distributed.context_parallel.families import ULYSSES_ATTENTION_MAP
from halo.distributed.context_parallel.model_rules import _UNSUPPORTED_SEQUENCE_AXIS_LAYERS
from halo.distributed.expert_parallel.families.roster import MOE_LAYER_MAP
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.distributed.parallelism_rules import AXES, SUPPORTED_AXIS_SETS
from halo.distributed.pipeline_parallel.families import PP_SPEC_MAP
from halo.distributed.tensor_parallel.parallelize_attention import TP_SHARDABLE_ATTENTION_CLASSES
from halo.models.moe.family_specs import MOE_FAMILY_SPECS, MoEFamilySpec, find_moe_family_spec
from scripts.docs.markdown import parse_tables
from tests.common.repo import REPO_ROOT
from tests.common.rosters import distributed_trainer_classes

_PARALLELISM_INDEX = REPO_ROOT / "agent-docs" / "parallelism" / "README.md"
_EXPERT_PARALLELISM = REPO_ROOT / "agent-docs" / "parallelism" / "expert-parallelism.md"
_TRAINER_ARCHITECTURE = REPO_ROOT / "agent-docs" / "reference" / "trainer-architecture.md"
_MODELS_INDEX = REPO_ROOT / "agent-docs" / "models" / "README.md"
_HUMAN_MATRIX = REPO_ROOT / "human-docs" / "supported-matrix.md"

# Doc column header -> whether the class declaration admits that axis. EP, ETP and TP run under every
# trainer, so the matrix carries the two axes a trainer declares.
_TRAINER_AXIS_SUPPORT = {"CP": lambda cls: cls.SUPPORTS_CP, "PP": lambda cls: cls.PP_REFUSAL is None}

# EP capabilities a family may switch off: its spec's ``supports_*`` fields (default True) and
# ``*_refusal`` fields (default None), and the EP layer's own ``_supports_*`` flags that default True
# over the whole MRO (the base is composed of mixins that each declare part of the surface). DERIVED,
# never listed: a capability restated here is a second copy of the declaration, and the one it forgets
# is unpinnable — the restriction table can then omit that family's row and still pass.
_EP_RESTRICTABLE_FLAGS = tuple(
    sorted(
        {
            *(
                f"FAMILY.{spec_field.name}"
                for spec_field in dataclasses.fields(MoEFamilySpec)
                if spec_field.name.startswith("supports_") or spec_field.name.endswith("_refusal")
            ),
            *(
                name
                for klass in EPMoELayerBase.__mro__
                for name, value in vars(klass).items()
                if name.startswith("_supports_") and value is True
            ),
        }
    )
)

# The catch-all row of the supported-combinations table; it names no axis set on purpose.
_CATCH_ALL_ROW = "anything else"

# The restriction table's preamble states how many flags a family may switch off. Spelled out, so the
# count is checked against the derivation rather than left to rot beside a table that grew.
_EP_FLAG_COUNT_SENTENCE = re.compile(r"declares (\w+) capabilities")
_NUMBER_WORDS = {word: n for n, word in enumerate(("four", "five", "six", "seven", "eight", "nine"), start=4)}

_FAMILY_AXES = ("EP", "ETP", "CP", "TP", "PP")

# The one remote-code family: hub modeling code, so no transformers module to read its classes off.
_REMOTE_CODE_CLASSES = {"bailing_moe": frozenset({"BailingMoeV2Attention", "BailingMoeV2Model"})}


def _table_with_headers(path: Path, *required: str) -> list[dict[str, str]]:
    """The one table in ``path`` whose header row contains every name in ``required``, as dicts."""
    matches = [
        [dict(zip(table.headers, row, strict=False)) for row in table.rows]
        for table in parse_tables(path.read_text(encoding="utf-8"))
        if set(required) <= set(table.headers)
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
        assert axis in {a.name for a in AXES}, (
            f"supported-combinations row {label!r} names {axis!r}, which is not an axis"
        )
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

    ``DistributedTrainerMixin`` raises off ``SUPPORTS_CP`` / ``PP_REFUSAL``; a doc cell that
    disagrees promises (or denies) a mode the trainer will refuse (or accept) at construction.
    """
    rows = _table_with_headers(_TRAINER_ARCHITECTURE, "Trainer", *_TRAINER_AXIS_SUPPORT)
    classes = {cls.__name__: cls for cls in distributed_trainer_classes()}

    documented = {_plain(row["Trainer"]) for row in rows}
    assert documented == set(classes), (
        f"trainer-compatibility rows missing from docs: {sorted(set(classes) - documented)}; "
        f"documented but not a DistributedTrainerMixin subclass: {sorted(documented - set(classes))}"
    )

    for row in rows:
        name = _plain(row["Trainer"])
        cls = classes[name]
        for column, supports in _TRAINER_AXIS_SUPPORT.items():
            documented_support = _plain(row[column]).lower().startswith("yes")
            assert documented_support == supports(cls), (
                f"{name} row, {column} column: docs say {row[column]!r} but the class "
                f"{'admits' if supports(cls) else 'refuses'} {column}"
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

    Each capability is on by default, so a family turning one off is the whole limitation. A stale row
    promises a restriction that no longer exists; a missing row hides one that does (Zaya under gradient checkpointing, Inkling under weight sync).
    """
    by_wrapper = {_ep_wrapper_class_name(row): _plain(row["Model"]) for row in _ep_model_rows()}
    documented = set()
    for row in _table_with_headers(_EXPERT_PARALLELISM, "Family", "Restricted capability", "Declaration"):
        family, attr = _plain(row["Family"]), _plain(row["Declaration"])
        assert family in by_wrapper.values(), (
            f"EP restriction row {family!r} names no family in the supported-models table above it"
        )
        assert attr in _EP_RESTRICTABLE_FLAGS, f"EP restriction row {family!r} cites unknown attribute {attr!r}"
        documented.add((family, attr))

    enforced = set()
    for cls in MOE_LAYER_MAP.values():
        for attr in _EP_RESTRICTABLE_FLAGS:
            value = operator.attrgetter(attr)(cls)
            # A ``*_refusal`` restricts when set; a ``supports_*`` / ``_supports_*`` flag when off.
            if value is not None if attr.endswith("_refusal") else not value:
                enforced.add((by_wrapper[cls.__name__], attr))
    assert documented == enforced, (
        f"restrictions enforced in code but undocumented: {sorted(enforced - documented)}; "
        f"documented but not enforced: {sorted(documented - enforced)}"
    )


def test_the_restriction_preamble_counts_the_flags_it_derives_from():
    """The prose above the table must state the number of capabilities a family can switch off.

    A capability added that no family switches off yet leaves the table itself correct and the
    sentence introducing it wrong, so nothing else here would notice.
    """
    match = _EP_FLAG_COUNT_SENTENCE.search(_EXPERT_PARALLELISM.read_text(encoding="utf-8"))
    assert match, (
        f"{_EXPERT_PARALLELISM.name} no longer states how many capabilities a family declares; the "
        f"restriction table's preamble is the reader's map into them"
    )
    word = match.group(1)
    documented = _NUMBER_WORDS.get(word.lower(), int(word) if word.isdigit() else None)
    assert documented is not None, (
        f"{_EXPERT_PARALLELISM.name} states the flag count as {match.group(1)!r}, which is neither a "
        f"digit nor a number word this pin can read — spell it out or use a digit"
    )
    assert documented == len(_EP_RESTRICTABLE_FLAGS), (
        f"{_EXPERT_PARALLELISM.name} says a family declares {match.group(1)!r} capabilities; it "
        f"declares {len(_EP_RESTRICTABLE_FLAGS)}: {list(_EP_RESTRICTABLE_FLAGS)}"
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
    # (``SUPPORTS_BIAS_BALANCING``), not restrictions.
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


def _family_rows(path: Path, *required: str) -> dict[str, dict[str, str]]:
    """The family table in ``path`` as ``{family label: {column: first word}}``, the catch-all row left out.

    A cell's first word drops markup and footnote markers: ``**No** ¹`` reads ``No``.
    """
    rows: dict[str, dict[str, str]] = {}
    for row in _table_with_headers(path, *required):
        label = _plain(row[required[0]])
        if label.lower().startswith("any other"):  # the catch-all row names no family on purpose
            continue
        assert label not in rows, f"{path.name}: family row {label!r} is a duplicate"
        rows[label] = {column: (_plain(cell).split() or [""])[0] for column, cell in row.items()}
    return rows


def _derived_family_verdicts(model_type: str) -> dict[str, str]:
    """The EP/ETP/CP/TP/PP verdicts the registries give the family ``model_type`` names, read off the classes its
    modeling module defines."""
    if model_type in _REMOTE_CODE_CLASSES:
        classes = _REMOTE_CODE_CLASSES[model_type]
    else:
        module = model_type_to_module_name(model_type)
        classes = frozenset(vars(importlib.import_module(f"transformers.models.{module}.modeling_{module}")))
    is_moe = find_moe_family_spec(model_type) is not None
    # CP needs a wrapped attention class and no layer carrying state along the sequence axis.
    cp = bool(classes & ULYSSES_ATTENTION_MAP.keys()) and not classes & _UNSUPPORTED_SEQUENCE_AXIS_LAYERS.keys()
    # A MoE takes the selective attention-only path; a dense model shards off its config's own HF plan.
    tp = (
        bool(classes & TP_SHARDABLE_ATTENTION_CLASSES)
        if is_moe
        else bool(CONFIG_MAPPING[model_type].base_model_tp_plan)
    )
    pp = any(PP_SPEC_MAP[name].validated for name in classes & PP_SPEC_MAP.keys())
    verdicts = {"EP": "Yes" if is_moe else "—", "ETP": "Yes" if is_moe else "—"}
    return verdicts | {axis: "Yes" if ok else "No" for axis, ok in (("CP", cp), ("TP", tp), ("PP", pp))}


@pytest.mark.parametrize("path, label_column", [(_MODELS_INDEX, "Model Family"), (_HUMAN_MATRIX, "Model family")])
def test_family_table_matches_the_registries(path: Path, label_column: str):
    """The agent compatibility matrix and its human copy must list a row per MoE family spec, each carrying the
    verdicts the registries give the ``model_type`` the agent matrix names for it.

    A cell the code contradicts sends a reader into a load-time raise, or away from a mode that works.
    """
    agent_rows = _family_rows(_MODELS_INDEX, "Model Family", "`model_type`", *_FAMILY_AXES)
    model_types = {label: row["`model_type`"] for label, row in agent_rows.items()}
    covered = {find_moe_family_spec(model_type) for model_type in model_types.values()}
    missing = sorted(spec.model_types[0] for spec in MOE_FAMILY_SPECS if spec not in covered)
    assert not missing, f"MoE families with no row in {_MODELS_INDEX.name}'s compatibility matrix: {missing}"
    rows = _family_rows(path, label_column, *_FAMILY_AXES)
    assert rows.keys() == model_types.keys(), (
        f"{path.name} family rows missing: {sorted(model_types.keys() - rows.keys())}; "
        f"absent from the agent matrix: {sorted(rows.keys() - model_types.keys())}"
    )
    mismatches = [
        f"{label} / {axis}: docs say {rows[label][axis]!r}, the code gives {verdict!r}"
        for label, model_type in model_types.items()
        for axis, verdict in _derived_family_verdicts(model_type).items()
        if rows[label][axis] != verdict
    ]
    assert not mismatches, f"{path.name} family table disagrees with the code: {mismatches}"


def test_human_trainer_table_matches_the_support_flags():
    """Each training-method row of ``human-docs/supported-matrix.md`` must state the CP / PP declarations of the
    trainers its scripts construct, and every distributed trainer must sit in some row."""
    classes = {cls.__name__: cls for cls in distributed_trainer_classes()}
    trainer_name = re.compile(r"\b(" + "|".join(classes) + r")\b")
    seen: set[str] = set()
    for row in _table_with_headers(_HUMAN_MATRIX, "Method", "Script", *_TRAINER_AXIS_SUPPORT):
        method, script = _plain(row["Method"]), _plain(row["Script"])
        # ``{dpo,kto}.py`` names one script per alternative; a trailing ``/`` names the directory's scripts.
        braces = re.fullmatch(r"(.*)\{(.*)\}(.*)", script)
        paths = [braces[1] + alt + braces[3] for alt in braces[2].split(",")] if braces else [script]
        files = [f for path in paths for f in (sorted((REPO_ROOT / path).glob("*.py")) or [REPO_ROOT / path])]
        trainers = {name for f in files for name in trainer_name.findall(f.read_text(encoding="utf-8"))}
        assert trainers, f"{_HUMAN_MATRIX.name} row {method!r} names no script that constructs a distributed trainer"
        seen |= trainers
        for column, supports in _TRAINER_AXIS_SUPPORT.items():
            declared = {supports(classes[name]) for name in trainers}
            assert declared == {_plain(row[column]).lower().startswith("yes")}, (
                f"{_HUMAN_MATRIX.name} row {method!r}, {column} column: docs say {row[column]!r}, but its "
                f"trainers {sorted(trainers)} {'admit' if declared == {True} else 'do not all admit'} {column}"
            )
    assert seen == set(classes), f"distributed trainers in no {_HUMAN_MATRIX.name} row: {sorted(set(classes) - seen)}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
