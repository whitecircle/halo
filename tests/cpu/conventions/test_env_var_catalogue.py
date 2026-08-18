"""The environment-variable catalogue and the code agree in both directions.

``agent-docs/reference/configuration-reference.md`` is the only place a user learns a toolkit env var
exists, so the two drift failures are symmetric and both silent: a knob added to ``src/`` that no
row announces is unreachable in practice, and a row for a variable nothing reads is a promise the
run does not keep. This sweep fails on either.

Run: pytest tests/cpu/conventions/test_env_var_catalogue.py
"""

import ast
import re

import pytest

from tests.common.utils import REPO_ROOT

CATALOGUE = REPO_ROOT / "agent-docs/reference/configuration-reference.md"

# Prefixes the toolkit owns. Launcher (``LOCAL_RANK``/``SLURM_*``), ``ACCELERATE_*`` and HF/OS
# variables are read raw at their owner by design (CLAUDE.md), so they owe the catalogue nothing —
# only a variable under one of these prefixes must carry a row.
TOOLKIT_PREFIXES = ("HALO_", "DIST_", "VLLM_", "SGLANG_", "NVLINK_", "EP_")

# Rows the catalogue documents for the user that this toolkit deliberately never reads, each with
# the reader that consumes it instead. Adding an entry must be a decision, not an omission.
THIRD_PARTY_ROWS = {
    "WANDB_RESUME": "read by the wandb SDK; the row exists to pair it with WANDB_RUN_ID",
    "HALO_SCRATCH": "consumed by the Makefile and compose files on the host; derives the in-container HALO_DATA_ROOT",
    "VLLM_USE_V2_MODEL_RUNNER": "read by the vLLM server; the row exists because rollout_max_thinking_tokens requires it",
}

# An env var name: SCREAMING_SNAKE with at least two segments, which is what every read spells.
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")


def _string_constants_in_value_position(tree: ast.AST):
    """String constants in a position that could make them an env key.

    Deliberately narrower than "every string literal in the file" — a docstring, an f-string's prose
    and a diagnostic dict's KEY spell the name without reading it, and counting those would let a log
    line stand in for a reader.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            slots = [node.value]
        elif isinstance(node, ast.Call):
            slots = [*node.args, *(kw.value for kw in node.keywords)]
        elif isinstance(node, ast.Compare):
            slots = [node.left, *node.comparators]
        elif isinstance(node, ast.Subscript):
            slots = [node.slice]
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            slots = list(node.elts)
        elif isinstance(node, ast.Dict):
            slots = [v for v in node.values if v is not None]  # values, not keys: a key is a report label
        else:
            continue
        for slot in slots:
            if isinstance(slot, ast.Constant) and isinstance(slot.value, str) and _ENV_NAME.match(slot.value):
                yield slot.value, node.lineno


def _env_names_read_by_the_toolkit() -> dict[str, str]:
    """Every candidate env-var name the shipped code names, mapped to its first ``file:line``."""
    found: dict[str, str] = {}
    for package in ("src", "scripts"):
        for path in sorted((REPO_ROOT / package).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for name, lineno in _string_constants_in_value_position(tree):
                found.setdefault(name, f"{path.relative_to(REPO_ROOT)}:{lineno}")
    return found


def _catalogued_variables() -> list[str]:
    """The Variable column of the ``## Environment variables`` table, one entry per code span."""
    section = CATALOGUE.read_text(encoding="utf-8").split("## Environment variables", 1)[1].split("\n---", 1)[0]
    names: list[str] = []
    for line in section.splitlines():
        if line.startswith("| `"):
            names += re.findall(r"`([A-Z][A-Z0-9_]+)`", line.split("|")[1])
    return names


_READ = _env_names_read_by_the_toolkit()
_CATALOGUED = _catalogued_variables()


def test_the_sweep_sees_both_sides():
    """Guards the two parsers: a rename that empties either side would otherwise pass everything."""
    assert len(_CATALOGUED) > 40, f"the catalogue table parsed to {len(_CATALOGUED)} rows"
    assert sum(name.startswith(TOOLKIT_PREFIXES) for name in _READ) > 40, "the source sweep found almost no env reads"


def test_catalogue_rows_are_unique():
    duplicates = sorted({name for name in _CATALOGUED if _CATALOGUED.count(name) > 1})
    assert not duplicates, f"documented twice in the environment-variable table: {duplicates}"


@pytest.mark.parametrize("variable", sorted(set(_CATALOGUED)), ids=lambda name: name)
def test_every_catalogued_variable_has_a_reader(variable):
    """A documented variable no module even names is a promise the run does not keep."""
    if variable in THIRD_PARTY_ROWS:
        pytest.skip(f"{variable}: {THIRD_PARTY_ROWS[variable]}")
    assert variable in _READ, (
        f"`{variable}` has a row in {CATALOGUE.name} but no module under src/ or scripts/ names it. "
        "Delete the row, or add it to THIRD_PARTY_ROWS with the reader that consumes it."
    )


@pytest.mark.parametrize(
    "variable", sorted(name for name in _READ if name.startswith(TOOLKIT_PREFIXES)), ids=lambda name: name
)
def test_every_toolkit_variable_the_code_reads_is_catalogued(variable):
    """A toolkit knob with no row is unreachable: the environment is the only way to set it."""
    assert variable in _CATALOGUED, (
        f"`{variable}` is read at {_READ[variable]} but has no row in the "
        f"'Environment variables' table of {CATALOGUE.name}."
    )


def test_third_party_rows_are_still_documented():
    """The exemption covers rows the catalogue keeps, not names it dropped."""
    for variable in THIRD_PARTY_ROWS:
        assert variable in _CATALOGUED, f"{variable} is exempted but no longer has a catalogue row"
        assert variable not in _READ, f"{variable} now has a reader — drop it from THIRD_PARTY_ROWS"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
