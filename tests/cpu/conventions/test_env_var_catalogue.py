"""The environment-variable catalogue and the code agree in both directions: a knob with no row is unreachable, and
a row nothing reads promises what the run does not do."""

import ast
import re

import pytest

from tests.common.repo import REPO_ROOT

CATALOGUE = REPO_ROOT / "agent-docs/reference/configuration-reference.md"

# Launcher, ``ACCELERATE_*`` and HF/OS variables are read raw at their owner by design, so they owe the catalogue
# nothing.
TOOLKIT_PREFIXES = ("HALO_", "DIST_", "VLLM_", "SGLANG_", "NVLINK_", "EP_")

# Rows the catalogue documents that the toolkit never reads, each with the reader that consumes it.
THIRD_PARTY_ROWS = {
    "WANDB_RESUME": "read by the wandb SDK; the row exists to pair it with WANDB_RUN_ID",
    "HALO_SCRATCH": "consumed by the Makefile and compose files on the host; derives the in-container HALO_DATA_ROOT",
    "VLLM_USE_V2_MODEL_RUNNER": (
        "read by the vLLM server; the row exists because a 0 there overrides the image's pinned Model Runner V2, "
        "which thinking budgets and R3 capture run on"
    ),
    "NCCL_CUMEM_ENABLE": "read by NCCL in the SGLang server; the row exists because docker-compose.sglang.yml must pre-set it",
    "PYTORCH_CUDA_ALLOC_CONF": (
        "read by PyTorch's CUDA caching allocator; the row exists because the dense examples launch with "
        "expandable segments and EP runs leave them off"
    ),
}

_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")


def _string_constants_in_value_position(tree: ast.AST):
    """String constants that could be an env key. A docstring, an f-string's prose or a diagnostic dict's key spells
    the name without reading it."""
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
    for package in ("halo", "scripts"):
        assert (REPO_ROOT / package).is_dir(), f"the sweep lost its {package}/ root"
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
    assert len(_CATALOGUED) > 40, f"the catalogue table parsed to {len(_CATALOGUED)} rows"
    assert sum(name.startswith(TOOLKIT_PREFIXES) for name in _READ) > 40, "the source sweep found almost no env reads"


def test_catalogue_rows_are_unique():
    duplicates = sorted({name for name in _CATALOGUED if _CATALOGUED.count(name) > 1})
    assert not duplicates, f"documented twice in the environment-variable table: {duplicates}"


@pytest.mark.parametrize("variable", sorted(set(_CATALOGUED)), ids=lambda name: name)
def test_every_catalogued_variable_has_a_reader(variable):
    if variable in THIRD_PARTY_ROWS:
        pytest.skip(f"{variable}: {THIRD_PARTY_ROWS[variable]}")
    assert variable in _READ, (
        f"`{variable}` has a row in {CATALOGUE.name} but no module under halo/ or scripts/ names it. "
        "Delete the row, or add it to THIRD_PARTY_ROWS with the reader that consumes it."
    )


@pytest.mark.parametrize(
    "variable", sorted(name for name in _READ if name.startswith(TOOLKIT_PREFIXES)), ids=lambda name: name
)
def test_every_toolkit_variable_the_code_reads_is_catalogued(variable):
    assert variable in _CATALOGUED, (
        f"`{variable}` is read at {_READ[variable]} but has no row in the "
        f"'Environment variables' table of {CATALOGUE.name}."
    )


def test_third_party_rows_are_still_documented():
    for variable in THIRD_PARTY_ROWS:
        assert variable in _CATALOGUED, f"{variable} is exempted but no longer has a catalogue row"
        assert variable not in _READ, f"{variable} now has a reader — drop it from THIRD_PARTY_ROWS"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
