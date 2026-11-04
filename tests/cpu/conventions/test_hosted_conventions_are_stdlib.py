"""Every test the hosted ``conventions`` job runs reaches only the standard library, pytest and first-party files
that do the same. ``.github/workflows/lint.yml`` installs pytest alone, so any other import fails only there."""

import ast
import functools
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from scripts.docs.references import (
    FIRST_PARTY,
    PACKAGE,
    LoadedModule,
    RepoIndex,
    dotted_name,
    loaded_module,
    string_constants,
)
from tests.common.repo import REPO_ROOT, load_script_module, probe_findings
from tests.common.sample_tree import spelled, write_sample_tree
from tests.gpu.manifest import LAUNCHER_ENTRYPOINTS

_WORKFLOW = REPO_ROOT / ".github/workflows/lint.yml"
_JOB = re.compile(r"^  conventions:\n(.*?)(?=^  \S|\Z)", re.MULTILINE | re.DOTALL)
_TEST_FILE = re.compile(r"\btests/[\w/]+\.py\b")
_INSTALLED = frozenset({"pytest"})
# What a listed test's probe script has pytest collect, which no import in the test's source names. The
# walk refuses any other probe, since it cannot read what one imports.
_PROBE_COLLECTS = {
    "tests/cpu/conventions/test_test_conventions.py": tuple(
        f"tests/gpu/{name}" for name in sorted(LAUNCHER_ENTRYPOINTS)
    ),
}
_INDEX = RepoIndex(REPO_ROOT)
uv_lock = load_script_module(".github/scripts/locked_version.py")


def _hosted_tests() -> list[str]:
    job = _JOB.search(_WORKFLOW.read_text(encoding="utf-8"))
    return sorted(set(_TEST_FILE.findall(job.group(1)))) if job else []


_HOSTED = _hosted_tests()


@functools.cache
def _tree(file: Path) -> ast.Module:
    return ast.parse(file.read_text(encoding="utf-8"), filename=str(file))


def _calls(file: Path) -> list[ast.Call]:
    return [node for node in ast.walk(_tree(file)) if isinstance(node, ast.Call)]


def _loads(file: Path) -> list[LoadedModule]:
    constants = string_constants(_tree(file))
    return [loaded for call in _calls(file) if (loaded := loaded_module(call, constants))]


def _pytest_runs(index: RepoIndex, test: Path) -> list[Path]:
    """The package ``__init__.py`` and ``conftest.py`` files pytest runs for ``test``, from the root down."""
    directories = [directory for directory in reversed(test.parents) if directory.is_relative_to(index.root)]
    return [
        file
        for directory in directories
        for file in (directory / "__init__.py", directory / "conftest.py")
        if file.is_file()
    ]


def _imported_modules(index: RepoIndex, file: Path) -> set[str]:
    """Each module ``file`` imports anywhere in it: an ``import`` target, a ``from`` source, a name a ``from``
    line imports that is a first-party submodule, and an ``import_module`` argument."""
    modules = set()
    for node in ast.walk(_tree(file)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            source = index.import_source(node, file)
            modules.add(source)
            modules.update(index.from_import_module(source, alias.name) for alias in node.names)
    return modules | {loaded.name for loaded in _loads(file) if loaded.path is None}


def _loaded_by_path(index: RepoIndex, file: Path) -> list[Path]:
    return [index.root / loaded.path for loaded in _loads(file) if loaded.path is not None]


def _runs_a_probe(file: Path) -> bool:
    return any((dotted_name(call.func) or "").rpartition(".")[2] == probe_findings.__name__ for call in _calls(file))


def _foreign_imports(index: RepoIndex, test: Path, probe_collects: Mapping[str, Sequence[str]]) -> list[str]:
    """Each import outside the standard library and pytest that running ``test`` reaches, as ``a -> b imports m``."""
    findings, seen = set(), set()
    pending = [(test,), *((test, file) for file in _pytest_runs(index, test))]
    while pending:
        chain = pending.pop()
        file = chain[-1]
        if file in seen:
            continue
        seen.add(file)
        reached = " -> ".join(str(step.relative_to(index.root)) for step in chain)
        for module in _imported_modules(index, file):
            top = module.partition(".")[0]
            if top in FIRST_PARTY and top != PACKAGE:
                parts = module.split(".")
                # Importing a.b.c runs a/__init__.py and a/b/__init__.py first.
                sources = (index.module_source(".".join(parts[:depth])) for depth in range(1, len(parts) + 1))
                pending += [(*chain, source) for source in sources if source is not None and source.is_file()]
            elif top not in sys.stdlib_module_names and top not in _INSTALLED:
                findings.add(f"{reached} imports {module}")
        pending += [(*chain, loaded) for loaded in _loaded_by_path(index, file)]
        if _runs_a_probe(file):
            collected = probe_collects.get(str(file.relative_to(index.root)))
            if collected is None:
                findings.add(f"{reached} runs a probe script, whose imports the walk cannot read")
            for module in map(index.root.joinpath, collected or ()):
                pending += [(*chain, step) for step in (*_pytest_runs(index, module), module)]
    return sorted(findings)


def test_the_job_runs_the_convention_tests_and_this_one():
    """A workflow edit that drops the list, or this check from it, would pass everything below."""
    assert str(Path(__file__).resolve().relative_to(REPO_ROOT)) in _HOSTED, _HOSTED
    assert len(_HOSTED) > 1, f"the conventions job lists only {_HOSTED}"
    missing = [path for path in _HOSTED if not (REPO_ROOT / path).is_file()]
    assert not missing, f"{_WORKFLOW.name} runs test files that do not exist: {missing}"


def test_every_probe_exception_names_a_listed_test_that_runs_one():
    stale = [path for path in _PROBE_COLLECTS if path not in _HOSTED or not _runs_a_probe(REPO_ROOT / path)]
    assert not stale, f"drop these from _PROBE_COLLECTS: {stale}"


@pytest.mark.parametrize("path", _HOSTED)
def test_a_hosted_test_reaches_only_the_standard_library_and_pytest(path):
    findings = _foreign_imports(_INDEX, REPO_ROOT / path, _PROBE_COLLECTS)
    assert not findings, (
        f"the hosted conventions job ({_WORKFLOW.relative_to(REPO_ROOT)}) installs only pytest, so these "
        "imports fail it: read the source with ast instead of importing it, move the import out of the "
        "shared helper, or drop the test from the job's list:\n  " + "\n  ".join(findings)
    )


def test_the_job_installs_the_locked_pytest():
    """The job installs uv.lock's pin, which is the one the image runs."""
    assert uv_lock.locked_version("pytest") == pytest.__version__
    with pytest.raises(SystemExit, match="no-such-package"):
        uv_lock.locked_version("no-such-package")


def test_the_walk_follows_every_load_to_a_foreign_import(tmp_path):
    write_sample_tree(
        tmp_path,
        {
            "$tests/__init__.py": "",
            "$tests/common/__init__.py": "",
            "$tests/common/clean.py": "import json\nfrom pathlib import Path\n",
            "$tests/common/heavy.py": "import torch\n",
            "$tests/heavy_package/__init__.py": "import numpy\n",
            "$tests/heavy_package/leaf.py": "",
            "$tests/heavy_tests/__init__.py": "import torch\n",
            "$tests/heavy_tests/test_in_package.py": "",
            "$tests/heavy_conftest/conftest.py": "import numpy\n",
            "$tests/heavy_conftest/test_beside_conftest.py": "",
            "$tests/collected/conftest.py": "import requests\n",
            "$tests/collected/test_launcher.py": "",
            "$scripts/tool.py": "import yaml\n",
            "$github/scripts/check.py": "import yaml\n",
            "$tests/cpu/__init__.py": "",
            "$tests/cpu/test_clean.py": """
                import pytest

                from $tests.common import clean
                from $tests.common.clean import Path
            """,
            "$tests/cpu/test_helper.py": "from $tests.common.heavy import thing\n",
            "$tests/cpu/test_relative.py": "from ..common import heavy\n",
            "$tests/cpu/test_package_init.py": "from $tests.heavy_package.leaf import thing\n",
            "$tests/cpu/test_script.py": "from $scripts.tool import run\n",
            "$tests/cpu/test_source.py": "from $pkg.env import env_flag\n",
            "$tests/cpu/test_local.py": "def load():\n    import transformers\n",
            "$tests/cpu/test_by_path.py": 'check = load_script_module("$github/scripts/check.py")\n',
            "$tests/cpu/test_by_named_path.py": '_CHECK = "$github/scripts/check.py"\ncheck = load_script_module(_CHECK)\n',
            "$tests/cpu/test_by_name.py": 'heavy = importlib.import_module("$tests.common.heavy")\n',
            "$tests/cpu/test_probe.py": 'found = probe_findings(SCRIPT, "FOUND:")\n',
            "$tests/cpu/test_collecting_probe.py": 'found = probe_findings(SCRIPT, "FOUND:")\n',
        },
    )
    index = RepoIndex(tmp_path)
    collecting_probe = {
        spelled("$tests/cpu/test_collecting_probe.py"): (spelled("$tests/collected/test_launcher.py"),)
    }

    def foreign(test: str) -> list[str]:
        return _foreign_imports(index, tmp_path / spelled(f"$tests/{test}"), collecting_probe)

    assert foreign("cpu/test_clean.py") == []
    expected = {
        "cpu/test_helper.py": "$tests/cpu/test_helper.py -> $tests/common/heavy.py imports torch",
        "cpu/test_relative.py": "$tests/cpu/test_relative.py -> $tests/common/heavy.py imports torch",
        "cpu/test_package_init.py": "$tests/cpu/test_package_init.py -> $tests/heavy_package/__init__.py imports numpy",
        "cpu/test_script.py": "$tests/cpu/test_script.py -> $scripts/tool.py imports yaml",
        "cpu/test_source.py": "$tests/cpu/test_source.py imports $pkg.env",
        "cpu/test_local.py": "$tests/cpu/test_local.py imports transformers",
        "cpu/test_by_path.py": "$tests/cpu/test_by_path.py -> $github/scripts/check.py imports yaml",
        "cpu/test_by_named_path.py": "$tests/cpu/test_by_named_path.py -> $github/scripts/check.py imports yaml",
        "cpu/test_by_name.py": "$tests/cpu/test_by_name.py -> $tests/common/heavy.py imports torch",
        "cpu/test_probe.py": "$tests/cpu/test_probe.py runs a probe script, whose imports the walk cannot read",
        "cpu/test_collecting_probe.py": (
            "$tests/cpu/test_collecting_probe.py -> $tests/collected/conftest.py imports requests"
        ),
        "heavy_tests/test_in_package.py": (
            "$tests/heavy_tests/test_in_package.py -> $tests/heavy_tests/__init__.py imports torch"
        ),
        "heavy_conftest/test_beside_conftest.py": (
            "$tests/heavy_conftest/test_beside_conftest.py -> $tests/heavy_conftest/conftest.py imports numpy"
        ),
    }
    for test, finding in expected.items():
        assert foreign(test) == [spelled(finding)], test


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
