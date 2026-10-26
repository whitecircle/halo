"""The move kit passes a pure move and a carve that keeps every definition whole, and fails each kind of change a
move must not carry.

A scratch repository commits a small package as the base; each case commits one change on top and runs the kit
from git, as ``make verify-move`` does. The kit also passes the real tree against itself, and a real module move
and a real package move with every reference rewritten across ``halo/``, ``scripts/`` and ``tests/``.

Run: python tests/cpu/conventions/test_move_kit.py
"""

import json
import re
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest

from tests.common.equivalence.move_kit import (
    CHECKS,
    GOLDENS_DIR,
    Checkout,
    RenameMap,
    compare_trees,
    module_name,
    overall_verdict,
    verify_move,
)
from tests.common.equivalence.revisions import Rename, write_evidence
from tests.common.repo import REPO_ROOT
from tests.common.sample_tree import spelled, write_sample_tree

_BASE_SOURCES = {
    "pyproject.toml": """
        [tool.hatch.build.targets.wheel]
        packages = ["$pkg"]
    """,
    "$pkg/__init__.py": '"""The package."""\n',
    "$pkg/core/__init__.py": '"""Core."""\n',
    "$pkg/core/helpers.py": '''
        """Helpers."""


        def scale(value):
            """Ten times ``value``."""
            return value * 10


        def _offset(value):
            return value + 1
    ''',
    "$pkg/core/registry.py": '''
        """The widget registry."""

        import logging

        from $pkg.core.helpers import scale

        logger = logging.getLogger(__name__)
        logging.captureWarnings(True)
        logging.disable(logging.NOTSET)
        REGISTRY = {}
        UNITS = frozenset({"cm", "mm"})


        def register(cls):
            REGISTRY[cls.__name__] = cls
            return cls


        def lookup(name):
            if name not in REGISTRY:
                raise KeyError(f"unknown widget {name!r}; widgets live under $pkg/core/")
            return REGISTRY[name]


        def scaled(value):
            return scale(value) * 2


        @register
        class Widget:
            """A widget."""

            size = 3

            def area(self):
                return scaled(self.size) ** 2


        GADGET = register(type("Gadget", (), {}))
    ''',
    "$pkg/plugins.py": """
        import $pkg.core.registry  # noqa: F401  registers the widgets
    """,
    "$tests/__init__.py": "",
    "$tests/test_widgets.py": """
        from unittest import mock


        def test_scaled():
            with mock.patch("$pkg.core.registry.scaled", return_value=0):
                pass
    """,
    f"{GOLDENS_DIR}/order.json": '{"widgets": ["Widget", "Gadget"]}\n',
    f"{GOLDENS_DIR}/order.py": """
        import json
        import sys
        from pathlib import Path

        import $pkg.plugins  # noqa: F401  registers the widgets
        from $pkg.core.registry import REGISTRY

        golden = Path(__file__).with_suffix(".json")
        if golden.read_text() != json.dumps({"widgets": list(REGISTRY)}) + "\\n":
            sys.exit(f"{golden.name} is stale")
    """,
}
BASE_TREE = {path: dedent(text).lstrip("\n") for path, text in _BASE_SOURCES.items()}
REGISTRY = "$pkg/core/registry.py"
HELPERS = "$pkg/core/helpers.py"
GOLDEN = f"{GOLDENS_DIR}/order.json"
GENERATOR = f"{GOLDENS_DIR}/order.py"
LOOKUP = BASE_TREE[REGISTRY][BASE_TREE[REGISTRY].index("def lookup") : BASE_TREE[REGISTRY].index("def scaled")]
DROPPED_REGISTRATION = ('GADGET = register(type("Gadget", (), {}))', 'GADGET = type("Gadget", (), {})')


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=kit", "-c", "user.email=kit@example.com", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _commit(repo: Path, files: dict[str, str], removed: tuple[str, ...] = ()) -> str:
    write_sample_tree(repo, files)
    for path in removed:
        (repo / spelled(path)).unlink()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "change")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path) -> Path:
    _git(tmp_path, "init", "-q")
    return tmp_path


def _edited(path: str, old: str, new: str) -> dict[str, str]:
    text = BASE_TREE[path]
    assert old in text, f"premise: {old!r} is in {path}"
    return {path: text.replace(old, new)}


def _failing(checks: dict) -> set[str]:
    return {name for name in CHECKS if checks[name]["verdict"] != "pass"}


def test_a_commit_against_itself_passes(repo):
    base = _commit(repo, BASE_TREE)
    checks = verify_move(repo, base, base)
    assert _failing(checks) == set()
    assert checks["definitions"]["compared"] == 15


def test_a_module_move_with_its_imports_rewritten_passes(repo):
    base = _commit(repo, BASE_TREE)
    moved = {
        "$pkg/util/__init__.py": '"""Utilities."""\n',
        "$pkg/util/helpers.py": BASE_TREE[HELPERS],
        **_edited(REGISTRY, "from $pkg.core.helpers import scale", "from $pkg.util.helpers import scale"),
    }
    head = _commit(repo, moved, removed=(HELPERS,))
    new_package = spelled("$pkg.util:__doc__")
    renames = RenameMap(files={spelled(HELPERS): spelled("$pkg/util/helpers.py")}, added=(new_package,))

    checks = verify_move(repo, base, head, renames)

    assert _failing(checks) == set()
    assert checks["renames"]["declared"] == [
        {"old": spelled(HELPERS), "new": spelled("$pkg/util/helpers.py"), "similarity": 100}
    ]
    assert checks["definitions"]["declared_added"] == [new_package]


def test_a_package_moved_whole_passes_with_its_directory_named_in_a_message(repo):
    """Every file of the core package moves to ``base/``, and the lookup error names the package's directory,
    which no file rename spells."""
    base = _commit(repo, BASE_TREE)
    core = [path for path in BASE_TREE if path.startswith("$pkg/core/")]
    moved = {
        path.replace("$pkg/core/", "$pkg/base/"): BASE_TREE[path]
        .replace("$pkg.core", "$pkg.base")
        .replace("$pkg/core/", "$pkg/base/")
        for path in core
    }
    rewritten = {
        path: BASE_TREE[path].replace("$pkg.core", "$pkg.base")
        for path in ("$pkg/plugins.py", "$tests/test_widgets.py", GENERATOR)
    }
    head = _commit(repo, {**moved, **rewritten}, removed=tuple(core))

    checks = verify_move(repo, base, head)

    assert _failing(checks) == set()
    assert {(rename["old"], rename["new"]) for rename in checks["renames"]["detected"]} >= {
        (spelled(path), spelled(path.replace("$pkg/core/", "$pkg/base/"))) for path in core
    }


def test_a_definition_carved_into_another_module_passes_and_lists_the_patch_to_review(repo):
    """``scaled`` and a library-built constant move to the helpers whole and the registry imports ``scaled``
    back: the definitions pair by name, the constant's build is no import-time effect, and the test's patch of
    ``registry.scaled`` now replaces only the registry's binding."""
    base = _commit(repo, BASE_TREE)
    scaled = "def scaled(value):\n    return scale(value) * 2\n"
    units = 'UNITS = frozenset({"cm", "mm"})\n'
    registry = BASE_TREE[REGISTRY].replace(f"{scaled}\n\n", "").replace(units, "")
    registry = registry.replace("import scale", "import scaled")
    helpers = f"{BASE_TREE[HELPERS]}\n{units}\n\n{scaled}"
    assert "def scaled" not in registry and "import scaled" in registry, "premise: the carve edits the registry"

    checks = verify_move(repo, base, _commit(repo, {REGISTRY: registry, HELPERS: helpers}))

    assert _failing(checks) == set()
    assert checks["definitions"]["moved"] == [
        spelled("$pkg.core.registry:UNITS -> $pkg.core.helpers:UNITS"),
        spelled("$pkg.core.registry:scaled -> $pkg.core.helpers:scaled"),
    ]
    assert checks["patch_targets"]["targets"] == [
        {
            "file": spelled("$tests/test_widgets.py"),
            "line": 5,
            "target": spelled("$pkg.core.registry.scaled"),
            "bound_from": spelled("$pkg.core.helpers.scaled"),
        }
    ]


def test_a_carve_of_library_decorated_definitions_passes(repo):
    """A dataclass and a cached function move whole: a library's decorator is no import-time effect of the module
    holding it, so both modules' ordered side effects still match."""
    decorated = dedent('''
        @dataclasses.dataclass(frozen=True)
        class Spec:
            """A widget spec."""

            name: str


        @functools.lru_cache
        def table():
            return {"cm": 10}
    ''')
    imports = "import dataclasses\nimport functools\n"
    registry = BASE_TREE[REGISTRY].replace("import logging\n", f"{imports}import logging\n")
    base = _commit(repo, {**BASE_TREE, REGISTRY: f"{registry}\n{decorated}"})
    helpers = BASE_TREE[HELPERS].replace('"""Helpers."""\n', f'"""Helpers."""\n\n{imports}')
    carved = registry.replace("import scale", "import Spec, scale, table")

    checks = verify_move(repo, base, _commit(repo, {REGISTRY: carved, HELPERS: f"{helpers}\n{decorated}"}))

    assert _failing(checks) == set()
    assert checks["definitions"]["moved"] == [
        spelled("$pkg.core.registry:Spec -> $pkg.core.helpers:Spec"),
        spelled("$pkg.core.registry:table -> $pkg.core.helpers:table"),
    ]


def test_a_carve_into_a_new_module_passes_once_its_additions_are_declared(repo):
    """``lookup`` moves to a new module with a docstring and a logger of its own: those two are additions, which
    the change must declare."""
    base = _commit(repo, BASE_TREE)
    header = '"""Widget lookup."""\n\nimport logging\n\nfrom $pkg.core.registry import REGISTRY\n\n'
    finder = f"{header}logger = logging.getLogger(__name__)\n\n\n{LOOKUP.rstrip()}\n"
    head = _commit(repo, {"$pkg/core/finder.py": finder, REGISTRY: BASE_TREE[REGISTRY].replace(LOOKUP, "")})
    additions = (spelled("$pkg.core.finder:__doc__"), spelled("$pkg.core.finder:logger"))

    assert _failing(verify_move(repo, base, head, RenameMap(added=additions))) == set()
    undeclared = verify_move(repo, base, head)
    assert _failing(undeclared) == {"definitions", "public_names"}
    assert undeclared["definitions"]["added"] == sorted(additions)


def test_a_declared_addition_that_is_no_addition_fails(repo):
    base = _commit(repo, BASE_TREE)
    checks = verify_move(repo, base, base, RenameMap(added=(spelled("$pkg.core.registry:logger"),)))
    assert _failing(checks) == {"definitions"}
    assert checks["definitions"]["declared_not_added"] == [spelled("$pkg.core.registry:logger")]


def test_a_declared_symbol_rename_passes_and_an_undeclared_one_fails(repo):
    base = _commit(repo, BASE_TREE)
    head = _commit(repo, _edited(REGISTRY, "def lookup(name):", "def find(name):"))
    declared = RenameMap(symbols={spelled("$pkg.core.registry:lookup"): spelled("$pkg.core.registry:find")})

    assert _failing(verify_move(repo, base, head, declared)) == set()
    undeclared = verify_move(repo, base, head)
    assert _failing(undeclared) == {"definitions", "public_names"}
    assert undeclared["definitions"]["removed"] == [spelled("$pkg.core.registry:lookup")]
    assert undeclared["definitions"]["added"] == [spelled("$pkg.core.registry:find")]


@pytest.mark.parametrize(
    ("path", "old", "new", "failing"),
    [
        pytest.param(HELPERS, "value * 10", "value * 11", {"definitions"}, id="a changed body"),
        pytest.param(
            REGISTRY,
            'f"unknown widget {name!r}',
            'f"no widget {name!r}',
            {"definitions", "string_literals", "raise_templates"},
            id="a changed message",
        ),
        pytest.param(
            REGISTRY,
            "raise KeyError",
            "raise LookupError",
            {"definitions", "raise_templates"},
            id="a changed exception",
        ),
        pytest.param(
            REGISTRY,
            "logging.captureWarnings(True)\nlogging.disable(logging.NOTSET)",
            "logging.disable(logging.NOTSET)\nlogging.captureWarnings(True)",
            {"side_effects"},
            id="reordered import-time calls",
        ),
        pytest.param(REGISTRY, "@register\n", "", {"definitions", "side_effects"}, id="a dropped decorator"),
        pytest.param(REGISTRY, *DROPPED_REGISTRATION, {"definitions", "side_effects"}, id="a dropped registration"),
        pytest.param(
            "$pkg/plugins.py",
            "import $pkg.core.registry  # noqa: F401  registers the widgets\n",
            "",
            {"side_effects"},
            id="a dropped side-effect import",
        ),
        pytest.param(
            HELPERS, "def _offset", "def offset", {"definitions", "public_names"}, id="a private name made public"
        ),
        pytest.param(
            HELPERS, '"""Ten times ``value``."""', '"""Ten times the value."""', {"definitions"}, id="a docstring"
        ),
        pytest.param(HELPERS, '"""Helpers."""', '"""Arithmetic."""', {"definitions"}, id="a module docstring"),
        pytest.param(GOLDEN, '"Widget", ', "", {"goldens"}, id="a golden"),
    ],
)
def test_each_kind_of_change_fails_its_checks(repo, path, old, new, failing):
    base = _commit(repo, BASE_TREE)
    head = _commit(repo, _edited(path, old, new))
    assert _failing(verify_move(repo, base, head)) == failing


def test_a_docstring_change_passes_when_docstrings_are_ignored(repo):
    base = _commit(repo, BASE_TREE)
    head = _commit(
        repo,
        {
            **_edited(HELPERS, '"""Ten times ``value``."""', '"""Ten times the value."""'),
            **_edited(REGISTRY, '"""The widget registry."""', '"""Where widgets register."""'),
        },
    )
    assert _failing(verify_move(repo, base, head, ignore_docstrings=True)) == set()


def test_a_declared_golden_may_change_and_an_undeclared_one_may_not(repo):
    """The registry module moves and a golden names it, so the golden changes with the move: declared, the move
    passes; undeclared, it fails; declared but unchanged, the stale declaration fails."""
    golden = {"widgets": ["Widget", "Gadget"], "module": "$pkg.core.registry"}
    base = _commit(repo, {**BASE_TREE, GOLDEN: json.dumps(golden) + "\n"})
    moved = {
        "$pkg/core/widgets.py": BASE_TREE[REGISTRY],
        **{
            path: BASE_TREE[path].replace("core.registry", "core.widgets")
            for path in ("$pkg/plugins.py", "$tests/test_widgets.py", GENERATOR)
        },
        GOLDEN: json.dumps({**golden, "module": "$pkg.core.widgets"}) + "\n",
    }
    head = _commit(repo, moved, removed=(REGISTRY,))
    declared = RenameMap(goldens=("order.json",))

    checks = verify_move(repo, base, head, declared)
    assert _failing(checks) == set()
    assert checks["goldens"]["declared"] == ["order.json"]
    undeclared = verify_move(repo, base, head)
    assert _failing(undeclared) == {"goldens"}
    assert undeclared["goldens"]["changed"] == ["order.json"]
    stale = verify_move(repo, base, base, declared)
    assert _failing(stale) == {"goldens"}
    assert stale["goldens"]["declared_unchanged"] == ["order.json"]


def test_a_golden_stale_against_head_code_fails_its_generator_check(repo):
    """At base the generator renders what the golden holds; a head registering one widget fewer leaves the golden
    byte-identical but stale against head's own code."""
    base = _commit(repo, BASE_TREE)
    assert verify_move(repo, base, base, check_goldens=True)["goldens"]["generator_checks"] == {"order.json": "pass"}

    head = _commit(repo, _edited(REGISTRY, *DROPPED_REGISTRATION))
    goldens = verify_move(repo, base, head, check_goldens=True)["goldens"]

    assert goldens["changed"] == []
    assert goldens["generator_checks"] == {"order.json": "order.json is stale"}
    assert goldens["verdict"] == "fail"


def test_a_golden_without_a_generator_fails_the_generator_check(repo):
    base = _commit(repo, BASE_TREE)
    head = _commit(repo, {f"{GOLDENS_DIR}/orphan.json": "{}\n"})

    goldens = verify_move(repo, base, head, RenameMap(goldens=("orphan.json",)), check_goldens=True)["goldens"]

    assert goldens["generator_checks"] == {"order.json": "pass", "orphan.json": "no generator"}
    assert goldens["verdict"] == "fail"


def test_a_declared_rename_below_the_similarity_bar_fails(repo):
    base = _commit(repo, BASE_TREE)
    rewritten = BASE_TREE[HELPERS].replace('"""Helpers."""', '"""Arithmetic."""').replace("+ 1", "+ 2")
    head = _commit(
        repo,
        {
            "$pkg/core/arithmetic.py": rewritten,
            **_edited(REGISTRY, "core.helpers import", "core.arithmetic import"),
        },
        removed=(HELPERS,),
    )
    renames = RenameMap(files={spelled(HELPERS): spelled("$pkg/core/arithmetic.py")})

    checks = verify_move(repo, base, head, renames)

    assert checks["renames"]["verdict"] == "fail"
    [short] = checks["renames"]["below_minimum"]
    assert 50 <= short["similarity"] < 90
    lowered = verify_move(repo, base, head, renames, min_similarity=short["similarity"])
    assert lowered["renames"]["verdict"] == "pass"


def test_a_declared_name_the_trees_lack_is_refused(repo):
    base = _commit(repo, BASE_TREE)
    missing = spelled("$pkg.core.registry:missing")
    with pytest.raises(ValueError, match="lack"):
        verify_move(repo, base, base, RenameMap(symbols={missing: spelled("$pkg.x:y")}))
    with pytest.raises(ValueError, match="lack"):
        verify_move(repo, base, base, RenameMap(added=(missing,)))


def test_the_rename_map_file_reads_every_field_and_refuses_unknown_keys(tmp_path):
    path = tmp_path / "renames.json"
    declared = {"files": {"a.py": "b.py"}, "symbols": {"a:x": "b:x"}, "added": ["b:logger"], "goldens": ["g.json"]}
    path.write_text(json.dumps(declared))
    assert RenameMap.read(path) == RenameMap({"a.py": "b.py"}, {"a:x": "b:x"}, ("b:logger",), ("g.json",))
    path.write_text(json.dumps({"files": {}, "modules": {}}))
    with pytest.raises(ValueError, match="unknown keys"):
        RenameMap.read(path)


def test_the_verdict_counts_only_the_required_checks(repo):
    base = _commit(repo, BASE_TREE)
    checks = verify_move(repo, base, _commit(repo, _edited(HELPERS, "value * 10", "value * 11")))
    assert overall_verdict(checks) == "fail"
    assert overall_verdict(checks, ("renames", "string_literals", "side_effects")) == "pass"


def test_the_evidence_names_both_commits_and_the_findings(tmp_path):
    path = write_evidence(
        "move_kit",
        base="b" * 40,
        head="h" * 40,
        image="halo:test",
        verdict="pass",
        findings={"checks": {}},
        directory=tmp_path,
    )
    record = json.loads(path.read_text())
    assert path.name == f"move_kit-{'h' * 40}.json"
    assert {key: record[key] for key in ("harness", "base", "head", "image", "verdict", "checks")} == {
        "harness": "move_kit",
        "base": "b" * 40,
        "head": "h" * 40,
        "image": "halo:test",
        "verdict": "pass",
        "checks": {},
    }
    assert record["command"]


def test_the_real_tree_passes_against_itself():
    checkout = Checkout.read(REPO_ROOT)
    checks = compare_trees(checkout, checkout, [])
    assert _failing(checks) == set()
    assert checks["definitions"]["compared"] > 5000, "the kit lost the package's definitions"


def _moved_paths(old: str, new: str) -> dict[str, str]:
    """Each file the move of ``old`` to ``new`` renames, by repo path."""
    if old.endswith(".py"):
        return {old: new}
    return {
        path.relative_to(REPO_ROOT).as_posix(): f"{new}/{path.relative_to(REPO_ROOT / old).as_posix()}"
        for path in sorted((REPO_ROOT / old).rglob("*.py"))
    }


@pytest.mark.parametrize(
    ("old", "new"),
    [
        pytest.param("$pkg/env.py", "$pkg/process/environment.py", id="a module"),
        pytest.param(
            "$pkg/distributed/expert_parallel/families", "$pkg/distributed/expert_parallel/layers", id="a package"
        ),
    ],
)
def test_a_real_move_with_every_reference_rewritten_passes(tmp_path, old, new):
    """The module or package moves, and every import, patch string, dotted name and path naming it follows: the
    canonical spelling absorbs all of it, a message naming the package's directory included."""
    for top in (*Checkout.read(REPO_ROOT).packages, "scripts", "tests"):
        shutil.copytree(REPO_ROOT / top, tmp_path / top, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(REPO_ROOT / "pyproject.toml", tmp_path)
    old, new = spelled(old), spelled(new)
    (tmp_path / new).parent.mkdir(exist_ok=True)
    (tmp_path / old).rename(tmp_path / new)
    renames = _moved_paths(old, new)
    old_dotted, new_dotted = (module_name(Path(path)) for path in (old, new))
    spellings = {
        re.compile(rf"\b{re.escape(old)}\b"): new,
        re.compile(rf"\b{re.escape(old_dotted)}\b"): new_dotted,
    }
    rewritten = 0
    for path in tmp_path.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        moved = text
        for pattern, spelling in spellings.items():
            moved = pattern.sub(spelling, moved)
        if moved != text:
            path.write_text(moved, encoding="utf-8")
            rewritten += 1
    assert rewritten > 20, "premise: the move is named across the tree"

    checks = compare_trees(
        Checkout.read(REPO_ROOT),
        Checkout.read(tmp_path),
        [Rename(source, target, 100) for source, target in renames.items()],
        RenameMap(files=renames),
    )

    assert _failing(checks) == set()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
