"""Every first-party import, patch target, attribute chain, dotted name and repo-root path outside the
docs resolves.

ruff does not resolve imports, and the GPU scripts, the ``scripts/`` tools and the image builds reach
modules no CPU test imports, so a moved module or a renamed symbol would surface only when that file
next runs. This sweep resolves them statically, through the index the docs gate checks markdown with
(:class:`scripts.docs.references.RepoIndex`):

* each import of a first-party package in the Python under ``src/``, ``scripts/``, ``tests/`` (the GPU
  scripts included), ``docker/`` and ``.github/``, from the module that defines what it takes: one that
  only re-binds the name by an import of its own is a second path, which would keep a moved symbol
  reachable through its old home;
* each ``patch`` / ``monkeypatch.setattr`` / ``delattr`` target, down to the attribute unless
  ``create=True`` / ``raising=False`` lets it be absent, and each assignment to an attribute of such an
  owner (a hand-rolled patch): a dotted string, or an object-form owner bound at module level by an import
  or a ``load_script_module`` / ``import_module`` call. A module's own top-level name must be one a
  function reads when it runs, one of that module or, outside the tests, one reading it through the module
  object; patching a name that is only bound, or read only at import, intercepts nothing, and the test
  passes vacuously;
* each attribute chain on such an owner (``sdpa_flex_sliding._SAVED_BYTES_PER_SCORE``) where no
  local name hides it. It must exist, but may name a binding the module only imports: that binding is
  what a patch replaces and restores through the module;
* each dotted name and repo-root path any other string literal or comment names: ``caplog`` loggers,
  ``sys.modules`` keys, ``import_module`` arguments, docstring roles, doc pages and their headings in
  messages; an f-string counts where it formats a module-level string constant;
* the same names and paths in every other text file of those trees (the GPU benchmark runners, the
  ``docker/`` patches, the workflows), the image builds, the ``Makefile``, ``pyproject.toml``,
  ``.env.example``, ``.pre-commit-config.yaml``, ``.devcontainer/`` and the recipes under ``examples/``
  and ``launcher-configs/``.

A part after a class must be an attribute its first-party hierarchy defines, or one the index lists for
a third-party base, so a renamed method fails under ``nn.Module`` or a TRL trainer too. Standard
library and pytest only: nothing it checks is imported.
"""

import ast
import io
import tokenize
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path
from typing import Literal, NamedTuple

import pytest

from scripts.docs.references import (
    COMPREHENSION_SCOPES,
    FIRST_PARTY,
    UNKNOWN_FIELD,
    RepoIndex,
    dotted_name,
    literal_text,
    own_bindings,
    patch_call,
    string_constants,
    whole_string_literals,
)
from tests.common.repo import REPO_ROOT
from tests.common.sample_tree import spelled, write_sample_tree

_PYTHON_TREES = (*FIRST_PARTY, "docker", ".github")
_TEXT_TREES = (*_PYTHON_TREES, ".devcontainer", "examples", "launcher-configs")
_TEXT_FILES_AT_ROOT = (
    "Dockerfile*",
    "docker-compose*.yml",
    "Makefile",
    "pyproject.toml",
    ".env.example",
    ".pre-commit-config.yaml",
)

# Text that reads as a reference but names nothing on purpose, by file.
_NOT_REFERENCES = {
    # The embedded Python's own `src` variable.
    "docker/vllm/patches/patch_vllm_disable_gptoss.sh": {"src.replace"},
    # The scratch trees the CLI's script and config lookups are tested on.
    "tests/cpu/config/test_cli.py": {
        "examples/sft/example.yaml",
        "examples/sft/missing.yaml",
        "scripts/inference/serve.py",
        "scripts/training/a/train.py",
        "scripts/training/b/train.py",
        "scripts/training/nested/reward.py",
        "scripts/training/nested/sft.py",
        "scripts/training/preference/constants.py",
    },
    # A page the release check plants in its scratch copy of the docs.
    "tests/cpu/config/test_release_version_in_sync.py": {"human-docs/planted.md"},
}

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_INDEX = RepoIndex(REPO_ROOT)


class Finding(NamedTuple):
    line: int
    kind: Literal["import", "patch target", "attribute", "name", "path"]
    text: str
    why: str | None  # why it names nothing; None when it resolves


def _python_files(root: Path) -> list[Path]:
    return sorted(path for tree in _PYTHON_TREES for path in (root / tree).rglob("*.py"))


def _is_text(path: Path) -> bool:
    """A file that decodes as UTF-8; a font or an image names nothing."""
    try:
        path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _text_files(root: Path) -> list[Path]:
    at_root = {path for pattern in _TEXT_FILES_AT_ROOT for path in root.glob(pattern)}
    in_trees = {
        path
        for tree in _TEXT_TREES
        for path in (root / tree).rglob("*")
        if path.is_file() and path.suffix != ".py" and "__pycache__" not in path.parts and _is_text(path)
    }
    return sorted(at_root | in_trees)


# This file's exemptions and sample tree name what does not exist, on purpose.
_FILES = [path for path in (*_python_files(REPO_ROOT), *_text_files(REPO_ROOT)) if path != Path(__file__).resolve()]


def _attribute_chains(tree: ast.Module, owners: dict[str, str]) -> list[tuple[int, str, bool]]:
    """``(line, dotted name, assigned)`` for each outermost attribute chain on a module-level name bound to a
    first-party owner, where no local binding hides that name; ``assigned`` when the chain is assigned to."""
    chains = []

    def visit(node: ast.AST, hidden: frozenset[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (*_FUNCTIONS, *COMPREHENSION_SCOPES)):
                visit(child, hidden | own_bindings(child))
            elif isinstance(child, ast.Attribute) and (chain := dotted_name(child)):
                root, _, attributes = chain.partition(".")
                if root in owners and root not in hidden:
                    assigned = isinstance(child.ctx, ast.Store)
                    chains.append((child.lineno, f"{owners[root]}.{attributes}", assigned))
            else:
                visit(child, hidden)

    visit(tree, frozenset())
    return chains


def _run_time_loads(tree: ast.Module) -> list[str]:
    """Each name, or attribute chain on one, a function of the module ``tree`` loads when it runs, where no
    binding of that function, or of one around it, hides the name. Decorators and defaults load where the
    function is defined."""
    loads = []

    def visit(node: ast.AST, hidden: frozenset[str] | None) -> None:
        if isinstance(node, _FUNCTIONS):
            arguments = node.args
            for evaluated in (*getattr(node, "decorator_list", ()), *arguments.defaults, *arguments.kw_defaults):
                if evaluated is not None:
                    visit(evaluated, hidden)
            inside = (hidden or frozenset()) | own_bindings(node)
            for statement in node.body if isinstance(node.body, list) else [node.body]:
                visit(statement, inside)
            return
        if hidden is not None:
            if isinstance(node, COMPREHENSION_SCOPES):
                hidden = hidden | own_bindings(node)
            elif isinstance(node, (ast.Name, ast.Attribute)) and (dotted := dotted_name(node)):
                # Assigning to a chain loads what it is set on.
                loaded = dotted if isinstance(node.ctx, ast.Load) else dotted.rpartition(".")[0]
                if loaded and loaded.partition(".")[0] not in hidden:
                    loads.append(loaded)
                return
        for child in ast.iter_child_nodes(node):
            visit(child, hidden)

    visit(tree, None)
    return loads


@cache
def _function_reads(source: Path) -> frozenset[str]:
    """The module-level names the functions of the module at ``source`` read when they run."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    return frozenset(loaded.partition(".")[0] for loaded in _run_time_loads(tree))


def _module_attribute(index: RepoIndex, dotted: str) -> tuple[str, str] | None:
    """``(module, name)``: the name ``dotted`` reads off its longest module prefix; None without one."""
    parts = dotted.split(".")
    for end in range(len(parts) - 1, 0, -1):
        module = ".".join(parts[:end])
        if index.module_source(module) is not None:
            return module, parts[end]
    return None


@cache
def _reads_through_modules(index: RepoIndex) -> dict[str, frozenset[str]]:
    """Per first-party module, the names functions in the swept Python outside the test modules read off it when
    they run, through a module-level name bound to the module itself (``module.name``)."""
    reads: dict[str, set[str]] = defaultdict(set)
    for path in _python_files(index.root):
        # A test reading back what it patched observes the patch, not the code under test.
        if path.name.startswith("test_") and path.is_relative_to(index.root / "tests"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        owners = index.owners(path, tree, string_constants(tree))
        modules = {name: owner for name, owner in owners.items() if index.module_source(owner) is not None}
        for loaded in _run_time_loads(tree):
            root, _, attributes = loaded.partition(".")
            if root in modules and attributes and (read := _module_attribute(index, f"{modules[root]}.{attributes}")):
                module, name = read
                reads[module].add(name)
    return {module: frozenset(names) for module, names in reads.items()}


def _unread_patch_target(target: str, index: RepoIndex) -> str | None:
    """Why patching ``target`` intercepts nothing, or None. Only a module's own top-level name can be dead this
    way, a class attribute being read through its class: one that no function reads when it runs, neither the
    module's own nor one elsewhere through the module object."""
    module, _, name = target.rpartition(".")
    source = index.module_source(module)
    if UNKNOWN_FIELD in target or source is None or source.is_dir() or index.module_source(target) is not None:
        return None
    if name in _function_reads(source) or name in _reads_through_modules(index).get(module, ()):
        return None
    return (
        f"no function reads {name!r} from {module} when it runs, so the patch intercepts nothing; patch the "
        f"binding the code under test reads"
    )


def _reexported_import(index: RepoIndex, module: str, name: str) -> str | None:
    """Why ``from module import name`` takes a second path to ``name``, or None when ``module`` defines it."""
    source = index.import_binding(module, name)
    return f"{module} only re-binds it from {source}; import it from there" if source else None


def _comment_blocks(source: str) -> list[tuple[int, str]]:
    """``(first line, text)`` for each run of comments on consecutive lines, so a path wrapped onto the
    next comment line reads whole."""
    blocks: list[tuple[int, list[str]]] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            line = token.start[0]
            if blocks and blocks[-1][0] + len(blocks[-1][1]) == line:
                blocks[-1][1].append(token.string)
            else:
                blocks.append((line, [token.string]))
    return [(line, "\n".join(comments)) for line, comments in blocks]


def _text_findings(text: str, first_line: int, index: RepoIndex, last_line: int | None = None) -> list[Finding]:
    """The references in ``text``, which starts on ``first_line``, each on the line it starts on; a string's
    ``last_line`` caps that where an escaped newline is no line of the source."""
    findings = []
    for reference in index.references(text):
        line = first_line + text.count("\n", 0, reference.start)
        findings.append(Finding(min(line, last_line or line), reference.kind, reference.text, reference.why))
    return findings


def _scan_python(path: Path, index: RepoIndex) -> list[Finding]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    constants = string_constants(tree)
    owners = index.owners(path, tree, constants)
    findings = []
    for line, dotted, assigned in _attribute_chains(tree, owners):
        why = index.unresolved_name(dotted)
        if assigned:
            findings.append(Finding(line, "patch target", dotted, why or _unread_patch_target(dotted, index)))
        else:
            findings.append(Finding(line, "attribute", dotted, why))
    # A patch call's target string is read as the target, not again as text.
    patch_strings = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            findings += [
                Finding(node.lineno, "import", alias.name, index.unresolved_name(alias.name))
                for alias in node.names
                if alias.name.partition(".")[0] in FIRST_PARTY
            ]
        elif isinstance(node, ast.ImportFrom):
            module = index.import_source(node, path)
            if module.partition(".")[0] in FIRST_PARTY:
                for alias in node.names:
                    name = module if alias.name == "*" else f"{module}.{alias.name}"
                    why = index.unresolved_name(name) or _reexported_import(index, module, alias.name)
                    findings.append(Finding(node.lineno, "import", name, why))
        elif isinstance(node, ast.Call) and (patch := patch_call(node, owners, constants)):
            if patch.string is not None:
                patch_strings.add(id(patch.string))
            unread = _unread_patch_target(patch.target, index) if patch.replaces_binding else None
            findings += [
                Finding(node.lineno, "patch target", reference.text, reference.why or unread)
                for reference in index.references(patch.target)
                if reference.kind == "name"
            ]
    for node in whole_string_literals(tree):
        if id(node) not in patch_strings:
            findings += _text_findings(literal_text(node, constants), node.lineno, index, node.end_lineno)
    for line, comments in _comment_blocks(source):
        findings += _text_findings(comments, line, index)
    return findings


def _scan(path: Path, index: RepoIndex) -> list[Finding]:
    if path.suffix == ".py":
        return _scan_python(path, index)
    return _text_findings(path.read_text(encoding="utf-8"), 1, index)


@cache
def _findings(path: Path) -> list[Finding]:
    return _scan(path, _INDEX)


def test_the_sweep_sees_every_kind_of_reference():
    """Anti-vacuity: a file list or an extractor that silently stops matching would pass every file below."""
    swept = {str(path.relative_to(REPO_ROOT)) for path in _FILES}
    assert {
        "Dockerfile",
        "Dockerfile.vllm",
        "Makefile",
        "pyproject.toml",
        ".env.example",
        ".pre-commit-config.yaml",
        "tests/gpu/profiling/run_baseline_benchmarks.sh",
        "docker/training/flash_attn_split_stubs_hopper.cpp",
    } <= swept
    kinds = Counter(finding.kind for path in _FILES for finding in _findings(path))
    gpu_scripts = [
        path
        for path in _FILES
        if path.is_relative_to(REPO_ROOT / "tests/gpu")
        and any(finding.kind == "import" for finding in _findings(path))
    ]
    text_references = sum(len(_findings(path)) for path in _FILES if path.suffix != ".py")
    assert len(gpu_scripts) > 150, f"only {len(gpu_scripts)} GPU scripts import a first-party module"
    assert kinds["patch target"] > 500, kinds
    assert kinds["attribute"] > 2000, kinds
    assert kinds["import"] > 8000, kinds
    assert kinds["name"] > 400, kinds
    assert kinds["path"] > 1500, kinds
    assert text_references > 400, f"the build files, runners and recipes name only {text_references} references"


@pytest.mark.parametrize("path", _FILES, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_every_reference_resolves(path):
    file = str(path.relative_to(REPO_ROOT))
    exempt = _NOT_REFERENCES.get(file, set())
    unresolved = [finding for finding in _findings(path) if finding.why and finding.text not in exempt]
    assert not unresolved, "\n".join(
        f"{file}:{finding.line}: {finding.kind} {finding.text} — {finding.why}" for finding in unresolved
    )


def test_every_exemption_is_still_needed():
    removable = [
        f"{file}: {text}"
        for file, texts in _NOT_REFERENCES.items()
        for text in sorted(texts - {finding.text for finding in _findings(REPO_ROOT / file) if finding.why})
    ]
    assert not removable, f"exempted text that no longer fails; drop it from the exemptions: {removable}"


def test_each_kind_of_stale_reference_is_reported(tmp_path):
    """A renamed module, a dead patch target or attribute, a stale doc path or heading and a stale
    image-build import each fail, while the live reference of the same kind beside each resolves."""
    sample = {
        "$pkg/__init__.py": "",
        "$pkg/sub/__init__.py": """
            from . import mod
            from .gone import thing
            from .tool import tool
        """,
        "$pkg/sub/tool.py": """
            def tool():
                pass


            def run():
                return tool()
        """,
        "$pkg/sub/base.py": """
            import torch


            class Base:
                inherited = 1

                def __init__(self):
                    self.field = 0


            class TorchBase(torch.nn.Module):
                def forward(self):
                    pass
        """,
        "$pkg/sub/mod.py": """
            import torch

            from $pkg.sub.base import Base, TorchBase

            VALUE = 1
            Dynamic = type("Dynamic", (), {})


            class Child(Base):
                def method(self):
                    pass


            class TorchChild(TorchBase):
                pass


            class DynamicChild(Dynamic):
                pass


            class Op(torch.autograd.Function):
                pass


            def helper():
                pass


            def run():
                return helper(), VALUE
        """,
        "$pkg/moved/__pycache__/left.cpython-312.pyc": "",
        "scripts/tool.py": """
            def run():
                pass
        """,
        "agent-docs/page.md": "# Page\n\n## Section\n",
        "sample_test.py": '''
            """Covers $pkg/sub/mod.py and agent-docs/page.md#section,
            not $pkg/sub/gone.py, $pkg/moved/ or agent-docs/page.md#gone."""

            import importlib
            from unittest import mock

            from $pkg.sub import mod
            from $pkg.sub.mod import Child, DynamicChild, TorchChild
            from $pkg.sub.renamed import thing

            _MOD = "$pkg.sub.mod"
            tool = load_script_module("scripts/tool.py")
            loaded = importlib.import_module("$pkg.sub.mod")
            _TOOL = "scripts/tool.py"
            named_tool = load_script_module(_TOOL)


            def test(monkeypatch, caplog):
                mock.patch("$pkg.sub.mod.helper")
                mock.patch("$pkg.sub.mod.gone_target")
                mock.patch("$pkg.sub.mod.created", create=True)
                mock.patch("$pkg.sub.tool.tool")
                mock.patch(f"{_MOD}.VALUE")
                mock.patch(f"{_MOD}.gone_value")
                mock.patch.object(Child, "inherited")
                mock.patch.object(Child, "field")
                mock.patch.object(Child, "gone_attribute")
                mock.patch.object(TorchChild, "forward")
                mock.patch.object(TorchChild, "train")
                mock.patch.object(DynamicChild, "anything")
                monkeypatch.setattr(mod, "helper", None)
                monkeypatch.setattr(mod, "gone_helper", None)
                monkeypatch.setattr(mod, "absent", None, raising=False)
                monkeypatch.setattr(mod.Child, "method", None)
                monkeypatch.setattr(loaded, "VALUE", 2)
                assert mod.VALUE and mod.__name__ and mod.Op.apply and tool.run and mod.GONE_VALUE and tool.gone_run and named_tool.gone_by_name
                caplog.set_level(10, logger="$pkg.sub.mod")
                caplog.set_level(10, logger="$pkg.sub.gone_logger")  # see $pkg.sub.gone_comment
                # Wraps $pkg/sub/
                # mod.py, and $pkg/sub/
                # wrapped_gone.py.


            def shadowed(mod):
                return mod.local_only
        ''',
        "Dockerfile": """
            RUN python -c "from $pkg.sub.mod import Child, Gone" && cp $pkg/sub/mod.py /opt/
            # Built from $pkg/sub/
            # base.py.
        """,
    }
    write_sample_tree(tmp_path, sample)
    index = RepoIndex(tmp_path)
    scanned = [tmp_path / "sample_test.py", tmp_path / "Dockerfile", tmp_path / spelled("$pkg/sub/__init__.py")]

    findings = [finding for path in scanned for finding in _scan(path, index)]

    found = {(finding.kind, finding.text): finding.why for finding in findings}
    unresolved = {reference for reference, why in found.items() if why}
    assert unresolved == {
        (kind, spelled(text))
        for kind, text in [
            ("path", "$pkg/sub/gone.py"),
            ("path", "$pkg/moved/"),
            ("path", "$pkg/sub/wrapped_gone.py"),
            ("path", "agent-docs/page.md#gone"),
            ("import", "$pkg.sub.renamed.thing"),
            ("import", "$pkg.sub.gone.thing"),
            ("patch target", "$pkg.sub.mod.gone_target"),
            ("patch target", "$pkg.sub.mod.gone_value"),
            ("patch target", "$pkg.sub.mod.Child.gone_attribute"),
            ("patch target", "$pkg.sub.mod.TorchChild.train"),
            ("patch target", "$pkg.sub.mod.gone_helper"),
            ("attribute", "$pkg.sub.mod.GONE_VALUE"),
            ("attribute", "scripts.tool.gone_run"),
            ("attribute", "scripts.tool.gone_by_name"),
            ("name", "$pkg.sub.gone_logger"),
            ("name", "$pkg.sub.gone_comment"),
            ("name", "from $pkg.sub.mod import Gone"),
        ]
    }, unresolved
    resolved = set(found) - unresolved
    assert resolved >= {
        (kind, spelled(text))
        for kind, text in [
            ("path", "$pkg/sub/mod.py"),
            ("path", "$pkg/sub/base.py"),
            ("path", "agent-docs/page.md#section"),
            ("import", "$pkg.sub.mod"),
            ("import", "$pkg.sub.tool.tool"),
            ("import", "$pkg.sub.mod.Child"),
            ("patch target", "$pkg.sub.mod.helper"),
            ("patch target", "$pkg.sub.mod"),
            ("patch target", "$pkg.sub.tool.tool"),
            ("patch target", "$pkg.sub.mod.VALUE"),
            ("patch target", "$pkg.sub.mod.Child.inherited"),
            ("patch target", "$pkg.sub.mod.Child.field"),
            ("patch target", "$pkg.sub.mod.TorchChild.forward"),
            ("patch target", "$pkg.sub.mod.DynamicChild.anything"),
            ("patch target", "$pkg.sub.mod.Child.method"),
            ("attribute", "$pkg.sub.mod.VALUE"),
            ("attribute", "$pkg.sub.mod.__name__"),
            ("attribute", "$pkg.sub.mod.Op.apply"),
            ("attribute", "scripts.tool.run"),
            ("name", "$pkg.sub.mod"),
            ("name", "from $pkg.sub.mod import Child"),
        ]
    }, resolved
    lines = {(finding.kind, finding.text): finding.line for finding in findings if finding.why}
    assert lines[("path", spelled("$pkg/sub/gone.py"))] == 2
    assert lines[("path", spelled("$pkg/sub/wrapped_gone.py"))] == 40


def test_an_import_through_a_reexport_and_a_patch_nothing_reads_are_reported(tmp_path):
    """An import from a module that only re-binds the name fails, as does a patch of a module's own name, by a
    patch call or by assignment, that no function reads when it runs: one read only at import time, here or
    through the module object elsewhere, only by a test, under a local that hides it, or nowhere. A
    defining-module import, a submodule import, a name a function reads (``global`` included, beside a nested
    function or comprehension binding the same name, or through the module object from another module), a
    class attribute and ``patch.dict`` pass."""
    sample = {
        "$pkg/__init__.py": "",
        "$pkg/sub/__init__.py": "",
        "$pkg/sub/defining.py": """
            REGISTRY = {}
            LIMIT = 1


            def helper():
                pass
        """,
        "$pkg/sub/scopes.py": """
            from $pkg.sub.defining import helper


            def other():
                pass


            def outer():
                def inner(value):
                    helper = value
                    return helper

                return helper(), inner


            def comprehension():
                return other(), [other for other in range(3)]
        """,
        "$pkg/sub/attr_only.py": """
            def target():
                pass


            def captured():
                pass
        """,
        "$pkg/sub/attr_user.py": """
            from $pkg.sub import attr_only

            CAPTURED = attr_only.captured


            def run():
                return attr_only.target()
        """,
        "$tests/test_reads_back.py": """
            from $pkg.sub import attr_only


            def test_reads_back():
                assert attr_only.captured
        """,
        "$pkg/sub/user.py": """
            from $pkg.sub.defining import REGISTRY, helper

            AT_IMPORT = helper
            COUNTER = 0


            class Thing:
                def method(self):
                    pass


            def tool():
                pass


            def run():
                global COUNTER
                COUNTER = COUNTER + 1
                return helper(), [tool for tool in REGISTRY]


            def shadowed(tool):
                return tool


            def rebound(value):
                tool = value
                return tool
        """,
        "sample_test.py": """
            from unittest import mock

            from $pkg.sub import attr_only, defining
            from $pkg.sub.defining import helper
            from $pkg.sub.user import REGISTRY, run


            def test(monkeypatch):
                mock.patch("$pkg.sub.user.helper")
                mock.patch("$pkg.sub.user.COUNTER")
                mock.patch("$pkg.sub.user.Thing.method")
                mock.patch.dict("$pkg.sub.defining.REGISTRY", {})
                mock.patch("$pkg.sub.user.AT_IMPORT")
                mock.patch("$pkg.sub.user.tool")
                monkeypatch.setattr(defining, "helper", None)
                mock.patch("$pkg.sub.scopes.helper")
                mock.patch("$pkg.sub.scopes.other")
                mock.patch("$pkg.sub.attr_only.target")
                mock.patch("$pkg.sub.attr_only.captured")
                attr_only.target = None
                defining.LIMIT = 2
        """,
    }
    write_sample_tree(tmp_path, sample)

    findings = _scan(tmp_path / "sample_test.py", RepoIndex(tmp_path))

    failed = {(finding.kind, finding.text) for finding in findings if finding.why}
    assert failed == {
        (kind, spelled(text))
        for kind, text in [
            ("import", "$pkg.sub.user.REGISTRY"),
            ("patch target", "$pkg.sub.user.AT_IMPORT"),
            ("patch target", "$pkg.sub.user.tool"),
            ("patch target", "$pkg.sub.defining.helper"),
            ("patch target", "$pkg.sub.defining.LIMIT"),
            ("patch target", "$pkg.sub.attr_only.captured"),
        ]
    }, findings
    passed = {(finding.kind, finding.text) for finding in findings if not finding.why}
    assert passed >= {
        (kind, spelled(text))
        for kind, text in [
            ("import", "$pkg.sub.defining"),
            ("import", "$pkg.sub.defining.helper"),
            ("import", "$pkg.sub.user.run"),
            ("patch target", "$pkg.sub.user.helper"),
            ("patch target", "$pkg.sub.user.COUNTER"),
            ("patch target", "$pkg.sub.user.Thing.method"),
            ("patch target", "$pkg.sub.defining.REGISTRY"),
            ("patch target", "$pkg.sub.scopes.helper"),
            ("patch target", "$pkg.sub.scopes.other"),
            ("patch target", "$pkg.sub.attr_only.target"),
        ]
    }, findings


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
