"""The import layering of ``src/``, read off the import graph: the sharding-agnostic trees reach no
``src.distributed``, ``torch.distributed`` or loader dispatch, no parallelism package imports the dispatch, the
packages import their members acyclically, toolkit env knobs go through ``src/env.py``, and every import follows the
final layer table ``_LAYERS``, which agent-docs/reference/architecture.md renders. ``TYPE_CHECKING`` imports and the
package's bootstrap ``__init__`` are exempt; a function-local import counts like any other."""

import ast
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import NamedTuple

import pytest

from scripts.docs.references import PACKAGE, RepoIndex, dotted_name
from tests.common.repo import REPO_ROOT
from tests.common.sample_tree import write_sample_tree

_SRC = REPO_ROOT / PACKAGE
_INDEX = RepoIndex(REPO_ROOT)
_ARCHITECTURE_DOC = REPO_ROOT / "agent-docs" / "reference" / "architecture.md"

# The package above every parallelism implementation, which nothing in src/distributed imports.
_DISPATCH_PACKAGE = "model_loading.parallel"
_DISPATCH_DIR = _SRC.joinpath(*_DISPATCH_PACKAGE.split("."))
# The collective half of the checkpoint tree, which the agnostic sweep skips and the dispatch sweep covers.
_CHECKPOINT_PARALLEL = "checkpoint.parallel"

# The sharding-agnostic trees and what they may not reach.
_AGNOSTIC_DIRS = ("checkpoint", "model_loading", "models")
_FORBIDDEN_ROOTS = (f"{PACKAGE}.distributed", "torch.distributed", f"{PACKAGE}.{_DISPATCH_PACKAGE}")

# EP_* stays out: EP_DISABLE_GIN / EP_SUPPRESS_NCCL_CHECK are DeepEP's own vars, read raw at
# their owner by design (src/env.py docstring).
_ENV_PREFIXES = ("HALO_", "DIST_", "VLLM_", "SGLANG_", "NVLINK_")
_ENV_OWNER = _SRC / "env.py"

_TYPE_CHECKING_GUARDS = ("TYPE_CHECKING", "typing.TYPE_CHECKING")

# The final layer table, bottom to top: a unit imports only units of lower rows.
_LAYERS = (
    ("env",),
    ("log",),
    ("hardware", "config_validation", "subclass_registry", "cli"),
    ("runtime", "kernels", "inference", "checkpoint.model_free"),
    ("models", "diagnostics", "rewards"),
    ("model_loading", "optimizers", "weight_sync"),
    ("checkpoint.model_aware", "data"),
    ("distributed", "environments"),
    ("model_loading.parallel", "checkpoint.parallel", "callbacks", "configs"),
    ("trainers",),
    ("entrypoints",),
)
_LEVEL = {unit: level for level, row in enumerate(_LAYERS) for unit in row}
# The src/checkpoint/ modules that read no model; the rest of the package reads one.
_MODEL_FREE_CHECKPOINT_MODULES = frozenset({"format", "atomic", "reading", "model_card"})
# What the standalone tools load, and what it may not reach: no process group, collective or rollout sync.
_TOOL_SAFE = frozenset(
    {"models", "model_loading", "checkpoint.model_free", "checkpoint.model_aware", "kernels", "inference"}
)
_TOOL_UNSAFE = frozenset({"runtime", "distributed", "weight_sync", "model_loading.parallel", "checkpoint.parallel"})
# The one torch.distributed import a tool-safe module makes: the DTensor type a low-precision Linear checks for.
_TOOL_SAFE_TORCH_DISTRIBUTED = {"kernels.lowp.linear": "torch.distributed.tensor"}
# Importing any module of the package runs its bootstrap, which imports torch; these units import nothing from it.
_PACKAGE_FREE_UNITS = frozenset({"cli"})
# Per package, the order its members import in, bottom to top: a member imports only members of lower rows.
_MEMBER_ORDERS = {
    "distributed": (
        ("group_layout", "mesh", "nvlink", "grad_reduce"),
        ("fsdp", "tensor_parallel", "expert_parallel"),
        ("rank_uniform_env", "context_parallel"),
        ("parallelism_rules",),
        ("parallelism_config",),
        ("pipeline_parallel",),
    ),
    "distributed.expert_parallel": (
        ("config",),
        ("deepep",),
        ("moe_layer",),
        ("families",),
        ("parallelize",),
        ("lazy_plan",),
        ("expert_fuser",),
    ),
    "distributed.expert_parallel.deepep": (("settings",), ("capacity",), ("backends",), ("dispatcher",)),
    "distributed.expert_parallel.moe_layer": (
        ("compute", "routing", "grad_sync", "expert_lora", "gather", "balancing_bias"),
        ("layer",),
        ("shared_experts_layer",),
    ),
}
_MEMBER_ROW = {
    package: {member: row for row, members in enumerate(rows) for member in members}
    for package, rows in _MEMBER_ORDERS.items()
}
# ``import deep_ep`` runs cuInit, which latches the CUDA environment: one module makes it, inside a function.
_DEEP_EP_IMPORTER = "distributed.expert_parallel.deepep.backends"
# Modules only their own package imports; everything else reaches their behavior through a layer instance.
_PACKAGE_INTERNAL = {
    f"distributed.expert_parallel.moe_layer.{name}": "distributed.expert_parallel"
    for name in ("compute", "routing", "grad_sync", "gather", "balancing_bias")
}
# Where each module that has not moved yet lands, by its longest entry: today's dotted path under the package
# root -> the final one. A move deletes its entry; a module staying in its unit and member needs none.
_PENDING_MOVES = {
    "distributed.nccl": "weight_sync.transport",
    "distributed.nccl._nccl_wrapper": "weight_sync.transport.nccl_library",
}


class _Import(NamedTuple):
    """One import a package module makes outside an ``if TYPE_CHECKING:`` block."""

    file: str  # relative to the index root
    importer: str  # dotted under the package root; "" for the package's own __init__
    line: int
    # Absolute and dotted: a first-party ``from`` names the submodule it loads, else its source module; a
    # third-party one names ``source.name``, which may be a module or an attribute.
    module: str
    lazy: bool  # made inside a function body


class _Break(NamedTuple):
    """One break of the final layer table."""

    rule: str
    importer: str  # a final path; a cycle's members
    imported: str  # a final path or a third-party module; "" for a cycle
    location: str  # "file:line"; "" for a cycle
    detail: str


def _module_name(package_dir: Path, path: Path) -> str:
    """``path``'s dotted name under the package root; "" for the root ``__init__``."""
    parts = path.relative_to(package_dir).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _file_imports(index: RepoIndex, path: Path) -> list[_Import]:
    file = path.relative_to(index.root).as_posix()
    importer = _module_name(index.root / PACKAGE, path)
    found: list[_Import] = []

    def imported(source: str, name: str) -> str:
        if source.partition(".")[0] == PACKAGE:
            return index.from_import_module(source, name)
        return f"{source}.{name}"

    def visit(node: ast.AST, lazy: bool) -> None:
        if isinstance(node, ast.If) and dotted_name(node.test) in _TYPE_CHECKING_GUARDS:
            for statement in node.orelse:
                visit(statement, lazy)
            return
        if isinstance(node, ast.Import):
            found.extend(_Import(file, importer, node.lineno, alias.name, lazy) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            source = index.import_source(node, path)
            found.extend(
                _Import(file, importer, node.lineno, imported(source, alias.name), lazy) for alias in node.names
            )
        lazy = lazy or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for child in ast.iter_child_nodes(node):
            visit(child, lazy)

    visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)), False)
    return found


def _package_files(index: RepoIndex) -> list[Path]:
    return sorted((index.root / PACKAGE).rglob("*.py"))


def _package_imports(index: RepoIndex) -> list[_Import]:
    """Every import the package's modules make."""
    return [found for path in _package_files(index) for found in _file_imports(index, path)]


def _in_package(module: str) -> str | None:
    """``module``'s dotted name under the package root, or None outside the package."""
    if module == PACKAGE:
        return ""
    return module.removeprefix(f"{PACKAGE}.") if module.startswith(f"{PACKAGE}.") else None


def _is_within(module: str, package: str) -> bool:
    return module == package or module.startswith(f"{package}.")


def _member(module: str, package: str) -> str | None:
    """The member of ``package`` that ``module`` is or sits in, both dotted under the package root; None
    when ``module`` is not inside ``package``."""
    if not module.startswith(f"{package}."):
        return None
    return module.removeprefix(f"{package}.").partition(".")[0]


def _final_path(module: str, moves: Mapping[str, str]) -> str:
    """Where ``module`` (dotted under the package root) lands, through the longest entry of ``moves`` naming
    it or a package above it."""
    parts = module.split(".")
    for end in range(len(parts), 0, -1):
        prefix = ".".join(parts[:end])
        if prefix in moves:
            return ".".join([moves[prefix], *parts[end:]])
    return module


_IMPORTS = _package_imports(_INDEX)
_AGNOSTIC_FILES = sorted(
    path
    for directory in _AGNOSTIC_DIRS
    for path in (_SRC / directory).rglob("*.py")
    if not path.is_relative_to(_DISPATCH_DIR) and not _is_within(_module_name(_SRC, path), _CHECKPOINT_PARALLEL)
)
_ENV_SWEPT_FILES = _package_files(_INDEX) + sorted((REPO_ROOT / "scripts").rglob("*.py"))


def _imports_of(path: Path) -> list[_Import]:
    file = path.relative_to(REPO_ROOT).as_posix()
    return [found for found in _IMPORTS if found.file == file]


def _env_literals_read_directly(path: Path) -> list[str]:
    """Toolkit env-var literals this file passes to ``os.environ.get`` / ``os.getenv`` / ``os.environ[...]``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []

    def _is_toolkit_literal(node) -> bool:
        return isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(_ENV_PREFIXES)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            reads_env = (isinstance(func, ast.Attribute) and func.attr in ("getenv", "get", "pop", "setdefault")) or (
                isinstance(func, ast.Name) and func.id == "getenv"
            )
            if reads_env:
                found += [arg.value for arg in node.args if _is_toolkit_literal(arg)]
        elif isinstance(node, ast.Subscript) and _is_toolkit_literal(node.slice):
            found.append(node.slice.value)
    return found


def _member_graph(imports: Iterable[_Import], package: str, node=None) -> dict[str, set[str]]:
    """Each member of ``package`` (dotted under the package root) -> the other members it imports; ``node`` maps
    a module to its graph node in place of its member."""
    node = node or (lambda module: _member(module, package))
    graph: dict[str, set[str]] = defaultdict(set)
    for found in imports:
        importer, imported = node(found.importer), node(_in_package(found.module) or "")
        if importer and imported and importer != imported:
            graph[importer].add(imported)
    return graph


def _reachable(graph: Mapping[str, Iterable[str]], start: str) -> set[str]:
    """The nodes one or more edges of ``graph`` lead to from ``start``."""
    reached: set[str] = set()
    pending = list(graph.get(start, ()))
    while pending:
        node = pending.pop()
        if node not in reached:
            reached.add(node)
            pending.extend(graph.get(node, ()))
    return reached


def _cycles(graph: Mapping[str, Iterable[str]]) -> list[list[str]]:
    """Every import cycle of ``graph``: the sorted members of each strongly connected component that holds one."""
    reach = {node: _reachable(graph, node) for node in graph}
    components = {frozenset(other for other in reach[node] if node in reach.get(other, ())) for node in graph}
    return sorted(sorted(component) for component in components if component)


def _unit(module: str) -> str:
    """The layer-table unit of the final path ``module``."""
    top, _, rest = module.partition(".")
    member = rest.partition(".")[0]
    if top in ("checkpoint", "model_loading") and member == "parallel":
        return f"{top}.parallel"
    if top == "checkpoint":
        is_model_free = not member or member in _MODEL_FREE_CHECKPOINT_MODULES
        return "checkpoint.model_free" if is_model_free else "checkpoint.model_aware"
    return top


def _misplaced(module: str) -> str | None:
    """Why the final path ``module`` has no place in the table, or None."""
    if _unit(module) not in _LEVEL:
        return f"its unit {_unit(module)!r} is in no row of _LAYERS"
    for package, rows in _MEMBER_ROW.items():
        member = _member(module, package)
        if member is not None and member not in rows:
            return f"{member!r} is in no row of the {package} order"
    return None


def _parallelism_package(module: str) -> str | None:
    """The parallelism package ``module`` (dotted under the package root) sits in: a member of ``distributed``, or
    the checkpoint tree's collective half; None outside them."""
    if _is_within(_final_path(module, _PENDING_MOVES), _CHECKPOINT_PARALLEL):
        return _CHECKPOINT_PARALLEL
    return _member(module, "distributed")


def _lands_below_dispatch(path: Path) -> bool:
    """Whether a parallelism-package module lands at or below the dispatch's row, outside the dispatch."""
    module = _module_name(_SRC, path)
    final = _final_path(module, _PENDING_MOVES)
    return (
        _parallelism_package(module) is not None
        and _LEVEL.get(_unit(final), 0) <= _LEVEL[_DISPATCH_PACKAGE]
        and not _is_within(final, _DISPATCH_PACKAGE)
    )


_BELOW_DISPATCH_FILES = [path for path in _package_files(_INDEX) if _lands_below_dispatch(path)]


def _edge_breaks(location: str, source: str, target: str) -> list[_Break]:
    """The rules a first-party import from final path ``source`` to final path ``target`` breaks."""
    breaks = []
    source_unit, target_unit = _unit(source), _unit(target)
    if source_unit != target_unit and _LEVEL[target_unit] >= _LEVEL[source_unit]:
        detail = f"{source} ({source_unit}, L{_LEVEL[source_unit]}) imports {target} ({target_unit}, L{_LEVEL[target_unit]})"
        breaks.append(_Break("layer order", source, target, location, detail))
    if source_unit in _TOOL_SAFE and target_unit in _TOOL_UNSAFE:
        detail = f"{source} ({source_unit}) imports {target} ({target_unit})"
        breaks.append(_Break("tool-safe", source, target, location, detail))
    for package, rows in _MEMBER_ROW.items():
        source_member, target_member = _member(source, package), _member(target, package)
        if source_member is None or target_member is None or source_member == target_member:
            continue
        if rows[target_member] >= rows[source_member]:
            detail = f"in {package}, {source_member} imports {target_member}"
            breaks.append(_Break("member order", source, target, location, detail))
    owner = _PACKAGE_INTERNAL.get(target)
    if owner and _member(source, owner) is None:
        detail = f"{source} imports {target}, internal to {owner}"
        breaks.append(_Break("package-internal", source, target, location, detail))
    return breaks


def _third_party_breaks(location: str, source: str, module: str, lazy: bool) -> list[_Break]:
    """The rules an import of the third-party ``module`` from final path ``source`` breaks."""
    breaks = []
    if _is_within(module, "deep_ep"):
        if source != _DEEP_EP_IMPORTER:
            detail = f"{source} imports {module}; only {_DEEP_EP_IMPORTER} does"
            breaks.append(_Break("deep_ep", source, module, location, detail))
        elif not lazy:
            breaks.append(_Break("deep_ep", source, module, location, f"{source} imports {module} at module level"))
    allowed = _TOOL_SAFE_TORCH_DISTRIBUTED.get(source)
    if (
        _is_within(module, "torch.distributed")
        and _unit(source) in _TOOL_SAFE
        and not (allowed and _is_within(module, allowed))
    ):
        detail = f"{source} ({_unit(source)}) imports {module}"
        breaks.append(_Break("tool-safe", source, module, location, detail))
    return breaks


def _layer_table_breaks(imports: Iterable[_Import], moves: Mapping[str, str]) -> list[_Break]:
    """Every break of the final layer table by ``imports``, each module placed through ``moves``."""
    breaks = []
    graph: dict[str, set[str]] = defaultdict(set)
    for found in imports:
        if not found.importer:
            continue
        location = f"{found.file}:{found.line}"
        source = _final_path(found.importer, moves)
        target = _in_package(found.module)
        if target is None:
            breaks += _third_party_breaks(location, source, found.module, found.lazy)
            continue
        target = _final_path(target, moves)
        if _unit(source) in _PACKAGE_FREE_UNITS:
            imported = target or PACKAGE
            breaks.append(_Break("package-free", source, imported, location, f"{source} imports {imported}"))
        if not target or target == source:
            continue
        unplaced = [f"{module}: {why}" for module in (source, target) if (why := _misplaced(module))]
        if unplaced:
            breaks += [_Break("unplaced", source, target, location, detail) for detail in unplaced]
            continue
        graph[source].add(target)
        breaks += _edge_breaks(location, source, target)
    for cycle in _cycles(graph):
        breaks.append(_Break("module cycle", ", ".join(cycle), "", "", "the module graph must be acyclic"))
    # One break per import line, however many names it takes.
    return sorted(set(breaks))


def _report(breaks: Iterable[_Break]) -> str:
    return "\n".join(f"{found.location or found.importer}: {found.rule}: {found.detail}" for found in breaks)


def test_the_sweep_sees_every_agnostic_tree():
    """Anti-vacuity: an empty or shrunken sweep would pass every assertion below."""
    assert len(_AGNOSTIC_FILES) > 10, f"the sharding-agnostic sweep collapsed to {len(_AGNOSTIC_FILES)} files"
    assert _DISPATCH_DIR.is_dir(), f"src/{_DISPATCH_PACKAGE.replace('.', '/')} moved; the sweep excludes it by path"
    for directory in _AGNOSTIC_DIRS:
        assert any(path.is_relative_to(_SRC / directory) for path in _AGNOSTIC_FILES), directory


@pytest.mark.parametrize("path", _AGNOSTIC_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_the_sharding_agnostic_layer_imports_no_distributed_module(path):
    offending = sorted(
        {
            found.module
            for found in _imports_of(path)
            if any(_is_within(found.module, root) for root in _FORBIDDEN_ROOTS)
        }
    )
    assert not offending, (
        f"{path.relative_to(REPO_ROOT)} imports {offending}. This layer is shared by the standalone "
        f"conversion tools and by every parallel save/load path, so it can hold no collective and no "
        f"parallelism knowledge — take the verdict as a parameter from the caller that owns it, or "
        f"move the function to the caller's side."
    )


def test_the_sweep_sees_the_dispatch_and_every_parallelism_package():
    """Anti-vacuity: a moved dispatch package leaves nothing to import, and the packages it reaches into must
    all be in the sweep."""
    assert _INDEX.module_source(f"{PACKAGE}.{_DISPATCH_PACKAGE}") is not None, f"no {_DISPATCH_PACKAGE} package"
    swept = {_parallelism_package(_module_name(_SRC, path)) for path in _BELOW_DISPATCH_FILES}
    assert {
        "expert_parallel",
        "context_parallel",
        "tensor_parallel",
        "pipeline_parallel",
        _CHECKPOINT_PARALLEL,
    } <= swept


@pytest.mark.parametrize("path", _BELOW_DISPATCH_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_distributed_package_imports_the_loader_dispatch(path):
    offending = sorted(
        {found.module for found in _imports_of(path) if _is_within(_in_package(found.module) or "", _DISPATCH_PACKAGE)}
    )
    assert not offending, (
        f"{path.relative_to(REPO_ROOT)} imports {offending}. src/model_loading/parallel/ dispatches to "
        f"every parallelism package, so an import back closes a package cycle — move what both sides "
        f"need into the lower package (the expert-key plan and fuser live in expert_parallel/)."
    )


def test_the_cycle_finder_finds_every_cycle():
    """Anti-vacuity for the acyclicity rules: each cycle is reported, not only the first one found."""
    graph = {"a": {"b"}, "b": {"c", "d"}, "c": {"a"}, "d": {"e"}, "e": {"d"}, "f": {"a"}}

    assert _cycles(graph) == [["a", "b", "c"], ["d", "e"]]
    assert _cycles({"a": {"b"}, "b": {"c"}}) == []


def test_a_type_checking_import_adds_no_edge_and_a_runtime_one_does(tmp_path):
    """Anti-vacuity for the exemption: an annotation-only import adds no edge, while a module-level import in
    the same file, and one inside a function, still do."""
    write_sample_tree(
        tmp_path,
        {
            "$pkg/__init__.py": "",
            "$pkg/tree/__init__.py": "",
            "$pkg/tree/a.py": """
                from typing import TYPE_CHECKING

                from $pkg.tree import b

                if TYPE_CHECKING:
                    from $pkg.tree.c import Annotation


                def lazy():
                    import $pkg.tree.d
            """,
            "$pkg/tree/b.py": "",
            "$pkg/tree/c.py": "Annotation = int\n",
            "$pkg/tree/d.py": "",
        },
    )

    assert _member_graph(_package_imports(RepoIndex(tmp_path)), "tree") == {"a": {"b", "d"}}


@pytest.mark.parametrize(
    ("package", "node", "required"),
    [
        ("distributed", _parallelism_package, {_CHECKPOINT_PARALLEL, "expert_parallel", "pipeline_parallel"}),
        (_DISPATCH_PACKAGE, None, {"loader", "frozen_models"}),
    ],
)
def test_the_packages_import_their_members_acyclically(package, node, required):
    graph = _member_graph(_IMPORTS, package, node)
    assert required <= set(graph), sorted(graph)
    cycles = _cycles(graph)
    assert not cycles, (
        f"src/{package.replace('.', '/')} import cycles: {cycles}. Each member must sit at one layer — move what "
        f"both sides need into the lower one rather than importing back up."
    )


@pytest.mark.parametrize("path", _ENV_SWEPT_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_toolkit_env_vars_are_read_only_through_the_env_helpers(path):
    if path == _ENV_OWNER:
        return
    offending = sorted(set(_env_literals_read_directly(path)))
    assert not offending, (
        f"{path.relative_to(REPO_ROOT)} reads {offending} from the environment directly; route it "
        f"through src/env.py (env_flag / env_int / env_float / env_str), which owns the default."
    )


def test_a_pending_move_places_a_module_by_its_longest_entry():
    moves = {"old": "new", "old.kept_apart": "elsewhere.module"}

    assert _final_path("old", moves) == "new"
    assert _final_path("old.sub.module", moves) == "new.sub.module"
    assert _final_path("old.kept_apart.inner", moves) == "elsewhere.module.inner"
    assert _final_path("older", moves) == "older"


def test_a_module_outside_every_row_has_no_place():
    assert _misplaced("args.sft_args") is not None
    assert _misplaced("distributed.new_member") is not None
    assert _misplaced("distributed.expert_parallel.deepep.extra") is not None
    assert _misplaced("distributed.expert_parallel.moe_layer.layer") is None
    assert _misplaced("checkpoint.parallel.save") is None
    assert [_unit(module) for module in ("checkpoint.format", "checkpoint.adapters", "model_loading.parallel.ep")] == [
        "checkpoint.model_free",
        "checkpoint.model_aware",
        "model_loading.parallel",
    ]


def test_each_break_of_the_layer_table_is_reported(tmp_path):
    """Anti-vacuity for the report: a sample package breaking each rule, beside imports each rule allows,
    yields exactly those breaks."""
    write_sample_tree(
        tmp_path,
        {
            "$pkg/__init__.py": """
                import $pkg.trainers.loop
                from $pkg import env
            """,
            "$pkg/env.py": "",
            "$pkg/cli.py": """
                import $pkg
                import $pkg.env
            """,
            "$pkg/old_ranks.py": "",
            "$pkg/kernels/__init__.py": "",
            "$pkg/kernels/dispatch.py": "import $pkg.env\n",
            "$pkg/kernels/fused.py": """
                def load():
                    import deep_ep
            """,
            "$pkg/kernels/first.py": "import $pkg.kernels.second\n",
            "$pkg/kernels/second.py": "import $pkg.kernels.first\n",
            "$pkg/kernels/third.py": "import $pkg.kernels.fourth\n",
            "$pkg/kernels/fourth.py": "import $pkg.kernels.third\n",
            "$pkg/kernels/lowp/__init__.py": "",
            "$pkg/kernels/lowp/linear.py": "from torch.distributed.tensor import DTensor\n",
            "$pkg/runtime/__init__.py": "",
            "$pkg/runtime/ranks.py": "import $pkg.kernels.dispatch\n",
            "$pkg/models/__init__.py": "",
            "$pkg/models/geometry.py": """
                import torch.distributed

                import $pkg.old_ranks
            """,
            "$pkg/models/typing_only.py": """
                from typing import TYPE_CHECKING

                from $pkg.data import rows

                if TYPE_CHECKING:
                    from $pkg.trainers.loop import Loop
            """,
            "$pkg/data/__init__.py": "",
            "$pkg/data/rows.py": "",
            "$pkg/trainers/__init__.py": "",
            "$pkg/trainers/loop.py": "from $pkg.distributed.expert_parallel.moe_layer import routing\n",
            "$pkg/distributed/__init__.py": "",
            "$pkg/distributed/mesh.py": "import $pkg.distributed.fsdp\n",
            "$pkg/distributed/fsdp.py": "",
            "$pkg/distributed/expert_parallel/__init__.py": "",
            "$pkg/distributed/expert_parallel/config.py": "import $pkg.distributed.expert_parallel.moe_layer.layer\n",
            "$pkg/distributed/expert_parallel/deepep/__init__.py": "",
            "$pkg/distributed/expert_parallel/deepep/backends.py": """
                import deep_ep


                def load():
                    import deep_ep
            """,
            "$pkg/distributed/expert_parallel/moe_layer/__init__.py": "",
            "$pkg/distributed/expert_parallel/moe_layer/layer.py": """
                import $pkg.distributed.expert_parallel.deepep.backends
                import $pkg.distributed.expert_parallel.moe_layer.gather
            """,
            "$pkg/distributed/expert_parallel/moe_layer/gather.py": "",
            "$pkg/distributed/expert_parallel/moe_layer/routing.py": (
                "import $pkg.distributed.expert_parallel.moe_layer.layer\n"
            ),
        },
    )

    breaks = _layer_table_breaks(_package_imports(RepoIndex(tmp_path)), {"old_ranks": "runtime.old_ranks"})

    assert Counter((found.rule, found.importer) for found in breaks) == Counter(
        {
            ("package-free", "cli"): 2,
            ("deep_ep", "kernels.fused"): 1,
            ("deep_ep", "distributed.expert_parallel.deepep.backends"): 1,
            ("layer order", "runtime.ranks"): 1,
            ("layer order", "models.typing_only"): 1,
            ("tool-safe", "models.geometry"): 2,
            ("package-internal", "trainers.loop"): 1,
            ("member order", "distributed.mesh"): 1,
            ("member order", "distributed.expert_parallel.config"): 1,
            ("member order", "distributed.expert_parallel.moe_layer.routing"): 1,
            ("module cycle", "kernels.first, kernels.second"): 1,
            ("module cycle", "kernels.fourth, kernels.third"): 1,
        }
    ), _report(breaks)


def test_every_module_has_a_place_in_the_final_table():
    """Each module lands in a unit of the table and, inside an ordered package, in one of its rows; a module
    the table cannot place would drop out of every rule."""
    unplaced = {
        module: why
        for path in _package_files(_INDEX)
        if (module := _module_name(_SRC, path)) and (why := _misplaced(_final_path(module, _PENDING_MOVES)))
    }
    assert not unplaced, f"place these through _LAYERS, _MEMBER_ORDERS or _PENDING_MOVES: {unplaced}"


def test_every_pending_move_is_still_pending():
    """A landed move leaves an entry naming a module that no longer exists."""
    landed = sorted(key for key in _PENDING_MOVES if _INDEX.module_source(f"{PACKAGE}.{key}") is None)
    assert not landed, f"these modules moved; drop their _PENDING_MOVES entries: {landed}"
    assert not [key for key, final in _PENDING_MOVES.items() if key == final]


def test_the_layer_table_sweep_sees_every_unit():
    """Anti-vacuity: a unit no module lands in is a misspelled row, and a collapsed import graph checks nothing."""
    placed = {_unit(_final_path(_module_name(_SRC, path), _PENDING_MOVES)) for path in _package_files(_INDEX)}
    assert not set(_LEVEL) - placed, f"no module lands in {sorted(set(_LEVEL) - placed)}"
    first_party = [found for found in _IMPORTS if found.importer and _in_package(found.module)]
    assert len(first_party) > 2000, f"only {len(first_party)} first-party imports"


def test_imports_follow_the_final_layer_table():
    breaks = _layer_table_breaks(_IMPORTS, _PENDING_MOVES)
    assert not breaks, f"these imports break the final layer table:\n{_report(breaks)}"


def test_architecture_doc_renders_the_layer_table():
    """The ``L<n>  unit, ...`` block under the Layer table heading of the architecture page is ``_LAYERS``."""
    section = _ARCHITECTURE_DOC.read_text(encoding="utf-8").partition("\n## Layer table\n")[2].partition("\n## ")[0]
    block = section.partition("```text\n")[2].partition("```")[0]
    rows = [re.fullmatch(r"L(\d+)\s+(.+)", line.strip()) for line in block.splitlines() if line.strip()]
    assert rows and all(rows), f"{_ARCHITECTURE_DOC.name} has no Layer table block of `L<n>  unit, ...` lines"
    assert [int(row[1]) for row in rows] == list(range(len(rows))), (
        "the documented levels must run L0, L1, ... in order"
    )
    assert tuple(tuple(re.split(r",\s*", row[2])) for row in rows) == _LAYERS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
