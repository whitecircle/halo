"""Four layering rules the tree enforces by construction, checked on the import graph itself.

  * ``src/checkpoint/`` and ``src/models/loading/`` are the SHARDING-AGNOSTIC halves — one artifact
    format and one unsharded load path, shared by the parallel savers/loaders and by the standalone
    ``scripts/`` tools. A ``src.distributed`` or ``torch.distributed`` import there means either a
    collective has moved into a layer whose callers do not enter it in lockstep, or a CPU-only
    conversion tool now drags the DeepEP dispatcher and the whole family roster into its process.
  * ``src/distributed/loading/`` dispatches to every parallelism implementation, so nothing else in
    ``src/distributed`` imports it: an import back from the EP/CP/TP/PP packages or the checkpoint
    layer closes a package cycle, and what both sides need belongs in the lower package.
  * more generally, the subpackages and top-level modules of ``src/distributed`` import each other
    acyclically, so each sits at one layer (the savers above the parallelism packages they gather).
  * every toolkit env knob (``HALO_``/``DIST_``/``VLLM_``/``SGLANG_``/``NVLINK_``) is read through
    ``src/env.py``, in ``src/`` and ``scripts/`` alike. A raw
    ``os.environ`` read elsewhere carries its own default, and two defaults for one knob differ.

Import-graph checks, not text matches: CLAUDE.md bans function-local imports, so a module-level
import is the only way a call site reaches one of these.

    python tests/cpu/conventions/test_import_layering.py
"""

import ast
from collections import defaultdict
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"

# The two sharding-agnostic trees, and what they may not reach.
_AGNOSTIC_DIRS = ("checkpoint", "models/loading")
_FORBIDDEN_ROOTS = ("src.distributed", "torch.distributed")

# The package above every parallelism implementation, and the tree that may not import it back.
_DISPATCH_PACKAGE = "src.distributed.loading"
_DISTRIBUTED = _SRC / "distributed"
_BELOW_DISPATCH_FILES = sorted(
    path for path in _DISTRIBUTED.rglob("*.py") if not path.is_relative_to(_DISTRIBUTED / "loading")
)

# EP_* stays out: EP_DISABLE_GIN / EP_SUPPRESS_NCCL_CHECK are DeepEP's own vars, read raw at
# their owner by design (src/env.py docstring).
_ENV_PREFIXES = ("HALO_", "DIST_", "VLLM_", "SGLANG_", "NVLINK_")
_ENV_OWNER = _SRC / "env.py"

_AGNOSTIC_FILES = sorted(path for directory in _AGNOSTIC_DIRS for path in (_SRC / directory).rglob("*.py"))
_SRC_FILES = sorted(_SRC.rglob("*.py"))
_ENV_SWEPT_FILES = _SRC_FILES + sorted((_REPO_ROOT / "scripts").rglob("*.py"))


def _imported_modules(path: Path) -> set[str]:
    """Every module name ``path`` imports, as written (``import a.b`` and ``from a.b import c``)."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules.add(node.module)
    return modules


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


def test_the_sweep_sees_both_agnostic_trees():
    """Anti-vacuity: an empty or shrunken sweep would pass every assertion below."""
    assert len(_AGNOSTIC_FILES) > 10, f"the sharding-agnostic sweep collapsed to {len(_AGNOSTIC_FILES)} files"
    for directory in _AGNOSTIC_DIRS:
        assert any(str(path).startswith(str(_SRC / directory)) for path in _AGNOSTIC_FILES), directory


@pytest.mark.parametrize("path", _AGNOSTIC_FILES, ids=lambda p: str(p.relative_to(_SRC.parent)))
def test_the_sharding_agnostic_layer_imports_no_distributed_module(path):
    offending = sorted(
        module
        for module in _imported_modules(path)
        if any(module == root or module.startswith(f"{root}.") for root in _FORBIDDEN_ROOTS)
    )
    assert not offending, (
        f"{path.relative_to(_SRC.parent)} imports {offending}. This layer is shared by the standalone "
        f"conversion tools and by every parallel save/load path, so it can hold no collective and no "
        f"parallelism knowledge — take the verdict as a parameter from the caller that owns it, or "
        f"move the function to the caller's side."
    )


def test_the_sweep_sees_every_parallelism_package():
    """Anti-vacuity: the packages the dispatcher reaches into must all be in the sweep."""
    swept = {path.relative_to(_DISTRIBUTED).parts[0] for path in _BELOW_DISPATCH_FILES}
    assert {"expert_parallel", "context_parallel", "tensor_parallel", "pipeline_parallel", "checkpoint"} <= swept


@pytest.mark.parametrize("path", _BELOW_DISPATCH_FILES, ids=lambda p: str(p.relative_to(_SRC.parent)))
def test_no_distributed_package_imports_the_loader_dispatch(path):
    offending = sorted(
        module
        for module in _imported_modules(path)
        if module == _DISPATCH_PACKAGE or module.startswith(f"{_DISPATCH_PACKAGE}.")
    )
    assert not offending, (
        f"{path.relative_to(_SRC.parent)} imports {offending}. src/distributed/loading/ dispatches to "
        f"every parallelism package, so an import back closes a package cycle — move what both sides "
        f"need into the lower package (the FP32-master selector and restore live in expert_parallel/)."
    )


def _distributed_package_graph() -> dict[str, set[str]]:
    """``src/distributed`` subpackage / top-level module → the ones it imports, as written."""
    graph: dict[str, set[str]] = defaultdict(set)
    for path in _DISTRIBUTED.rglob("*.py"):
        node = path.relative_to(_DISTRIBUTED).parts[0].removesuffix(".py")
        if node == "__init__":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for statement in ast.walk(tree):
            if isinstance(statement, ast.Import):
                modules = [alias.name for alias in statement.names]
            elif isinstance(statement, ast.ImportFrom) and statement.module and not statement.level:
                # ``from src.distributed import runtime`` names the node in the imported names.
                if statement.module == "src.distributed":
                    modules = [f"src.distributed.{alias.name}" for alias in statement.names]
                else:
                    modules = [statement.module]
            else:
                continue
            for module in modules:
                parts = module.split(".")
                if parts[:2] == ["src", "distributed"] and len(parts) > 2 and parts[2] != node:
                    graph[node].add(parts[2])
    return graph


def _a_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    """One import cycle of ``graph`` as its node path, or ``None`` when it is acyclic."""
    visiting: list[str] = []
    done: set[str] = set()

    def visit(node: str) -> list[str] | None:
        if node in visiting:
            return visiting[visiting.index(node) :] + [node]
        if node in done:
            return None
        visiting.append(node)
        for target in sorted(graph.get(node, ())):
            if cycle := visit(target):
                return cycle
        visiting.pop()
        done.add(node)
        return None

    return next((cycle for node in sorted(graph) if (cycle := visit(node))), None)


def test_the_cycle_finder_finds_a_cycle():
    """Anti-vacuity for the acyclicity rule below."""
    assert _a_cycle({"a": {"b"}, "b": {"c"}, "c": {"a"}}) == ["a", "b", "c", "a"]
    assert _a_cycle({"a": {"b"}, "b": {"c"}}) is None


def test_the_distributed_packages_import_each_other_acyclically():
    graph = _distributed_package_graph()
    assert {"loading", "checkpoint", "expert_parallel", "pipeline_parallel"} <= set(graph), sorted(graph)
    cycle = _a_cycle(graph)
    assert cycle is None, (
        f"src/distributed import cycle: {' -> '.join(cycle)}. Each package must sit at one layer — move "
        f"what both sides need into the lower one rather than importing back up."
    )


@pytest.mark.parametrize("path", _ENV_SWEPT_FILES, ids=lambda p: str(p.relative_to(_SRC.parent)))
def test_toolkit_env_vars_are_read_only_through_the_env_helpers(path):
    if path == _ENV_OWNER:
        return
    offending = sorted(set(_env_literals_read_directly(path)))
    assert not offending, (
        f"{path.relative_to(_SRC.parent)} reads {offending} from the environment directly; route it "
        f"through src/env.py (env_flag / env_int / env_float / env_str), which owns the default."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
