"""AST sweeps over the repo's own source for the load-coverage tests: calls read off syntax, so a docstring mention is
not a call."""

import ast
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from halo.checkpoint.tools.offline_export import load_task_model
from halo.model_loading.checkpoint_coverage import from_pretrained_verified
from halo.model_loading.model_preparation import auto_load_model
from tests.common.repo import REPO_ROOT

# A model build is one of these, a SentenceTransformer from a path, or a factory on any class but tokenizer, processor
# or config.
TOOLKIT_LOAD_CALLS = frozenset({from_pretrained_verified.__name__, auto_load_model.__name__, load_task_model.__name__})
MODEL_LOAD_CALLS = TOOLKIT_LOAD_CALLS | {"SentenceTransformer"}
MODEL_FACTORIES = frozenset({"from_pretrained", "from_config", "_from_config"})
NON_MODEL_CLASSES = frozenset({"AutoTokenizer", "AutoProcessor", "AutoConfig", "GenerationConfig", "PeftConfig"})

Function = ast.FunctionDef | ast.AsyncFunctionDef


def imports_name(path: Path, name: str) -> bool:
    """Whether the module at ``path`` binds ``name`` at import; function-local imports are banned, so that is how a
    call site reaches a symbol."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        isinstance(node, ast.ImportFrom | ast.Import)
        and any(alias.name == name or alias.asname == name for alias in node.names)
        for node in ast.walk(tree)
    )


def _imported_package_files(path: str, package: str) -> set[str]:
    """Repo-relative files of the ``package`` modules a file imports; ``from package.mod import name`` counts ``name``
    where it is a submodule."""
    modules = set()
    for node in ast.walk(ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    files = set()
    for module in modules:
        if module.split(".")[0] != package:
            continue
        base = REPO_ROOT.joinpath(*module.split("."))
        for candidate in (base.with_suffix(".py"), base / "__init__.py"):
            if candidate.is_file():
                files.add(candidate.relative_to(REPO_ROOT).as_posix())
                break
    return files


def imported_module_files(paths: Iterable[str], package: str) -> tuple[str, ...]:
    """Repo-relative files of every ``package`` module ``paths`` import, directly or through another ``package``
    module."""
    files: set[str] = set()
    frontier = list(paths)
    while frontier:
        found = _imported_package_files(frontier.pop(), package) - files
        files |= found
        frontier.extend(found)
    return tuple(sorted(files))


def call_name(call: ast.Call) -> str | None:
    """``f`` in ``f()`` and in ``x.f()``; None for any other callee."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def called_names(function: Function) -> frozenset[str]:
    """Every name ``function`` calls through, its nested functions' calls included."""
    return frozenset(
        name for node in ast.walk(function) if isinstance(node, ast.Call) and (name := call_name(node)) is not None
    )


def _receiver(func: ast.Attribute) -> str:
    """``X`` in ``X.f()`` and ``a.X.f()``, ``super`` in ``super().f()``."""
    value = func.value.func if isinstance(func.value, ast.Call) else func.value
    return value.id if isinstance(value, ast.Name) else getattr(value, "attr", "")


def builds_a_model(call: ast.Call) -> bool:
    """Whether ``call`` builds a model: a load entry point, or a factory on anything but a non-model class."""
    name = call_name(call)
    if name in MODEL_LOAD_CALLS:
        return True
    return (
        name in MODEL_FACTORIES
        and isinstance(call.func, ast.Attribute)
        and _receiver(call.func) not in NON_MODEL_CLASSES
    )


def functions_in(paths: Iterable[str], *, top_level: bool = False) -> Iterator[tuple[str, Function]]:
    """``(repo-relative path, function)`` for every function under ``paths``, nested ones and methods included unless
    ``top_level``."""
    for root in paths:
        base = REPO_ROOT / root
        for path in [base] if base.is_file() else sorted(base.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            rel = path.relative_to(REPO_ROOT).as_posix()
            yield from (
                (rel, node) for node in (tree.body if top_level else ast.walk(tree)) if isinstance(node, Function)
            )


def functions_calling(
    paths: Iterable[str], predicate: Callable[[ast.Call], bool]
) -> dict[tuple[str, str], frozenset[str]]:
    """``(path, function name) -> called_names`` for every function under ``paths`` making a call ``predicate``
    accepts."""
    return {
        (rel, function.name): called_names(function)
        for rel, function in functions_in(paths)
        if any(isinstance(node, ast.Call) and predicate(node) for node in ast.walk(function))
    }
