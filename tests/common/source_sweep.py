"""AST sweeps over the repository's own source: which names a module imports and which functions make which calls.

The load-coverage tests pin that every function building a model also runs the repairs a run needs (the
run-dtype cast, ``finalize_loaded_model``). They walk the tree rather than a hand list, so a new function
is discovered by the call it makes, and read calls off the syntax, so a mention in a docstring or a
comment is not a call.
"""

import ast
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from halo.checkpoint.tools.offline_export import load_task_model
from halo.model_loading.checkpoint_coverage import from_pretrained_verified
from halo.model_loading.model_preparation import auto_load_model
from tests.common.repo import REPO_ROOT

# The toolkit's eager load entry points (the standalone tools' per-task load among them); a model build is
# one of them, a SentenceTransformer built from a checkpoint path, or a factory on any class but a tokenizer,
# processor or config one.
TOOLKIT_LOAD_CALLS = frozenset({from_pretrained_verified.__name__, auto_load_model.__name__, load_task_model.__name__})
MODEL_LOAD_CALLS = TOOLKIT_LOAD_CALLS | {"SentenceTransformer"}
MODEL_FACTORIES = frozenset({"from_pretrained", "from_config", "_from_config"})
NON_MODEL_CLASSES = frozenset({"AutoTokenizer", "AutoProcessor", "AutoConfig", "GenerationConfig", "PeftConfig"})

Function = ast.FunctionDef | ast.AsyncFunctionDef


def imports_name(path: Path, name: str) -> bool:
    """Whether the module at ``path`` imports ``name``.

    The binding is the check rather than a text match: CLAUDE.md bans function-local imports, so a
    module-level import is the only way a call site reaches a symbol, while a mention in a docstring
    or a comment is not a use. ``asname`` counts, since an aliased import binds the name too.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        isinstance(node, ast.ImportFrom | ast.Import)
        and any(alias.name == name or alias.asname == name for alias in node.names)
        for node in ast.walk(tree)
    )


def _imported_package_files(path: str, package: str) -> set[str]:
    """The repo-relative files of the ``package`` modules the ``.py`` file at ``path`` imports by absolute name;
    ``from package.mod import name`` counts ``name`` too where it is a submodule."""
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
    """The repo-relative files of every ``package`` module the ``.py`` files at ``paths`` import, directly or
    through another ``package`` module, sorted."""
    files: set[str] = set()
    frontier = list(paths)
    while frontier:
        found = _imported_package_files(frontier.pop(), package) - files
        files |= found
        frontier.extend(found)
    return tuple(sorted(files))


def call_name(call: ast.Call) -> str | None:
    """The name a call goes through: ``f`` in ``f()`` and in ``x.f()``; ``None`` for any other callee."""
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
    """The name a method is called on: ``X`` in ``X.f()`` and ``a.X.f()``, ``super`` in ``super().f()``."""
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
    """``(repo-relative path, function)`` for every function defined in the ``.py`` files under ``paths``
    (repo-relative files or directories), nested ones and methods included unless ``top_level``."""
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
    """``(repo-relative path, function name) -> called_names`` for every function under ``paths`` making a
    call ``predicate`` accepts."""
    return {
        (rel, function.name): called_names(function)
        for rel, function in functions_in(paths)
        if any(isinstance(node, ast.Call) and predicate(node) for node in ast.walk(function))
    }
