"""Repo pin: every module in halo/ and scripts/ logs through one stdlib logger form.

The module logger is ``logger = logging.getLogger(__name__)``, or ``info_logger(__name__)`` where it needs
its INFO pin. accelerate's adapter drops every level off global rank 0 and raises before its state exists,
so a rank-0 line is guarded at its call with ``is_global_main_process()``, never by a logger keyword.
"""

import ast

import pytest

from tests.common.repo import REPO_ROOT

_SWEPT_ROOTS = ("halo", "scripts")
_FOREIGN_LOGGING_MODULES = frozenset({"accelerate.logging", "transformers.utils.logging"})
_LOGGER_FORMS = frozenset({"logging.getLogger(__name__)", "info_logger(__name__)"})
_LOGGER_FACTORIES = frozenset({"getLogger", "get_logger", "info_logger"})


def _is_logger_binding(node: ast.stmt) -> bool:
    """A module-level assignment to a ``*logger`` name or from a logger factory."""
    if not isinstance(node, ast.Assign):
        return False
    names = [target.id for target in node.targets if isinstance(target, ast.Name)]
    func = node.value.func if isinstance(node.value, ast.Call) else None
    factory = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    return factory in _LOGGER_FACTORIES or any(name.lower().endswith("logger") for name in names)


def _logger_form_offences(tree: ast.Module) -> list[str]:
    """Each break of the logger form in ``tree``, as ``"line N: ..."``."""
    offences = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported = {node.module, *(f"{node.module}.{alias.name}" for alias in node.names)}
        elif isinstance(node, ast.Import):
            imported = {alias.name for alias in node.names}
        else:
            imported = set()
        if imported & _FOREIGN_LOGGING_MODULES:
            offences.append(f"line {node.lineno}: imports {sorted(imported & _FOREIGN_LOGGING_MODULES)}")
        if isinstance(node, ast.Call) and any(keyword.arg == "main_process_only" for keyword in node.keywords):
            offences.append(f"line {node.lineno}: passes main_process_only=")
    for node in tree.body:
        if not _is_logger_binding(node):
            continue
        form = f"{ast.unparse(node.targets[0])} = {ast.unparse(node.value)}"
        if ast.unparse(node.targets[0]) != "logger" or ast.unparse(node.value) not in _LOGGER_FORMS:
            offences.append(f"line {node.lineno}: module logger {form}")
    return offences


def _sweep() -> tuple[dict[str, list[str]], int]:
    """Parse halo/ and scripts/ once: each offending file's offences, and how many files bind a module logger."""
    assert all((REPO_ROOT / root).is_dir() for root in _SWEPT_ROOTS), f"the sweep lost a root: {_SWEPT_ROOTS}"
    offences, bound = {}, 0
    for path in sorted(path for root in _SWEPT_ROOTS for path in (REPO_ROOT / root).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if found := _logger_form_offences(tree):
            offences[path.relative_to(REPO_ROOT).as_posix()] = found
        bound += any(_is_logger_binding(node) for node in tree.body)
    return offences, bound


_OFFENCES, _MODULE_LOGGER_COUNT = _sweep()


def test_every_module_uses_the_stdlib_logger_form():
    assert not _OFFENCES, (
        "bind `logger = logging.getLogger(__name__)` and guard a rank-0-only line with "
        f"`if is_global_main_process():` (halo.runtime.ranks): {_OFFENCES}"
    )


def test_the_sweep_sees_the_module_loggers():
    """Anti-vacuity: a collapsed sweep would pass the form check."""
    assert _MODULE_LOGGER_COUNT > 150, f"only {_MODULE_LOGGER_COUNT} module loggers found"


@pytest.mark.parametrize(
    ("source", "offence"),
    [
        ("from accelerate.logging import get_logger\n", "imports ['accelerate.logging']"),
        ("from transformers.utils import logging\n", "imports ['transformers.utils.logging']"),
        ("import accelerate.logging\n", "imports ['accelerate.logging']"),
        ("logger = logging.getLogger('halo')\n", "module logger logger = logging.getLogger('halo')"),
        ("log = logging.getLogger(__name__)\n", "module logger log = logging.getLogger(__name__)"),
        ("_tee_logger = logging.getLogger(__name__)\n", "module logger _tee_logger = logging.getLogger(__name__)"),
        ("logger.info('x', main_process_only=False)\n", "passes main_process_only="),
    ],
)
def test_each_break_of_the_logger_form_is_flagged(source, offence):
    assert _logger_form_offences(ast.parse(source)) == [f"line 1: {offence}"]


def test_the_stdlib_forms_pass():
    source = "import logging\n\nfrom halo.log import info_logger\n\nlogger = logging.getLogger(__name__)\n"
    assert _logger_form_offences(ast.parse(source)) == []
    assert _logger_form_offences(ast.parse("logger = info_logger(__name__)\n")) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
