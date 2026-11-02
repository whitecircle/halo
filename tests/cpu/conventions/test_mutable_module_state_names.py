"""``UPPER_SNAKE`` names an immutable value only: no module under ``halo/`` binds an ``UPPER_SNAKE`` name to an empty
mutable container, the shape a registry, cache or warn-once set starts from (CLAUDE.md, Code style)."""

import ast

import pytest

from tests.common.repo import REPO_ROOT

# Constructors whose result is mutable whatever they are called with, and the empty-call ones.
_ALWAYS_MUTABLE_CALLS = frozenset({"defaultdict", "WeakKeyDictionary", "WeakValueDictionary", "WeakSet"})
_EMPTY_MUTABLE_CALLS = frozenset({"set", "dict", "list", "OrderedDict", "Counter", "deque"}) | _ALWAYS_MUTABLE_CALLS


def _is_upper_snake(name: str) -> bool:
    stem = name.lstrip("_")
    return bool(stem) and stem[0].isalpha() and stem == stem.upper()


def _starts_mutable_state(value: ast.expr) -> bool:
    """Whether ``value`` is an empty mutable container: ``{}``, ``[]``, an argument-free mutable constructor, or a
    constructor that is mutable whatever it is given."""
    if isinstance(value, ast.Dict):
        return not value.keys
    if isinstance(value, ast.List):
        return not value.elts
    if not isinstance(value, ast.Call):
        return False
    func = value.func
    name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
    if name in _ALWAYS_MUTABLE_CALLS:
        return True
    return name in _EMPTY_MUTABLE_CALLS and not value.args and not value.keywords


def _upper_snake_mutable_bindings(source: str) -> list[str]:
    """The module-level ``UPPER_SNAKE`` names ``source`` binds to an empty mutable container."""
    found = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        found += [
            t.id for t in targets if isinstance(t, ast.Name) and _is_upper_snake(t.id) and _starts_mutable_state(value)
        ]
    return found


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("_WARNED: set[str] = set()", ["_WARNED"]),
        ("REGISTRY = {}", ["REGISTRY"]),
        ("_SEEN = []", ["_SEEN"]),
        ("CACHE = collections.defaultdict(list)", ["CACHE"]),
        ("_warned: set[str] = set()", []),
        ("_LEVELS = (0.0, 0.5)", []),
        ("NAMES = frozenset({'a'})", []),
        ("TABLE = {'a': 1}", []),
        ("def f():\n    LOCAL = set()", []),
    ],
)
def test_detector_reads_module_level_empty_mutable_bindings(source, expected):
    assert _upper_snake_mutable_bindings(source) == expected


def test_no_upper_snake_name_holds_mutable_state():
    modules = sorted((REPO_ROOT / "halo").rglob("*.py"))
    assert len(modules) > 100, "the sweep lost the halo/ tree"
    offending = {
        str(module.relative_to(REPO_ROOT)): names
        for module in modules
        if (names := _upper_snake_mutable_bindings(module.read_text()))
    }
    assert not offending, (
        f"{offending}: UPPER_SNAKE is for immutable values. Name mutable module state in lowercase and reach it "
        "through a public object or function."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
