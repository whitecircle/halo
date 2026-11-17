"""A class's ``forward`` is replaced through ``patch_layer_forward`` only: transformers wraps a checkpointed layer's
``forward`` at class creation, so a bare assignment on such a class silently turns its gradient checkpointing off.
Instance patches (``backbone.forward = ...``) are not a class's forward."""

import ast

import pytest

from tests.common.repo import REPO_ROOT

_HELPER_MODULE = "halo/models/gradient_checkpointing.py"
# Names that bind one module or model instance at the patches under halo/.
_INSTANCE_NAMES = frozenset({"backbone", "model", "module", "linear"})


def _is_forward_target(target: ast.expr) -> bool:
    return (
        isinstance(target, ast.Attribute)
        and target.attr == "forward"
        and not (isinstance(target.value, ast.Name) and target.value.id in _INSTANCE_NAMES)
    )


def _bare_forward_assignments(tree: ast.Module) -> list[str]:
    """Each assignment of a non-instance ``.forward`` and each ``setattr(..., "forward", ...)``, as ``"line N"``."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(_is_forward_target(target) for target in targets):
                found.append(f"line {node.lineno}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) > 1
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "forward"
        ):
            found.append(f"line {node.lineno}")
    return found


def test_no_class_forward_is_assigned_outside_the_helper():
    offences = {}
    for path in sorted((REPO_ROOT / "halo").rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative != _HELPER_MODULE and (found := _bare_forward_assignments(ast.parse(path.read_text("utf-8")))):
            offences[relative] = found
    assert not offences, (
        f"replace a class's forward with patch_layer_forward (halo.models.gradient_checkpointing): {offences}"
    )


@pytest.mark.parametrize(
    "source",
    [
        "layer_cls.forward = forward",
        "owner.forward = forward",
        "family.sparse.forward = forward",
        "type(layer).forward = forward",
        "setattr(layer_cls, 'forward', forward)",
        "layer_cls.forward: Callable = forward",
    ],
)
def test_each_bare_class_forward_assignment_is_flagged(source):
    assert _bare_forward_assignments(ast.parse(source)) == ["line 1"]


@pytest.mark.parametrize(
    "source",
    ["backbone.forward = forward", "module.forward = functools.partial(forward, module)", "patch_layer_forward(c, f)"],
)
def test_instance_patches_and_the_helper_pass(source):
    assert _bare_forward_assignments(ast.parse(source)) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
