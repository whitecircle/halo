"""No first-party code reads a RoPE parameter as a flat config attribute.

transformers 5 keeps ``rope_theta``, ``partial_rotary_factor`` and the scaling spec in
``config.rope_parameters``, and from 5.20 it moves them there for remote-code configs written against
v4 as well, so ``config.partial_rotary_factor`` raises ``AttributeError`` on every config the toolkit
loads. The GPU scripts are never imported on CPU, so a flat read there surfaces only on a GPU run; this
sweep finds one statically in every Python file under ``halo/``, ``scripts/`` and ``tests/``.
"""

import ast

import pytest

from tests.common.repo import REPO_ROOT

FLAT_ROPE_ATTRIBUTES = frozenset({"rope_theta", "partial_rotary_factor", "rope_scaling"})
SCANNED_TREES = ("halo", "scripts", "tests")


def _flat_reads(path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        f"{path.relative_to(REPO_ROOT)}:{node.lineno}: .{node.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in FLAT_ROPE_ATTRIBUTES and isinstance(node.ctx, ast.Load)
    ]


def test_no_flat_rope_attribute_reads():
    assert all((REPO_ROOT / tree).is_dir() for tree in SCANNED_TREES), f"the sweep lost a root: {SCANNED_TREES}"
    reads = [
        read
        for tree in SCANNED_TREES
        for path in sorted((REPO_ROOT / tree).rglob("*.py"))
        for read in _flat_reads(path)
    ]
    assert not reads, "read these through config.rope_parameters:\n" + "\n".join(reads)


def test_the_sweep_sees_a_flat_read(tmp_path, monkeypatch):
    """Anti-vacuity: the parent tree's GPU-script spelling is caught."""
    monkeypatch.setattr("tests.cpu.conventions.test_flat_rope_attributes.REPO_ROOT", tmp_path)
    script = tmp_path / "probe.py"
    script.write_text("rotary_dim = int(config.head_dim * config.partial_rotary_factor)\n")
    assert _flat_reads(script) == ["probe.py:1: .partial_rotary_factor"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
