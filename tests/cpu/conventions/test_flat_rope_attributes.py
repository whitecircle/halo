"""No first-party code reads a RoPE parameter as a flat config attribute. transformers 5 keeps them in
``config.rope_parameters`` (from 5.20 for v4 remote-code configs too), so a flat read raises ``AttributeError``."""

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
    monkeypatch.setattr("tests.cpu.conventions.test_flat_rope_attributes.REPO_ROOT", tmp_path)
    script = tmp_path / "probe.py"
    script.write_text("rotary_dim = int(config.head_dim * config.partial_rotary_factor)\n")
    assert _flat_reads(script) == ["probe.py:1: .partial_rotary_factor"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
