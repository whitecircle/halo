#!/usr/bin/env python
"""TRL's ``compute_loss`` reads ``outputs.token_accuracy``, which a plain ``MoeCausalLMOutputWithPast`` lacks, so
``mean_token_accuracy`` would be silently absent for the run."""

from __future__ import annotations

import ast
import dataclasses

import pytest
from liger_kernel.transformers.model.output_classes import LigerMoeCausalLMOutputWithPast

from tests.common.repo import REPO_ROOT

LIGER_PACKAGE = REPO_ROOT / "halo/model_loading/liger"

# One for the whole roster, so nobody else holds it to this contract.
TOOLKIT_LCE_FORWARD = LIGER_PACKAGE / "cross_entropy.py"


def test_the_output_class_actually_carries_the_field():
    """Premise: if Liger renames it, every assertion below passes vacuously."""
    fields = {f.name for f in dataclasses.fields(LigerMoeCausalLMOutputWithPast)}
    assert "token_accuracy" in fields, f"Liger's MoE output no longer carries token_accuracy: {sorted(fields)}"


def test_no_family_ships_a_second_fused_forward():
    """The contract is enforced on one file, so a per-family forward would escape it."""
    builders = {
        path.name
        for path in LIGER_PACKAGE.glob("*.py")
        if "LigerMoeCausalLMOutputWithPast(" in path.read_text(encoding="utf-8")
    }
    assert builders == {TOOLKIT_LCE_FORWARD.name}, (
        f"the fused causal-LM output is built in {sorted(builders)}; it belongs only in "
        f"{TOOLKIT_LCE_FORWARD.name}, where this file holds it to TRL's contract"
    )


def test_the_forward_returns_ligers_output_class():
    """A plain transformers output silently drops the field TRL reads."""
    source = ast.parse(TOOLKIT_LCE_FORWARD.read_text(encoding="utf-8"))
    returned = {
        node.value.func.id
        for node in ast.walk(source)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
    }
    assert "LigerMoeCausalLMOutputWithPast" in returned, (
        f"the fused forward returns {sorted(returned)}; TRL reads outputs.token_accuracy off this object"
    )
    assert "MoeCausalLMOutputWithPast" not in returned


def test_the_forward_forwards_the_accuracy_it_computed():
    """Unpacking into ``loss, _, _, _`` discards the accuracy in slot 2."""
    source = TOOLKIT_LCE_FORWARD.read_text(encoding="utf-8")
    tree = ast.parse(source)

    unpacks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "unpack_cross_entropy_result"
    ]
    assert unpacks, "the fused forward no longer routes through unpack_cross_entropy_result"
    for node in unpacks:
        names = [getattr(t, "id", None) for t in node.targets[0].elts]
        assert names[2] == "token_accuracy", (
            f"the fused forward discards slot 2 of unpack_cross_entropy_result (got {names}); that "
            f"slot IS the token accuracy TRL asked for"
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "LigerMoeCausalLMOutputWithPast":
            passed = {kw.arg for kw in node.keywords}
            assert "token_accuracy" in passed, f"the output is built without token_accuracy: {sorted(passed)}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
