#!/usr/bin/env python
"""CPU tests for the token-accuracy contract the toolkit's own Liger appliers must honour.

TRL's ``SFTTrainer.compute_loss`` sets ``return_token_accuracy=True`` on every forward whenever
``use_liger_kernel`` is on — the toolkit default — and reads ``outputs.token_accuracy`` back. An
applier that returns a plain ``MoeCausalLMOutputWithPast`` has no such field, so
``mean_token_accuracy`` is silently absent for the whole run and TRL warns once per step telling the
user to report it to the liger-kernel repository, which did not write the forward.

Run: ``pytest -m cpu tests/cpu/kernels/test_liger_token_accuracy.py``
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib

import pytest
from liger_kernel.transformers.model.output_classes import LigerMoeCausalLMOutputWithPast

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

LIGER_PACKAGE = REPO_ROOT / "src/kernels/liger"

# The toolkit's own FLCE forward — one for the whole roster (the families upstream Liger does not
# cover), so nobody else is holding it to this contract.
TOOLKIT_LCE_FORWARD = LIGER_PACKAGE / "lce_forward.py"


def test_the_output_class_actually_carries_the_field():
    """Premise check: if Liger renames it, every assertion below would pass vacuously."""
    fields = {f.name for f in dataclasses.fields(LigerMoeCausalLMOutputWithPast)}
    assert "token_accuracy" in fields, f"Liger's MoE output no longer carries token_accuracy: {sorted(fields)}"


def test_no_family_ships_a_second_fused_forward():
    """The contract below is enforced on ONE file, so a per-family forward would escape it.

    Reintroducing a bespoke ``lce_forward`` beside the generic one is exactly how the token-accuracy
    regression happened the first time: two forwards, one of them holding the contract.
    """
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
    """Computing it and dropping it on the floor is the failure this pins — deepseek_v4 unpacked the
    result into ``loss, _, _, _`` while the accuracy sat in slot 2."""
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
