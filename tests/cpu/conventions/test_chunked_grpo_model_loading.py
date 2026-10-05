#!/usr/bin/env python
"""The chunked GRPO lifecycle loads identical Qwen3 norm implementations before and after training."""

import pytest

from tests.common.utils import probe_findings


def test_qk_norms_match_after_late_trainer_patch_and_reload():
    script = """
import tempfile

from accelerate import PartialState
from liger_kernel.transformers import _apply_liger_kernel_to_instance
from transformers import Qwen3Config, Qwen3ForCausalLM

from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.utils import load_script_module

PartialState()
suite = load_script_module("tests/gpu/trainers/grpo/test_offline_grpo_chunked.py")

def qk_forwards(model):
    targets = {
        name: module.forward.__func__
        for name, module in model.named_modules()
        if name.endswith((".self_attn.q_norm", ".self_attn.k_norm"))
    }
    assert targets, "the fixture must exercise Qwen3 attention q/k norms"
    return targets

with tempfile.TemporaryDirectory() as checkpoint:
    Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG)).save_pretrained(checkpoint)
    original = suite._load_model(checkpoint, attn_implementation="eager")
    _apply_liger_kernel_to_instance(original, cross_entropy=False, fused_linear_cross_entropy=False)
    restored = suite._load_model(checkpoint, attn_implementation="eager")
    before, after = qk_forwards(original), qk_forwards(restored)
    findings = ["target sets differ"] if before.keys() != after.keys() else []
    findings.extend(name for name in before.keys() & after.keys() if before[name] is not after[name])
    print("MISMATCHES:" + "|".join(findings))
"""
    assert not probe_findings(script, "MISMATCHES:")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
