#!/usr/bin/env python
"""The merged EP save writes base-model key names, or the checkpoint is silently wrong.
``lora_modules_to_save: [router]`` makes the trained-copy collapse reachable under EP."""

from __future__ import annotations

import pytest

from halo.checkpoint.parallel.ep import _save_key_remap

PEFT_PREFIX = "lora_"  # PeftModel.prefix for LoRA; read off the live model in production.

LAYER = "base_model.model.model.layers.0"


def _remap(*, cp: bool = False, peft: bool = True):
    return _save_key_remap(cp_key_remap=cp, peft_prefix=PEFT_PREFIX if peft else None)


def test_lora_wrapped_base_weight_gets_its_plain_name():
    assert _remap()(f"{LAYER}.self_attn.q_proj.base_layer.weight") == "model.layers.0.self_attn.q_proj.weight"


def test_adapter_tensors_are_dropped():
    """Their delta is in the base weight; writing them repeats it on a PEFT reload."""
    for suffix in ("lora_A.default.weight", "lora_B.default.weight"):
        assert _remap()(f"{LAYER}.self_attn.q_proj.{suffix}") is None


def test_modules_to_save_pair_collapses_to_the_trained_copy():
    """Exactly one key is written."""
    trained = _remap()(f"{LAYER}.mlp.gate.modules_to_save.default.weight")
    frozen = _remap()(f"{LAYER}.mlp.gate.original_module.weight")
    assert trained == "model.layers.0.mlp.gate.weight"
    assert frozen is None


def test_cp_and_peft_corrections_compose():
    key = f"{LAYER}.self_attn.original_attention.q_proj.base_layer.weight"
    assert _remap(cp=True)(key) == "model.layers.0.self_attn.q_proj.weight"


def test_non_peft_keys_are_untouched():
    remap = _remap(peft=False)
    assert remap("model.layers.0.mlp.experts.gate_up_proj") == "model.layers.0.mlp.experts.gate_up_proj"
    # `lora_` appears inside the native grouped adapter attrs; without a peft_prefix nothing is dropped.
    assert remap("model.layers.0.mlp.gate_up_proj_lora_A") == "model.layers.0.mlp.gate_up_proj_lora_A"


def test_cp_only_remap_is_unchanged_without_peft():
    assert _remap(cp=True, peft=False)("model.layers.0.self_attn.original_attention.q_proj.weight") == (
        "model.layers.0.self_attn.q_proj.weight"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
