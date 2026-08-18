#!/usr/bin/env python
"""The merged EP save must write BASE-model key names, and must refuse to write unmerged adapters.

``merge_expert_lora_on_save`` routes a LoRA run's save through ``save_ep_model``, which walks a
PEFT-wrapped module tree. Three things have to happen to every key or the resulting checkpoint is
silently wrong rather than broken:

  * the ``base_model.model.`` prefix and ``.base_layer`` infix PEFT introduces must come off, or
    ``from_pretrained`` matches nothing and leaves those weights randomly initialised;
  * adapter tensors must be dropped — ``merged_adapters`` has already folded their delta into the
    base weight, so writing them too would apply it twice on a PEFT-aware reload;
  * a ``modules_to_save`` pair must collapse to the TRAINED copy, not the frozen ``original_module``
    one. Reachable under EP via ``lora_modules_to_save: [router]``, which the EP layer's own gather
    walks.

And the whole fold is opt-in, so ``save_ep_model`` refuses a PeftModel that arrives without it —
otherwise the gathered base weights carry no delta and the checkpoint is base-quality while looking
trained.

Run: pytest tests/cpu/checkpoint/test_merged_adapter_save_keys.py
"""

from __future__ import annotations

import pytest
from peft import LoraConfig, get_peft_model
from torch import nn

from src.distributed.expert_parallel.saving import _save_key_remap, save_ep_model

PEFT_PREFIX = "lora_"  # PeftModel.prefix for LoRA; read off the live model in production.

LAYER = "base_model.model.model.layers.0"


def _remap(*, cp: bool = False, peft: bool = True):
    return _save_key_remap(cp_key_remap=cp, peft_prefix=PEFT_PREFIX if peft else None)


class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.q_proj(x)


def test_lora_wrapped_base_weight_gets_its_plain_name():
    """``q_proj.base_layer.weight`` is what a LoRA-wrapped Linear calls its frozen weight."""
    assert _remap()(f"{LAYER}.self_attn.q_proj.base_layer.weight") == "model.layers.0.self_attn.q_proj.weight"


def test_adapter_tensors_are_dropped():
    """Their delta is already inside the base weight; writing them repeats it on a PEFT reload."""
    for suffix in ("lora_A.default.weight", "lora_B.default.weight"):
        assert _remap()(f"{LAYER}.self_attn.q_proj.{suffix}") is None


def test_modules_to_save_pair_collapses_to_the_trained_copy():
    """The frozen half drops; the trained half takes the base name — so exactly one key is written."""
    trained = _remap()(f"{LAYER}.mlp.gate.modules_to_save.default.weight")
    frozen = _remap()(f"{LAYER}.mlp.gate.original_module.weight")
    assert trained == "model.layers.0.mlp.gate.weight"
    assert frozen is None


def test_cp_and_peft_corrections_compose():
    """EP+CP+LoRA: the Ulysses wrapper level and the PEFT levels must both come off the same key."""
    key = f"{LAYER}.self_attn.original_attention.q_proj.base_layer.weight"
    assert _remap(cp=True)(key) == "model.layers.0.self_attn.q_proj.weight"


def test_non_peft_keys_are_untouched():
    """The non-adapter save path must be byte-for-byte unchanged by the PEFT seam."""
    remap = _remap(peft=False)
    assert remap("model.layers.0.mlp.experts.gate_up_proj") == "model.layers.0.mlp.experts.gate_up_proj"
    # `lora_` appears inside the native grouped adapter attrs; without a peft_prefix nothing is dropped.
    assert remap("model.layers.0.mlp.gate_up_proj_lora_A") == "model.layers.0.mlp.gate_up_proj_lora_A"


def test_cp_only_remap_is_unchanged_without_peft():
    assert _remap(cp=True, peft=False)("model.layers.0.self_attn.original_attention.q_proj.weight") == (
        "model.layers.0.self_attn.q_proj.weight"
    )


def test_save_ep_model_refuses_an_unmerged_peft_model(tmp_path):
    """Without the fold the gathered base carries no delta — a base-quality checkpoint that looks trained."""
    peft_model = get_peft_model(_TinyLM(), LoraConfig(target_modules=["q_proj"]))
    with pytest.raises(ValueError, match="adapters_merged"):
        save_ep_model(peft_model, str(tmp_path))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
